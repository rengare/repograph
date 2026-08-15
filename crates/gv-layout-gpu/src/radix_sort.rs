//! A stable GPU radix sort, the foundation of the on-device octree build.
//!
//! See `shaders/radix_sort.wgsl` for why stability is a requirement rather than
//! a nicety, and how the three dispatches per digit fit together.

use anyhow::Result;
use gv_gpu::GpuContext;
use wgpu::util::DeviceExt;

/// Elements per workgroup.
///
/// Trades two things off. The scan is a single workgroup walking
/// `256 * group_count` entries, so a small tile — many groups — makes that pass
/// long. The scatter runs one thread per group, so a large tile makes *that*
/// pass long.
///
/// Measured at 100k keys: 128 → 4.80 ms, 256 → 3.97, 512 → 3.70, 1024 → 3.72.
/// Flat across the top, which says the scatter's serial length is *not* what
/// binds. Sorting 10k keys costs 0.99 ms against 100k's 3.62, so roughly 1 ms
/// of it is fixed cost — twelve dispatches at about 80 µs of launch overhead
/// each — and only the remainder scales. Cutting that floor means fewer passes
/// (10-bit digits would need three rather than four) or a properly parallel
/// scatter; retuning this constant will not do it.
const TILE: u32 = 512;

const RADIX: u32 = 256;
const DIGIT_BITS: u32 = 8;
const PASSES: u32 = u32::BITS / DIGIT_BITS;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    count: u32,
    shift: u32,
    tile: u32,
    group_count: u32,
}

/// Sorts `(key, value)` pairs by key, in place on the GPU.
///
/// Holds both halves of the ping-pong internally, so callers hand it a buffer
/// and get that same buffer back sorted.
pub struct RadixSort {
    histogram: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,

    uniforms: wgpu::Buffer,
    histograms: wgpu::Buffer,
    /// The scratch half of the ping-pong.
    scratch_keys: wgpu::Buffer,
    scratch_values: wgpu::Buffer,

    capacity: u32,
    group_count: u32,
}

impl RadixSort {
    pub fn new(context: &GpuContext, capacity: u32) -> Result<Self> {
        let device = &context.device;
        let group_count = capacity.div_ceil(TILE).max(1);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radix sort"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/radix_sort.wgsl").into()),
        });

        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radix sort"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                entry(1, true),
                entry(2, true),
                entry(3, false),
                entry(4, false),
                entry(5, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("radix sort"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        // One uniform block per pass, at dynamic offsets: the passes are all
        // recorded into one encoder before any of them runs, so a single block
        // rewritten between them would give every pass the last one's shift.
        let stride = uniform_stride(context);
        let mut blocks = vec![0u8; (stride * PASSES) as usize];
        for pass in 0..PASSES {
            let uniforms = Uniforms {
                count: capacity,
                shift: pass * DIGIT_BITS,
                tile: TILE,
                group_count,
            };
            let start = (pass * stride) as usize;
            blocks[start..start + size_of::<Uniforms>()]
                .copy_from_slice(bytemuck::bytes_of(&uniforms));
        }

        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radix sort uniforms"),
            contents: &blocks,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scratch = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(capacity.max(1)) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        Ok(Self {
            histogram: pipeline("histogram"),
            scan: pipeline("scan"),
            scatter: pipeline("scatter"),
            layout,
            uniforms,
            histograms: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("radix sort histograms"),
                size: u64::from(RADIX * group_count) * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            scratch_keys: scratch("radix sort scratch keys"),
            scratch_values: scratch("radix sort scratch values"),
            capacity,
            group_count,
        })
    }

    /// Records the whole sort. `keys` and `values` are sorted in place.
    ///
    /// An even number of digit passes means the ping-pong lands back in the
    /// caller's buffers, so no copy is needed to finish.
    pub fn record(
        &self,
        context: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
    ) {
        debug_assert_eq!(PASSES % 2, 0, "an odd pass count would end in the scratch buffers");

        let stride = uniform_stride(context);

        for pass in 0..PASSES {
            // Ping-pong: even passes read the caller's buffers, odd passes read
            // the scratch they were written into.
            let (source_keys, source_values, target_keys, target_values) = if pass % 2 == 0 {
                (keys, values, &self.scratch_keys, &self.scratch_values)
            } else {
                (&self.scratch_keys, &self.scratch_values, keys, values)
            };

            let bind_group = context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("radix sort"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.uniforms,
                                offset: 0,
                                size: wgpu::BufferSize::new(size_of::<Uniforms>() as u64),
                            }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: source_keys.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: source_values.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: target_keys.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: target_values.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: self.histograms.as_entire_binding() },
                    ],
                });

            let offset = pass * stride;

            // A pass per dispatch, never batched: each stage consumes what the
            // last one wrote to the histogram buffer, and dispatches inside one
            // compute pass are not ordered against each other.
            for (label, pipeline, groups) in [
                ("radix histogram", &self.histogram, self.group_count),
                ("radix scan", &self.scan, 1),
                ("radix scatter", &self.scatter, self.group_count),
            ] {
                let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                compute.set_pipeline(pipeline);
                compute.set_bind_group(0, &bind_group, &[offset]);
                compute.dispatch_workgroups(groups, 1, 1);
            }
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

/// Uniform blocks have to start at a multiple of the adapter's minimum dynamic
/// offset alignment, which is 256 on most hardware and never smaller than the
/// struct here.
fn uniform_stride(context: &GpuContext) -> u32 {
    let alignment = context.limits().min_uniform_buffer_offset_alignment;
    (size_of::<Uniforms>() as u32).div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sorts on the GPU and reads the result back.
    fn sort(pairs: &[(u32, u32)]) -> Vec<(u32, u32)> {
        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        let device = &context.device;

        let keys: Vec<u32> = pairs.iter().map(|(key, _)| *key).collect();
        let values: Vec<u32> = pairs.iter().map(|(_, value)| *value).collect();

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let key_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&keys),
            usage,
        });
        let value_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&values),
            usage,
        });

        let sorter = RadixSort::new(&context, pairs.len() as u32).expect("pipelines build");
        let mut encoder = device.create_command_encoder(&Default::default());
        sorter.record(&context, &mut encoder, &key_buffer, &value_buffer);
        context.queue.submit([encoder.finish()]);

        let keys = read_back(&context, &key_buffer, pairs.len());
        let values = read_back(&context, &value_buffer, pairs.len());
        keys.into_iter().zip(values).collect()
    }

    fn read_back(context: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
        let bytes = (count * 4) as u64;
        let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = context.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        context.queue.submit([encoder.finish()]);

        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        receiver.recv().expect("map").expect("mapped");

        let view = staging.slice(..).get_mapped_range();
        let result = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        staging.unmap();
        result
    }

    /// Deterministic pseudo-random keys.
    fn noise(count: usize, mask: u32) -> Vec<(u32, u32)> {
        let mut state = 0x2545F491_4F6CDD1Du64;
        (0..count)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 32) as u32 & mask, index as u32)
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn sorts_the_full_key_range() {
        let input = noise(5000, u32::MAX);
        let mut expected = input.clone();
        expected.sort_by_key(|(key, _)| *key);

        assert_eq!(sort(&input), expected);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn is_stable_across_duplicate_keys() {
        // The property the octree build depends on, and the one an atomic-
        // counter scatter would not give: with only 16 distinct keys across
        // 4000 elements, every key has hundreds of duplicates, and their
        // payloads must come out in input order.
        let input = noise(4000, 0xF);
        let mut expected = input.clone();
        expected.sort_by_key(|(key, _)| *key);

        let actual = sort(&input);
        assert_eq!(actual, expected, "equal keys were reordered");

        // Spelled out rather than left to `sort_by_key`'s own stability.
        for window in actual.windows(2) {
            let ((key, value), (next_key, next_value)) = (window[0], window[1]);
            assert!(key <= next_key, "not sorted: {key} then {next_key}");
            if key == next_key {
                assert!(value < next_value, "unstable: {value} then {next_value}");
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn sorts_a_graph_sized_input() {
        // More than one workgroup's tile, and not a multiple of it either.
        let input = noise(100_003, u32::MAX);
        let mut expected = input.clone();
        expected.sort_by_key(|(key, _)| *key);

        assert_eq!(sort(&input), expected);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn an_already_sorted_input_is_unchanged() {
        let input: Vec<(u32, u32)> = (0..3000).map(|i| (i * 7, i)).collect();
        assert_eq!(sort(&input), input);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn a_reversed_input_comes_out_forwards() {
        let input: Vec<(u32, u32)> = (0..3000).map(|i| (3000 - i, i)).collect();
        let mut expected = input.clone();
        expected.sort_by_key(|(key, _)| *key);

        assert_eq!(sort(&input), expected);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn every_key_identical_preserves_the_payload_order() {
        let input: Vec<(u32, u32)> = (0..2048).map(|i| (42, i)).collect();
        assert_eq!(sort(&input), input);
    }

    #[test]
    #[ignore = "benchmark; requires a GPU adapter"]
    fn stress_100k() {
        use std::time::Instant;

        const COUNT: u32 = 100_000;
        const STEPS: u32 = 50;

        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        let input = noise(COUNT as usize, u32::MAX);
        let keys: Vec<u32> = input.iter().map(|(key, _)| *key).collect();
        let values: Vec<u32> = input.iter().map(|(_, value)| *value).collect();

        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let key_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&keys),
                usage,
            });
        let value_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&values),
                usage,
            });

        let sorter = RadixSort::new(&context, COUNT).expect("pipelines build");

        let mut warm = context.device.create_command_encoder(&Default::default());
        sorter.record(&context, &mut warm, &key_buffer, &value_buffer);
        context.queue.submit([warm.finish()]);
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        let started = Instant::now();
        for _ in 0..STEPS {
            let mut encoder = context.device.create_command_encoder(&Default::default());
            sorter.record(&context, &mut encoder, &key_buffer, &value_buffer);
            context.queue.submit([encoder.finish()]);
        }
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        let per_sort = started.elapsed().as_secs_f64() * 1000.0 / f64::from(STEPS);
        println!("radix sort over {COUNT} keys: {per_sort:.3} ms");
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn a_single_element_sorts() {
        assert_eq!(sort(&[(7, 0)]), vec![(7, 0)]);
    }
}
