//! Barnes-Hut with the octree built on the device.
//!
//! The CPU tree in `gv_layout::octree` stays the oracle: same flattened cells,
//! same escape-index walk, same opening criterion. What differs is that nothing
//! here can recurse or allocate, so each stage of the build is a compute pass
//! over an array.
//!
//! Pipeline, in order:
//!
//! 1. `clear_bounds` / `reduce_bounds` — the root cube, by atomic min/max.
//! 2. `morton` — one code per body, x in the low bit of each octant digit.
//! 3. [`RadixSort`] — bodies into Morton order, stably.
//! 4. `enumerate` — one node per distinct subtree range, appended by atomic.
//! 5. [`RadixSort`] again — nodes into depth-first order, by a key unique to
//!    each, which is what makes the atomic append above reproducible.
//! 6. `link` — escape indices and cell widths.
//! 7. `centres` — centres of mass, one dispatch per level, deepest first.
//! 8. `repulsive` — the stackless walk, in place of `fr.wgsl`'s O(n²) loop.
//!
//! Attraction and the position update are then `fr.wgsl`'s, unchanged.

use anyhow::Result;
use gv_gpu::{GpuContext, GraphBuffers};
use gv_layout::LayoutParams;

use crate::radix_sort::RadixSort;
use crate::{FrGpuLayout, GpuLayout};

/// Bits per axis in a Morton code, and so levels in the tree.
///
/// Ten rather than the CPU's twenty-one: WGSL has no `u64`, and 30 bits keeps
/// the code in a `u32` so the four-pass radix sort applies unchanged. The lost
/// resolution costs nothing because a leaf holds a body *range* — bodies
/// sharing a code are resolved by iterating them, not by aggregation.
pub const LEVELS: u32 = 10;

const WORKGROUP: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    node_count: u32,
    three_d: u32,
    levels: u32,
    _pad: u32,
}

/// The bounding-cube and Morton-code stages, and the buffers they fill.
pub struct Prepare {
    clear_bounds: wgpu::ComputePipeline,
    reduce_bounds: wgpu::ComputePipeline,
    morton: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    uniforms: wgpu::Buffer,
    pub bounds: wgpu::Buffer,
    /// Morton code per body, sorted in place by [`RadixSort`].
    pub codes: wgpu::Buffer,
    /// Body indices, permuted alongside `codes` into Morton order.
    pub order: wgpu::Buffer,

    sort: RadixSort,
    node_count: u32,
}

impl Prepare {
    pub fn new(context: &GpuContext, buffers: &GraphBuffers, three_d: bool) -> Result<Self> {
        let device = &context.device;
        let node_count = buffers.node_count.max(1);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bh prepare"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bh_prepare.wgsl").into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bh uniforms"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context.queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&Uniforms {
                node_count: buffers.node_count,
                three_d: u32::from(three_d),
                levels: LEVELS,
                _pad: 0,
            }),
        );

        let scratch = |label: &str, elements: u32| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(elements) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let bounds = scratch("bh bounds", 6);
        let codes = scratch("bh codes", node_count);
        let order = scratch("bh order", node_count);

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
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
            label: Some("bh prepare"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, false),
                storage(4, false),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bh prepare"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffers.nodes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bounds.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: codes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: order.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bh prepare"),
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

        Ok(Self {
            clear_bounds: pipeline("clear_bounds"),
            reduce_bounds: pipeline("reduce_bounds"),
            morton: pipeline("morton"),
            bind_group,
            uniforms,
            bounds,
            codes,
            order,
            sort: RadixSort::new(context, node_count)?,
            node_count: buffers.node_count,
        })
    }

    /// Records the cube, the codes, and the sort that puts bodies in Morton
    /// order.
    pub fn record(&self, context: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
        if self.node_count == 0 {
            return;
        }

        let groups = self.node_count.div_ceil(WORKGROUP);

        // A pass per stage: the cube has to be complete before any code is
        // quantised against it, and dispatches inside one compute pass are not
        // ordered against each other.
        for (label, pipeline, count) in [
            ("bh clear bounds", &self.clear_bounds, 1),
            ("bh reduce bounds", &self.reduce_bounds, groups),
            ("bh morton", &self.morton, groups),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(count, 1, 1);
        }

        self.sort
            .record(context, encoder, &self.codes, &self.order);
    }

    /// Rewrites the uniforms, for a 2D/3D switch between steps.
    pub fn set_three_d(&self, context: &GpuContext, three_d: bool) {
        context.queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&Uniforms {
                node_count: self.node_count,
                three_d: u32::from(three_d),
                levels: LEVELS,
                _pad: 0,
            }),
        );
    }
}

/// One cell of the flattened tree, mirroring `bh_tree.wgsl`'s `Cell`.
///
/// Deliberately *not* `gv_layout::octree::Cell`. That one carries a `body`
/// index and a `_pad`; this carries the body *range* `first..=last`, because a
/// 10-bit code is coarse enough that distinct bodies can collide in one and a
/// leaf has to resolve them by iteration rather than by aggregation. Same 32
/// bytes, same escape-index walk, different leaf contract — see the header of
/// `shaders/bh_tree.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCell {
    /// Centre of mass of every body beneath this cell.
    pub center: [f32; 3],
    /// Number of bodies beneath this cell.
    pub mass: f32,
    /// Width of the cell's cube, which the opening criterion compares against
    /// distance.
    pub width: f32,
    /// Next cell to visit when this subtree is skipped.
    pub escape: u32,
    /// The body range this cell covers, as indices into [`Prepare::order`].
    pub first: u32,
    pub last: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TreeUniforms {
    node_count: u32,
    cell_capacity: u32,
    levels: u32,
    level: u32,
    three_d: u32,
    _pad: [u32; 3],
}

/// The tree build: enumeration, depth-first ordering, escapes and masses.
pub struct Tree {
    context: GpuContext,

    clear: wgpu::ComputePipeline,
    enumerate: wgpu::ComputePipeline,
    link: wgpu::ComputePipeline,
    centres: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    uniforms: wgpu::Buffer,
    /// `first * 16 + level` per node, sorted into depth-first order.
    keys: wgpu::Buffer,
    /// `last` per node, carried through that sort.
    values: wgpu::Buffer,
    /// The flattened tree, [`GpuCell`] each.
    pub cells: wgpu::Buffer,
    /// How many of `cells` the build filled.
    pub counter: wgpu::Buffer,

    sort: RadixSort,
    node_count: u32,
    /// Slots allocated for cells. See [`Self::capacity_for`].
    capacity: u32,
    uniform_stride: u32,
}

impl Tree {
    /// Cells to allocate for `node_count` bodies.
    ///
    /// The emitted ranges form a laminar family — every subtree is a contiguous
    /// run, and path compression means a node is emitted only where its range
    /// first appears, so every internal node splits into at least two children.
    /// A laminar family over `n` elements with that property has at most
    /// `2n - 1` members, and the root is always one of them. The extra slot is
    /// slack, not necessity: `enumerate` drops rather than writes past the end,
    /// and [`Self::read_cell_count`] is what would report the overflow.
    pub fn capacity_for(node_count: u32) -> u32 {
        2 * node_count.max(1) + 1
    }

    pub fn new(
        context: &GpuContext,
        buffers: &GraphBuffers,
        prepare: &Prepare,
        three_d: bool,
    ) -> Result<Self> {
        let device = &context.device;
        let node_count = buffers.node_count;
        let capacity = Self::capacity_for(node_count);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bh tree"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bh_tree.wgsl").into()),
        });

        // One uniform block per level, at dynamic offsets: `centres` runs once
        // per level and every dispatch is recorded before any of them runs, so
        // a single block rewritten between them would give every dispatch the
        // last one's level. The same trick, for the same reason, as
        // [`RadixSort`]'s per-digit shift.
        let uniform_stride = uniform_stride::<TreeUniforms>(context);
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bh tree uniforms"),
            size: u64::from(uniform_stride) * u64::from(LEVELS + 1),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scratch = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let keys = scratch("bh tree keys", u64::from(capacity) * 4);
        let values = scratch("bh tree values", u64::from(capacity) * 4);
        let cells = scratch("bh tree cells", u64::from(capacity) * size_of::<GpuCell>() as u64);
        let counter = scratch("bh tree counter", 4);

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
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
            label: Some("bh tree"),
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
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, true),
                storage(5, false),
                storage(6, false),
                storage(7, false),
                storage(8, false),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bh tree"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniforms,
                        offset: 0,
                        size: wgpu::BufferSize::new(size_of::<TreeUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: buffers.nodes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: prepare.bounds.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: prepare.codes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: prepare.order.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: keys.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: values.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: counter.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bh tree"),
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

        let tree = Self {
            context: context.clone(),
            clear: pipeline("clear"),
            enumerate: pipeline("enumerate"),
            link: pipeline("link"),
            centres: pipeline("centres"),
            bind_group,
            uniforms,
            keys,
            values,
            cells,
            counter,
            sort: RadixSort::new(context, capacity)?,
            node_count,
            capacity,
            uniform_stride,
        };
        tree.set_three_d(&context.queue, three_d);
        Ok(tree)
    }

    /// Rewrites every level's uniform block, for a 2D/3D switch between steps.
    pub fn set_three_d(&self, queue: &wgpu::Queue, three_d: bool) {
        let mut blocks = vec![0u8; (self.uniform_stride * (LEVELS + 1)) as usize];
        for level in 0..=LEVELS {
            let uniforms = TreeUniforms {
                node_count: self.node_count,
                cell_capacity: self.capacity,
                levels: LEVELS,
                level,
                three_d: u32::from(three_d),
                _pad: [0; 3],
            };
            let start = (level * self.uniform_stride) as usize;
            blocks[start..start + size_of::<TreeUniforms>()]
                .copy_from_slice(bytemuck::bytes_of(&uniforms));
        }
        queue.write_buffer(&self.uniforms, 0, &blocks);
    }

    /// Records the build. [`Prepare::record`] must already have run this step.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.node_count == 0 {
            return;
        }

        let over_cells = self.capacity.div_ceil(WORKGROUP);
        let over_bodies = self.node_count.div_ceil(WORKGROUP);

        // A pass per dispatch throughout. Every stage here consumes what the
        // previous one wrote to the *same* buffer in the *same* usage, so wgpu's
        // tracker sees no state transition and emits no barrier inside a single
        // compute pass — the phase 3 bug, which was invisible until the graph
        // outgrew one workgroup.
        self.pass(encoder, "bh clear", &self.clear, over_cells, 1, 0);
        self.pass(encoder, "bh enumerate", &self.enumerate, over_bodies, LEVELS + 1, 0);

        // Depth-first order, by a key unique to each node. This is what makes
        // the atomic append in `enumerate` reproducible.
        self.sort
            .record(&self.context, encoder, &self.keys, &self.values);

        self.pass(encoder, "bh link", &self.link, over_cells, 1, 0);

        // Deepest level first, so every child is summed before its parent.
        for level in (0..=LEVELS).rev() {
            self.pass(
                encoder,
                "bh centres",
                &self.centres,
                over_cells,
                1,
                level * self.uniform_stride,
            );
        }
    }

    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        groups_x: u32,
        groups_y: u32,
        uniform_offset: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[uniform_offset]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WalkUniforms {
    node_count: u32,
    three_d: u32,
    _pad0: [u32; 2],
    k: f32,
    theta: f32,
    _pad1: [f32; 2],
}

/// Fruchterman-Reingold with the repulsion term approximated by an octree the
/// device builds and walks itself.
///
/// The node array never reaches host memory: positions go in as buffer
/// contents, the cube, the codes, the sort, the tree and the walk all read and
/// write buffers, and what comes out is the next step's positions. That is the
/// whole point — the CPU tree in `gv_layout::barnes_hut` is fast because it
/// wins on complexity, but it still pays an upload of the node array every
/// step, and its walk is the part that belongs on a device.
pub struct BhGpuLayout {
    /// Opening angle. Smaller is more accurate and slower; 0.5 is the usual
    /// default, 0 degrades to brute force — and, because a leaf here iterates
    /// its bodies rather than aggregating them, degrades to it *exactly*.
    pub theta: f32,

    context: GpuContext,
    prepare: Prepare,
    tree: Tree,

    repulsive: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,

    /// Attraction and the position update, unchanged from the exact GPU path.
    fr: FrGpuLayout,
    node_count: u32,
}

/// The opening angle every Barnes-Hut path in this workspace defaults to.
pub const DEFAULT_THETA: f32 = 0.5;

impl BhGpuLayout {
    /// `three_d` is not a constructor argument: [`Self::record_step`] writes it
    /// from [`LayoutParams`] every step, so the picker can build the layout
    /// before anyone has chosen a dimension.
    pub fn new(context: &GpuContext, buffers: &GraphBuffers) -> Result<Self> {
        let device = &context.device;

        let prepare = Prepare::new(context, buffers, false)?;
        let tree = Tree::new(context, buffers, &prepare, false)?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bh walk"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bh_walk.wgsl").into()),
        });

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bh walk uniforms"),
            size: size_of::<WalkUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
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
            label: Some("bh walk"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, false),
                storage(2, true),
                storage(3, true),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bh walk"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffers.nodes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tree.cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: prepare.order.as_entire_binding() },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bh walk"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        Ok(Self {
            theta: DEFAULT_THETA,
            context: context.clone(),
            prepare,
            tree,
            repulsive: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bh repulsive"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("repulsive"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group,
            uniforms,
            fr: FrGpuLayout::new(context, buffers)?,
            node_count: buffers.node_count,
        })
    }

    /// Cells the last recorded build actually emitted.
    ///
    /// A device read, so it stalls; it is for tests and diagnostics, not the
    /// frame path. The walk finds the count on the device instead, out of the
    /// root's escape index.
    pub async fn read_cell_count(&self) -> Result<u32> {
        read_u32(&self.context, &self.tree.counter).await
    }

    /// The tree as the device built it, for validation against the CPU oracle.
    pub async fn read_cells(&self) -> Result<Vec<GpuCell>> {
        let count = self.read_cell_count().await?.min(self.tree.capacity);
        let bytes = u64::from(count) * size_of::<GpuCell>() as u64;
        let raw = read_bytes(&self.context, &self.tree.cells, bytes).await?;
        Ok(bytemuck::cast_slice::<u8, GpuCell>(&raw).to_vec())
    }

    /// Bodies in Morton order — the permutation a cell's `first..=last` indexes
    /// into. Also a device read, so also not for the frame path.
    pub async fn read_order(&self) -> Result<Vec<u32>> {
        let bytes = u64::from(self.node_count) * 4;
        let raw = read_bytes(&self.context, &self.prepare.order, bytes).await?;
        Ok(bytemuck::cast_slice::<u8, u32>(&raw).to_vec())
    }
}

impl GpuLayout for BhGpuLayout {
    fn name(&self) -> &'static str {
        "F-R gpu barnes-hut"
    }

    fn record_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        params: &LayoutParams,
    ) -> Result<()> {
        if self.node_count == 0 {
            return Ok(());
        }

        self.prepare.set_three_d(&self.context, params.three_d);
        self.tree.set_three_d(queue, params.three_d);
        queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&WalkUniforms {
                node_count: self.node_count,
                three_d: u32::from(params.three_d),
                _pad0: [0; 2],
                k: params.k(self.node_count as usize),
                theta: self.theta,
                _pad1: [0.0; 2],
            }),
        );
        self.fr.write_uniforms(queue, params);

        // The tree is rebuilt every step, from the positions the previous step
        // left in the node buffer. Nothing is cached across steps: the layout
        // moves every body, so a tree that survived one would be a tree of
        // where the graph used to be.
        self.prepare.record(&self.context, encoder);
        self.tree.record(encoder);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bh repulsive"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.repulsive);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.node_count.div_ceil(WORKGROUP), 1, 1);
        }

        self.fr.record_after_repulsion(encoder);
        Ok(())
    }
}

/// Uniform blocks have to start at a multiple of the adapter's minimum dynamic
/// offset alignment, which is 256 on most hardware and never smaller than the
/// structs here.
fn uniform_stride<T>(context: &GpuContext) -> u32 {
    let alignment = context.limits().min_uniform_buffer_offset_alignment;
    (size_of::<T>() as u32).div_ceil(alignment) * alignment
}

async fn read_bytes(context: &GpuContext, buffer: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bh readback"),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes.max(4));
    context.queue.submit([encoder.finish()]);

    let (sender, receiver) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    context.device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv()??;

    let view = staging.slice(..).get_mapped_range();
    let out = view[..bytes as usize].to_vec();
    drop(view);
    staging.unmap();
    Ok(out)
}

async fn read_u32(context: &GpuContext, buffer: &wgpu::Buffer) -> Result<u32> {
    let bytes = read_bytes(context, buffer, 4).await?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// The CPU mirror of the shader's Morton code, for validation.
///
/// Deliberately a separate implementation from `gv_layout::octree`, which uses
/// 21 bits in a `u64`; this is the 10-bit form the shader computes.
pub fn morton_10(position: [f32; 3], center: [f32; 3], half: f32) -> u32 {
    fn spread(value: u32) -> u32 {
        let mut x = value & 0x3FF;
        x = (x | (x << 16)) & 0x030000FF;
        x = (x | (x << 8)) & 0x0300F00F;
        x = (x | (x << 4)) & 0x030C30C3;
        x = (x | (x << 2)) & 0x09249249;
        x
    }

    let scale = (1u32 << LEVELS) as f32;
    let quantise = |axis: usize| {
        let normalised = (position[axis] - (center[axis] - half)) / (2.0 * half);
        (normalised * scale).clamp(0.0, scale - 1.0) as u32
    };

    spread(quantise(0)) | spread(quantise(1)) << 1 | spread(quantise(2)) << 2
}

/// Centre and half-width of the cube the shader derives, for validation.
pub fn bounding_cube(bodies: &[[f32; 3]]) -> ([f32; 3], f32) {
    let mut low = bodies[0];
    let mut high = bodies[0];
    for body in bodies {
        for axis in 0..3 {
            low[axis] = low[axis].min(body[axis]);
            high[axis] = high[axis].max(body[axis]);
        }
    }

    let center = [
        (low[0] + high[0]) * 0.5,
        (low[1] + high[1]) * 0.5,
        (low[2] + high[2]) * 0.5,
    ];
    let extent = (0..3)
        .map(|axis| high[axis] - low[axis])
        .fold(0.0f32, f32::max);
    let half = if extent > 0.0 { extent * 0.5 * 1.001 } else { 1.0 };

    (center, half)
}

#[cfg(test)]
mod tree_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use gv_graph::GraphData;

    struct Prepared {
        codes: Vec<u32>,
        order: Vec<u32>,
    }

    fn prepare(graph: &GraphData, three_d: bool) -> Prepared {
        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        let buffers = GraphBuffers::upload(&context, graph).expect("upload");
        let prepare = Prepare::new(&context, &buffers, three_d).expect("pipelines build");

        let mut encoder = context.device.create_command_encoder(&Default::default());
        prepare.record(&context, &mut encoder);
        context.queue.submit([encoder.finish()]);

        let count = graph.node_count();
        Prepared {
            codes: read_back(&context, &prepare.codes, count),
            order: read_back(&context, &prepare.order, count),
        }
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

    fn bodies_of(graph: &GraphData, three_d: bool) -> Vec<[f32; 3]> {
        graph
            .nodes
            .iter()
            .map(|node| {
                let p = node.position;
                [p[0], p[1], if three_d { p[2] } else { 0.0 }]
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn the_device_computes_the_same_codes_as_the_host() {
        // Validates the bounding-cube reduction and the code together: the cube
        // feeds the quantisation, so a wrong bound shows up as wrong codes.
        let graph = gv_graph::testing::path(4096);
        let bodies = bodies_of(&graph, true);
        let (center, half) = bounding_cube(&bodies);

        let prepared = prepare(&graph, true);

        let mut expected: Vec<(u32, u32)> = bodies
            .iter()
            .enumerate()
            .map(|(body, position)| (morton_10(*position, center, half), body as u32))
            .collect();
        expected.sort_by_key(|(code, _)| *code);

        let actual: Vec<(u32, u32)> = prepared
            .codes
            .iter()
            .copied()
            .zip(prepared.order.iter().copied())
            .collect();

        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn codes_come_back_sorted_and_order_is_a_permutation() {
        let graph = gv_graph::testing::path(10_000);
        let prepared = prepare(&graph, true);

        assert!(
            prepared.codes.windows(2).all(|w| w[0] <= w[1]),
            "codes are not in Morton order"
        );

        let mut seen = prepared.order.clone();
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..graph.node_count() as u32).collect::<Vec<_>>(),
            "order lost or duplicated a body"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn two_d_flattens_the_z_lane_before_the_cube_is_taken() {
        // The 3D fixture carries a spread of z. In 2D every code must land in
        // the low half of the z axis, since every body sits at z = 0 and the
        // cube is centred on it.
        let graph = gv_graph::testing::path(2048);
        let bodies = bodies_of(&graph, false);
        let (center, half) = bounding_cube(&bodies);

        let prepared = prepare(&graph, false);

        let mut expected: Vec<u32> = bodies
            .iter()
            .map(|position| morton_10(*position, center, half))
            .collect();
        expected.sort_unstable();

        assert_eq!(prepared.codes, expected);
    }

    #[test]
    #[ignore = "benchmark; requires a GPU adapter"]
    fn stress_100k() {
        use std::time::Instant;

        const BODIES: usize = 100_000;
        const STEPS: u32 = 50;

        let graph = gv_graph::testing::path(BODIES);
        let context = pollster::block_on(GpuContext::new(None)).expect("adapter available");
        let buffers = GraphBuffers::upload(&context, &graph).expect("upload");
        let prepare = Prepare::new(&context, &buffers, true).expect("pipelines build");

        // One untimed run first: the first submission pays for pipeline
        // compilation and buffer residency, which is not what is being measured.
        let mut encoder = context.device.create_command_encoder(&Default::default());
        prepare.record(&context, &mut encoder);
        context.queue.submit([encoder.finish()]);
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");

        let started = Instant::now();
        for _ in 0..STEPS {
            let mut encoder = context.device.create_command_encoder(&Default::default());
            prepare.record(&context, &mut encoder);
            context.queue.submit([encoder.finish()]);
        }
        // The clock stops after the device drains, not after the last submit:
        // submitting is nearly free and would report a speed that is not real.
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let elapsed = started.elapsed();

        let per_step = elapsed.as_secs_f64() * 1000.0 / f64::from(STEPS);
        println!(
            "prepare (cube + morton + sort) over {BODIES} bodies: {per_step:.3} ms/step \
             across {STEPS} steps"
        );

        // Correctness is not suspended for a benchmark: a sort that quietly
        // dropped work would otherwise look like a speedup.
        let codes = read_back(&context, &prepare.codes, BODIES);
        let order = read_back(&context, &prepare.order, BODIES);
        assert!(codes.windows(2).all(|w| w[0] <= w[1]), "codes not sorted");
        let mut seen = order;
        seen.sort_unstable();
        assert_eq!(seen, (0..BODIES as u32).collect::<Vec<_>>());
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn coincident_bodies_share_a_code_without_a_degenerate_cube() {
        // Zero extent would give a zero-width root that no octant divides.
        let mut graph = gv_graph::testing::dust(64);
        for node in &mut graph.nodes {
            node.position = [5.0, -3.0, 2.0, 1.0];
        }

        let prepared = prepare(&graph, true);
        assert!(
            prepared.codes.iter().all(|code| *code == prepared.codes[0]),
            "coincident bodies disagreed on their code"
        );
    }
}
