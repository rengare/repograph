// Stable least-significant-digit radix sort over u32 keys with u32 payloads.
//
// The octree build needs bodies in Morton order, and that order has to be
// reproducible: the layout's determinism guarantee is that identical input
// gives byte-identical output, which an unstable sort would break the moment
// two bodies share a Morton code.
//
// Stability rules out the obvious GPU sort — histogram with a global atomic
// counter and scatter — because the order two equal keys come out in then
// depends on which invocation reached the atomic first. So the offsets are
// computed up front instead, and the scatter only reads them.
//
// One pass per 8-bit digit, four passes for a u32, ping-ponging between two
// key/value buffers. Each pass is three dispatches:
//
//   1. `histogram`  — per-workgroup counts of each digit, laid out digit-major
//   2. `scan`       — one workgroup turns those counts into output offsets
//   3. `scatter`    — each workgroup walks its own tile in order, consuming
//                     its offsets
//
// The digit-major histogram layout is what makes step 2 a single workgroup's
// work: thread `d` owns the whole row for digit `d`, so summing the row, the
// scan across digits, and writing the running offsets back are all it does.

struct Uniforms {
    count: u32,
    // Bit position of the digit this pass sorts on.
    shift: u32,
    // Elements per workgroup.
    tile: u32,
    group_count: u32,
};

const RADIX: u32 = 256u;
const THREADS: u32 = 256u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> values_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> values_out: array<u32>;
// Per-workgroup digit counts, becoming per-workgroup output offsets after
// `scan`. `RADIX * group_count`, indexed `digit * group_count + group`.
@group(0) @binding(5) var<storage, read_write> buckets: array<u32>;

var<workgroup> counts: array<atomic<u32>, RADIX>;
var<workgroup> totals: array<u32, RADIX>;

@compute @workgroup_size(THREADS)
fn histogram(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    atomicStore(&counts[local.x], 0u);
    workgroupBarrier();

    let start = group.x * u.tile;
    let end = min(start + u.tile, u.count);
    for (var i = start + local.x; i < end; i = i + THREADS) {
        let digit = (keys_in[i] >> u.shift) & (RADIX - 1u);
        atomicAdd(&counts[digit], 1u);
    }
    workgroupBarrier();

    buckets[local.x * u.group_count + group.x] = atomicLoad(&counts[local.x]);
}

// Turns per-workgroup counts into per-workgroup output offsets.
//
// Thread `d` owns digit `d`'s row. It sums the row, the workgroup scans those
// sums to find where each digit's block starts in the output, and then the
// thread writes each workgroup's offset within that block. One workgroup is
// enough because there are only ever RADIX rows to scan across.
@compute @workgroup_size(THREADS)
fn scan(@builtin(local_invocation_id) local: vec3<u32>) {
    let digit = local.x;
    let row = digit * u.group_count;

    var total = 0u;
    for (var g = 0u; g < u.group_count; g = g + 1u) {
        total = total + buckets[row + g];
    }
    totals[digit] = total;
    workgroupBarrier();

    // Hillis-Steele inclusive scan, then shifted to exclusive. Double-buffered
    // through a barrier pair so no thread reads a slot another has overwritten.
    var value = total;
    for (var offset = 1u; offset < RADIX; offset = offset * 2u) {
        var addend = 0u;
        if (digit >= offset) {
            addend = totals[digit - offset];
        }
        workgroupBarrier();
        value = value + addend;
        totals[digit] = value;
        workgroupBarrier();
    }

    var base = value - total;

    for (var g = 0u; g < u.group_count; g = g + 1u) {
        let count = buckets[row + g];
        buckets[row + g] = base;
        base = base + count;
    }
}

// Places each element at the offset computed above.
//
// One thread per workgroup, walking the tile front to back. That is what makes
// the sort stable: equal keys are emitted in the order they appear, because
// there is exactly one writer per (workgroup, digit) counter and it advances
// in input order. Occupancy is poor, but the tile size is chosen so there are
// enough workgroups in flight to cover it, and this pass is a small fraction
// of a step next to the tree walk it feeds.
@compute @workgroup_size(THREADS)
fn scatter(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    if (local.x != 0u) {
        return;
    }

    let start = group.x * u.tile;
    let end = min(start + u.tile, u.count);

    for (var i = start; i < end; i = i + 1u) {
        let key = keys_in[i];
        let digit = (key >> u.shift) & (RADIX - 1u);
        let slot = buckets[digit * u.group_count + group.x];
        buckets[digit * u.group_count + group.x] = slot + 1u;

        keys_out[slot] = key;
        values_out[slot] = values_in[i];
    }
}
