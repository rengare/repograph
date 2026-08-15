// The octree itself, read off the sorted Morton codes.
//
// Nothing here recurses or allocates, so the build `gv_layout::octree` does in
// one depth-first pass becomes four stages over flat arrays. The keys are
// sorted, so *every subtree is a contiguous range of `codes`* — that single
// fact is what the whole file rests on. Notation, matching the CPU's:
//
//   prefix(c, l)     = c >> (3 * (LEVELS - l))     the octant path down to l
//   is_boundary(i,l) = i == 0 || prefix differs from i - 1
//   run_end(i, l)    = first index past the run of prefix(codes[i], l)
//
// # Where this tree differs from the CPU's, deliberately
//
// **It is path-compressed.** A node is emitted only at the *shallowest* level
// at which its range first appears. Without that, bodies sharing a long prefix
// produce a chain of single-child cells all covering the same range, which on a
// clustered graph is most of the tree. So cell counts do not match the CPU's,
// and are not meant to.
//
// **A leaf holds a body range, not a body.** Codes are 10 bits per axis rather
// than 21 (WGSL has no u64), so distinct bodies can collide in one code. An
// aggregate of them would be wrong in a way the CPU's never is: a body would
// feel its own mass from a centre a hair away, and k²/d with a tiny d is an
// explosion, not a rounding error. Instead a leaf carries `first`/`last` and
// the walk iterates the bodies exactly. That also makes theta = 0 exact.

struct Node {
    position: vec4<f32>,
    color: vec4<f32>,
    size: f32,
    dx: f32,
    dy: f32,
    dz: f32,
};

// 32 bytes. `center` is three separate f32 rather than a vec3, which WGSL
// would give 16-byte alignment and so pad the struct to 48.
struct Cell {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    /// Bodies beneath this cell.
    mass: f32,
    /// Width of the cell's cube, which the opening criterion compares against
    /// distance. Derived from the level, not from the bodies: a compressed node
    /// still stands for the cube of its level.
    width: f32,
    /// Next cell to visit when this subtree is skipped.
    escape: u32,
    /// The body range this cell covers, as indices into `order`.
    first: u32,
    last: u32,
};

struct Uniforms {
    node_count: u32,
    cell_capacity: u32,
    levels: u32,
    /// The level `centres` is summing on this dispatch. One uniform block per
    /// level, bound at a dynamic offset, because all the dispatches are
    /// recorded before any of them runs.
    level: u32,
    three_d: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
// min xyz then max xyz, as the order-preserving u32 `bh_prepare.wgsl` wrote.
@group(0) @binding(2) var<storage, read> bounds: array<u32, 6>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> order: array<u32>;
// `first * 16 + level`, unique per node — sorting by it *is* depth-first order.
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
// `last`, carried through the sort alongside the key.
@group(0) @binding(6) var<storage, read_write> values: array<u32>;
@group(0) @binding(7) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(8) var<storage, read_write> counter: atomic<u32>;

fn from_orderable(key: u32) -> f32 {
    if ((key & 0x80000000u) != 0u) {
        return bitcast<f32>(key & 0x7FFFFFFFu);
    }
    return bitcast<f32>(~key);
}

// The root cube, recomputed from the same bounds the codes were quantised
// against. Cheap enough to redo per invocation, and reading it back out of the
// reduction is what keeps it identical to what `morton` used.
fn cube_center() -> vec3<f32> {
    let low = vec3<f32>(
        from_orderable(bounds[0]),
        from_orderable(bounds[1]),
        from_orderable(bounds[2]),
    );
    let high = vec3<f32>(
        from_orderable(bounds[3]),
        from_orderable(bounds[4]),
        from_orderable(bounds[5]),
    );
    return (low + high) * 0.5;
}

fn root_width() -> f32 {
    let low = vec3<f32>(
        from_orderable(bounds[0]),
        from_orderable(bounds[1]),
        from_orderable(bounds[2]),
    );
    let high = vec3<f32>(
        from_orderable(bounds[3]),
        from_orderable(bounds[4]),
        from_orderable(bounds[5]),
    );
    let extent = max(max(high.x - low.x, high.y - low.y), high.z - low.z);
    if (extent > 0.0) {
        return extent * 1.001;
    }
    return 2.0;
}

fn body_of(index: u32) -> vec3<f32> {
    let position = nodes[index].position;
    var z = 0.0;
    if (u.three_d != 0u) {
        z = position.z;
    }
    return vec3<f32>(position.x, position.y, z);
}

fn prefix(code: u32, level: u32) -> u32 {
    return code >> (3u * (u.levels - level));
}

fn is_boundary(i: u32, level: u32) -> bool {
    if (i == 0u) {
        return true;
    }
    return prefix(codes[i], level) != prefix(codes[i - 1u], level);
}

// First index past the run of bodies sharing `codes[i]`'s prefix at `level`.
//
// A binary search rather than a scan: the prefixes are non-decreasing because
// the codes are sorted, and a run can be the whole array.
fn run_end(i: u32, level: u32) -> u32 {
    let p = prefix(codes[i], level);
    var lo = i + 1u;
    var hi = u.node_count;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        if (prefix(codes[mid], level) == p) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(256)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u) {
        atomicStore(&counter, 0u);
    }
    if (gid.x >= u.cell_capacity) {
        return;
    }
    // Unused slots sort to the end and are never read. They carry identical
    // key *and* value, so the stable sort cannot order them two ways.
    keys[gid.x] = 0xFFFFFFFFu;
    values[gid.x] = 0u;
}

// One invocation per (level, body): x is the body, y is the level.
//
// Emits a node at (l, i) when i starts a run at l and that run is not simply
// the run it already started at l - 1. The append order is non-deterministic
// because the slot comes from an atomic — which is only acceptable because the
// sort that follows keys on `first * 16 + level`, unique per node, and so puts
// every node in one fixed place. Weakening that key would make the layout
// irreproducible.
@compute @workgroup_size(256)
fn enumerate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let level = gid.y;
    if (i >= u.node_count || level > u.levels) {
        return;
    }
    if (!is_boundary(i, level)) {
        return;
    }

    let end = run_end(i, level);

    if (level > 0u && is_boundary(i, level - 1u)) {
        // The parent run starts here too, so this level is only worth a node if
        // it cut the parent short. It did iff the body just past this run is
        // still inside the parent's — otherwise the two ranges are equal and
        // the parent already stands for this one.
        if (end >= u.node_count) {
            return;
        }
        if (prefix(codes[end], level - 1u) != prefix(codes[i], level - 1u)) {
            return;
        }
    }

    let slot = atomicAdd(&counter, 1u);
    if (slot >= u.cell_capacity) {
        // Cannot happen: the emitted ranges form a laminar family in which
        // every internal node has at least two children, so there are at most
        // 2n - 1 of them and the capacity is 2n + 1. Dropping rather than
        // writing past the end keeps a miscount from corrupting memory, and
        // `cell_count` reports the overflow to the host.
        return;
    }
    keys[slot] = i * 16u + level;
    values[slot] = end - 1u;
}

// Escape indices and widths, once the nodes are in depth-first order.
//
// A node's subtree is exactly the nodes after it whose range falls inside its
// own, and those are contiguous, so the escape is the first node with
// `first > last` — a binary search for `(last + 1) * 16` over the sorted keys.
// A leaf is then `escape == index + 1`, which is what the walk tests.
@compute @workgroup_size(256)
fn link(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let count = min(atomicLoad(&counter), u.cell_capacity);
    if (index >= count) {
        return;
    }

    let key = keys[index];
    let first = key >> 4u;
    let level = key & 15u;
    let last = values[index];

    // `target` would be nicer; it is a reserved keyword in WGSL.
    let past_last = (last + 1u) * 16u;
    var lo = index + 1u;
    var hi = count;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        if (keys[mid] < past_last) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    cells[index].first = first;
    cells[index].last = last;
    cells[index].escape = lo;
    cells[index].width = root_width() * exp2(-f32(level));
}

// Centres of mass, one dispatch per level, deepest first.
//
// Not a prefix sum over positions: each node sums its *direct children* by
// walking `j = index + 1` while `j < escape`, stepping `j = escape[j]`. Total
// work is O(cells), it needs no scan primitive, and because a child's level is
// always deeper than its parent's, running the levels in reverse guarantees
// every child is finished before the parent reads it.
@compute @workgroup_size(256)
fn centres(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let count = min(atomicLoad(&counter), u.cell_capacity);
    if (index >= count) {
        return;
    }
    if ((keys[index] & 15u) != u.level) {
        return;
    }

    // Summed relative to the cube centre so the accumulator sits near zero and
    // f32 keeps its low bits — the CPU accumulates in f64 for the same reason.
    let origin = cube_center();
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    var mass = 0.0;

    let escape = cells[index].escape;
    if (escape == index + 1u) {
        // A leaf: a run of bodies sharing one code, summed directly.
        for (var b = cells[index].first; b <= cells[index].last; b = b + 1u) {
            sum = sum + (body_of(order[b]) - origin);
            mass = mass + 1.0;
        }
    } else {
        var j = index + 1u;
        while (j < escape) {
            let child = cells[j];
            let center = vec3<f32>(child.center_x, child.center_y, child.center_z);
            sum = sum + (center - origin) * child.mass;
            mass = mass + child.mass;
            j = child.escape;
        }
    }

    // Every cell covers at least one body, so `mass` is never zero.
    let center = origin + sum / mass;
    cells[index].center_x = center.x;
    cells[index].center_y = center.y;
    cells[index].center_z = center.z;
    cells[index].mass = mass;
}
