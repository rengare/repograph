// Front of the on-device octree build: bounding cube, then Morton codes.
//
// Both mirror `gv_layout::octree` exactly, because that CPU tree is the oracle
// the GPU one is validated against. Where the two differ it is written down.

struct Node {
    position: vec4<f32>,
    color: vec4<f32>,
    size: f32,
    dx: f32,
    dy: f32,
    dz: f32,
};

struct Uniforms {
    node_count: u32,
    three_d: u32,
    // Levels of the tree, and so bits per axis in a code.
    levels: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
// Six lanes: min xyz then max xyz, as order-preserving u32 (see `orderable`).
@group(0) @binding(2) var<storage, read_write> bounds: array<atomic<u32>, 6>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> order: array<u32>;

// Maps a f32 onto a u32 whose unsigned ordering matches the float's ordering.
//
// `atomicMin`/`atomicMax` take integers only, so a float reduction needs a
// monotonic encoding. For a non-negative float the raw bits already sort
// correctly once the sign bit is set; for a negative one the bits sort
// backwards, so every bit is inverted.
fn orderable(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

fn from_orderable(key: u32) -> f32 {
    if ((key & 0x80000000u) != 0u) {
        return bitcast<f32>(key & 0x7FFFFFFFu);
    }
    return bitcast<f32>(~key);
}

// In 2D the z lane is flattened rather than ignored, so the tree subdivides in
// the plane the forces actually act in.
fn body_of(index: u32) -> vec3<f32> {
    let position = nodes[index].position;
    var z = 0.0;
    if (u.three_d != 0u) {
        z = position.z;
    }
    return vec3<f32>(position.x, position.y, z);
}

@compute @workgroup_size(256)
fn clear_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 6u) {
        return;
    }
    // Identities for the reduction: min starts at the largest key, max at the
    // smallest.
    if (gid.x < 3u) {
        atomicStore(&bounds[gid.x], 0xFFFFFFFFu);
    } else {
        atomicStore(&bounds[gid.x], 0u);
    }
}

@compute @workgroup_size(256)
fn reduce_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.node_count) {
        return;
    }
    let body = body_of(gid.x);
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let key = orderable(body[axis]);
        atomicMin(&bounds[axis], key);
        atomicMax(&bounds[axis + 3u], key);
    }
}

// Spreads 10 bits so each occupies every third bit position.
//
// The CPU uses 21 bits per axis in a u64; WGSL has no u64, so this is 10 bits
// in a u32 — a coarser grid. That is affordable because a leaf here holds a
// *range* of bodies rather than one: bodies sharing a code are resolved by
// iterating them, not by being lumped into an aggregate.
fn spread(value: u32) -> u32 {
    var x = value & 0x3FFu;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x << 8u)) & 0x0300F00Fu;
    x = (x | (x << 4u)) & 0x030C30C3u;
    x = (x | (x << 2u)) & 0x09249249u;
    return x;
}

@compute @workgroup_size(256)
fn morton(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.node_count) {
        return;
    }

    var low = vec3<f32>(
        from_orderable(atomicLoad(&bounds[0])),
        from_orderable(atomicLoad(&bounds[1])),
        from_orderable(atomicLoad(&bounds[2])),
    );
    var high = vec3<f32>(
        from_orderable(atomicLoad(&bounds[3])),
        from_orderable(atomicLoad(&bounds[4])),
        from_orderable(atomicLoad(&bounds[5])),
    );

    // The cube is the longest side, with a hair of slack so a body sitting
    // exactly on the boundary stays inside it. A single point, or a set of
    // coincident points, would otherwise give a zero-width root.
    let center = (low + high) * 0.5;
    let extent = max(max(high.x - low.x, high.y - low.y), high.z - low.z);
    var half = 1.0;
    if (extent > 0.0) {
        half = extent * 0.5 * 1.001;
    }

    let scale = f32(1u << u.levels);
    let body = body_of(gid.x);
    let normalised = (body - (center - vec3<f32>(half))) / (2.0 * half);
    let quantised = clamp(normalised * scale, vec3<f32>(0.0), vec3<f32>(scale - 1.0));

    // x in the low bit of each three-bit group, so a code's digits are the
    // octant path from the root.
    codes[gid.x] = spread(u32(quantised.x))
        | (spread(u32(quantised.y)) << 1u)
        | (spread(u32(quantised.z)) << 2u);
    order[gid.x] = gid.x;
}
