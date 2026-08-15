// Fruchterman-Reingold as three compute passes.
//
// This is a transcription of `gv_layout::fr_cpu::FrCpuLayout::step`, and it has
// to stay one: the CPU path is the oracle the milestone test compares against,
// so any divergence here is a bug even when it looks like an improvement.
//
// `Node` must match `gv_graph::Node` byte for byte. `disp` is three separate
// f32 rather than a `vec3<f32>` because WGSL gives vec3 a 16-byte alignment,
// which would push it from offset 36 to 48 and break the 48-byte layout the
// no-copy design depends on.
struct Node {
    position: vec4<f32>,  // offset  0
    color: vec4<f32>,     // offset 16
    size: f32,            // offset 32
    dx: f32,              // offset 36
    dy: f32,              // offset 40
    dz: f32,              // offset 44
};                        // size    48

struct Uniforms {
    node_count: u32,
    edge_count: u32,
    three_d: u32,
    _pad: u32,

    // Derived on the host once per step rather than recomputed by every
    // invocation, which is what the original's GLSL did.
    k: f32,
    speed_scale: f32,
    gravity: f32,
    max_displace: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> csr_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> csr_neighbors: array<u32>;

// The vector from `a` to `b` in xyz, and its length in w.
//
// In 2D the z lane is dropped rather than merely ignored, so a node still
// carrying a stale z from a 3D run cannot influence the distance.
fn separation(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    var delta = vec3<f32>(a.x - b.x, a.y - b.y, 0.0);
    if (u.three_d != 0u) {
        delta.z = a.z - b.z;
    }
    return vec4<f32>(delta, length(delta));
}

fn displacement_of(i: u32) -> vec3<f32> {
    return vec3<f32>(nodes[i].dx, nodes[i].dy, nodes[i].dz);
}

fn store_displacement(i: u32, displacement: vec3<f32>) {
    nodes[i].dx = displacement.x;
    nodes[i].dy = displacement.y;
    nodes[i].dz = displacement.z;
}

// Repulsion: k² / d, away from every other node.
//
// Two corrections to the original's `repulsive.comp`. Its bound check was
// `if (globalIndex > graphDataSize) return;`, so one invocation past the end
// read out of bounds on every single dispatch; this uses `>=`. And it looped
// `j` from `globalIndex + 1`, computing each pair once but applying the force
// to only one of the two nodes, which is not a symmetric force; this is the
// full `0..n`.
//
// Writes only `nodes[i]`'s displacement, and reads only positions, which no
// pass writes until the update below. That is what makes it race-free.
@compute @workgroup_size(256)
fn repulsive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.node_count) {
        return;
    }

    let position = nodes[i].position;
    var displacement = vec3<f32>(0.0, 0.0, 0.0);

    for (var j = 0u; j < u.node_count; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let s = separation(position, nodes[j].position);
        let distance = s.w;
        if (distance > 0.0) {
            let magnitude = (u.k * u.k) / distance;
            displacement = displacement + s.xyz / distance * magnitude;
        }
    }

    store_displacement(i, displacement);
}

// Attraction: d² / k, toward every neighbour.
//
// The pass that justifies the whole CSR structure. The original ran one
// invocation *per edge*, doing an unsynchronised read-modify-write of
// `data[from]` and `data[to]`; every edge incident to a node raced with every
// other, so contributions were silently lost and no two runs agreed. This is
// one invocation *per node*, gathering over that node's CSR row, so each
// invocation writes only its own node and nothing is lost.
@compute @workgroup_size(256)
fn attractive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.node_count) {
        return;
    }

    let position = nodes[i].position;
    var displacement = displacement_of(i);

    let start = csr_offsets[i];
    let end = csr_offsets[i + 1u];
    for (var e = start; e < end; e = e + 1u) {
        let neighbor = csr_neighbors[e];
        if (neighbor == i) {
            continue;
        }
        let s = separation(position, nodes[neighbor].position);
        let distance = s.w;
        if (distance > 0.0) {
            let magnitude = (distance * distance) / u.k;
            displacement = displacement - s.xyz / distance * magnitude;
        }
    }

    store_displacement(i, displacement);
}

// Gravity, speed scale, clamp, write back.
//
// Each invocation reads and writes only its own node, so this is the one pass
// that may touch positions.
@compute @workgroup_size(256)
fn position_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.node_count) {
        return;
    }

    var position = nodes[i].position;
    var displacement = displacement_of(i);

    // Gravity, toward the origin. A node exactly at the origin has no
    // direction to be pulled in; the original divided by that zero.
    let radius = separation(position, vec4<f32>(0.0, 0.0, 0.0, 0.0)).w;
    if (radius > 0.0) {
        let pull = 0.01 * u.k * u.gravity * radius;
        displacement = displacement - pull * position.xyz / radius;
    }

    displacement = displacement * u.speed_scale;

    let travel = length(displacement);
    if (travel > 0.0) {
        let ceiling = u.max_displace * u.speed_scale;
        let step = min(ceiling, travel);
        let moved = position.xyz + displacement / travel * step;
        position = vec4<f32>(moved, position.w);
    }

    if (u.three_d == 0u) {
        position.z = 0.0;
    }

    nodes[i].position = position;
    // Stored unclamped and after the speed scale, matching the CPU reference.
    store_displacement(i, displacement);
}
