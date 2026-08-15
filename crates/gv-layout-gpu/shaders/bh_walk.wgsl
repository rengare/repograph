// Barnes-Hut repulsion: the octree walk, replacing `fr.wgsl`'s O(n²) loop.
//
// A transcription of `gv_layout::octree::Octree::repulsion` — same stackless
// escape-index loop, same squared-comparison criterion, same rule that a cell
// at distance zero can never be accepted. Attraction and the position update
// are `fr.wgsl`'s, unchanged: Barnes-Hut replaces one of the three passes and
// nothing else, which is what makes the comparison against the exact paths mean
// something.

struct Node {
    position: vec4<f32>,
    color: vec4<f32>,
    size: f32,
    dx: f32,
    dy: f32,
    dz: f32,
};

// Mirrors `bh_tree.wgsl`'s Cell byte for byte.
struct Cell {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    mass: f32,
    width: f32,
    escape: u32,
    first: u32,
    last: u32,
};

struct Uniforms {
    node_count: u32,
    three_d: u32,
    _pad0: u32,
    _pad1: u32,

    k: f32,
    /// Opening angle. A cell is taken whole when its width over the distance to
    /// it falls below this; zero opens everything and degrades to brute force.
    theta: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> cells: array<Cell>;
@group(0) @binding(3) var<storage, read> order: array<u32>;

// In 2D the z lane is dropped rather than merely ignored, matching
// `fr.wgsl`'s `separation`: a node still carrying a stale z from a 3D run
// cannot influence a distance.
fn body_of(index: u32) -> vec3<f32> {
    let position = nodes[index].position;
    var z = 0.0;
    if (u.three_d != 0u) {
        z = position.z;
    }
    return vec3<f32>(position.x, position.y, z);
}

@compute @workgroup_size(256)
fn repulsive(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.node_count) {
        return;
    }

    // Bodies are walked in Morton order, not index order — `gid.x` selects a
    // slot in `order`, not a node. The walk is bound by memory latency rather
    // than arithmetic, and adjacent slots are adjacent in space, so neighbouring
    // invocations traverse nearly the same cells: the same locality argument
    // `Octree::order` makes for the CPU, and on a GPU it buys branch coherence
    // within the subgroup as well. Each invocation still writes only its own
    // node, so the scatter costs one store and no synchronisation.
    let slot = gid.x;
    let i = order[slot];
    let position = body_of(i);

    var displacement = vec3<f32>(0.0, 0.0, 0.0);
    let theta_squared = u.theta * u.theta;
    let k_squared = u.k * u.k;

    // The root escapes past the last cell, so its escape *is* the cell count —
    // which saves binding the tree's node counter here.
    let cell_count = cells[0].escape;

    var index = 0u;
    while (index < cell_count) {
        let cell = cells[index];
        let delta = position - vec3<f32>(cell.center_x, cell.center_y, cell.center_z);
        let distance_squared = dot(delta, delta);

        // A cell with nothing after it but its own successor has no children,
        // so it cannot be opened. Where the CPU then accepts its aggregate,
        // this iterates the bodies: a 10-bit code is coarse enough that two
        // distinct bodies can share one, and a body that met its own mass at a
        // centre a hair away would feel k²/d for a vanishing d. Runs are one
        // body wide except where bodies really are that close together.
        if (cell.escape == index + 1u) {
            // `order` is a permutation and this invocation owns slot `slot`, so
            // "is this body me" is a comparison of slots — no load, and no need
            // to know the body index at all.
            if (cell.first == cell.last) {
                // The overwhelmingly common leaf: one body, whose centre of
                // mass *is* its position. Taking it from the cell already in
                // hand saves a dependent 48-byte-strided load into `nodes` on
                // the hottest path in the walk.
                if (slot != cell.first && distance_squared > 0.0) {
                    let distance = sqrt(distance_squared);
                    let magnitude = k_squared / distance;
                    displacement = displacement + delta / distance * magnitude;
                }
            } else {
                for (var b = cell.first; b <= cell.last; b = b + 1u) {
                    if (b == slot) {
                        continue;
                    }
                    let separation = position - body_of(order[b]);
                    let distance = length(separation);
                    if (distance > 0.0) {
                        let magnitude = k_squared / distance;
                        displacement = displacement + separation / distance * magnitude;
                    }
                }
            }
            index = cell.escape;
        } else if (distance_squared > 0.0
            && cell.width * cell.width < theta_squared * distance_squared) {
            // Accepted whole. A cell at distance zero never satisfies this,
            // which is what stops a cell containing this very body from being
            // taken with its own mass included.
            let distance = sqrt(distance_squared);
            let magnitude = cell.mass * k_squared / distance;
            displacement = displacement + delta / distance * magnitude;
            index = cell.escape;
        } else {
            // Descend: in depth-first order a cell's first child is the next.
            index = index + 1u;
        }
    }

    nodes[i].dx = displacement.x;
    nodes[i].dy = displacement.y;
    nodes[i].dz = displacement.z;
}
