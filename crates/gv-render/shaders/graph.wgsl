// Shared bindings and types for the node and edge draw pipelines.
//
// `Node` must match `gv_graph::Node` byte for byte. Note that `disp` is three
// separate f32 rather than a `vec3<f32>`: WGSL gives vec3 a 16-byte alignment,
// which would push it from offset 36 to offset 48 and break the layout the
// whole no-copy design depends on. The original's GLSL declared it the same
// way, for the same reason.
struct Node {
    position: vec4<f32>,  // offset  0
    color: vec4<f32>,     // offset 16
    size: f32,            // offset 32
    dx: f32,              // offset 36
    dy: f32,              // offset 40
    dz: f32,              // offset 44
};                        // size    48

// `gv_graph::Edge` calls these `from` and `to`, but `from` is a reserved
// keyword in WGSL, so the endpoints are named for their role instead. Only the
// names differ; the layout is two u32 either way.
struct Edge {
    tail: u32,
    head: u32,
};

struct Camera {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    viewport: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> edges: array<Edge>;
// One flag per node: 0 hides the node and every edge touching it. Driven by the
// GUI's per-kind visibility checkboxes; all 1 when unfiltered.
@group(0) @binding(3) var<storage, read> visible: array<u32>;
// One factor per node: 1.0 full brightness, <1.0 dimmed toward black. Driven by
// a node selection (the selected node and its edge-connected neighbours stay at
// 1.0, everything else fades); all 1.0 when nothing is selected.
@group(0) @binding(4) var<storage, read> dim: array<f32>;

struct NodeOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Position within the sprite, in -1..1. The fragment shader masks the
    // quad to a disc with it, as `circle.frag` did with `gl_PointCoord`.
    @location(1) offset: vec2<f32>,
};

// Two triangles. A triangle strip would join consecutive nodes into one
// connected ribbon, so this is a triangle list with the corners repeated.
const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
);

// Somewhere off-screen, for vertices that must not be drawn. z = 2 is beyond
// the far plane in wgpu's 0..1 depth range, so the clipper discards it.
const CULLED = vec4<f32>(0.0, 0.0, 2.0, 1.0);

@vertex
fn vs_nodes(@builtin(vertex_index) vertex_index: u32) -> NodeOut {
    var out: NodeOut;

    let node_index = vertex_index / 6u;
    let node = nodes[node_index];
    let corner = CORNERS[vertex_index % 6u];
    let view_position = camera.view * vec4<f32>(node.position.xyz, 1.0);

    out.color = vec4<f32>(node.color.rgb * dim[node_index], node.color.a);
    out.offset = corner;

    // Filtered out by kind, or behind the camera (where -view_z would flip the
    // sprite inside out): nothing to draw.
    if (visible[node_index] == 0u || view_position.z >= 0.0) {
        out.clip_position = CULLED;
        return out;
    }

    // `circle.vert` sized its point as `size * (500 / -viewPosition.z)` in
    // pixels. Reproduced here by converting that pixel diameter into a clip
    // space offset: ndc = pixels / viewport * 2, and clip = ndc * w.
    let diameter = node.size * (500.0 / -view_position.z);
    var clip = camera.projection * view_position;
    let offset = corner * diameter * 0.5 / camera.viewport * 2.0 * clip.w;
    clip.x += offset.x;
    clip.y += offset.y;

    out.clip_position = clip;
    return out;
}

@fragment
fn fs_nodes(in: NodeOut) -> @location(0) vec4<f32> {
    // The disc mask. `circle.frag` compared against 0.5 because gl_PointCoord
    // runs 0..1; this offset runs -1..1, so the radius is 1.
    if (length(in.offset) > 1.0) {
        discard;
    }
    return in.color;
}

struct EdgeOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// No vertex buffer and no per-step copy: the edge is derived from the vertex
// index, the endpoint id is read out of `edges`, and the position out of
// `nodes`. This is what makes the original's `lines` compute pass and its
// duplicate edge vertex array unnecessary.
@vertex
fn vs_edges(@builtin(vertex_index) vertex_index: u32) -> EdgeOut {
    var out: EdgeOut;

    let edge = edges[vertex_index / 2u];
    let node_index = select(edge.head, edge.tail, vertex_index % 2u == 0u);
    let node = nodes[node_index];

    let view_position = camera.view * vec4<f32>(node.position.xyz, 1.0);
    // An edge dims with whichever endpoint is dimmer.
    let edge_dim = min(dim[edge.tail], dim[edge.head]);
    out.color = vec4<f32>(node.color.rgb * edge_dim, node.color.a);

    // Hide the edge if either endpoint's kind is filtered out, or is behind
    // the camera.
    if (visible[edge.tail] == 0u || visible[edge.head] == 0u || view_position.z >= 0.0) {
        out.clip_position = CULLED;
        return out;
    }

    out.clip_position = camera.projection * view_position;
    return out;
}

@fragment
fn fs_edges(in: EdgeOut) -> @location(0) vec4<f32> {
    return in.color;
}
