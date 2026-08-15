//! Layout parameters and the CPU layout implementations.
//!
//! # Why there are two layout traits
//!
//! A CPU layout owns the node array and mutates it in place; a GPU layout owns
//! a `wgpu::Buffer` the CPU never sees and records commands into an encoder.
//! Forcing both through one trait would mean either round-tripping the GPU
//! buffer to host memory every frame or handing the CPU path a device it does
//! not need. So [`CpuLayout`] lives here and `gv_layout_gpu::GpuLayout` lives
//! in its own crate; `gv-app` holds an enum over the two and drives whichever
//! is selected.
//!
//! The shared vocabulary — [`LayoutParams`] and the force constants — lives
//! here, because both paths must agree on it for their results to be
//! comparable.

pub mod barnes_hut;
pub mod fr_cpu;
pub mod octree;
pub mod random;

use gv_graph::GraphData;

/// Divisor applied to `speed` before it scales a displacement. Named
/// `SPEED_DIVISOR` in the original's GLSL.
pub const SPEED_DIVISOR: f32 = 400.0;

/// Multiplier applied to `area` when deriving the ideal edge length `k`.
/// Named `AREA_MULTIPLICATOR` in the original's GLSL.
pub const AREA_MULTIPLIER: f32 = 500.0;

/// The three knobs the original exposed in its "Graph settings" ImGui panel,
/// plus the 2D/3D switch that lived in the "Algorithms" panel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayoutParams {
    pub speed: f32,
    pub area: f32,
    pub gravity: f32,
    pub three_d: bool,
}

impl Default for LayoutParams {
    fn default() -> Self {
        // The defaults the original's FRModel constructed with.
        Self {
            speed: 100.0,
            area: 1000.0,
            gravity: 1.0,
            three_d: false,
        }
    }
}

impl LayoutParams {
    /// Ideal edge length: `k = (AREA_MULTIPLIER * area) / (1 + n)`.
    pub fn k(&self, node_count: usize) -> f32 {
        (AREA_MULTIPLIER * self.area) / (1.0 + node_count as f32)
    }

    /// Per-step displacement ceiling, before the speed scale is applied.
    pub fn max_displace(&self) -> f32 {
        (AREA_MULTIPLIER * self.area).sqrt() / 10.0
    }

    /// The factor displacements are scaled by each step.
    pub fn speed_scale(&self) -> f32 {
        self.speed / SPEED_DIVISOR
    }
}

/// Separation between two nodes: the vector from `a` to `b`, and its length.
///
/// In 2D the z lane is dropped rather than merely ignored, so a node that still
/// carries a stale z from a 3D run cannot influence the distance.
#[inline]
pub fn separation(a: &[f32; 4], b: &[f32; 4], three_d: bool) -> ([f32; 3], f32) {
    let delta = [
        a[0] - b[0],
        a[1] - b[1],
        if three_d { a[2] - b[2] } else { 0.0 },
    ];
    let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    (delta, length)
}

/// Attraction on node `i`: `d² / k` toward each of its neighbours.
///
/// Gathered over the node's CSR row, so this invocation writes only its own
/// displacement — the structure that removes the original's per-edge race.
#[inline]
pub fn attraction(
    index: usize,
    positions: &[[f32; 4]],
    adjacency: &gv_graph::Csr,
    k: f32,
    three_d: bool,
) -> [f32; 3] {
    let mut displacement = [0.0f32; 3];
    let position = positions[index];

    for &neighbor in adjacency.neighbors_of(index) {
        if neighbor as usize == index {
            continue;
        }
        let (delta, distance) = separation(&position, &positions[neighbor as usize], three_d);
        if distance > 0.0 {
            let magnitude = (distance * distance) / k;
            for (component, delta) in displacement.iter_mut().zip(&delta) {
                *component -= delta / distance * magnitude;
            }
        }
    }

    displacement
}

/// Applies gravity, the speed scale and the displacement ceiling, then writes
/// the new positions.
///
/// Shared by every CPU layout so they cannot drift in how a step is closed out:
/// only the force accumulation above differs between them, and that is the only
/// thing that should.
pub fn integrate(
    graph: &mut GraphData,
    displacements: Vec<[f32; 3]>,
    params: &LayoutParams,
    k: f32,
) {
    let speed_scale = params.speed_scale();
    let ceiling = params.max_displace() * speed_scale;
    let three_d = params.three_d;

    for (node, mut displacement) in graph.nodes.iter_mut().zip(displacements) {
        // Gravity, toward the origin. A node exactly at the origin has no
        // direction to be pulled in; the original divided by that zero.
        let (_, radius) = separation(&node.position, &[0.0; 4], three_d);
        if radius > 0.0 {
            let pull = 0.01 * k * params.gravity * radius;
            for (component, axis) in displacement.iter_mut().zip(&node.position) {
                *component -= pull * axis / radius;
            }
        }

        for component in &mut displacement {
            *component *= speed_scale;
        }

        let length = (displacement[0] * displacement[0]
            + displacement[1] * displacement[1]
            + displacement[2] * displacement[2])
            .sqrt();

        if length > 0.0 {
            let step = ceiling.min(length);
            for (axis, component) in node.position.iter_mut().zip(&displacement) {
                *axis += component / length * step;
            }
        }

        if !three_d {
            node.position[2] = 0.0;
        }

        node.disp = displacement;
    }
}

/// A layout that advances the node array in host memory.
pub trait CpuLayout {
    /// Name shown in the algorithm picker.
    fn name(&self) -> &'static str;

    /// Advances the layout by one step.
    fn step(&mut self, graph: &mut GraphData, params: &LayoutParams);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_matches_the_glsl_formula() {
        let params = LayoutParams { area: 1000.0, ..Default::default() };
        // (500 * 1000) / (1 + 999) == 500
        assert_eq!(params.k(999), 500.0);
    }

    #[test]
    fn max_displace_matches_the_glsl_formula() {
        let params = LayoutParams { area: 2.0, ..Default::default() };
        // sqrt(500 * 2) / 10 == 10 * sqrt(10) / 10
        assert!((params.max_displace() - 1000.0_f32.sqrt() / 10.0).abs() < 1e-6);
    }
}
