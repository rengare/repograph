//! A flattened octree for Barnes-Hut repulsion.
//!
//! # Why the tree is flat
//!
//! The obvious octree is a tree of boxed nodes walked recursively. That cannot
//! go on a GPU — no allocation, no recursion, no pointers — and a second,
//! differently-shaped implementation for the GPU would be a second place for
//! the approximation to be subtly wrong.
//!
//! So there is one representation: [`Cell`]s in depth-first order, each holding
//! an *escape index* — the cell to jump to when this subtree is skipped. That
//! turns the walk into a loop over an array with no stack at all:
//!
//! ```text
//! i = 0
//! while i < cells.len():
//!     if accept(cells[i]): accumulate; i = cells[i].escape   # skip the subtree
//!     else:                            i = i + 1             # descend
//! ```
//!
//! Descending is `i + 1` because in depth-first order a cell's first child is
//! always the next cell. The array is `Pod`, so the same bytes the CPU walks
//! are what gets uploaded for the GPU to walk.

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;

/// Absent index, for `escape` and `body`.
pub const NONE: u32 = u32::MAX;

/// Depth at which subdivision gives up and the cell becomes an aggregate.
///
/// Coincident bodies never separate no matter how far the tree subdivides, so
/// without a cap the build recurses until it runs out of stack. At this depth a
/// cell spans the root width over 2²⁴, which is far below any distance the
/// force computation can distinguish, so lumping whatever is left changes
/// nothing: bodies that close are at distance zero to each other and the
/// repulsion term skips them anyway.
const MAX_DEPTH: u32 = 24;

/// One cell of the flattened tree: an aggregate body plus where to skip to.
///
/// 32 bytes, `repr(C)`, and laid out so the WGSL mirror needs no padding games
/// — `center` is three separate lanes followed by `mass`, which packs as two
/// 16-byte chunks with no implicit alignment gaps.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Cell {
    /// Centre of mass of every body beneath this cell.
    pub center: [f32; 3],
    /// Number of bodies beneath this cell.
    pub mass: f32,
    /// Width of the cell's cube. The Barnes-Hut criterion compares this
    /// against distance.
    pub width: f32,
    /// Next cell to visit when this subtree is skipped.
    pub escape: u32,
    /// Body index when this cell holds exactly one, else [`NONE`].
    ///
    /// Needed to exclude a body from its own repulsion. An aggregate never
    /// carries one, which is why a cell containing the body being solved for
    /// has to be opened rather than accepted.
    pub body: u32,
    pub _pad: u32,
}

/// The flattened tree.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Octree {
    pub cells: Vec<Cell>,
    /// Body indices in Morton order — the order the build sorted them into.
    ///
    /// Kept because walking the tree in this order rather than in index order
    /// is worth a large constant factor. The walk is bound by memory latency,
    /// not arithmetic: the cell array outgrows L2 well before the graphs that
    /// need Barnes-Hut do. Bodies adjacent in Morton order are adjacent in
    /// space and so traverse nearly the same cells, which turns a stream of
    /// cache misses into a stream of hits.
    pub order: Vec<u32>,
}

/// Bits of Morton code per axis, and so the number of levels the tree can
/// distinguish. 21 fits three axes in a `u64` and resolves the root cube to one
/// part in two million, far finer than any distance the forces resolve.
const LEVELS: u32 = 21;

/// Spreads 21 bits so each occupies every third bit position.
///
/// The standard Morton dilation: five shift-and-mask rounds, which beats a
/// per-bit loop by roughly an order of magnitude and is what makes computing
/// 100k codes free next to sorting them.
#[inline]
fn spread(value: u32) -> u64 {
    let mut x = u64::from(value) & 0x1F_FFFF;
    x = (x | x << 32) & 0x001F_0000_0000_FFFF;
    x = (x | x << 16) & 0x001F_0000_FF00_00FF;
    x = (x | x << 8) & 0x100F_00F0_0F00_F00F;
    x = (x | x << 4) & 0x10C3_0C30_C30C_30C3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

/// Morton (Z-order) code of a position within the root cube.
///
/// x lands in the low bit of each three-bit group, matching [`octant_of`], so
/// the code's digits *are* the octant path from the root — which is what lets
/// the build read the tree straight off a sorted array.
#[inline]
fn morton(position: [f32; 3], center: [f32; 3], half: f32) -> u64 {
    let scale = (1u32 << LEVELS) as f32;
    let quantise = |axis: usize| {
        let normalised = (position[axis] - (center[axis] - half)) / (2.0 * half);
        (normalised * scale).clamp(0.0, scale - 1.0) as u32
    };
    spread(quantise(0)) | spread(quantise(1)) << 1 | spread(quantise(2)) << 2
}

/// The octant digit of a code at a given depth, root first.
#[inline]
fn digit(code: u64, depth: u32) -> usize {
    ((code >> (3 * (LEVELS - 1 - depth))) & 7) as usize
}

struct Builder<'a> {
    /// `(morton code, body index)`, sorted. Every subtree is a contiguous
    /// slice of this, which is the whole reason the build is fast: it walks
    /// memory forwards instead of chasing child pointers around a heap.
    keys: &'a [(u64, u32)],
    positions: &'a [[f32; 3]],
    cells: Vec<Cell>,
}

impl Builder<'_> {
    /// Emits the subtree covering `keys[lo..hi]` in depth-first order and
    /// returns its summed positions and body count.
    ///
    /// Summed in f64: 100k single-precision additions into one accumulator lose
    /// enough low bits to move a centre of mass visibly.
    fn emit(
        &mut self,
        lo: usize,
        hi: usize,
        depth: u32,
        center: [f32; 3],
        half: f32,
    ) -> ([f64; 3], u32) {
        let here = self.cells.len();
        self.cells.push(Cell::default());

        let count = (hi - lo) as u32;
        let mut sum = [0.0f64; 3];
        let mut body = NONE;

        // A single body is a leaf. So is a run of identical codes, which means
        // bodies closer together than the finest level resolves — and so is
        // running out of depth. Without those two the recursion would never
        // terminate on coincident input.
        let exhausted = depth >= MAX_DEPTH.min(LEVELS) || self.keys[lo].0 == self.keys[hi - 1].0;

        if count == 1 {
            body = self.keys[lo].1;
            let position = self.positions[body as usize];
            sum = [
                f64::from(position[0]),
                f64::from(position[1]),
                f64::from(position[2]),
            ];
        } else if exhausted {
            for (_, index) in &self.keys[lo..hi] {
                let position = self.positions[*index as usize];
                for (sum, axis) in sum.iter_mut().zip(&position) {
                    *sum += f64::from(*axis);
                }
            }
        } else {
            // The slice is sorted, so each octant is a contiguous run. One scan
            // splits them all.
            let child_half = half * 0.5;
            let mut start = lo;
            while start < hi {
                let octant = digit(self.keys[start].0, depth);
                let mut end = start + 1;
                while end < hi && digit(self.keys[end].0, depth) == octant {
                    end += 1;
                }

                let (child_sum, _) = self.emit(
                    start,
                    end,
                    depth + 1,
                    child_center(center, child_half, octant),
                    child_half,
                );
                for (sum, child) in sum.iter_mut().zip(&child_sum) {
                    *sum += child;
                }

                start = end;
            }
        }

        let mass = f64::from(count);
        self.cells[here] = Cell {
            center: [
                (sum[0] / mass) as f32,
                (sum[1] / mass) as f32,
                (sum[2] / mass) as f32,
            ],
            mass: count as f32,
            width: half * 2.0,
            escape: self.cells.len() as u32,
            body,
            _pad: 0,
        };

        (sum, count)
    }
}

impl Octree {
    /// Builds the tree over `positions`.
    ///
    /// In 2D the caller is expected to have already flattened z, so that the
    /// tree subdivides in the plane the forces actually act in.
    pub fn build(positions: &[[f32; 3]]) -> Self {
        if positions.is_empty() {
            return Self::default();
        }

        let (center, half) = bounding_cube(positions);

        let mut keys: Vec<(u64, u32)> = positions
            .par_iter()
            .enumerate()
            .map(|(body, position)| (morton(*position, center, half), body as u32))
            .collect();
        // Sorting by code puts every subtree in a contiguous run, which is what
        // turns the build into a linear scan. Sorting the pair rather than the
        // code alone keeps ties in body order, so the tree — and every force
        // summed from it — is reproducible.
        keys.par_sort_unstable();

        let mut builder = Builder {
            keys: &keys,
            positions,
            cells: Vec::with_capacity(2 * positions.len()),
        };
        builder.emit(0, keys.len(), 0, center, half);

        Self {
            cells: builder.cells,
            order: keys.into_iter().map(|(_, body)| body).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Repulsion on `body` at `position`, as `k² / d` away from every other
    /// body, with distant subtrees aggregated.
    ///
    /// `theta` is the opening angle: a cell is accepted whole when its width
    /// over the distance to it falls below `theta`. Zero opens everything and
    /// degrades to brute force.
    pub fn repulsion(&self, body: u32, position: [f32; 3], k: f32, theta: f32) -> [f32; 3] {
        let mut displacement = [0.0f32; 3];
        let mut index = 0usize;
        // Comparing squares keeps the criterion off the square-root path.
        // Every cell the walk *opens* is tested and then discarded, and those
        // are the majority of visits, so the root taken only on acceptance is
        // a large share of the walk's cost.
        let theta_squared = theta * theta;

        while index < self.cells.len() {
            let cell = self.cells[index];
            let delta = [
                position[0] - cell.center[0],
                position[1] - cell.center[1],
                position[2] - cell.center[2],
            ];
            let distance_squared =
                delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

            // A cell with nothing after it but its own successor has no
            // children, so it cannot be opened and must be taken whole.
            //
            // Testing `body != NONE` instead would be wrong: an aggregate leaf
            // of coincident bodies carries no body index, so it would fail both
            // this and the criterion below, and "descending" would step into a
            // subtree that does not exist — silently dropping every body in it.
            //
            // Otherwise the criterion decides, and a cell at distance zero can
            // never satisfy it, which is what stops the cell containing this
            // very body from being accepted with its own mass included.
            let is_leaf = cell.escape as usize == index + 1;
            let accepted = distance_squared > 0.0
                && cell.width * cell.width < theta_squared * distance_squared;

            if is_leaf || accepted {
                if distance_squared > 0.0 && cell.body != body {
                    let distance = distance_squared.sqrt();
                    let magnitude = cell.mass * (k * k) / distance;
                    for (component, delta) in displacement.iter_mut().zip(&delta) {
                        *component += delta / distance * magnitude;
                    }
                }
                index = cell.escape as usize;
            } else {
                index += 1;
            }
        }

        displacement
    }
}

/// Centre and half-width of a cube containing every position.
fn bounding_cube(positions: &[[f32; 3]]) -> ([f32; 3], f32) {
    let mut min = positions[0];
    let mut max = positions[0];

    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }

    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];

    // The longest side sets the cube, and a hair of slack keeps a body sitting
    // exactly on the boundary inside it. A single point, or a set of coincident
    // points, would otherwise give a zero-width root that no octant divides.
    let extent = (0..3)
        .map(|axis| max[axis] - min[axis])
        .fold(0.0f32, f32::max);
    let half = if extent > 0.0 { extent * 0.5 * 1.001 } else { 1.0 };

    (center, half)
}

/// Which of the eight octants of `center` a position falls in.
///
/// Only used to pin the Morton bit convention in the tests — the build reads
/// octants out of the code's digits instead.
#[cfg(test)]
fn octant_of(position: [f32; 3], center: [f32; 3]) -> usize {
    usize::from(position[0] >= center[0])
        | usize::from(position[1] >= center[1]) << 1
        | usize::from(position[2] >= center[2]) << 2
}

/// Centre of the given octant of a parent, where `half` is the *child's*
/// half-width.
fn child_center(center: [f32; 3], half: f32, octant: usize) -> [f32; 3] {
    let offset = |bit: usize| if octant & (1 << bit) != 0 { half } else { -half };
    [
        center[0] + offset(0),
        center[1] + offset(1),
        center[2] + offset(2),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(positions: &[[f32; 3]], body: usize, k: f32) -> [f32; 3] {
        let mut displacement = [0.0f32; 3];
        let position = positions[body];

        for (other, target) in positions.iter().enumerate() {
            if other == body {
                continue;
            }
            let delta = [
                position[0] - target[0],
                position[1] - target[1],
                position[2] - target[2],
            ];
            let distance =
                (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            if distance > 0.0 {
                let magnitude = (k * k) / distance;
                for (component, delta) in displacement.iter_mut().zip(&delta) {
                    *component += delta / distance * magnitude;
                }
            }
        }

        displacement
    }

    /// Deterministic pseudo-random cloud, so failures reproduce.
    fn cloud(count: usize) -> Vec<[f32; 3]> {
        let mut state = 0x2545F491_4F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16777216.0) * 2000.0 - 1000.0
        };
        (0..count).map(|_| [next(), next(), next()]).collect()
    }

    #[test]
    fn the_morton_digit_is_the_octant() {
        // The convention the whole build rests on: because x sits in the low
        // bit of each three-bit group, a code's digit at depth d *is* the
        // octant that position occupies at that depth. If these ever disagree,
        // the tree's cell centres stop matching the bodies inside them and the
        // widths fed to the opening criterion become nonsense.
        let center = [0.0, 0.0, 0.0];
        let half = 64.0;

        for (index, position) in [
            [-10.0f32, -10.0, -10.0],
            [10.0, -10.0, -10.0],
            [-10.0, 10.0, -10.0],
            [10.0, 10.0, 10.0],
            [-1.0, 30.0, -60.0],
        ]
        .into_iter()
        .enumerate()
        {
            let code = morton(position, center, half);
            assert_eq!(
                digit(code, 0),
                octant_of(position, center),
                "position {index} ({position:?}) disagrees at the root"
            );
        }
    }

    #[test]
    fn an_empty_cloud_builds_an_empty_tree() {
        assert!(Octree::build(&[]).is_empty());
    }

    #[test]
    fn a_single_body_is_its_own_leaf() {
        let tree = Octree::build(&[[1.0, 2.0, 3.0]]);
        assert_eq!(tree.cells.len(), 1);
        assert_eq!(tree.cells[0].body, 0);
        assert_eq!(tree.cells[0].mass, 1.0);
        assert_eq!(tree.cells[0].center, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_body_feels_nothing_from_itself() {
        let tree = Octree::build(&[[5.0, 0.0, 0.0]]);
        assert_eq!(tree.repulsion(0, [5.0, 0.0, 0.0], 10.0, 0.5), [0.0; 3]);
    }

    #[test]
    fn the_root_mass_is_the_body_count() {
        let positions = cloud(200);
        let tree = Octree::build(&positions);
        assert_eq!(tree.cells[0].mass, 200.0);
    }

    #[test]
    fn the_root_centre_of_mass_is_the_mean_position() {
        let positions = cloud(100);
        let tree = Octree::build(&positions);

        for axis in 0..3 {
            let mean =
                positions.iter().map(|p| f64::from(p[axis])).sum::<f64>() / positions.len() as f64;
            let error = (f64::from(tree.cells[0].center[axis]) - mean).abs();
            assert!(error < 1e-2, "axis {axis} centre of mass off by {error}");
        }
    }

    #[test]
    fn every_escape_index_points_past_its_own_subtree() {
        // The invariant the whole stackless walk rests on: escape must be
        // strictly greater than the cell's own index, or the loop cannot
        // terminate; and it must not exceed the array.
        let tree = Octree::build(&cloud(500));
        for (index, cell) in tree.cells.iter().enumerate() {
            assert!(
                cell.escape as usize > index,
                "cell {index} escapes backwards to {}",
                cell.escape
            );
            assert!(cell.escape as usize <= tree.cells.len(), "cell {index} escapes past the end");
        }
    }

    #[test]
    fn a_cells_first_child_is_the_next_cell() {
        // The other half of that invariant: descending is `index + 1`, so any
        // cell that is not a leaf must be followed immediately by its subtree.
        let tree = Octree::build(&cloud(300));
        for (index, cell) in tree.cells.iter().enumerate() {
            if cell.body == NONE && cell.mass > 1.0 {
                assert!(
                    cell.escape as usize > index + 1,
                    "internal cell {index} has no subtree after it"
                );
            }
        }
    }

    #[test]
    fn theta_zero_visits_every_body_and_matches_brute_force() {
        // The exactness anchor. With nothing ever accepted as an aggregate the
        // walk reaches every leaf, so the only difference from brute force is
        // the order the contributions are summed in.
        let positions = cloud(300);
        let tree = Octree::build(&positions);
        let k = 50.0;

        for body in [0usize, 7, 42, 299] {
            let approximate = tree.repulsion(body as u32, positions[body], k, 0.0);
            let exact = brute_force(&positions, body, k);

            for axis in 0..3 {
                let scale = exact[axis].abs().max(1.0);
                let error = (approximate[axis] - exact[axis]).abs() / scale;
                assert!(
                    error < 1e-4,
                    "body {body} axis {axis}: tree {} vs brute force {} ({error})",
                    approximate[axis],
                    exact[axis]
                );
            }
        }
    }

    #[test]
    fn the_default_theta_stays_within_a_few_percent_of_brute_force() {
        let positions = cloud(1000);
        let tree = Octree::build(&positions);
        let k = 50.0;

        for body in [0usize, 13, 500, 999] {
            let approximate = tree.repulsion(body as u32, positions[body], k, 0.5);
            let exact = brute_force(&positions, body, k);

            let magnitude = (exact[0] * exact[0] + exact[1] * exact[1] + exact[2] * exact[2]).sqrt();
            let error = ((approximate[0] - exact[0]).powi(2)
                + (approximate[1] - exact[1]).powi(2)
                + (approximate[2] - exact[2]).powi(2))
            .sqrt();

            assert!(
                error / magnitude.max(1e-6) < 0.05,
                "body {body}: approximation is {:.1}% off",
                100.0 * error / magnitude
            );
        }
    }

    #[test]
    fn a_larger_theta_visits_fewer_cells() {
        // The entire point of the structure: opening less must cost less.
        let positions = cloud(2000);
        let tree = Octree::build(&positions);

        let visited = |theta: f32| {
            let mut count = 0usize;
            let mut index = 0usize;
            while index < tree.cells.len() {
                count += 1;
                let cell = tree.cells[index];
                let delta = [
                    positions[0][0] - cell.center[0],
                    positions[0][1] - cell.center[1],
                    positions[0][2] - cell.center[2],
                ];
                let distance =
                    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                if cell.body != NONE || (distance > 0.0 && cell.width < theta * distance) {
                    index = cell.escape as usize;
                } else {
                    index += 1;
                }
            }
            count
        };

        let exact = visited(0.0);
        let approximate = visited(0.5);
        assert!(
            approximate < exact / 2,
            "theta 0.5 visited {approximate} cells against {exact} for brute force"
        );
    }

    #[test]
    fn coincident_bodies_terminate_the_build() {
        // Without the depth cap this recurses until the stack runs out: no
        // subdivision ever separates bodies at the same point.
        let positions = vec![[3.0, 3.0, 3.0]; 64];
        let tree = Octree::build(&positions);

        assert_eq!(tree.cells[0].mass, 64.0);
        // All at one point, so every distance is zero and nothing is applied.
        let displacement = tree.repulsion(0, positions[0], 10.0, 0.5);
        assert!(displacement.iter().all(|component| component.is_finite()));
        assert_eq!(displacement, [0.0; 3]);
    }

    #[test]
    fn two_coincident_bodies_do_not_hide_a_third() {
        // The aggregate leaf must still repel a body that is not in it.
        let mut positions = vec![[0.0, 0.0, 0.0]; 8];
        positions.push([100.0, 0.0, 0.0]);
        let tree = Octree::build(&positions);

        let displacement = tree.repulsion(8, [100.0, 0.0, 0.0], 10.0, 0.5);
        assert!(displacement[0] > 0.0, "pushed the wrong way: {displacement:?}");
        // Eight bodies at one point, so eight times the single-body force.
        let single = 100.0f32 / 100.0 * (10.0 * 10.0 / 100.0);
        assert!((displacement[0] - 8.0 * single).abs() < 1e-3, "{displacement:?}");
    }

    #[test]
    fn the_walk_terminates_on_a_degenerate_cloud() {
        // Every body on one axis: octants collapse to two, the tree goes deep.
        let positions: Vec<[f32; 3]> = (0..500).map(|i| [i as f32 * 0.001, 0.0, 0.0]).collect();
        let tree = Octree::build(&positions);
        let displacement = tree.repulsion(0, positions[0], 1.0, 0.5);
        assert!(displacement.iter().all(|component| component.is_finite()));
    }

    #[test]
    fn the_cell_layout_is_what_the_shader_expects() {
        assert_eq!(size_of::<Cell>(), 32);
        assert_eq!(size_of::<Cell>() % 16, 0);
    }
}
