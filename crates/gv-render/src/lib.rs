//! Surface management, camera, and the node and edge draw pipelines.
//!
//! Knows nothing about layout algorithms: it draws whatever is currently in
//! the shared node buffer, whether a CPU or a GPU layout put it there.

pub mod camera;
pub mod edges;
pub mod nodes;
pub mod renderer;

pub use camera::Camera;
pub use renderer::Renderer;
