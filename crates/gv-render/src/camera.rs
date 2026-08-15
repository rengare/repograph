//! Perspective fly camera.
//!
//! Controls are the original's: `W`/`S` forward/back, `A`/`D` strafe, `R`/`F`
//! up/down, arrow keys rotate, right-drag to look, `Space` to reset.
//!
//! The matrix composition follows `Camera::Update` exactly —
//! `view = Rx(pitch) * Ry(yaw) * T(position)` — including the detail that
//! `position` accumulates translations expressed in the *world* frame, not the
//! camera's, because the original post-multiplied a pure-translation matrix.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// What the vertex shaders need, in one uniform block.
///
/// `viewport` is here because the node sprite reproduces the original's
/// pixel-space point size, and converting pixels to clip space needs the
/// framebuffer dimensions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub viewport: [f32; 2],
    pub _pad: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    /// Euler rotation in degrees, as the original stored it.
    pub rotation: Vec3,
    pub velocity: Vec3,
    pub rotation_velocity: Vec3,
    /// Degrees per arrow-key press.
    pub rotation_ratio: f32,

    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
    aspect: f32,
    viewport: [f32; 2],
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            // The original's main() placed the camera at z = -700 with a far
            // plane of 200_000 to keep a 1000-unit scatter in view.
            position: Vec3::new(0.0, 0.0, -700.0),
            rotation: Vec3::ZERO,
            velocity: Vec3::splat(100.0),
            rotation_velocity: Vec3::splat(0.5),
            rotation_ratio: 1.0,
            fov_degrees: 45.0,
            near: 0.1,
            far: 200_000.0,
            aspect: 1.0,
            viewport: [1.0, 1.0],
        }
    }
}

impl Camera {
    /// Recomputes the aspect ratio on resize.
    ///
    /// The original computed `width / height` on two `int`s, so a 1280x620
    /// window got an aspect of 2 instead of 2.064 and the image was subtly
    /// stretched. This divides as f32.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.aspect = width as f32 / height as f32;
            self.viewport = [width as f32, height as f32];
        }
    }

    /// Right-handed perspective onto wgpu's 0..=1 depth range.
    ///
    /// Not the OpenGL -1..=1 range the original's `glm::perspective` produced:
    /// wgpu follows the DirectX/Vulkan convention, so the GL-style projection
    /// would compress everything into the near half of the depth buffer.
    pub fn projection_matrix(&self) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(
            self.fov_degrees.to_radians(),
            self.aspect,
            self.near,
            self.far,
        )
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_rotation_x(self.rotation.x.to_radians())
            * Mat4::from_rotation_y(self.rotation.y.to_radians())
            * Mat4::from_translation(self.position)
    }

    pub fn uniforms(&self) -> CameraUniforms {
        CameraUniforms {
            view: self.view_matrix().to_cols_array_2d(),
            projection: self.projection_matrix().to_cols_array_2d(),
            viewport: self.viewport,
            _pad: [0.0; 2],
        }
    }

    fn translate(&mut self, delta: Vec3) {
        self.position += delta;
    }

    /// Moves along the view direction by `amount` steps of the camera velocity —
    /// positive is forward (zoom in), negative back. `W`/`S` are `zoom(±1)`; the
    /// mouse wheel passes its notch count so a flick travels proportionally.
    pub fn zoom(&mut self, amount: f32) {
        let yaw = (-self.rotation.y).to_radians();
        let pitch = self.rotation.x.to_radians();
        self.translate(
            Vec3::new(
                self.velocity.x * yaw.sin(),
                self.velocity.y * pitch.sin(),
                self.velocity.z * yaw.cos(),
            ) * amount,
        );
    }

    pub fn forward(&mut self) {
        self.zoom(1.0);
    }

    pub fn back(&mut self) {
        self.zoom(-1.0);
    }

    pub fn left(&mut self) {
        let yaw = self.rotation.y.to_radians();
        self.translate(Vec3::new(
            self.velocity.x * yaw.cos(),
            0.0,
            self.velocity.z * yaw.sin(),
        ));
    }

    pub fn right(&mut self) {
        let yaw = self.rotation.y.to_radians();
        self.translate(-Vec3::new(
            self.velocity.x * yaw.cos(),
            0.0,
            self.velocity.z * yaw.sin(),
        ));
    }

    pub fn up(&mut self) {
        self.translate(Vec3::new(0.0, -self.velocity.y, 0.0));
    }

    pub fn down(&mut self) {
        self.translate(Vec3::new(0.0, self.velocity.y, 0.0));
    }

    pub fn rotate_left(&mut self) {
        self.rotation.y -= self.rotation_ratio;
    }

    pub fn rotate_right(&mut self) {
        self.rotation.y += self.rotation_ratio;
    }

    pub fn rotate_up(&mut self) {
        self.rotation.x -= self.rotation_ratio;
    }

    pub fn rotate_down(&mut self) {
        self.rotation.x += self.rotation_ratio;
    }

    /// Applies a mouse drag, in pixels.
    pub fn mouse_look(&mut self, delta_x: f32, delta_y: f32) {
        self.rotation.y += delta_x * self.rotation_velocity.x;
        self.rotation.x += delta_y * self.rotation_velocity.y;
    }

    /// Clears the rotation, keeping the position — what `Camera::Reset` did.
    pub fn reset(&mut self) {
        self.rotation = Vec3::ZERO;
    }

    /// Frames `target` dead-centre at `standoff` units in front of the camera,
    /// clearing rotation. Used to jump to a searched node.
    ///
    /// With rotation cleared the view is a pure translation (`view = T(position)`),
    /// so a world point `p` lands at `p + position` in camera space. Placing
    /// `target` at `(0, 0, -standoff)` — the same framing the default camera gives
    /// the origin — means `position = (0,0,-standoff) - target`.
    pub fn focus_on(&mut self, target: Vec3, standoff: f32) {
        self.rotation = Vec3::ZERO;
        self.position = Vec3::new(-target.x, -target.y, -standoff - target.z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_is_not_integer_divided() {
        let mut camera = Camera::default();
        camera.resize(1280, 620);
        // The C++ original produced exactly 2.0 here.
        assert!((camera.aspect - 1280.0 / 620.0).abs() < 1e-6);
        assert_ne!(camera.aspect, 2.0);
    }

    #[test]
    fn resize_to_zero_height_does_not_divide_by_zero() {
        let mut camera = Camera::default();
        camera.resize(1280, 0);
        assert!(camera.aspect.is_finite());
    }

    #[test]
    fn uniforms_satisfy_wgsl_uniform_alignment() {
        assert_eq!(size_of::<CameraUniforms>(), 144);
        assert_eq!(size_of::<CameraUniforms>() % 16, 0);
    }

    #[test]
    fn uniforms_carry_the_viewport_for_the_point_size_conversion() {
        let mut camera = Camera::default();
        camera.resize(800, 600);
        assert_eq!(camera.uniforms().viewport, [800.0, 600.0]);
    }

    #[test]
    fn the_projection_maps_the_near_plane_to_zero_depth() {
        // wgpu's depth range is 0..1, not OpenGL's -1..1.
        let mut camera = Camera::default();
        camera.resize(100, 100);
        let clip = camera.projection_matrix() * glam::Vec4::new(0.0, 0.0, -camera.near, 1.0);
        assert!((clip.z / clip.w).abs() < 1e-4, "near plane at {}", clip.z / clip.w);
    }

    #[test]
    fn the_projection_maps_the_far_plane_to_unit_depth() {
        let mut camera = Camera::default();
        camera.resize(100, 100);
        let clip = camera.projection_matrix() * glam::Vec4::new(0.0, 0.0, -camera.far, 1.0);
        assert!((clip.z / clip.w - 1.0).abs() < 1e-3, "far plane at {}", clip.z / clip.w);
    }

    #[test]
    fn forward_and_back_are_inverses() {
        let mut camera = Camera {
            rotation: Vec3::new(15.0, 30.0, 0.0),
            ..Default::default()
        };
        let start = camera.position;

        camera.forward();
        assert_ne!(camera.position, start);
        camera.back();

        assert!((camera.position - start).length() < 1e-3);
    }

    #[test]
    fn left_and_right_are_inverses() {
        let mut camera = Camera::default();
        camera.rotation.y = 42.0;
        let start = camera.position;

        camera.left();
        camera.right();

        assert!((camera.position - start).length() < 1e-3);
    }

    #[test]
    fn up_and_down_are_inverses() {
        let mut camera = Camera::default();
        let start = camera.position;
        camera.up();
        camera.down();
        assert_eq!(camera.position, start);
    }

    #[test]
    fn zoom_scales_forward_motion_by_the_amount() {
        // One wheel notch of zoom equals a W press; several notches move
        // proportionally further along the same direction.
        let mut wheel = Camera::default();
        let mut key = Camera::default();
        wheel.zoom(3.0);
        for _ in 0..3 {
            key.forward();
        }
        assert!((wheel.position - key.position).length() < 1e-3);

        // Negative zoom is a back step.
        let mut c = Camera::default();
        let start = c.position;
        c.zoom(2.0);
        c.zoom(-2.0);
        assert!((c.position - start).length() < 1e-3);
    }

    #[test]
    fn forward_with_no_rotation_moves_along_positive_z() {
        // The camera sits at negative z looking toward the origin, and the
        // view translation is applied before rotation, so "forward" increases z.
        let mut camera = Camera::default();
        let start = camera.position;
        camera.forward();

        assert!(camera.position.z > start.z);
        assert!((camera.position.x - start.x).abs() < 1e-4);
    }

    #[test]
    fn arrow_rotation_is_symmetric() {
        let mut camera = Camera::default();
        camera.rotate_left();
        camera.rotate_right();
        assert_eq!(camera.rotation, Vec3::ZERO);

        camera.rotate_up();
        camera.rotate_down();
        assert_eq!(camera.rotation, Vec3::ZERO);
    }

    #[test]
    fn mouse_look_scales_by_the_rotation_velocity() {
        let mut camera = Camera {
            rotation_velocity: Vec3::splat(0.5),
            ..Default::default()
        };
        camera.mouse_look(10.0, -4.0);

        assert_eq!(camera.rotation.y, 5.0);
        assert_eq!(camera.rotation.x, -2.0);
    }

    #[test]
    fn focus_on_centres_the_target_at_the_standoff() {
        let mut camera = Camera {
            rotation: Vec3::new(20.0, 40.0, 0.0),
            ..Default::default()
        };
        let target = Vec3::new(300.0, -150.0, 900.0);
        camera.focus_on(target, 700.0);

        // Rotation cleared, so the view is a pure translation.
        assert_eq!(camera.rotation, Vec3::ZERO);
        let camera_space = camera.view_matrix() * target.extend(1.0);
        assert!((camera_space.truncate() - Vec3::new(0.0, 0.0, -700.0)).length() < 1e-3);
    }

    #[test]
    fn reset_clears_rotation_but_keeps_position() {
        // Faithful to `Camera::Reset`, which rebuilt the translation from the
        // current position and zeroed only the rotation.
        let mut camera = Camera::default();
        camera.forward();
        let moved = camera.position;
        camera.rotation = Vec3::new(10.0, 20.0, 0.0);

        camera.reset();

        assert_eq!(camera.rotation, Vec3::ZERO);
        assert_eq!(camera.position, moved);
    }

    #[test]
    fn the_view_matrix_composes_rotation_then_translation() {
        // view = Rx * Ry * T, matching Camera::Update's `rotationMatrix * translationMatrix`.
        let camera = Camera {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Vec3::new(30.0, 45.0, 0.0),
            ..Default::default()
        };

        let expected = Mat4::from_rotation_x(30f32.to_radians())
            * Mat4::from_rotation_y(45f32.to_radians())
            * Mat4::from_translation(camera.position);

        assert!((camera.view_matrix() - expected).abs_diff_eq(Mat4::ZERO, 1e-5));
    }
}
