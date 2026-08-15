//! Held-key and mouse-drag state, applied to the camera once per frame.
//!
//! The original's `InputManager` kept a `map<int, bool>` of pressed keys and
//! polled it from `App::ProcessInput`. Same idea, but the key set is a fixed
//! struct rather than a map, so a typo is a compile error.
//!
//! Movement is applied per *frame*, not per key event: holding a key must
//! produce continuous motion, and key repeat rates are a platform setting.

use gv_render::Camera;
use winit::keyboard::KeyCode;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct InputState {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub rotate_left: bool,
    pub rotate_right: bool,
    pub rotate_up: bool,
    pub rotate_down: bool,
    pub reset: bool,

    looking: bool,
    cursor: Option<(f32, f32)>,
}

impl InputState {
    /// Records a key transition. Unknown keys are ignored.
    pub fn set_key(&mut self, code: KeyCode, pressed: bool) {
        let slot = match code {
            KeyCode::KeyW => &mut self.forward,
            KeyCode::KeyS => &mut self.back,
            KeyCode::KeyA => &mut self.left,
            KeyCode::KeyD => &mut self.right,
            KeyCode::KeyR => &mut self.up,
            KeyCode::KeyF => &mut self.down,
            KeyCode::ArrowLeft => &mut self.rotate_left,
            KeyCode::ArrowRight => &mut self.rotate_right,
            KeyCode::ArrowUp => &mut self.rotate_up,
            KeyCode::ArrowDown => &mut self.rotate_down,
            KeyCode::Space => &mut self.reset,
            _ => return,
        };
        *slot = pressed;
    }

    /// Starts or ends a right-drag look.
    pub fn set_looking(&mut self, looking: bool) {
        self.looking = looking;
        if !looking {
            // Dropping the anchor stops the next press from being interpreted
            // as one enormous drag from wherever the cursor was left.
            self.cursor = None;
        }
    }

    /// Feeds a cursor position, rotating the camera if a drag is in progress.
    pub fn set_cursor(&mut self, x: f32, y: f32, camera: &mut Camera) {
        if self.looking {
            if let Some((previous_x, previous_y)) = self.cursor {
                camera.mouse_look(x - previous_x, y - previous_y);
            }
            self.cursor = Some((x, y));
        } else {
            self.cursor = None;
        }
    }

    /// Applies every held key to the camera. Called once per frame.
    pub fn apply_to(&self, camera: &mut Camera) {
        if self.forward {
            camera.forward();
        }
        if self.back {
            camera.back();
        }
        if self.left {
            camera.left();
        }
        if self.right {
            camera.right();
        }
        if self.up {
            camera.up();
        }
        if self.down {
            camera.down();
        }
        if self.rotate_left {
            camera.rotate_left();
        }
        if self.rotate_right {
            camera.rotate_right();
        }
        if self.rotate_up {
            camera.rotate_up();
        }
        if self.rotate_down {
            camera.rotate_down();
        }
        if self.reset {
            camera.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holding_a_key_keeps_moving_the_camera() {
        // The reason movement is applied per frame rather than per event.
        let mut input = InputState::default();
        let mut camera = Camera::default();
        input.set_key(KeyCode::KeyW, true);

        input.apply_to(&mut camera);
        let after_one = camera.position;
        input.apply_to(&mut camera);

        assert_ne!(camera.position, after_one);
    }

    #[test]
    fn releasing_a_key_stops_the_motion() {
        let mut input = InputState::default();
        let mut camera = Camera::default();

        input.set_key(KeyCode::KeyW, true);
        input.set_key(KeyCode::KeyW, false);
        let before = camera.position;
        input.apply_to(&mut camera);

        assert_eq!(camera.position, before);
    }

    #[test]
    fn unmapped_keys_are_ignored() {
        let mut input = InputState::default();
        input.set_key(KeyCode::KeyQ, true);
        assert_eq!(input, InputState::default());
    }

    #[test]
    fn every_movement_key_maps_to_something() {
        let keys = [
            KeyCode::KeyW, KeyCode::KeyS, KeyCode::KeyA, KeyCode::KeyD,
            KeyCode::KeyR, KeyCode::KeyF, KeyCode::ArrowLeft, KeyCode::ArrowRight,
            KeyCode::ArrowUp, KeyCode::ArrowDown, KeyCode::Space,
        ];
        for key in keys {
            let mut input = InputState::default();
            input.set_key(key, true);
            assert_ne!(input, InputState::default(), "{key:?} is not mapped");
        }
    }

    #[test]
    fn the_first_drag_sample_only_anchors() {
        // Without this the camera would snap by the full distance from the
        // origin to wherever the cursor happened to be when the button went
        // down.
        let mut input = InputState::default();
        let mut camera = Camera::default();
        input.set_looking(true);

        input.set_cursor(400.0, 300.0, &mut camera);

        assert_eq!(camera.rotation.x, 0.0);
        assert_eq!(camera.rotation.y, 0.0);
    }

    #[test]
    fn dragging_rotates_by_the_delta() {
        let mut input = InputState::default();
        let mut camera = Camera::default();
        camera.rotation_velocity = glam::Vec3::splat(0.5);
        input.set_looking(true);

        input.set_cursor(100.0, 100.0, &mut camera);
        input.set_cursor(110.0, 90.0, &mut camera);

        assert_eq!(camera.rotation.y, 5.0);
        assert_eq!(camera.rotation.x, -5.0);
    }

    #[test]
    fn moving_without_the_button_held_does_not_rotate() {
        let mut input = InputState::default();
        let mut camera = Camera::default();

        input.set_cursor(100.0, 100.0, &mut camera);
        input.set_cursor(200.0, 200.0, &mut camera);

        assert_eq!(camera.rotation, glam::Vec3::ZERO);
    }

    #[test]
    fn releasing_and_re_pressing_does_not_produce_one_giant_jump() {
        let mut input = InputState::default();
        let mut camera = Camera::default();
        input.set_looking(true);
        input.set_cursor(100.0, 100.0, &mut camera);
        input.set_cursor(110.0, 100.0, &mut camera);
        let after_drag = camera.rotation;

        input.set_looking(false);
        // Cursor travels far while the button is up.
        input.set_cursor(900.0, 700.0, &mut camera);
        input.set_looking(true);
        input.set_cursor(905.0, 700.0, &mut camera);

        // Only the anchor was re-established; no rotation from the gap.
        assert_eq!(camera.rotation, after_drag);
    }
}
