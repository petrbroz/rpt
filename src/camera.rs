use rand::rngs::ThreadRng;
use rand::{ Rng };
use super::vec3::{ Vec3, cross, length };
use super::ray::Ray;

pub trait Camera {
    fn generate_ray(self: &Self, u: f32, v: f32, rng: &mut ThreadRng) -> Ray;
}

pub struct PerspectiveCamera {
    origin: Vec3,
    target: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
    viewport_width: f32,
    viewport_height: f32,
    focal_distance: f32,
    lens_radius: f32,
}

impl PerspectiveCamera {
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3, fov: f32, aspect_ratio: f32, focal_distance: f32, lens_radius: f32) -> PerspectiveCamera {
        let mut dir = &target - &eye;
        let theta = fov / 180.0 * std::f32::consts::PI;
        let h = (0.5 * theta).tan();
        let viewport_width = 2.0 * h * length(&dir);
        let viewport_height = aspect_ratio * viewport_width;
        dir.normalize();
        let mut u_axis = cross(&up, &dir);
        u_axis.normalize();
        let v_axis = cross(&dir, &u_axis);
        PerspectiveCamera {
            origin: eye,
            target,
            u_axis,
            v_axis,
            viewport_width,
            viewport_height,
            focal_distance,
            lens_radius,
        }
    }
}

impl Camera for PerspectiveCamera {
    fn generate_ray(self: &Self, u: f32, v: f32, rng: &mut ThreadRng) -> Ray {
        let mut target = self.target;
        target += &(u * self.viewport_width * &self.u_axis);
        target += &(v * self.viewport_height * &self.v_axis);
        let mut dir = &target - &self.origin;
        dir.normalize();
        let mut ray = Ray::new(self.origin, dir);

        let focus_point = ray.point_at(self.focal_distance);
        loop {
            let (u, v): (f32, f32) = (rng.gen(), rng.gen());
            if u * u + v * v < 1.0 {
                ray.o += &(u * self.lens_radius * &self.u_axis);
                ray.o += &(v * self.lens_radius * &self.v_axis);
                break;
            }
        }
        ray.d = &focus_point - &ray.o;
        ray.d.normalize();

        ray
    }
}
