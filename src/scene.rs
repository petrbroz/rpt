use super::vec3::{ Vec3, dot, length_squared };
use super::ray::Ray;

#[derive(Debug, Copy, Clone)]
pub enum Material {
    Diffuse(Vec3),
    Metal(Vec3, f32 /* roughness */),
    Light(Vec3),
    Glass(Vec3 /* attenuation */, f32 /* ior */),
    Normal,
}

pub struct Hit {
    pub p: Vec3,
    pub n: Vec3,
    pub t: f32,
    pub m: Material,
}

impl Hit {
    pub fn new(p: Vec3, n: Vec3, t: f32, m: Material ) -> Hit {
        Hit { p, n, t, m }
    }
}

pub trait Hitable {
    fn hit(&self, ray: &Ray) -> Option<Hit>;
}

pub struct Scene {
    pub spheres: Vec<Sphere>,
}

impl Scene {
    pub fn new(spheres: Vec<Sphere>) -> Scene {
        Scene { spheres }
    }
}

impl Hitable for Scene {
    fn hit(&self, ray: &Ray) -> Option<Hit> {
        let mut smallest_t = std::f32::MAX;
        let mut closest_hit: Option<Hit> = None;
        for sphere in &self.spheres {
            if let Some(hit) = sphere.hit(ray) {
                if hit.t < smallest_t {
                    smallest_t = hit.t;
                    closest_hit = Some(hit);
                }
            }
        }
        closest_hit
    }
}

pub struct Sphere {
    c: Vec3,
    r: f32,
    m: Material,
}

impl Sphere {
    pub fn new(c: Vec3, r: f32, m: Material) -> Sphere {
        Sphere { c, r, m }
    }
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray) -> Option<Hit> {
        let oc = &ray.o - &self.c;
        let a = length_squared(&ray.d);
        let b = 2.0 * dot(&oc, &ray.d);
        let c = length_squared(&oc) - self.r * self.r;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant > 0.0 {
            let dsqrt = discriminant.sqrt();
            let t1 = (-b - dsqrt) / (2.0 * a);
            let t2 = (-b + dsqrt) / (2.0 * a);
            if t1 > 0.0 {
                let p = ray.point_at(t1);
                let mut n = &p - &self.c;
                n.normalize();
                Some(Hit::new(p, n, t1, self.m))
            } else if t2 > 0.0 {
                let p = ray.point_at(t2);
                let mut n = &p - &self.c;
                n.normalize();
                Some(Hit::new(p, n, t2, self.m))
            } else {
                None
            }
        } else {
            None
        }
    }
}
