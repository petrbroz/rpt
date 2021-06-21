use super::vec3::Vec3;

#[derive(Debug, Copy, Clone)]
pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,
}

impl Ray {
    pub fn new(o: Vec3, d: Vec3) -> Ray {
        Ray { o, d }
    }

    pub fn point_at(&self, t: f32) -> Vec3 {
        Vec3::new(
            self.o.x + t * self.d.x,
            self.o.y + t * self.d.y,
            self.o.z + t * self.d.z,
        )
    }
}
