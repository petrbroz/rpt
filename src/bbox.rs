use super::vec3::{ Vec3, distance };
use std::ops;

/// Axis aligned bounding box.
#[derive(Debug, Copy, Clone)]
pub struct BBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BBox {
    /// Create new, empty bounding box, with max values set to infinity, and min values set to -infinity.
    #[inline(always)]
    pub fn new() -> BBox {
        BBox {
            min: Vec3::new(std::f32::INFINITY, std::f32::INFINITY, std::f32::INFINITY),
            max: Vec3::new(std::f32::NEG_INFINITY, std::f32::NEG_INFINITY, std::f32::NEG_INFINITY),
        }
    }

    /// Create new bounding box for single point.
    #[inline(always)]
    pub fn new_from_point(p: Vec3) -> BBox {
        BBox { min: p, max: p }
    }

    /// Create new bounding box for two specific points.
    #[inline(always)]
    pub fn new_from_points(p1: &Vec3, p2: &Vec3) -> BBox {
        BBox {
            min: Vec3::new(p1.x.min(p2.x), p1.y.min(p2.y), p1.z.min(p2.z)),
            max: Vec3::new(p1.x.max(p2.x), p1.y.max(p2.y), p1.z.max(p2.z)),
        }
    }

    /// Check whether the bounding box partially overlaps another one.
    #[inline(always)]
    pub fn overlaps(self, bbox: BBox) -> bool {
        self.max.x >= bbox.min.x && self.min.x <= bbox.max.x
        && self.max.y >= bbox.min.y && self.min.y <= bbox.max.y
        && self.max.z >= bbox.min.z && self.min.z <= bbox.max.z
    }

    /// Check whether the bounding box contains given point.
    #[inline(always)]
    pub fn contains(self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
        && p.y >= self.min.y && p.y <= self.max.y
        && p.z >= self.min.z && p.z <= self.max.z
    }

    /// Expand the bounding box by specific delta in each direction.
    #[inline(always)]
    pub fn expand(&mut self, delta: f32) {
        self.min.x -= delta;
        self.min.y -= delta;
        self.min.z -= delta;
        self.max.x += delta;
        self.max.y += delta;
        self.max.z += delta;
    }

    /// Compute the surface area.
    #[inline(always)]
    pub fn surface(self) -> f32 {
        let d = &self.max - &self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.x * d.z)
    }

    /// Compute the volume.
    #[inline(always)]
    pub fn volume(self) -> f32 {
        let d = &self.max - &self.min;
        d.x * d.y * d.z
    }

    /// Transform relative XYZ value within the bounding box to absolute position.
    #[inline(always)]
    pub fn rel_to_abs(self, t: Vec3) -> Vec3 {
        let d = &self.max - &self.min;
        Vec3::new(
            self.min.x + t.x * d.x,
            self.min.y + t.y * d.y,
            self.min.z + t.z * d.z,
        )
    }

    /// Transform absolute position to relative XYZ value within the bounding box.
    #[inline(always)]
    pub fn abs_to_rel(self, p: Vec3) -> Vec3 {
        let d = &self.max - &self.min;
        Vec3::new(
            (p.x - self.min.x) / d.x,
            (p.y - self.min.y) / d.y,
            (p.z - self.min.z) / d.z,
        )
    }

    /// Compute center point of the bounding box.
    #[inline(always)]
    pub fn center(self) -> Vec3 {
        Vec3::new(
            0.5 * (self.min.x + self.max.x),
            0.5 * (self.min.y + self.max.y),
            0.5 * (self.min.z + self.max.z),
        )
    }

    /// Compute bounding sphere.
    ///
    /// # Returns
    ///
    /// (sphere center, sphere radius).
    #[inline(always)]
    pub fn bounding_shpere(self) -> (Vec3, f32) {
        let center = self.center();
        (center, distance(&self.min, &center))
    }
}

impl ops::Add<&Vec3> for &BBox {
    type Output = BBox;

    /// Create new bounding box by expanding this one with another point.
    #[inline(always)]
    fn add(self, p: &Vec3) -> Self::Output {
        BBox {
            min: Vec3::new(self.min.x.min(p.x), self.min.y.min(p.y), self.min.z.min(p.z)),
            max: Vec3::new(self.max.x.max(p.x), self.max.y.max(p.y), self.max.z.max(p.z)),
        }
    }
}

impl ops::AddAssign<&Vec3> for BBox {
    /// Expand this bounding box with another point.
    #[inline(always)]
    fn add_assign(&mut self, v: &Vec3) {
        self.min.x = self.min.x.min(v.x);
        self.min.y = self.min.y.min(v.y);
        self.min.z = self.min.y.min(v.z);
        self.max.x = self.max.x.max(v.x);
        self.max.y = self.max.y.max(v.y);
        self.max.z = self.max.y.max(v.z);
    }
}

impl ops::Add<&BBox> for &BBox {
    type Output = BBox;

    /// Create new bounding box as a union of this bounding box with another one.
    #[inline(always)]
    fn add(self, bbox: &BBox) -> Self::Output {
        BBox {
            min: Vec3::new(self.min.x.min(bbox.min.x), self.min.y.min(bbox.min.y), self.min.z.min(bbox.min.z)),
            max: Vec3::new(self.max.x.max(bbox.max.x), self.max.y.max(bbox.max.y), self.max.z.max(bbox.max.z)),
        }
    }
}

impl ops::AddAssign<&BBox> for BBox {
    /// Expand this bounding box with another one.
    #[inline(always)]
    fn add_assign(&mut self, bbox: &BBox) {
        self.min.x = self.min.x.min(bbox.min.x);
        self.min.y = self.min.y.min(bbox.min.y);
        self.min.z = self.min.y.min(bbox.min.z);
        self.max.x = self.max.x.max(bbox.max.x);
        self.max.y = self.max.y.max(bbox.max.y);
        self.max.z = self.max.y.max(bbox.max.z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_bboxes() {
        let bbox1 = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(0.75, 0.75, 0.75));
        let bbox2 = BBox::new_from_points(&Vec3::new(1.0, 1.0, 1.0), &Vec3::new(2.0, 2.0, 2.0));
        let bbox3 = BBox::new_from_points(&Vec3::new(0.5, 0.5, 0.5), &Vec3::new(1.5, 1.5, 1.5));
        assert_eq!(bbox1.overlaps(bbox2), false);
        assert_eq!(bbox1.overlaps(bbox3), true);
    }

    #[test]
    fn contains_point() {
        let bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(0.75, 0.75, 0.75));
        assert_eq!(bbox.contains(Vec3::new(0.0, 0.0, 0.0)), true);
        assert_eq!(bbox.contains(Vec3::new(1.0, 0.0, 0.0)), false);
    }

    #[test]
    fn expand_bbox() {
        let mut bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(0.75, 0.75, 0.75));
        bbox.expand(0.25);
        assert_eq!(bbox.min.x, -1.25); assert_eq!(bbox.min.y, -1.25); assert_eq!(bbox.min.z, -1.25);
        assert_eq!(bbox.max.x, 1.0); assert_eq!(bbox.max.y, 1.0); assert_eq!(bbox.max.z, 1.0);
    }

    #[test]
    fn bbox_surface() {
        let bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(bbox.surface(), 24.0);
    }

    #[test]
    fn bbox_volume() {
        let bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(bbox.volume(), 8.0);
    }

    #[test]
    fn rel_vs_abs_coords() {
        let bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(1.0, 1.0, 1.0));
        let abs = bbox.rel_to_abs(Vec3::new(0.75, 0.75, 0.75));
        assert_eq!(abs.x, 0.5); assert_eq!(abs.y, 0.5); assert_eq!(abs.z, 0.5);
        let rel = bbox.abs_to_rel(Vec3::new(-0.5, -0.5, -0.5));
        assert_eq!(rel.x, 0.25); assert_eq!(rel.y, 0.25); assert_eq!(rel.z, 0.25);
    }

    #[test]
    fn compute_center() {
        let bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(1.0, 1.0, 1.0));
        let center = bbox.center();
        assert_eq!(center.x, 0.0); assert_eq!(center.y, 0.0); assert_eq!(center.z, 0.0);
    }

    #[test]
    fn compute_bounding_sphere() {
        let bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(1.0, 1.0, 1.0));
        let (center, radius) = bbox.bounding_shpere();
        assert_eq!(center.x, 0.0); assert_eq!(center.y, 0.0); assert_eq!(center.z, 0.0);
        assert_eq!(radius, 1.7320508);
    }

    #[test]
    fn add_bboxes() {
        let bbox1 = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(0.75, 0.75, 0.75));
        let bbox2 = BBox::new_from_points(&Vec3::new(1.0, 1.0, 1.0), &Vec3::new(2.0, 2.0, 2.0));
        let sum = &bbox1 + &bbox2;
        assert_eq!(sum.min.x, -1.0); assert_eq!(sum.min.y, -1.0); assert_eq!(sum.min.z, -1.0);
        assert_eq!(sum.max.x, 2.0); assert_eq!(sum.max.y, 2.0); assert_eq!(sum.max.z, 2.0);
    }

    #[test]
    fn add_point_to_bbox() {
        let bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(0.75, 0.75, 0.75));
        let p = Vec3::new(2.0, 2.0, 2.0);
        let sum = &bbox + &p;
        assert_eq!(sum.min.x, -1.0); assert_eq!(sum.min.y, -1.0); assert_eq!(sum.min.z, -1.0);
        assert_eq!(sum.max.x, 2.0); assert_eq!(sum.max.y, 2.0); assert_eq!(sum.max.z, 2.0);
    }

    #[test]
    fn add_bbox_to_self() {
        let mut bbox1 = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(0.75, 0.75, 0.75));
        let bbox2 = BBox::new_from_points(&Vec3::new(1.0, 1.0, 1.0), &Vec3::new(2.0, 2.0, 2.0));
        bbox1 += &bbox2;
        assert_eq!(bbox1.min.x, -1.0); assert_eq!(bbox1.min.y, -1.0); assert_eq!(bbox1.min.z, -1.0);
        assert_eq!(bbox1.max.x, 2.0); assert_eq!(bbox1.max.y, 2.0); assert_eq!(bbox1.max.z, 2.0);
    }

    #[test]
    fn add_point_to_self() {
        let mut bbox = BBox::new_from_points(&Vec3::new(-1.0, -1.0, -1.0), &Vec3::new(0.75, 0.75, 0.75));
        let p = Vec3::new(2.0, 2.0, 2.0);
        bbox += &p;
        assert_eq!(bbox.min.x, -1.0); assert_eq!(bbox.min.y, -1.0); assert_eq!(bbox.min.z, -1.0);
        assert_eq!(bbox.max.x, 2.0); assert_eq!(bbox.max.y, 2.0); assert_eq!(bbox.max.z, 2.0);
    }
}
