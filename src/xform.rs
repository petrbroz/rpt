use std::ops;
use super::math::{ Degrees, deg_to_rad };
use super::vec3::{ Vec3, normalize, cross };
use super::mat4::{ Mat4, inverse, transpose };
use super::ray::Ray;
use super::bbox::BBox;

/// Affine transform represented by a 4x4 matrix,
/// also storing the inverse of the matrix.
#[derive(Debug, Copy, Clone)]
pub struct Transform {
    pub matrix: Mat4,
    pub inverse: Mat4,
    swaps_handedness: bool,
}

impl Transform {
    /// Create new transform with specific matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Transform matrix.
    /// * `inverse` - Inverse transform matrix.
    #[inline(always)]
    pub fn new(matrix: Mat4, inverse: Mat4) -> Transform {
        let swaps_handedness = {
            let m = &matrix;
            let det = m.m00 * (m.m11 * m.m22 - m.m12 * m.m21)
                - m.m01 * (m.m10 * m.m22 - m.m12 * m.m20)
                + m.m02 * (m.m10 * m.m21 - m.m11 * m.m20);
            det < 0.0
        };
        Transform { matrix, inverse, swaps_handedness }
    }

    /// Create new transform translating by given delta.
    ///
    /// # Arguments
    ///
    /// * `dx` - Translation along the X axis.
    /// * `dy` - Translation along the Y axis.
    /// * `dz` - Translation along the Z axis.
    #[inline(always)]
    pub fn translate(dx: f32, dy: f32, dz: f32) -> Transform {
        Transform::new(
            Mat4::new(
                1.0, 0.0, 0.0, dx,
                0.0, 1.0, 0.0, dy,
                0.0, 0.0, 1.0, dz,
                0.0, 0.0, 0.0, 1.0
            ),
            Mat4::new(
                1.0, 0.0, 0.0, -dx,
                0.0, 1.0, 0.0, -dy,
                0.0, 0.0, 1.0, -dz,
                0.0, 0.0, 0.0, 1.0
            )
        )
    }

    /// Create new transform scaling by given factors.
    ///
    /// # Arguments
    ///
    /// * `sx` - Scale along the X axis.
    /// * `sy` - Scale along the Y axis.
    /// * `sz` - Scale along the Z axis.
    #[inline(always)]
    pub fn scale(sx: f32, sy: f32, sz: f32) -> Transform {
        Transform::new(
            Mat4::new(
                sx, 0.0, 0.0, 0.0,
                0.0, sy, 0.0, 0.0,
                0.0, 0.0, sz, 0.0,
                0.0, 0.0, 0.0, 1.0
            ),
            Mat4::new(
                1.0 / sx, 0.0, 0.0, 0.0,
                0.0, 1.0 / sy, 0.0, 0.0,
                0.0, 0.0, 1.0 / sz, 0.0,
                0.0, 0.0, 0.0, 1.0
            )
        )
    }

    /// Create new transform rotating around the X axis.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in degrees.
    #[inline(always)]
    pub fn rotate_x(angle: Degrees) -> Transform {
        let sin_t = deg_to_rad(angle).sin();
        let cos_t = deg_to_rad(angle).cos();
        let m = Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, cos_t, -sin_t, 0.0,
            0.0, sin_t, cos_t, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
        Transform::new(m, transpose(&m))
    }

    /// Create new transform rotating around the Y axis.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in degrees.
    #[inline(always)]
    pub fn rotate_y(angle: Degrees) -> Transform {
        let sin_t = deg_to_rad(angle).sin();
        let cos_t = deg_to_rad(angle).cos();
        let m = Mat4::new(
            cos_t, 0.0, sin_t, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin_t, 0.0, cos_t, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
        Transform::new(m, transpose(&m))
    }

    /// Create new transform rotating around the Z axis.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in degrees.
    #[inline(always)]
    pub fn rotate_z(angle: Degrees) -> Transform {
        let sin_t = deg_to_rad(angle).sin();
        let cos_t = deg_to_rad(angle).cos();
        let m = Mat4::new(
            cos_t, -sin_t, 0.0, 0.0,
            sin_t, cos_t, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        );
        Transform::new(m, transpose(&m))
    }

    /// Create new transform rotating around arbitrary axis.
    ///
    /// # Arguments
    ///
    /// * `angle` - Rotation angle in degrees.
    /// * `axis` - Vector representing the rotation axis.
    #[inline(always)]
    pub fn rotate(angle: Degrees, axis: &Vec3) -> Transform {
        let a = normalize(axis);
        let s = deg_to_rad(angle).sin();
        let c = deg_to_rad(angle).cos();
        let m = Mat4::new(
            a.x * a.x + (1.0 - a.x * a.x) * c,
            a.x * a.y * (1.0 - c) - a.z * s,
            a.x * a.z * (1.0 - c) + a.y * s,
            0.0,
            a.x * a.y + (1.0 - c) + a.z * s,
            a.y * a.y * (1.0 - a.y * a.y) * c,
            a.y * a.z * (1.0 - c) - a.x * s,
            0.0,
            a.x * a.z + (1.0 - c) - a.y * s,
            a.y * a.z * (1.0 - c) + a.x * s,
            a.z * a.z * (1.0 - a.z * a.z) * c,
            0.0,
            0.0, 0.0, 0.0, 1.0
        );
        Transform::new(m, transpose(&m))
    }

    /// Create a "look at" transform that transforms input values
    /// to a new frame of reference that is "looking in a specific direction."
    ///
    /// # Arguments
    ///
    /// * `eye` - Desired origin.
    /// * `target` - Desired target (direction).
    /// * `up` - Desired up vector.
    #[inline(always)]
    pub fn look_at(eye: &Vec3, target: &Vec3, up: &Vec3) -> Transform {
        let dir = normalize(&(target - eye));
        let left = normalize(&cross(&normalize(up), &dir));
        let new_up = cross(&dir, &left);
        let m = Mat4::new(
            left.x, new_up.x, dir.x, eye.x,
            left.y, new_up.y, dir.y, eye.y,
            left.z, new_up.z, dir.z, eye.z,
            0.0, 0.0, 0.0, 1.0
        );
        Transform::new(inverse(&m), m)
    }

    /// Create orthographic projection transform.
    #[inline(always)]
    pub fn orthographic(near: f32, far: f32) -> Transform {
        let mut xform = Transform::translate(0.0, 0.0, -near);
        xform *= &Transform::scale(1.0, 1.0, 1.0 / (far - near));
        xform
    }

    /// Create perspective projection transform.
    #[inline(always)]
    pub fn perspective(fov: Degrees, near: f32, far: f32) -> Transform {
        let mut m = Mat4::identity();
        m.m22 = far / (far - near);
        m.m23 = -far * near / (far - near);
        m.m32 = 1.0;
        m.m33 = 0.0;
        let inv_tan = 1.0 / (0.5 * deg_to_rad(fov)).tan();
        let mut xform = Transform::new(m, inverse(&m));
        xform *= &Transform::scale(inv_tan, inv_tan, 1.0);
        xform
    }

    /// Transform vector (assuming the homogeneous coordinate to be 0.0).
    ///
    /// # Arguments
    ///
    /// * `v` - Input vector.
    #[inline(always)]
    pub fn apply_to_vector(&self, v: &Vec3) -> Vec3 {
        let x = self.matrix.m00 * v.x + self.matrix.m01 * v.y + self.matrix.m02 * v.z;
        let y = self.matrix.m10 * v.x + self.matrix.m11 * v.y + self.matrix.m12 * v.z;
        let z = self.matrix.m20 * v.x + self.matrix.m21 * v.y + self.matrix.m22 * v.z;
        Vec3::new(x, y, z)
    }

    /// Transform vector as a point (assuming the homogeneous coordinate to be 1.0).
    ///
    /// # Arguments
    ///
    /// * `p` - Input point.
    #[inline(always)]
    pub fn apply_to_point(&self, p: &Vec3) -> Vec3 {
        let x = self.matrix.m00 * p.x + self.matrix.m01 * p.y + self.matrix.m02 * p.z + self.matrix.m03;
        let y = self.matrix.m10 * p.x + self.matrix.m11 * p.y + self.matrix.m12 * p.z + self.matrix.m13;
        let z = self.matrix.m20 * p.x + self.matrix.m21 * p.y + self.matrix.m22 * p.z + self.matrix.m23;
        let w = self.matrix.m30 * p.x + self.matrix.m31 * p.y + self.matrix.m32 * p.z + self.matrix.m33;
        if w != 1.0 {
            let inv = 1.0 / w;
            Vec3::new(inv * x, inv * y, inv * z)
        } else {
            Vec3::new(x, y, z)
        }
    }

    /// Transform vector as a normal.
    ///
    /// # Arguments
    ///
    /// * `n` - Input normal.
    #[inline(always)]
    pub fn apply_to_normal(&self, n: &Vec3) -> Vec3 {
        let x = self.inverse.m00 * n.x + self.inverse.m10 * n.y + self.inverse.m20 * n.z;
        let y = self.inverse.m01 * n.x + self.inverse.m11 * n.y + self.inverse.m21 * n.z;
        let z = self.inverse.m02 * n.x + self.inverse.m12 * n.y + self.inverse.m22 * n.z;
        Vec3::new(x, y, z)
    }

    /// Transform ray.
    ///
    /// # Arguments
    ///
    /// * `r` - Input ray.
    #[inline(always)]
    pub fn apply_to_ray(&self, r: &Ray) -> Ray {
        Ray::new(
            self.apply_to_point(&r.o),
            self.apply_to_vector(&r.d),
        )
    }

    /// Transform bounding box.
    ///
    /// # Arguments
    ///
    /// * `bbox` - Input bounding box.
    #[inline(always)]
    pub fn apply_to_bbox(&self, bbox: &BBox) -> BBox {
        let mut new_bbox = BBox::new();
        let mut p_in = Vec3::new(0.0, 0.0, 0.0);
        let mut p_out;
        p_in.x = bbox.min.x; p_in.y = bbox.min.y; p_in.z = bbox.min.z; p_out = self.apply_to_point(&p_in); new_bbox += &p_out;
        p_in.x = bbox.min.x; p_in.y = bbox.min.y; p_in.z = bbox.max.z; p_out = self.apply_to_point(&p_in); new_bbox += &p_out;
        p_in.x = bbox.min.x; p_in.y = bbox.max.y; p_in.z = bbox.min.z; p_out = self.apply_to_point(&p_in); new_bbox += &p_out;
        p_in.x = bbox.min.x; p_in.y = bbox.max.y; p_in.z = bbox.max.z; p_out = self.apply_to_point(&p_in); new_bbox += &p_out;
        p_in.x = bbox.max.x; p_in.y = bbox.min.y; p_in.z = bbox.min.z; p_out = self.apply_to_point(&p_in); new_bbox += &p_out;
        p_in.x = bbox.max.x; p_in.y = bbox.min.y; p_in.z = bbox.max.z; p_out = self.apply_to_point(&p_in); new_bbox += &p_out;
        p_in.x = bbox.max.x; p_in.y = bbox.max.y; p_in.z = bbox.min.z; p_out = self.apply_to_point(&p_in); new_bbox += &p_out;
        p_in.x = bbox.max.x; p_in.y = bbox.max.y; p_in.z = bbox.max.z; p_out = self.apply_to_point(&p_in); new_bbox += &p_out;
        new_bbox
    }

    /// Invert transformation.
    pub fn invert(&self) -> Transform {
        Transform::new(self.inverse, self.matrix)
    }
}

impl ops::Mul<&Transform> for &Transform {
    type Output = Transform;

    /// Create new transform by combining two transforms.
    #[inline(always)]
    fn mul(self, xform: &Transform) -> Self::Output {
        Transform::new(
            &self.matrix * &xform.matrix,
            &xform.inverse * &self.inverse,
        )
    }
}

impl ops::MulAssign<&Transform> for Transform {
    /// Combine this transform with another transform.
    #[inline(always)]
    fn mul_assign(&mut self, xform: &Transform) {
        self.matrix = &self.matrix * &xform.matrix;
        self.inverse = &xform.inverse * &self.inverse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_point() {
        let t = Transform::translate(0.5, -0.5, 1.0);
        let p = Vec3::new(2.0, 3.0, 4.0);
        let tp = t.apply_to_point(&p);
        assert_eq!(tp.x, 2.5); assert_eq!(tp.y, 2.5); assert_eq!(tp.z, 5.0);
    }

    #[test]
    fn translate_vector() {
        let t = Transform::translate(0.5, -0.5, 1.0);
        let v = Vec3::new(2.0, 3.0, 4.0);
        let tv = t.apply_to_vector(&v);
        assert_eq!(tv.x, 2.0); assert_eq!(tv.y, 3.0); assert_eq!(tv.z, 4.0);
    }

    #[test]
    fn translate_normal() {
        let t = Transform::translate(0.5, -0.5, 1.0);
        let n = Vec3::new(2.0, 3.0, 4.0);
        let tn = t.apply_to_normal(&n);
        assert_eq!(tn.x, 2.0); assert_eq!(tn.y, 3.0); assert_eq!(tn.z, 4.0);
    }

    #[test]
    fn scale_point() {
        let t = Transform::scale(0.5, -0.5, 1.0);
        let p = Vec3::new(2.0, 3.0, 4.0);
        let tp = t.apply_to_point(&p);
        assert_eq!(tp.x, 1.0); assert_eq!(tp.y, -1.5); assert_eq!(tp.z, 4.0);
    }

    #[test]
    fn scale_vector() {
        let t = Transform::scale(0.5, -0.5, 1.0);
        let v = Vec3::new(2.0, 3.0, 4.0);
        let tv = t.apply_to_vector(&v);
        assert_eq!(tv.x, 1.0); assert_eq!(tv.y, -1.5); assert_eq!(tv.z, 4.0);
    }

    #[test]
    fn scale_normal() {
        let t = Transform::scale(0.5, -0.5, 1.0);
        let n = Vec3::new(2.0, 3.0, 4.0);
        let tn = t.apply_to_normal(&n);
        assert_eq!(tn.x, 4.0); assert_eq!(tn.y, -6.0); assert_eq!(tn.z, 4.0);
    }

    #[test]
    fn rotate_point() {
        let t = Transform::rotate_z(90.0);
        let p = Vec3::new(1.0, 1.0, 0.0);
        let tp = t.apply_to_point(&p);
        assert_eq!(tp.x, -1.0); assert_eq!(tp.y, 0.99999994); assert_eq!(tp.z, 0.0);
    }

    #[test]
    fn rotate_vector() {
        let t = Transform::rotate_z(90.0);
        let v = Vec3::new(1.0, 1.0, 0.0);
        let tv = t.apply_to_vector(&v);
        assert_eq!(tv.x, -1.0); assert_eq!(tv.y, 0.99999994); assert_eq!(tv.z, 0.0);
    }

    #[test]
    fn rotate_normal() {
        let t = Transform::rotate_z(90.0);
        let n = Vec3::new(1.0, 1.0, 0.0);
        let tn = t.apply_to_normal(&n);
        assert_eq!(tn.x, -1.0); assert_eq!(tn.y, 0.99999994); assert_eq!(tn.z, 0.0);
    }

    #[test]
    fn combine_xforms() {
        let t1 = Transform::rotate_z(90.0);
        let t2 = Transform::translate(0.5, -0.5, 1.0);
        let t = &t2 * &t1;
        let p = Vec3::new(1.0, 1.0, 0.0);
        let tp = t.apply_to_point(&p);
        assert_eq!(tp.x, -0.5); assert_eq!(tp.y, 0.49999994); assert_eq!(tp.z, 1.0);
    }

    #[test]
    fn add_xform_to_self() {
        let mut t = Transform::translate(0.5, -0.5, 1.0);
        let t2 = Transform::rotate_z(90.0);
        t *= &t2;
        let p = Vec3::new(1.0, 1.0, 0.0);
        let tp = t.apply_to_point(&p);
        assert_eq!(tp.x, -0.5); assert_eq!(tp.y, 0.49999994); assert_eq!(tp.z, 1.0);
    }
}
