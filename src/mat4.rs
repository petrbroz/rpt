use std::ops;

/// Matrix of 4x4 floats.
#[derive(Debug, Copy, Clone)]
pub struct Mat4 {
    pub m00: f32, pub m01: f32, pub m02: f32, pub m03: f32,
    pub m10: f32, pub m11: f32, pub m12: f32, pub m13: f32,
    pub m20: f32, pub m21: f32, pub m22: f32, pub m23: f32,
    pub m30: f32, pub m31: f32, pub m32: f32, pub m33: f32,
}

impl Mat4 {
    /// Create new matrix with all 16 values specified.
    #[inline(always)]
    pub fn new(
        m00: f32, m01: f32, m02: f32, m03: f32,
        m10: f32, m11: f32, m12: f32, m13: f32,
        m20: f32, m21: f32, m22: f32, m23: f32,
        m30: f32, m31: f32, m32: f32, m33: f32,
    ) -> Mat4 {
        Mat4 {
            m00, m01, m02, m03,
            m10, m11, m12, m13,
            m20, m21, m22, m23,
            m30, m31, m32, m33,
        }
    }

    /// Create new identity matrix.
    #[inline(always)]
    pub fn identity() -> Mat4 {
        Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Check if matrix has any NaN values.
    #[inline(always)]
    pub fn has_nans(&self) -> bool {
        self.m00.is_nan() || self.m01.is_nan() || self.m02.is_nan() || self.m03.is_nan() ||
        self.m10.is_nan() || self.m11.is_nan() || self.m12.is_nan() || self.m13.is_nan() ||
        self.m20.is_nan() || self.m21.is_nan() || self.m22.is_nan() || self.m23.is_nan() ||
        self.m30.is_nan() || self.m31.is_nan() || self.m32.is_nan() || self.m33.is_nan()
    }
}

/// Transpose matrix.
///
/// # Arguments
///
/// * `m` - Input matrix.
#[inline(always)]
pub fn transpose(m: &Mat4) -> Mat4 {
    Mat4 {
        m00: m.m00, m01: m.m10, m02: m.m20, m03: m.m30,
        m10: m.m01, m11: m.m11, m12: m.m21, m13: m.m31,
        m20: m.m02, m21: m.m12, m22: m.m22, m23: m.m32,
        m30: m.m03, m31: m.m13, m32: m.m23, m33: m.m33,
    }
}

/// Compute matrix determinant.
///
/// # Arguments
///
/// * `m` - Input matrix.
#[inline(always)]
pub fn determinant(m: &Mat4) -> f32 {
    m.m03*m.m12*m.m21*m.m30 - m.m02*m.m13*m.m21*m.m30 - m.m03*m.m11*m.m22*m.m30 + m.m01*m.m13*m.m22*m.m30
    + m.m02*m.m11*m.m23*m.m30 - m.m01*m.m12*m.m23*m.m30 - m.m03*m.m12*m.m20*m.m31 + m.m02*m.m13*m.m20*m.m31
    + m.m03*m.m10*m.m22*m.m31 - m.m00*m.m13*m.m22*m.m31 - m.m02*m.m10*m.m23*m.m31 + m.m00*m.m12*m.m23*m.m31
    + m.m03*m.m11*m.m20*m.m32 - m.m01*m.m13*m.m20*m.m32 - m.m03*m.m10*m.m21*m.m32 + m.m00*m.m13*m.m21*m.m32
    + m.m01*m.m10*m.m23*m.m32 - m.m00*m.m11*m.m23*m.m32 - m.m02*m.m11*m.m20*m.m33 + m.m01*m.m12*m.m20*m.m33
    + m.m02*m.m10*m.m21*m.m33 - m.m00*m.m12*m.m21*m.m33 - m.m01*m.m10*m.m22*m.m33 + m.m00*m.m11*m.m22*m.m33
}

/// Compute inverted matrix.
///
/// # Arguments
///
/// * `m` - Input matrix.
///
/// # Panics
///
/// When the matrix is singular (determinant is zero).
#[inline(always)]
pub fn inverse(m: &Mat4) -> Mat4 {
    let det = determinant(m);
    debug_assert_ne!(det, 0.0);
    let mut adj = Mat4::new(
        m.m12*m.m23*m.m31 - m.m13*m.m22*m.m31 + m.m13*m.m21*m.m32 - m.m11*m.m23*m.m32 - m.m12*m.m21*m.m33 + m.m11*m.m22*m.m33,
        m.m03*m.m22*m.m31 - m.m02*m.m23*m.m31 - m.m03*m.m21*m.m32 + m.m01*m.m23*m.m32 + m.m02*m.m21*m.m33 - m.m01*m.m22*m.m33,
        m.m02*m.m13*m.m31 - m.m03*m.m12*m.m31 + m.m03*m.m11*m.m32 - m.m01*m.m13*m.m32 - m.m02*m.m11*m.m33 + m.m01*m.m12*m.m33,
        m.m03*m.m12*m.m21 - m.m02*m.m13*m.m21 - m.m03*m.m11*m.m22 + m.m01*m.m13*m.m22 + m.m02*m.m11*m.m23 - m.m01*m.m12*m.m23,
        m.m13*m.m22*m.m30 - m.m12*m.m23*m.m30 - m.m13*m.m20*m.m32 + m.m10*m.m23*m.m32 + m.m12*m.m20*m.m33 - m.m10*m.m22*m.m33,
        m.m02*m.m23*m.m30 - m.m03*m.m22*m.m30 + m.m03*m.m20*m.m32 - m.m00*m.m23*m.m32 - m.m02*m.m20*m.m33 + m.m00*m.m22*m.m33,
        m.m03*m.m12*m.m30 - m.m02*m.m13*m.m30 - m.m03*m.m10*m.m32 + m.m00*m.m13*m.m32 + m.m02*m.m10*m.m33 - m.m00*m.m12*m.m33,
        m.m02*m.m13*m.m20 - m.m03*m.m12*m.m20 + m.m03*m.m10*m.m22 - m.m00*m.m13*m.m22 - m.m02*m.m10*m.m23 + m.m00*m.m12*m.m23,
        m.m11*m.m23*m.m30 - m.m13*m.m21*m.m30 + m.m13*m.m20*m.m31 - m.m10*m.m23*m.m31 - m.m11*m.m20*m.m33 + m.m10*m.m21*m.m33,
        m.m03*m.m21*m.m30 - m.m01*m.m23*m.m30 - m.m03*m.m20*m.m31 + m.m00*m.m23*m.m31 + m.m01*m.m20*m.m33 - m.m00*m.m21*m.m33,
        m.m01*m.m13*m.m30 - m.m03*m.m11*m.m30 + m.m03*m.m10*m.m31 - m.m00*m.m13*m.m31 - m.m01*m.m10*m.m33 + m.m00*m.m11*m.m33,
        m.m03*m.m11*m.m20 - m.m01*m.m13*m.m20 - m.m03*m.m10*m.m21 + m.m00*m.m13*m.m21 + m.m01*m.m10*m.m23 - m.m00*m.m11*m.m23,
        m.m12*m.m21*m.m30 - m.m11*m.m22*m.m30 - m.m12*m.m20*m.m31 + m.m10*m.m22*m.m31 + m.m11*m.m20*m.m32 - m.m10*m.m21*m.m32,
        m.m01*m.m22*m.m30 - m.m02*m.m21*m.m30 + m.m02*m.m20*m.m31 - m.m00*m.m22*m.m31 - m.m01*m.m20*m.m32 + m.m00*m.m21*m.m32,
        m.m02*m.m11*m.m30 - m.m01*m.m12*m.m30 - m.m02*m.m10*m.m31 + m.m00*m.m12*m.m31 + m.m01*m.m10*m.m32 - m.m00*m.m11*m.m32,
        m.m01*m.m12*m.m20 - m.m02*m.m11*m.m20 + m.m02*m.m10*m.m21 - m.m00*m.m12*m.m21 - m.m01*m.m10*m.m22 + m.m00*m.m11*m.m22
    );
    adj *= 1.0 / det;
    adj
}

impl ops::Mul<&Mat4> for &Mat4 {
    type Output = Mat4;

    /// Compute new matrix by multiplying this matrix with another one.
    #[inline(always)]
    fn mul(self, m: &Mat4) -> Self::Output {
        Mat4 {
            m00: self.m00 * m.m00 + self.m01 * m.m10 + self.m02 * m.m20 + self.m03 * m.m30,
            m01: self.m00 * m.m01 + self.m01 * m.m11 + self.m02 * m.m21 + self.m03 * m.m31,
            m02: self.m00 * m.m02 + self.m01 * m.m12 + self.m02 * m.m22 + self.m03 * m.m32,
            m03: self.m00 * m.m03 + self.m01 * m.m13 + self.m02 * m.m23 + self.m03 * m.m33,
            m10: self.m10 * m.m00 + self.m11 * m.m10 + self.m12 * m.m20 + self.m13 * m.m30,
            m11: self.m10 * m.m01 + self.m11 * m.m11 + self.m12 * m.m21 + self.m13 * m.m31,
            m12: self.m10 * m.m02 + self.m11 * m.m12 + self.m12 * m.m22 + self.m13 * m.m32,
            m13: self.m10 * m.m03 + self.m11 * m.m13 + self.m12 * m.m23 + self.m13 * m.m33,
            m20: self.m20 * m.m00 + self.m21 * m.m10 + self.m22 * m.m20 + self.m23 * m.m30,
            m21: self.m20 * m.m01 + self.m21 * m.m11 + self.m22 * m.m21 + self.m23 * m.m31,
            m22: self.m20 * m.m02 + self.m21 * m.m12 + self.m22 * m.m22 + self.m23 * m.m32,
            m23: self.m20 * m.m03 + self.m21 * m.m13 + self.m22 * m.m23 + self.m23 * m.m33,
            m30: self.m30 * m.m00 + self.m31 * m.m10 + self.m32 * m.m20 + self.m33 * m.m30,
            m31: self.m30 * m.m01 + self.m31 * m.m11 + self.m32 * m.m21 + self.m33 * m.m31,
            m32: self.m30 * m.m02 + self.m31 * m.m12 + self.m32 * m.m22 + self.m33 * m.m32,
            m33: self.m30 * m.m03 + self.m31 * m.m13 + self.m32 * m.m23 + self.m33 * m.m33
        }
    }
}

impl ops::MulAssign<&Mat4> for Mat4 {
    /// Multiply this matrix by another one.
    #[inline(always)]
    fn mul_assign(&mut self, m: &Mat4) {
        let mut tmp1 = self.m00; let mut tmp2 = self.m01; let mut tmp3 = self.m02; let mut tmp4 = self.m03;
        self.m00 = tmp1 * m.m00 + tmp2 * m.m10 + tmp3 * m.m20 + tmp4 * m.m30;
        self.m01 = tmp1 * m.m01 + tmp2 * m.m11 + tmp3 * m.m21 + tmp4 * m.m31;
        self.m02 = tmp1 * m.m02 + tmp2 * m.m12 + tmp3 * m.m22 + tmp4 * m.m32;
        self.m03 = tmp1 * m.m03 + tmp2 * m.m13 + tmp3 * m.m23 + tmp4 * m.m33;
        tmp1 = self.m10; tmp2 = self.m11; tmp3 = self.m12; tmp4 = self.m13;
        self.m10 = tmp1 * m.m00 + tmp2 * m.m10 + tmp3 * m.m20 + tmp4 * m.m30;
        self.m11 = tmp1 * m.m01 + tmp2 * m.m11 + tmp3 * m.m21 + tmp4 * m.m31;
        self.m12 = tmp1 * m.m02 + tmp2 * m.m12 + tmp3 * m.m22 + tmp4 * m.m32;
        self.m13 = tmp1 * m.m03 + tmp2 * m.m13 + tmp3 * m.m23 + tmp4 * m.m33;
        tmp1 = self.m20; tmp2 = self.m21; tmp3 = self.m22; tmp4 = self.m23;
        self.m20 = tmp1 * m.m00 + tmp2 * m.m10 + tmp3 * m.m20 + tmp4 * m.m30;
        self.m21 = tmp1 * m.m01 + tmp2 * m.m11 + tmp3 * m.m21 + tmp4 * m.m31;
        self.m22 = tmp1 * m.m02 + tmp2 * m.m12 + tmp3 * m.m22 + tmp4 * m.m32;
        self.m23 = tmp1 * m.m03 + tmp2 * m.m13 + tmp3 * m.m23 + tmp4 * m.m33;
        tmp1 = self.m30; tmp2 = self.m31; tmp3 = self.m32; tmp4 = self.m33;
        self.m30 = tmp1 * m.m00 + tmp2 * m.m10 + tmp3 * m.m20 + tmp4 * m.m30;
        self.m31 = tmp1 * m.m01 + tmp2 * m.m11 + tmp3 * m.m21 + tmp4 * m.m31;
        self.m32 = tmp1 * m.m02 + tmp2 * m.m12 + tmp3 * m.m22 + tmp4 * m.m32;
        self.m33 = tmp1 * m.m03 + tmp2 * m.m13 + tmp3 * m.m23 + tmp4 * m.m33;
    }
}

impl ops::MulAssign<f32> for Mat4 {
    /// Multiply this matrix by scalar.
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) {
        self.m00 *= s; self.m01 *= s; self.m02 *= s; self.m03 *= s;
        self.m10 *= s; self.m11 *= s; self.m12 *= s; self.m13 *= s;
        self.m20 *= s; self.m21 *= s; self.m22 *= s; self.m23 *= s;
        self.m30 *= s; self.m31 *= s; self.m32 *= s; self.m33 *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_mat4_eq(m1: &Mat4, m2: &Mat4) {
        assert_eq!(m1.m00, m2.m00); assert_eq!(m1.m01, m2.m01); assert_eq!(m1.m02, m2.m02); assert_eq!(m1.m03, m2.m03);
        assert_eq!(m1.m10, m2.m10); assert_eq!(m1.m11, m2.m11); assert_eq!(m1.m12, m2.m12); assert_eq!(m1.m13, m2.m13);
        assert_eq!(m1.m20, m2.m20); assert_eq!(m1.m21, m2.m21); assert_eq!(m1.m22, m2.m22); assert_eq!(m1.m23, m2.m23);
        assert_eq!(m1.m30, m2.m30); assert_eq!(m1.m31, m2.m31); assert_eq!(m1.m32, m2.m32); assert_eq!(m1.m33, m2.m33);
    }

    #[test]
    fn transpose_matrix() {
        let m = Mat4::new(
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6,
        );
        let transposed = transpose(&m);
        let expected = Mat4::new(
            0.1, 0.5, 0.9, 1.3,
            0.2, 0.6, 1.0, 1.4,
            0.3, 0.7, 1.1, 1.5,
            0.4, 0.8, 1.2, 1.6,
        );
        assert_mat4_eq(&transposed, &expected);
    }

    #[test]
    fn get_determinant() {
        let m = Mat4::new(
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.6,
        );
        assert_eq!(determinant(&m), -0.000000029802322);
    }

    #[test]
    fn get_determinant_identity() {
        let m = Mat4::identity();
        assert_eq!(determinant(&m), 1.0);
    }

    #[test]
    fn invert_matrix() {
        let m = Mat4::new(
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, -3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0,
        );
        let inverted = inverse(&m);
        let expected = Mat4::new(
            1.0, 0.0, 0.0, -2.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, -4.0,
            0.0, 0.0, 0.0, 1.0,
        );
        assert_mat4_eq(&inverted, &expected);
    }

    #[test]
    #[should_panic]
    fn invert_singular_matrix() {
        let m = Mat4::new(
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        let _ = inverse(&m);
    }

    #[test]
    fn multiply_matrices() {
        let m1 = Mat4::new(
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, -3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0,
        );
        let m2 = Mat4::new(
            2.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        let multiplied = &m1 * &m2;
        let expected = Mat4::new(
            2.0, 0.0, 0.0, 2.0,
            0.0, 3.0, 0.0, -3.0,
            0.0, 0.0, 4.0, 4.0,
            0.0, 0.0, 0.0, 1.0,
        );
        assert_mat4_eq(&multiplied, &expected);
    }

    #[test]
    fn multiply_self() {
        let mut m = Mat4::new(
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, -3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0,
        );
        let m2 = Mat4::new(
            2.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );
        m *= &m2;
        let expected = Mat4::new(
            2.0, 0.0, 0.0, 2.0,
            0.0, 3.0, 0.0, -3.0,
            0.0, 0.0, 4.0, 4.0,
            0.0, 0.0, 0.0, 1.0,
        );
        assert_mat4_eq(&m, &expected);
    }

    #[test]
    fn scale_matrix() {
        let mut m = Mat4::new(
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, -3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0,
        );
        m *= 2.0;
        let expected = Mat4::new(
            2.0, 0.0, 0.0, 4.0,
            0.0, 2.0, 0.0, -6.0,
            0.0, 0.0, 2.0, 8.0,
            0.0, 0.0, 0.0, 2.0,
        );
        assert_mat4_eq(&m, &expected);
    }
}
