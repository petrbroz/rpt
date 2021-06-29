use std::ops;

#[derive(Debug, Copy, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[inline(always)]
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        //debug_assert_ne!(x, std::f32::NAN);
        //debug_assert_ne!(y, std::f32::NAN);
        //debug_assert_ne!(z, std::f32::NAN);
        Vec3 { x, y, z }
    }

    #[inline(always)]
    pub fn normalize(&mut self) {
        let len = length(self);
        //debug_assert_ne!(len, 0.0);
        let inv_len = 1.0 / len;
        self.x *= inv_len;
        self.y *= inv_len;
        self.z *= inv_len;
    }
}

#[inline(always)]
pub fn cross(v1: &Vec3, v2: &Vec3) -> Vec3 {
    Vec3 {
        x: v1.y * v2.z - v1.z * v2.y,
        y: v1.z * v2.x - v1.x * v2.z,
        z: v1.x * v2.y - v1.y * v2.x,
    }
}

#[inline(always)]
pub fn dot(v1: &Vec3, v2: &Vec3) -> f32 {
    v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
}

#[inline(always)]
pub fn length_squared(v: &Vec3) -> f32 {
    dot(v, v)
}

#[inline(always)]
pub fn length(v: &Vec3) -> f32 {
    length_squared(v).sqrt()
}

#[inline(always)]
pub fn normalize(v: &Vec3) -> Vec3 {
    let len = length(v);
    (1.0 / len) * v
}

#[inline(always)]
pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    //debug_assert!(is_normalized(n));
    let dot = -v.x * n.x + -v.y * n.y + -v.z * n.z;
    Vec3 {
        x: -v.x + 2.0 * (dot * n.x + v.x),
        y: -v.y + 2.0 * (dot * n.y + v.y),
        z: -v.z + 2.0 * (dot * n.z + v.z),
    }
}

#[inline(always)]
pub fn refract(v: &Vec3, n: &Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let _v = normalize(v);
    let cos_theta = (-_v.x * n.x - _v.y * n.y - _v.z * n.z).min(1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    if ni_over_nt * sin_theta > 1.0 {
        None
    } else {
        let r_out_perp = Vec3::new(
            ni_over_nt * (_v.x + cos_theta * n.x),
            ni_over_nt * (_v.y + cos_theta * n.y),
            ni_over_nt * (_v.z + cos_theta * n.z),
        );
        let r_out_parallel = -((1.0 - length_squared(&r_out_perp)).abs()).sqrt() * n;
        Some(&r_out_perp + &r_out_parallel)
    }
}

impl ops::Neg for &Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl ops::Add<&Vec3> for &Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn add(self, v: &Vec3) -> Self::Output {
        Vec3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl ops::AddAssign<&Vec3> for Vec3 {
    #[inline(always)]
    fn add_assign(&mut self, v: &Vec3) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl ops::Sub<&Vec3> for &Vec3 {
    type Output = Vec3;

    #[inline(always)]
    fn sub(self, v: &Vec3) -> Self::Output {
        Vec3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl ops::SubAssign<&Vec3> for Vec3 {
    #[inline(always)]
    fn sub_assign(&mut self, v: &Vec3) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl ops::Mul<&Vec3> for f32 {
    type Output = Vec3;

    #[inline(always)]
    fn mul(self, v: &Vec3) -> Self::Output {
        Vec3::new(self * v.x, self * v.y, self * v.z)
    }
}

impl ops::MulAssign<f32> for Vec3 {
    #[inline(always)]
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}
