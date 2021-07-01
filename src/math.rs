pub const PI: f32 = std::f32::consts::PI;
pub const INV_PI: f32 = 1.0 / std::f32::consts::PI;

pub type Degrees = f32;
pub type Radians = f32;

#[inline(always)]
pub fn almost_zero(v: f32) -> bool {
    v.abs() <= std::f32::EPSILON
}

#[inline(always)]
pub fn almost_one(v: f32) -> bool {
    (v - 1.0).abs() <= std::f32::EPSILON
}

#[inline(always)]
pub fn lerp(v0: f32, v1: f32, t: f32) -> f32 {
    v0 + t * (v1 - v0)
}

#[inline(always)]
pub fn deg_to_rad(angle: Degrees) -> Radians {
    angle / 180.0 * PI
}

#[inline(always)]
pub fn rad_to_deg(angle: Radians) -> Degrees {
    angle * INV_PI * 180.0
}

pub enum Roots {
    Two(f32, f32),
    One(f32),
    None
}

// Find roots of quadratic equation `a * x^2 + b * x + c = 0`.
#[inline(always)]
pub fn quadratic(a: f32, b: f32, c: f32) -> Roots {
    let discrim = b * b - 4.0 * a * c;
    if discrim > 0.0 {
        let discrim_sqrt = discrim.sqrt();
        Roots::Two(0.5 * (-b + discrim_sqrt) / a, 0.5 * (-b - discrim_sqrt) / a)
    } else if discrim == 0.0 {
        Roots::One(-0.5 * b / a)
    } else {
        Roots::None
    }
}
