mod vec3;
mod ray;
mod scene;

extern crate png;
extern crate rand;

use std::path::Path;
use std::fs::File;
use std::io::BufWriter;
use rand::{ Rng };
use rand::rngs::ThreadRng;
use vec3::{ Vec3, normalize, length_squared, reflect, refract, dot };
use ray::Ray;
use scene::{ Hitable, Scene, Sphere, Material };

const IMAGE_WIDTH: u32 = 1024;
const IMAGE_HEIGHT: u32 = 1024;
const PIXEL_SAMPLES: u32 = 255;
const MAX_DEPTH: u32 = 8;
const LENS_RADIUS: f32 = 0.05;
const FOCAL_DISTANCE: f32 = 5.0;

fn generate_ray((x, y): (u32, u32), rng: &mut ThreadRng) -> Ray {
    let pixel_sample_u: f32 = rng.gen();
    let pixel_sample_v: f32 = rng.gen();
    let pixel_u: f32 = (x as f32 + pixel_sample_u) / IMAGE_WIDTH as f32;
    let pixel_v: f32 = 1.0 - (y as f32 + pixel_sample_v) / IMAGE_HEIGHT as f32;
    let mut ray = Ray::new(
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(-1.0 + pixel_u * 2.0, -1.0 + pixel_v * 2.0, 1.0),
    );
    ray.d.normalize();
    let focus_point = ray.point_at(FOCAL_DISTANCE / ray.d.z);
    let mut lens_sample = Vec3::new(0.0, 0.0, 0.0);
    loop {
        let (u, v): (f32, f32) = (rng.gen(), rng.gen());
        lens_sample.x = 2.0 * u - 1.0;
        lens_sample.y = 2.0 * v - 1.0;
        lens_sample.z = 0.0;
        if length_squared(&lens_sample) < 1.0 {
            break;
        }
    }
    lens_sample.normalize();
    ray.o.x += lens_sample.x * LENS_RADIUS;
    ray.o.y += lens_sample.y * LENS_RADIUS;
    ray.d = &focus_point - &ray.o;
    ray.d.normalize();
    ray
}

fn render_scene(scene: &Scene, output_filename: &String) {
    let mut rng = rand::thread_rng();
    let mut data = [0; IMAGE_WIDTH as usize * IMAGE_HEIGHT as usize * 4];
    let mut i = 0;
    for y in 0..IMAGE_HEIGHT {
        for x in 0..IMAGE_WIDTH {
            let mut color = Vec3::new(0.0, 0.0, 0.0);
            for _sample in 0..PIXEL_SAMPLES {
                let ray = generate_ray((x, y), &mut rng);
                let c = trace_ray(&scene, &ray, &mut rng, 0);
                color += &c;
            }
            color *= 1.0 / PIXEL_SAMPLES as f32;
            data[i + 0] = (255.99 * color.x.sqrt()) as u8;
            data[i + 1] = (255.99 * color.y.sqrt()) as u8;
            data[i + 2] = (255.99 * color.z.sqrt()) as u8;
            data[i + 3] = 255;
            i += 4;
        }
    }
    let file = File::create(Path::new(&output_filename)).unwrap();
    let ref mut buf_writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(buf_writer, IMAGE_WIDTH, IMAGE_HEIGHT);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header().unwrap();
    png_writer.write_image_data(&data).unwrap();
}

fn trace_ray(scene: &Scene, ray: &Ray, rng: &mut ThreadRng, depth: u32) -> Vec3 {
    if depth >= MAX_DEPTH {
        return Vec3::new(0.0, 0.0, 0.0);
    }

    if let Some(hit) = scene.hit(ray) {
        match hit.m {
            Material::Diffuse(albedo) => {
                let mut rand = Vec3::new(0.0, 0.0, 0.0);
                loop {
                    let (u, v, w): (f32, f32, f32) = (rng.gen(), rng.gen(), rng.gen());
                    rand.x = 2.0 * u - 1.0;
                    rand.y = 2.0 * v - 1.0;
                    rand.z = 2.0 * w - 1.0;
                    if length_squared(&rand) < 1.0 {
                        break;
                    }
                }
                let mut target = &hit.n + &rand;
                target.normalize();
                let mut new_ray = Ray::new(hit.p, target);
                new_ray.o.x += 0.001 * new_ray.d.x;
                new_ray.o.y += 0.001 * new_ray.d.y;
                new_ray.o.z += 0.001 * new_ray.d.z;
                let c = trace_ray(scene, &new_ray, rng, depth + 1);
                Vec3::new(
                    albedo.x * c.x,
                    albedo.y * c.y,
                    albedo.z * c.z,
                )
            },
            Material::Metal(albedo, roughness) => {
                let mut target = reflect(&ray.d, &hit.n);
                if roughness > 0.0 {
                    let mut rand = Vec3::new(0.0, 0.0, 0.0);
                    loop {
                        let (u, v, w): (f32, f32, f32) = (rng.gen(), rng.gen(), rng.gen());
                        rand.x = 2.0 * u - 1.0;
                        rand.y = 2.0 * v - 1.0;
                        rand.z = 2.0 * w - 1.0;
                        if length_squared(&rand) < roughness {
                            break;
                        }
                    }
                    target += &rand;
                }
                target.normalize();
                let mut new_ray = Ray::new(hit.p, target);
                new_ray.o.x += 0.001 * new_ray.d.x;
                new_ray.o.y += 0.001 * new_ray.d.y;
                new_ray.o.z += 0.001 * new_ray.d.z;
                let c = trace_ray(scene, &new_ray, rng, depth + 1);
                Vec3::new(
                    albedo.x * c.x,
                    albedo.y * c.y,
                    albedo.z * c.z,
                )
            },
            Material::Light(color) => {
                color
            },
            Material::Normal => {
                Vec3::new(
                    0.5 * (hit.n.x + 1.0),
                    0.5 * (hit.n.y + 1.0),
                    0.5 * (hit.n.z + 1.0),
                )
            }
        }
    } else {
        // Render background
        let normalized = normalize(&ray.d);
        let t = 0.5 * (normalized.y + 1.0);
        Vec3::new(
            (1.0 - t) * 1.0 + t * 0.5,
            (1.0 - t) * 1.0 + t * 0.7,
            (1.0 - t) * 1.0 + t * 0.9,
        )
    }
}

fn main() {
    let spheres = vec!(
        Sphere::new(Vec3::new(0., 7.0, 2.0), 5.0, Material::Metal(Vec3::new(0.75, 0.75, 0.5), 0.0)),
        Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0, Material::Metal(Vec3::new(1.0, 1.0, 1.0), 0.0)),
        Sphere::new(Vec3::new(-2.1, 0.0, 0.0), 1.0, Material::Diffuse(Vec3::new(1.0, 1.0, 1.0))),
        Sphere::new(Vec3::new(2.1, 0.0, 0.0), 1.0, Material::Normal),
        Sphere::new(Vec3::new(-1.5, -0.5, -2.5), 0.5, Material::Light(Vec3::new(1.0, 1.0, 0.0))),
        Sphere::new(Vec3::new(1.5, -0.5, -2.5), 0.5, Material::Metal(Vec3::new(1.0, 1.0, 1.0), 0.1)),
        //Sphere::new(Vec3::new(0.0, -0.5, -2.5), 0.5, Material::Dielectric(Vec3::new(1.0, 1.0, 1.0), 1.25)),
        Sphere::new(Vec3::new(0.0, -100.0, 0.0), 99.0, Material::Diffuse(Vec3::new(0.9, 0.7, 0.5))),
    );
    let scene = Scene::new(spheres);
    render_scene(&scene, &String::from("output.png"));
}
