mod vec3;
mod ray;
mod scene;

extern crate png;
extern crate rand;

use std::sync::Arc;
use std::path::Path;
use std::fs::File;
use std::io::BufWriter;
use std::thread;
use rand::{ Rng };
use rand::rngs::ThreadRng;
use vec3::{ Vec3, normalize, length_squared, reflect, dot, refract };
use ray::Ray;
use scene::{ Hitable, Scene, Sphere, Material };

const IMAGE_WIDTH: u32 = 1024;
const IMAGE_HEIGHT: u32 = 1024;
const PIXEL_SAMPLES: u32 = 255;
const MAX_DEPTH: u32 = 8;
const LENS_RADIUS: f32 = 0.05;
const FOCAL_DISTANCE: f32 = 5.0;
const NUM_THREADS: u32 = 16;

struct Tile {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

impl Tile {
    fn new(min_x: u32, min_y: u32, max_x: u32, max_y: u32) -> Tile {
        Tile { min_x, min_y, max_x, max_y }
    }
}

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
            Material::Glass(ior) => {
                let mut refraction_ratio = ior;
                let mut normal = hit.n;
                if dot(&ray.d, &hit.n) < 0.0 {
                    refraction_ratio = 1.0 / ior;
                } else {
                    normal = -&hit.n;
                }

                let schlick = {
                    let mut v = -&ray.d;
                    v.normalize();
                    let cos_theta = (v.x * normal.x + v.y * normal.y + v.z * normal.z).min(1.0);
                    let mut r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
                    r0 = r0 * r0;
                    r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5)
                };
                let rand: f32 = rng.gen();

                let mut new_ray = if let Some(mut refracted) = refract(&ray.d, &normal, refraction_ratio) {
                    if schlick > rand {
                        let mut reflected = reflect(&ray.d, &normal);
                        reflected.normalize();
                        Ray::new(hit.p, reflected)
                    } else {
                        refracted.normalize();
                        Ray::new(hit.p, refracted)
                    }
                } else {
                    let mut reflected = reflect(&ray.d, &normal);
                    reflected.normalize();
                    Ray::new(hit.p, reflected)                
                };
                new_ray.o.x += 0.001 * new_ray.d.x;
                new_ray.o.y += 0.001 * new_ray.d.y;
                new_ray.o.z += 0.001 * new_ray.d.z;
                let c = trace_ray(scene, &new_ray, rng, depth + 1);
                Vec3::new(
                    1.0 * c.x,
                    1.0 * c.y,
                    1.0 * c.z,
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

fn render_tile(scene: Arc<Scene>, tile: &Tile) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let size = (tile.max_y - tile.min_y) * (tile.max_x - tile.min_x) * 4;
    let mut output: Vec<u8> = vec![0; size as usize];
    let mut i = 0;
    for y in tile.min_y..tile.max_y {
        for x in tile.min_x..tile.max_x {
            let mut color = Vec3::new(0.0, 0.0, 0.0);
            for _sample in 0..PIXEL_SAMPLES {
                let ray = generate_ray((x, y), &mut rng);
                let c = trace_ray(&scene, &ray, &mut rng, 0);
                color += &c;
            }
            color *= 1.0 / PIXEL_SAMPLES as f32;
            output[i + 0] = (255.99 * color.x.sqrt()) as u8;
            output[i + 1] = (255.99 * color.y.sqrt()) as u8;
            output[i + 2] = (255.99 * color.z.sqrt()) as u8;
            output[i + 3] = 255;
            i = i + 4;
        }
    }
    output
}

fn render_scene(scene: Arc<Scene>, num_threads: u32) -> Vec<u8> {
    let mut handles: Vec<std::thread::JoinHandle<Vec<u8>>> = Vec::new();
    let tile_height = IMAGE_HEIGHT / num_threads;
    for i in 0..num_threads {
        let _scene = scene.clone();
        handles.push(thread::spawn(move || { render_tile(_scene, &Tile::new(0, i * tile_height, IMAGE_WIDTH, (i + 1) * tile_height)) }));
    }
    let mut result: Vec<u8> = Vec::new();
    for handle in handles {
        let mut tile = handle.join().unwrap();
        result.append(&mut tile);
    }
    result
}

fn main() {
    let spheres: Vec<Sphere> = vec!(
        Sphere::new(Vec3::new(0., 7.0, 2.0), 5.0, Material::Metal(Vec3::new(0.175, 0.75, 0.5), 0.0)),
        Sphere::new(Vec3::new(0.0, 0.0, 0.0), 1.0, Material::Metal(Vec3::new(1.0, 1.0, 1.0), 0.0)),
        Sphere::new(Vec3::new(-2.1, 0.0, 0.0), 1.0, Material::Diffuse(Vec3::new(1.0, 1.0, 1.0))),
        Sphere::new(Vec3::new(2.1, 0.0, 0.0), 1.0, Material::Normal),
        //Sphere::new(Vec3::new(-1.5, -0.5, -2.5), 0.5, Material::Light(Vec3::new(1.0, 1.0, 0.0))),
        Sphere::new(Vec3::new(-1.5, -0.5, -2.5), 0.5, Material::Glass(1.5)),
        Sphere::new(Vec3::new(1.5, -0.5, -2.5), 0.5, Material::Metal(Vec3::new(1.0, 1.0, 1.0), 0.1)),
        Sphere::new(Vec3::new(0.0, -100.0, 0.0), 99.0, Material::Diffuse(Vec3::new(0.9, 0.7, 0.5))),
    );
    let scene = Arc::new(Scene::new(spheres));

    let buff = render_scene(scene, NUM_THREADS);
    let file = File::create(Path::new(&String::from("output.png"))).unwrap();
    let ref mut buf_writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(buf_writer, IMAGE_WIDTH, IMAGE_HEIGHT);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder.write_header().unwrap();
    png_writer.write_image_data(&buff).unwrap();
}
