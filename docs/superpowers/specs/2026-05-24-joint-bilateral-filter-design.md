# Joint Bilateral Filter for Path Tracer Denoising

## Context
Lab5 builds on the Lab4 path tracer (Cornell Box scene). The tracer produces noisy output, primarily from indirect illumination. We implement a joint/cross bilateral filter using G-buffers to denoise while preserving edges.

## Architecture
Single standalone file: `lab5/main.py`. Contains:
1. Modified path tracer that outputs G-buffers
2. Joint bilateral filter function
3. Tone mapping + gamma correction + save

## G-Buffer Generation
Modified renderer outputs per-pixel:
- `direct_light` (H,W,3) — NEE contribution from first diffuse hit
- `indirect_light` (H,W,3) — all other contributions (emission hits, specular bounces, etc.)
- `depth_map` (H,W) — primary ray t-distance
- `normal_map` (H,W,3) — surface normal at primary hit
- `object_index` (H,W) — triangle index at primary hit

One primary ray per pixel fills G-buffers. SPP samples accumulate light.

## Filter Design
Filter only `indirect_light`, keep `direct_light` untouched, output `direct + filtered_indirect`.

### Formula
g(p) = (1/W_p) * sum_q f(q) * G_s(||p-q||) * G_r(p,q)

### Range Weights
- G_r_object = 1 if same object, 0 otherwise
- G_r_depth = exp(-dz^2 / (2 * sigma_z^2))
- G_r_normal = exp(-(1 - dot(n_p, n_q)) / (2 * sigma_n^2))
- G_r = G_r_object * G_r_depth * G_r_normal

### Default Parameters
- sigma_s = 4.0 (spatial), kernel radius = 8
- sigma_z = 0.05 (depth)
- sigma_n = 0.15 (normal)

### Energy Conservation
Guaranteed by normalization: W_p = sum(G_s * G_r). Weighted average preserves total energy.

## Implementation
Pure NumPy vectorized. Pad input, iterate over kernel offsets with stride tricks or rolling windows.

## Output
Linear RGB → tone mapping → gamma correction → PPM + PNG (if PIL available).
