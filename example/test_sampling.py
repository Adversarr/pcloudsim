#!/usr/bin/env python3
"""Test mesh sampling functions using bunny.obj.

Loads an OBJ mesh, converts it to NumPy arrays, and runs both uniform and
Poisson-disk sampling exposed via the Python API.

Requirements:
- Install the extension first (e.g., `pip install -e .`).

TODO: If the input OBJ contains non-triangular faces, currently we fan-triangulate.
      This assumes manifold-like triangulation; adjust per dataset if needed.
"""

import argparse
import os
import sys
import time
from typing import Tuple, List

import numpy as np

try:
    from pcloudsim import sample_points_uniformly, sample_points_poisson_disk
except Exception as e:
    print("Failed to import pcloudsim sampling API. Did you build/install the extension?", file=sys.stderr)
    raise


def load_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices (N,3) float64 and faces (M,3) int32 from an OBJ file.

    - Handles 'v x y z' lines for vertices.
    - Handles 'f i j k' or 'f i/j/k ...' faces, converting 1-based to 0-based.
    - Triangulates polygon faces via fan if necessary.
    """
    vs: List[List[float]] = []
    fs: List[List[int]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('v '):
                parts = line.split()
                if len(parts) < 4:
                    # TODO: Handle malformed vertex lines more strictly if needed.
                    continue
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vs.append([x, y, z])
            elif line.startswith('f '):
                parts = line.split()[1:]
                idxs: List[int] = []
                for tok in parts:
                    # Support formats like i, i/j, i//k, i/j/k
                    base = tok.split('/')[0]
                    if base == '':
                        # TODO: Decide behavior for empty tokens in faces.
                        continue
                    try:
                        idx = int(base)
                    except ValueError:
                        # TODO: Handle non-integer face tokens.
                        continue
                    idxs.append(idx - 1)  # OBJ is 1-based
                if len(idxs) < 3:
                    continue
                # Triangulate via fan (v0, v{i}, v{i+1})
                v0 = idxs[0]
                for i in range(1, len(idxs) - 1):
                    fs.append([v0, idxs[i], idxs[i + 1]])

    V = np.array(vs, dtype=np.float64)
    F = np.array(fs, dtype=np.int32)
    return V, F


def save_ply(path: str, points: np.ndarray, colors: np.ndarray = None):
    """Save point cloud to PLY format.
    
    Args:
        path: Output file path
        points: (N,3) array of point positions
        colors: Optional (N,3) array of RGB colors (0-255)
    """
    with open(path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points (and colors if provided)
        for i in range(len(points)):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f}")
                f.write(f" {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description="Test sampling on a triangle mesh")
    ap.add_argument('--obj', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'bunny.obj'),
                    help="Path to OBJ file (default: bunny.obj in repo root)")
    ap.add_argument('--num', type=int, default=1000, help="Number of points to sample")
    ap.add_argument('--init', type=int, default=5, help="Oversampling factor for Poisson sampling")
    ap.add_argument('--outdir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'dist'),
                    help="Output directory to save samples")
    ap.add_argument('--format', type=str, choices=['npy', 'ply'], default='npy',
                    help="Output format for point clouds (default: npy)")
    args = ap.parse_args()

    V, F = load_obj(args.obj)
    print(f"Loaded mesh: vertices={V.shape}, faces={F.shape}")

    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.time()
    U = sample_points_uniformly(V, F, args.num)
    t1 = time.time()
    print(f"Uniform sampling: {U.shape} in {(t1 - t0)*1000:.2f} ms")
    
    # Save uniform samples
    if args.format == 'npy':
        np.save(os.path.join(args.outdir, 'uniform.npy'), U)
    else:
        save_ply(os.path.join(args.outdir, 'uniform.ply'), U)

    t2 = time.time()
    P = sample_points_poisson_disk(V, F, args.num, args.init)
    t3 = time.time()
    print(f"Poisson sampling: {P.shape} in {(t3 - t2)*1000:.2f} ms")
    
    # Save Poisson samples
    if args.format == 'npy':
        np.save(os.path.join(args.outdir, 'poisson.npy'), P)
    else:
        save_ply(os.path.join(args.outdir, 'poisson.ply'), P)

    # Simple sanity: bounds and basic stats
    print(f"Uniform bbox min={U.min(axis=0)}, max={U.max(axis=0)}")
    print(f"Poisson bbox min={P.min(axis=0)}, max={P.max(axis=0)}")


if __name__ == '__main__':
    main()