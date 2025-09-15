# Point Cloud Simplification (pcloudsim)

A high-performance Python library for point cloud simplification and filtering operations. This library provides efficient implementations of statistical outlier removal, radius-based filtering, and voxel downsampling for 3D point clouds.

## Features

- **Statistical Outlier Removal**: Remove points that are statistical outliers based on their distance to neighboring points
- **Radius-based Filtering**: Filter points based on the density of neighboring points within a specified radius
- **Voxel Downsampling**: Simplify point clouds by averaging points within voxel grids
- **High Performance**: Implemented in C++ with OpenMP parallelization and efficient spatial indexing
- **Python Integration**: Easy-to-use Python API with NumPy array support

## Installation

### Prerequisites

- Python 3.11+
- CMake 3.15+
- C++17 compatible compiler
- OpenMP (optional, for parallel processing)

### From Source

```bash
git clone https://github.com/Adversarr/pcloudsim
cd pcloudsim
pip install -e .
```

The build process will automatically:
- Download and build Eigen3 for linear algebra operations
- Use nanobind for Python-C++ bindings
- Enable OpenMP if available for parallel processing

## Quick Start

```python
import numpy as np
from pcloudsim import RemovalParams, simplify_point_cloud

# Load your point cloud data
points = np.random.rand(10000, 3).astype(np.float32)  # (N, 3) array
features = np.random.rand(10000, 3).astype(np.float32)  # (N, 3) array (e.g., RGB colors)

# Configure processing parameters
params = RemovalParams(
    enable_statistical_outliers=True,
    n_neighbors_stats=20,
    std_dev_mul=2.0,
    
    enable_radius_outliers=True,
    radius=0.05,
    min_points_radius=16,
    
    enable_voxel_simplify=True,
    voxel_size=0.01
)

# Process the point cloud
simplified_points, simplified_features = simplify_point_cloud(points, features, params)

print(f"Original points: {points.shape[0]}")
print(f"Simplified points: {simplified_points.shape[0]}")
```

## API Reference

### RemovalParams

Configuration class for point cloud processing operations.

#### Parameters

- **Statistical Outlier Removal**:
  - `enable_statistical_outliers` (bool): Enable statistical outlier removal
  - `n_neighbors_stats` (int): Number of neighbors for statistical analysis (default: 20)
  - `std_dev_mul` (float): Standard deviation multiplier for outlier threshold (default: 2.0)

- **Radius-based Filtering**:
  - `enable_radius_outliers` (bool): Enable radius-based outlier removal
  - `radius` (float): Search radius for neighbor detection (default: 0.05)
  - `min_points_radius` (int): Minimum number of neighbors required (default: 16)

- **Voxel Downsampling**:
  - `enable_voxel_simplify` (bool): Enable voxel-based simplification
  - `voxel_size` (float): Size of each voxel for downsampling (default: 0.01)

### simplify_point_cloud(points, features, params)

Main function for point cloud simplification.

#### Arguments

- `points` (np.ndarray): Point coordinates as (N, 3) float32 array
- `features` (np.ndarray): Point features as (N, 3) float32 array (e.g., RGB colors)
- `params` (RemovalParams): Processing parameters

#### Returns

- `tuple`: (simplified_points, simplified_features) as (M, 3) arrays where M ≤ N

## Processing Pipeline

The library applies filters in the following order:

1. **Statistical Outlier Removal**: Removes points whose average distance to neighbors exceeds mean + std_dev_mul × standard_deviation
2. **Radius-based Filtering**: Removes points that don't have enough neighbors within the specified radius
3. **Voxel Downsampling**: Groups remaining points into voxels and averages their positions and features

## Performance

The library is optimized for performance with:

- **Efficient Spatial Indexing**: Uses nanoflann KD-tree for fast neighbor searches
- **Parallel Processing**: OpenMP parallelization when available
- **Memory Efficiency**: Uses unordered_dense hash maps for voxel operations
- **SIMD Optimization**: Compiled with native CPU optimizations when using GCC

## Dependencies

### Core Dependencies

- [Eigen3](https://eigen.tuxfamily.org/): Linear algebra library
- [nanobind](https://github.com/wjakob/nanobind): Python-C++ binding framework
- [nanoflann](https://github.com/jlblancoc/nanoflann): KD-tree library for efficient neighbor searches

### Python Dependencies

- NumPy ≥ 2.0.0
- setuptools ≥ 75.8.2

## Building from Source

### CMake Options

The project uses CMake with the following key features:

- Automatic dependency management via CPM
- Optional OpenMP support for parallelization
- Native CPU optimizations when using GCC
- Support for both Debug and Release builds

### Build Commands

```bash
# Development build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Release build (recommended for production)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Examples

### Basic Denoising

```python
# Remove statistical outliers only
params = RemovalParams(
    enable_statistical_outliers=True,
    n_neighbors_stats=10,
    std_dev_mul=1.5
)
clean_points, clean_features = simplify_point_cloud(points, features, params)
```

### Density-based Filtering

```python
# Keep only points in dense regions
params = RemovalParams(
    enable_radius_outliers=True,
    radius=0.02,
    min_points_radius=8
)
dense_points, dense_features = simplify_point_cloud(points, features, params)
```

### Point Cloud Downsampling

```python
# Reduce point cloud size while preserving structure
params = RemovalParams(
    enable_voxel_simplify=True,
    voxel_size=0.005
)
reduced_points, reduced_features = simplify_point_cloud(points, features, params)
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Author

**Adversarr** - yangzherui2001@foxmail.com