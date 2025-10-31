"""Simple Point Cloud Processing Library

A Python wrapper for point cloud simplification and filtering operations.
"""

from . import pcloudsim_binding as _pcloudsim_binding
import numpy as np


class RemovalParams:
    """Parameters for point cloud filtering and simplification operations.
    
    This class provides a convenient interface to configure various point cloud
    processing operations including statistical outlier removal, radius-based
    filtering, and voxel downsampling.
    """
    
    def __init__(self,
                 enable_statistical_outliers=False,
                 n_neighbors_stats=20,
                 std_dev_mul=2.0,
                 enable_radius_outliers=False,
                 radius=0.05,
                 min_points_radius=16,
                 enable_voxel_simplify=False,
                 voxel_size=0.01):
        """Initialize removal parameters.
        
        Args:
            enable_statistical_outliers (bool): Enable statistical outlier removal
            n_neighbors_stats (int): Number of neighbors for statistical outlier detection
            std_dev_mul (float): Standard deviation multiplier for outlier threshold
            enable_radius_outliers (bool): Enable radius-based outlier removal
            radius (float): Search radius for radius-based outlier detection
            min_points_radius (int): Minimum number of points within radius
            enable_voxel_simplify (bool): Enable voxel-based point cloud simplification
            voxel_size (float): Size of voxels for simplification
        """
        self.internal_params = _pcloudsim_binding.RemovalParams()
        self.internal_params.enable_statistical_outliers = enable_statistical_outliers
        self.internal_params.n_neighbors_stats = n_neighbors_stats
        self.internal_params.std_dev_mul = std_dev_mul
        self.internal_params.enable_radius_outliers = enable_radius_outliers
        self.internal_params.radius = radius
        self.internal_params.min_points_radius = min_points_radius
        self.internal_params.enable_voxel_simplify = enable_voxel_simplify
        self.internal_params.voxel_size = voxel_size
    
    @property
    def enable_statistical_outliers(self):
        return self.internal_params.enable_statistical_outliers
    
    @enable_statistical_outliers.setter
    def enable_statistical_outliers(self, value):
        self.internal_params.enable_statistical_outliers = value
    
    @property
    def n_neighbors_stats(self):
        return self.internal_params.n_neighbors_stats
    
    @n_neighbors_stats.setter
    def n_neighbors_stats(self, value):
        self.internal_params.n_neighbors_stats = value
    
    @property
    def std_dev_mul(self):
        return self.internal_params.std_dev_mul
    
    @std_dev_mul.setter
    def std_dev_mul(self, value):
        self.internal_params.std_dev_mul = value
    
    @property
    def enable_radius_outliers(self):
        return self.internal_params.enable_radius_outliers
    
    @enable_radius_outliers.setter
    def enable_radius_outliers(self, value):
        self.internal_params.enable_radius_outliers = value
    
    @property
    def radius(self):
        return self.internal_params.radius
    
    @radius.setter
    def radius(self, value):
        self.internal_params.radius = value
    
    @property
    def min_points_radius(self):
        return self.internal_params.min_points_radius
    
    @min_points_radius.setter
    def min_points_radius(self, value):
        self.internal_params.min_points_radius = value
    
    @property
    def enable_voxel_simplify(self):
        return self.internal_params.enable_voxel_simplify
    
    @enable_voxel_simplify.setter
    def enable_voxel_simplify(self, value):
        self.internal_params.enable_voxel_simplify = value
    
    @property
    def voxel_size(self):
        return self.internal_params.voxel_size
    
    @voxel_size.setter
    def voxel_size(self, value):
        self.internal_params.voxel_size = value

# Convenience function for direct usage
def simplify_point_cloud(points, features, params = RemovalParams()):
    """Simplify a point cloud with the given parameters.
    
    Args:
        points (np.ndarray): Point coordinates as (N, 3) array
        features (np.ndarray): Point features as (N, 3) array
        params (RemovalParams, optional): Parameters for simplification

    Returns:
        tuple: (simplified_points, simplified_features)
    """
    s_points, s_features = _pcloudsim_binding.simplify(points, features, params.internal_params)
    return s_points.T, s_features.T


# Export public API
__all__ = ['RemovalParams', 'simplify_point_cloud']


def sample_points_uniformly(vertices: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """Sample points uniformly over a triangle mesh surface.

    Args:
        vertices (np.ndarray): Vertex array shaped (N, 3) float64.
        faces (np.ndarray): Triangle index array shaped (M, 3) int32.
        num_points (int): Number of points to sample.

    Returns:
        np.ndarray: Sampled points shaped (num_points, 3) float64.

    Notes:
        - Faces are assumed to be 0-based indices.
        - Output is converted from internal column-major (3, K) to (K, 3) for Python convenience.
        - Uses double precision internally.
    """
    sampled = _pcloudsim_binding.sample_points_uniformly(vertices, faces, num_points)
    return sampled.T


def sample_points_poisson_disk(vertices: np.ndarray, faces: np.ndarray, num_points: int, init_factor: int = 5) -> np.ndarray:
    """Sample points with Poisson-disk-like spacing via exact greedy elimination.

    Args:
        vertices (np.ndarray): Vertex array shaped (N, 3) float64.
        faces (np.ndarray): Triangle index array shaped (M, 3) int32.
        num_points (int): Target number of points after elimination.
        init_factor (int, optional): Oversampling multiplier. Defaults to 5.

    Returns:
        np.ndarray: Sampled points shaped (num_points, 3) float64.

    Notes:
        - Uses Euclidean distances between sampled points; geodesic spacing is not used.
        - Output is converted from internal column-major (3, K) to (K, 3).
        - Double precision is used internally for robustness.
        - TODO: If faces are 1-based in some inputs, consider adjusting indices before calling.
    """
    sampled = _pcloudsim_binding.sample_points_poisson_disk(vertices, faces, num_points, init_factor)
    return sampled.T


# Export public API (updated)
__all__ = [
    'RemovalParams',
    'simplify_point_cloud',
    'sample_points_uniformly',
    'sample_points_poisson_disk',
]