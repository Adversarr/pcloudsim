"""Simple Point Cloud Processing Library

A Python wrapper for point cloud simplification and filtering operations.
"""

from . import pcloudsim_binding as _pcloudsim_binding


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