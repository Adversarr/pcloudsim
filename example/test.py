import numpy as np
from pcloudsim import RemovalParams, simplify_point_cloud
import timeit

points = np.loadtxt('/data/accgs/1747834320424/inputs/slam/points3D.txt', dtype=np.float32, usecols=(1, 2, 3))
features = np.loadtxt('/data/accgs/1747834320424/inputs/slam/points3D.txt', dtype=np.uint8, usecols=(4, 5, 6)).astype(np.float32) / 255.0
params = RemovalParams()
print(points.shape, features.shape)

params.enable_voxel_simplify = True
params.voxel_size = 0.001

# Correct.
params.enable_statistical_outliers = True
params.k = 20
params.enable_radius_outliers= True
params.radius = 0.05
params.min_points = 16

def run():
    s_points, s_features = simplify_point_cloud(points, features, params)

n = 100
t = timeit.timeit(run, number=n) / n
s_points, s_features = simplify_point_cloud(points, features, params)
print(s_points.shape)
print(f"Time: {t:.6f}s")


