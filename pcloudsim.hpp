#pragma once

#include <Eigen/Eigen>
#include <nanobind/nanobind.h>
#include <vector>
struct RemovalParams {
    bool enable_statistical_outliers = false;
    int n_neighbors_stats = 10;
    double std_dev_mul = 1.0;

    bool enable_radius_outliers = false;
    double radius = 0.0;
    int min_points_radius = 0;

    bool enable_voxel_simplify = false;
    double voxel_size = 0.0;
};

struct PointCloudData{
    Eigen::Matrix3Xf points;
    Eigen::Matrix3Xf features;
};

PointCloudData simplify(
    const PointCloudData& data,
    const RemovalParams& params
);