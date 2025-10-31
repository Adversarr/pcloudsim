#include "pcloudsim.hpp"
#include "sampling.hpp"

#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>

namespace nb = nanobind;
using namespace nanobind::literals;

std::pair<Eigen::Matrix3Xf, Eigen::MatrixXf> simplify_interface(
    nb::ndarray<float, nb::numpy, nb::shape<-1, 3>, nb::c_contig> points,
    nb::ndarray<float, nb::numpy, nb::shape<-1, -1>, nb::c_contig> features,
    RemovalParams params
) {
    if (points.shape(0) != features.shape(0)) {
        throw std::runtime_error("points and features must have the same number of rows");
    }

    PointCloudData pcd{
        Eigen::Map<Eigen::Matrix3Xf>(points.data(), points.shape(1), points.shape(0)),
        Eigen::Map<Eigen::Matrix3Xf>(features.data(), features.shape(1), features.shape(0)),
    };

    auto out = simplify(pcd, params);
    return {std::move(out.points), std::move(out.features)};
}

// Wrapper for uniform mesh surface sampling using double precision.
// Accepts vertices as (N,3) float64 and faces as (M,3) int32 arrays.
// Returns a (3,K) float64 Eigen matrix of sampled points (column-major).
// TODO: Confirm whether faces are guaranteed to be 0-based indices; if not, adjust accordingly.
Eigen::Matrix3Xd sample_points_uniformly_interface(
    nb::ndarray<double, nb::numpy, nb::shape<-1, 3>, nb::c_contig> vertices,
    nb::ndarray<int, nb::numpy, nb::shape<-1, 3>, nb::c_contig> faces,
    int num_of_points
) {
    if (vertices.shape(1) != 3 || faces.shape(1) != 3) {
        throw std::runtime_error("vertices and faces must be shaped (*, 3)");
    }
    Eigen::Map<Eigen::Matrix3Xd> V(vertices.data(), vertices.shape(1), vertices.shape(0));
    Eigen::Map<Eigen::Matrix3Xi> F(faces.data(), faces.shape(1), faces.shape(0));
    return sample_points_uniformly(V, F, num_of_points);
}

// Wrapper for Poisson-disk sampling via exact greedy elimination.
// Accepts vertices as (N,3) float64 and faces as (M,3) int32 arrays.
// Returns a (3,K) float64 Eigen matrix of sampled points (column-major).
// TODO: If geodesic distances are required instead of Euclidean, expose an optional flag.
Eigen::Matrix3Xd sample_points_poisson_disk_interface(
    nb::ndarray<double, nb::numpy, nb::shape<-1, 3>, nb::c_contig> vertices,
    nb::ndarray<int, nb::numpy, nb::shape<-1, 3>, nb::c_contig> faces,
    int num_of_points,
    int init_factor
) {
    if (vertices.shape(1) != 3 || faces.shape(1) != 3) {
        throw std::runtime_error("vertices and faces must be shaped (*, 3)");
    }
    Eigen::Map<Eigen::Matrix3Xd> V(vertices.data(), vertices.shape(1), vertices.shape(0));
    Eigen::Map<Eigen::Matrix3Xi> F(faces.data(), faces.shape(1), faces.shape(0));
    return sample_points_poisson_disk(V, F, num_of_points, init_factor);
}

NB_MODULE(pcloudsim_binding, m) {
    // Bind RemovalParams struct with all its configuration fields
    nb::class_<RemovalParams>(m, "RemovalParams")
        .def(nb::init<>())
        .def_rw("enable_statistical_outliers", &RemovalParams::enable_statistical_outliers,
                "Enable statistical outlier removal")
        .def_rw("n_neighbors_stats", &RemovalParams::n_neighbors_stats,
                "Number of neighbors for statistical outlier detection")
        .def_rw("std_dev_mul", &RemovalParams::std_dev_mul,
                "Standard deviation multiplier for outlier threshold")
        .def_rw("enable_radius_outliers", &RemovalParams::enable_radius_outliers,
                "Enable radius-based outlier removal")
        .def_rw("radius", &RemovalParams::radius,
                "Search radius for radius-based outlier detection")
        .def_rw("min_points_radius", &RemovalParams::min_points_radius,
                "Minimum number of points within radius")
        .def_rw("enable_voxel_simplify", &RemovalParams::enable_voxel_simplify,
                "Enable voxel-based point cloud simplification")
        .def_rw("voxel_size", &RemovalParams::voxel_size,
                "Size of voxels for simplification");

    // Bind the main simplify function interface
    m.def("simplify", &simplify_interface,
          "points"_a, "features"_a, "params"_a,
          "Simplify point cloud data using various filtering methods");

    // Bind mesh surface sampling utilities
    m.def("sample_points_uniformly", &sample_points_uniformly_interface,
          "vertices"_a, "faces"_a, "num_of_points"_a,
          "Sample points uniformly over the mesh surface using triangle-area weighting (double precision).");

    m.def("sample_points_poisson_disk", &sample_points_poisson_disk_interface,
          "vertices"_a, "faces"_a, "num_of_points"_a, "init_factor"_a,
          "Sample points with Poisson-disk-like spacing via exact greedy elimination (double precision).");
}