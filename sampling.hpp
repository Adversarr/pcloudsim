#pragma once

#include <Eigen/Dense>

// Sample points uniformly over a triangle mesh surface.
// Inputs:
// - `vertices`: 3xN matrix of vertex positions (columns are xyz).
// - `faces`: 3xM matrix of triangle indices into `vertices` (columns are i,j,k).
// - `num_of_points`: number of points to draw uniformly w.r.t. surface area.
// Returns:
// - 3xK matrix of sampled points (K == num_of_points).
// TODO: Clarify coordinate convention (row-major vs column-major) if needed.
// NOTE: Uses double precision for robustness.
Eigen::Matrix3Xd sample_points_uniformly(
    const Eigen::Matrix3Xd& vertices,
    const Eigen::Matrix3Xi& faces,
    int num_of_points
);

// Sample points with approximate Poisson-disk spacing using sample elimination
// similar to Open3D's method based on Yuksel (2015).
// Inputs:
// - `vertices`: 3xN matrix of vertex positions.
// - `faces`: 3xM matrix of triangle indices.
// - `num_of_points`: target number of points after elimination.
// - `init_factor`: multiplier for initial uniform samples before elimination
//   (e.g., 5 means start with 5 * num_of_points uniform samples).
// Returns:
// - 3xK matrix of sampled points (K == num_of_points).
// This implementation performs exact greedy sample elimination maximizing
// nearest-neighbor distances using Euclidean metrics on sampled points.
// TODO: If geodesic distances on the mesh are required, additional logic is needed.
// NOTE: Uses double precision for robustness.
Eigen::Matrix3Xd sample_points_poisson_disk(
    const Eigen::Matrix3Xd& vertices,
    const Eigen::Matrix3Xi& faces,
    int num_of_points,
    int init_factor
);