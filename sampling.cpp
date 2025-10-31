// Implementation of mesh surface sampling utilities similar to Open3D.
#include "./sampling.hpp"
#include <random>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>

namespace {
// Compute area of a single triangle using double precision for robustness.
inline double tri_area(const Eigen::Matrix3Xd& V, const Eigen::Vector3i& f) {
    const Eigen::Vector3d a = V.col(f[0]);
    const Eigen::Vector3d b = V.col(f[1]);
    const Eigen::Vector3d c = V.col(f[2]);
    const Eigen::Vector3d ab = b - a;
    const Eigen::Vector3d ac = c - a;
    const double area = 0.5 * ab.cross(ac).norm();
    return area;
}

// Sample a single point uniformly within a triangle using barycentric method.
inline Eigen::Vector3d sample_point_in_triangle(const Eigen::Vector3d& a,
                                                const Eigen::Vector3d& b,
                                                const Eigen::Vector3d& c,
                                                std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u = dist(rng);
    double v = dist(rng);
    if (u + v > 1.0) {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    const double w = 1.0 - u - v;
    return u * a + v * b + w * c;
}

// Build cumulative distribution of triangle areas for weighted triangle selection.
// Filters invalid faces (out-of-bounds indices), non-finite vertices, and degenerate triangles.
inline std::vector<double> build_cdf(const Eigen::Matrix3Xd& V,
                                     const Eigen::Matrix3Xi& F,
                                     std::vector<int>& face_map) {
    const int m = static_cast<int>(F.cols());
    face_map.clear();
    face_map.reserve(m);
    std::vector<double> cdf;
    cdf.reserve(m);
    double acc = 0.0;
    constexpr double AREA_EPS = 1e-12;
    const int nV = static_cast<int>(V.cols());
    for (int i = 0; i < m; ++i) {
        const Eigen::Vector3i f = F.col(i);
        // Bounds check
        if (f[0] < 0 || f[1] < 0 || f[2] < 0 || f[0] >= nV || f[1] >= nV || f[2] >= nV) {
            continue;
        }
        const Eigen::Vector3d a = V.col(f[0]);
        const Eigen::Vector3d b = V.col(f[1]);
        const Eigen::Vector3d c = V.col(f[2]);
        // Finite check
        auto finite3 = [](const Eigen::Vector3d& p) {
            return std::isfinite(p.x()) && std::isfinite(p.y()) && std::isfinite(p.z());
        };
        if (!finite3(a) || !finite3(b) || !finite3(c)) {
            continue;
        }
        const double area = tri_area(V, f);
        if (!(std::isfinite(area)) || area <= AREA_EPS) {
            continue;
        }
        acc += area;
        cdf.push_back(acc);
        face_map.push_back(i);
    }
    if (acc <= 0.0 || cdf.empty()) {
        throw std::runtime_error("No valid non-degenerate triangles found (invalid indices or zero area).");
    }
    // normalize to [0,1]
    for (double& v : cdf) v /= acc;
    return cdf;
}

inline int sample_triangle_index(const std::vector<double>& cdf, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double r = dist(rng);
    auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
    int idx = static_cast<int>(std::distance(cdf.begin(), it));
    if (idx >= static_cast<int>(cdf.size())) idx = static_cast<int>(cdf.size()) - 1;
    return idx;
}
} // namespace

Eigen::Matrix3Xd sample_points_uniformly(
    const Eigen::Matrix3Xd& vertices,
    const Eigen::Matrix3Xi& faces,
    int num_of_points
) {
    if (num_of_points <= 0) {
        throw std::runtime_error("num_of_points must be positive.");
    }
    if (vertices.rows() != 3) {
        throw std::runtime_error("vertices must be 3xN.");
    }
    if (faces.rows() != 3) {
        throw std::runtime_error("faces must be 3xM (triangle indices).");
    }
    if (faces.size() == 0) {
        throw std::runtime_error("faces must be non-empty.");
    }

    std::mt19937 rng{std::random_device{}()};
    std::vector<int> face_map;
    auto cdf = build_cdf(vertices, faces, face_map);
    Eigen::Matrix3Xd out(3, num_of_points);
    for (int i = 0; i < num_of_points; ++i) {
        const int fi = sample_triangle_index(cdf, rng);
        const Eigen::Vector3i f = faces.col(face_map[fi]);
        const Eigen::Vector3d a = vertices.col(f[0]);
        const Eigen::Vector3d b = vertices.col(f[1]);
        const Eigen::Vector3d c = vertices.col(f[2]);
        out.col(i) = sample_point_in_triangle(a, b, c, rng);
    }
    return out;
}

Eigen::Matrix3Xd sample_points_poisson_disk(
    const Eigen::Matrix3Xd& vertices,
    const Eigen::Matrix3Xi& faces,
    int num_of_points,
    int init_factor
) {
    if (num_of_points <= 0) {
        throw std::runtime_error("num_of_points must be positive.");
    }
    if (init_factor <= 0) {
        throw std::runtime_error("init_factor must be positive.");
    }
    // Step 1: initial uniform oversampling
    const int n_init = num_of_points * init_factor;
    Eigen::Matrix3Xd init = sample_points_uniformly(vertices, faces, n_init);

    // TODO: Use mesh geodesic-aware spacing; currently using Euclidean spacing.

    // Step 2: exact greedy sample elimination maximizing nearest-neighbor distance.
    // Compute full pairwise squared distance matrix D for oversampled points.
    // D(i,j) = ||p_i - p_j||^2, with diagonal set to +inf.
    const int M = n_init;
    // Normalize points to improve numeric stability in distance computations
    Eigen::Vector3d minV = init.rowwise().minCoeff();
    Eigen::Vector3d maxV = init.rowwise().maxCoeff();
    Eigen::Vector3d ext = (maxV - minV).cwiseMax(1e-12);
    double scale = std::max(std::max(ext.x(), ext.y()), ext.z());
    if (!std::isfinite(scale) || scale <= 0.0) {
        scale = 1.0; // TODO: If scale is not finite, consider rejecting input.
    }
    Eigen::Matrix3Xd init_norm = init.colwise() - minV;
    init_norm.array() /= scale;

    Eigen::MatrixXd P = init_norm.transpose(); // Mx3 normalized
    Eigen::VectorXd s = P.rowwise().squaredNorm(); // Mx1
    Eigen::MatrixXd D = s.replicate(1, M) + s.transpose().replicate(M, 1) - 2.0 * (P * P.transpose());
    const double INF = std::numeric_limits<double>::infinity();
    // OpenMP: setting diagonal elements is embarrassingly parallel
    // Parallelizing here is safe because iterations are independent.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < M; ++i) D(i, i) = INF;

    std::vector<char> active(M, 1);
    std::vector<int> nn_idx(M, -1);
    std::vector<double> nn_dist(M, INF);

    auto recompute_nn = [&](int i) {
        double best = INF; int bestj = -1;
        for (int j = 0; j < M; ++j) {
            if (!active[j] || j == i) continue;
            double dij = D(i, j);
            if (!std::isfinite(dij)) continue;
            if (dij < best) { best = dij; bestj = j; }
        }
        nn_dist[i] = best;
        nn_idx[i] = bestj;
    };

    // OpenMP: initial nearest neighbor computation for each point
    // Safe to parallelize since each iteration writes to disjoint indices.
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < M; ++i) {
        recompute_nn(i);
    }

    int remaining = M;
    while (remaining > num_of_points) {
        // Find active point with smallest NN distance (worst packed)
        double worstDist = INF; int worstIdx = -1;
        for (int i = 0; i < M; ++i) {
            if (!active[i]) continue;
            if (nn_dist[i] < worstDist) { worstDist = nn_dist[i]; worstIdx = i; }
        }
        if (worstIdx < 0) break; // safety

        // Eliminate this point
        active[worstIdx] = 0;
        --remaining;
        // Invalidate distances touching this index
        // OpenMP: independent assignments across j, safe to parallelize
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < M; ++j) { D(worstIdx, j) = INF; D(j, worstIdx) = INF; }

        // Update only those whose NN was the removed point or whose NN is no longer active
        // OpenMP: each iteration writes to distinct i; D and active are read-only here.
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < M; ++i) {
            if (!active[i]) continue;
            if (nn_idx[i] == worstIdx || (nn_idx[i] >= 0 && !active[nn_idx[i]])) {
                recompute_nn(i);
            }
        }
    }

    // Collect final points
    Eigen::Matrix3Xd out(3, num_of_points);
    int j = 0;
    for (int i = 0; i < M && j < num_of_points; ++i) {
        if (active[i]) { out.col(j++) = init.col(i); }
    }
    // If due to edge cases we have fewer points, pad by uniform samples
    if (j < num_of_points) {
        for (; j < num_of_points; ++j) {
            out.col(j) = init.col(j % M);
            // TODO: Better fallback strategy if elimination undershoots.
        }
    }
    return out;
}