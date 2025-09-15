#include "nanoflann.hpp"
#include "pcloudsim.hpp"
#include "unordered_dense.h"

using Knn = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix3Xf, 3,
                                                nanoflann::metric_L2, false>;


constexpr Eigen::Index PCLOUDSIM_MAX_SEARCH = 32;


// Hash helpers taken from tiny-cuda-nn
inline uint32_t prime_hash(const Eigen::Vector3i &pos_grid) {
  constexpr uint32_t factors[3] = {1958374283u, 2654435761u, 805459861u};
  uint32_t result = 0;
#pragma unroll
  for (uint32_t i = 0; i < 3; ++i) {
    result ^= pos_grid[i] * factors[i];
  }
  return result;
}

// voxel id: only allow positive grid index.
struct Vox {
    Eigen::Vector3f pos{0.f, 0.f, 0.f};
    Eigen::Vector3f color{0.f, 0.f, 0.f};
    int cnt = 0;

    void add(const Eigen::Vector3f &p, const Eigen::Vector3f &c) {
        pos += p;
        color += c;
        ++cnt;
    }
};

unsigned int estim_threads(unsigned int estim) {
    // ensure power of 2
    if (5 <= estim && estim <= 7) {
        return 4;
    } else if (8 <= estim && estim <= 15) {
        return 8;
    } else if (16 <= estim && estim <= 31) {
        return 16;
    } else if (32 <= estim && estim <= 63) {
        return 32;
    } else {
        return estim;
    }
}

template <>
struct std::hash<Eigen::Vector3i> {
  size_t operator()(const Eigen::Vector3i &v) const { return prime_hash(v); }
};

PointCloudData simplify(const PointCloudData &data,
                        const RemovalParams &params) {
  const int n_points = static_cast<int>(data.points.cols());
  // ensure boundary

#ifdef PCLOUDSIM_HAS_OPENMP
  const unsigned int nthreads = estim_threads(std::thread::hardware_concurrency());
#else
  const unsigned int nthreads = 1;
#endif
  if (n_points == 0) {
    throw std::runtime_error("No points given.");
  } else if (n_points != data.features.cols()) {
    throw std::runtime_error("Points/Features size does not match. " +
                             std::to_string(n_points) + " != " +
                             std::to_string(data.features.cols()));
  } else if (params.n_neighbors_stats + 1 > PCLOUDSIM_MAX_SEARCH || params.min_points_radius + 1> PCLOUDSIM_MAX_SEARCH) {
    throw std::runtime_error("k or min_points is larger than max search.");
  }

  Knn knn(3, data.points, 10, n_points > 10'000 ? nthreads : 1);

  std::vector<bool> valid(n_points, true);

  if (params.enable_statistical_outliers && params.n_neighbors_stats > 0 &&
      params.std_dev_mul > 0.0) {
    Eigen::VectorXf avg_dist(n_points);
    avg_dist.setConstant(0.f);
    size_t total_valid = 0;

#ifdef PCLOUDSIM_HAS_OPENMP
#pragma omp parallel for reduction(+ : total_valid) if (n_points > 4000) num_threads(nthreads)
#endif
    for (int i = 0; i < n_points; ++i) {
      Eigen::Index out[PCLOUDSIM_MAX_SEARCH];
      float out_dist_sqr[PCLOUDSIM_MAX_SEARCH];
      Eigen::Vector3f q = data.points.col(i);
      nanoflann::KNNResultSet<float, Eigen::Index> result(params.n_neighbors_stats + 1);
      result.init(out, out_dist_sqr);
      knn.index_->findNeighbors(result, q.data(),
                                nanoflann::SearchParameters(10));

      if (result.size() <= 1) {
        valid[i] = false;
        continue;
      }

      // Get the average distance, excluding self.
      float avg = 0.0f;
      for (int j = 1; j < result.size(); ++j) {
        avg += std::sqrt(out_dist_sqr[j]);
      }

      avg /= (float)(result.size() - 1);
      avg_dist[i] = avg; // + FLT_EPSILON; // It is ok.
      ++total_valid;
    }

    if (total_valid > 1) {
      const float mean_dist = avg_dist.sum() / (float)total_valid;
      float std = 0.0f;
      for (int i = 0; i < n_points; ++i) {
        if (avg_dist[i] > 0.0f) {
          std += (avg_dist[i] - mean_dist) * (avg_dist[i] - mean_dist);
        }
      }
      std = std::sqrt(std / (float)(total_valid - 1));

      // Only apply outlier removal if standard deviation is meaningful
      if (std > 1e-6f) { // Avoid division by near-zero std
        for (int i = 0; i < n_points; ++i) {
          if (avg_dist[i] > mean_dist + params.std_dev_mul * std) {
              valid[i] = false;
          }
        }
      }
    }
  }

  if (params.enable_radius_outliers && params.radius > 0.0 &&
      params.min_points_radius > 0) {
#ifdef PCLOUDSIM_HAS_OPENMP
#pragma omp parallel for if (n_points > 4000) num_threads(nthreads)
#endif
    for (int i = 0; i < n_points; ++i) {
      if (!valid[i]) {
        continue;
      }

      Eigen::Index out[PCLOUDSIM_MAX_SEARCH];
      float out_dist_sqr[PCLOUDSIM_MAX_SEARCH];
      Eigen::Vector3f q = data.points.col(i);
      nanoflann::KNNResultSet<float, Eigen::Index> result(PCLOUDSIM_MAX_SEARCH);
      result.init(out, out_dist_sqr);
      knn.index_->findNeighbors(result, q.data(),
                                nanoflann::SearchParameters(10));

      // guard against no neighbors.
      if (result.size() <= 1) {
        valid[i] = false;
        continue;
      }

      size_t valid_count = 0;
      for (int j = 1; j < result.size(); ++j) {
        if (out_dist_sqr[j] <= params.radius * params.radius) {
          ++valid_count;
        }
      }

      if (valid_count < params.min_points_radius) {
        valid[i] = false;
      }
    }
  }

  const size_t removal_left = std::count(valid.begin(), valid.end(), true);
  PointCloudData median_data;
  median_data.points.resize(3, removal_left);
  median_data.features.resize(3, removal_left);
  size_t idx = 0;
  for (int i = 0; i < n_points; ++i) {
    if (valid[i]) {
      median_data.points.col(idx) = data.points.col(i);
      median_data.features.col(idx) = data.features.col(i);
      ++idx;
    }
  }

  if (!params.enable_voxel_simplify || params.voxel_size <= 0.0) {
    return median_data;
  }

  assert(idx == removal_left);
  if (removal_left < 1) {
    throw std::runtime_error("Removal criteria is too strict that no points left.");
  }

  auto coord = [vs = params.voxel_size](const Eigen::Vector3f &p) -> Eigen::Vector3i {
    return (p.array() / vs).round().cast<int>();
  };

//   std::unordered_map<Eigen::Vector3i, Vox> voxs;
  ankerl::unordered_dense::map<Eigen::Vector3i, Vox> voxs;
  for (int i = 0; i < removal_left; ++i) {
    auto c = coord(median_data.points.col(i));
    voxs[c].add(median_data.points.col(i), median_data.features.col(i));
  }

  PointCloudData output;
  output.points.resize(3, voxs.size());
  output.features.resize(3, voxs.size());
  idx = 0;
  for (auto &[k, v] : voxs) {
    output.points.col(idx) = v.pos / v.cnt;
    output.features.col(idx) = v.color / v.cnt;
    ++idx;
  }

  return output;
}
