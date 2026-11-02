// =============================================================================
//  Particulate denoising  (Stage A, step 0 — multi-terrain)
// -----------------------------------------------------------------------------
//  Strips airborne particulate (blowing snow / dust / sand, rain) that a
//  spinning LiDAR reports as sparse, isolated, usually low-intensity returns.
//  This is a *sensor-noise* filter and is DISTINCT from the two other
//  corruptions Stage A handles:
//      * non-rigid in-place jitter (wind on foliage)  -> rigidity / trunk band
//      * translating actors (people/animals/vehicles)  -> dynamic removal
//  It must run FIRST so particulate never reaches the ground fit or front-end.
//
//  Algorithm: DROR (Dynamic Radius Outlier Removal, Charron 2018) with an
//  optional intensity gate (the DSOR-style refinement, Kurup 2021).
//
//    A real surface return has many neighbours; a snow/dust speck is isolated.
//    A FIXED radius wrongly deletes far-but-real points (LiDAR rings spread with
//    range), so the search radius SCALES WITH RANGE:
//
//        sr = clamp( multiplier * r_xy * azimuth_res ,  sr_min ,  sr_max )
//
//    where r_xy is the horizontal range and azimuth_res the beam spacing [rad].
//    A point with fewer than `min_neighbors` neighbours inside sr is noise.
//    Optionally a point is spared if its intensity exceeds `max_intensity`
//    (strong returns are real surfaces, not particulate).
//
//  Performance: the internal voxel grid uses edge = sr_max, so the search
//  sphere (radius <= sr_max) is always contained in the 3x3x3 voxel
//  neighbourhood -> exactly 27 bucket lookups per point, with early-exit once
//  `min_neighbors` is reached (real points exit immediately; only the rare
//  isolated speck pays the full scan). O(N), a few ms on a 32-beam scan.
//  Self-contained: STL only.
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace cop
{

struct DenoiseConfig
{
  // Defaults calibrated for a sparse 32-beam scanner (VLP-32 @10Hz). The classic
  // DROR tuning (beta=3, knn=3) is for dense 64-beam automotive data and deletes
  // ~30% of a forest scan; a sparse scanner needs a more permissive radius so
  // only genuinely isolated specks (snow/dust) are removed (~4% on clean data).
  double azimuth_res_deg  = 0.20;  // LiDAR horizontal angular resolution     [deg]
  double multiplier       = 8.0;   // beta: sr = beta * r_xy * azimuth_res
  double min_search_radius = 0.20; // floor on the adaptive search radius      [m]
  double max_search_radius = 0.80; // cap on the adaptive search radius        [m]
  int    min_neighbors    = 2;     // < this many neighbours inside sr -> noise
  double max_intensity    = -1.0;  // spare returns brighter than this; <0 = off
  double min_range        = 0.0;   // skip denoising inside this range (dense)  [m]
};

class ParticulateDenoiser
{
public:
  explicit ParticulateDenoiser(const DenoiseConfig & cfg = DenoiseConfig()) : cfg_(cfg) {}
  void configure(const DenoiseConfig & cfg) { cfg_ = cfg; }
  const DenoiseConfig & config() const { return cfg_; }

  // Fills `keep` (1 = keep, 0 = particulate) aligned to `pts`. Order-preserving:
  // the caller emits surviving points in their original scan order. Returns the
  // number of points removed. `pts[i]` must expose .x .y .z .intensity (floats).
  template <typename PointVec>
  std::size_t denoise(const PointVec & pts, std::vector<uint8_t> & keep) const
  {
    const std::size_t n = pts.size();
    keep.assign(n, 1u);
    if (n == 0) return 0;

    const double vs  = std::max(cfg_.max_search_radius, cfg_.min_search_radius);
    const double inv = 1.0 / vs;
    const double ang = cfg_.azimuth_res_deg * M_PI / 180.0;
    const bool use_int = cfg_.max_intensity >= 0.0;
    const int knn = std::max(1, cfg_.min_neighbors);

    // ---- voxel grid (edge = sr_max) : key -> point indices -----------------
    std::unordered_map<int64_t, std::vector<int>> grid;
    grid.reserve(n);
    std::vector<int64_t> vx(n), vy(n), vz(n);
    for (std::size_t i = 0; i < n; ++i)
    {
      const auto & p = pts[i];
      const int64_t ix = fl(p.x * inv), iy = fl(p.y * inv), iz = fl(p.z * inv);
      vx[i] = ix; vy[i] = iy; vz[i] = iz;
      grid[pack(ix, iy, iz)].push_back(static_cast<int>(i));
    }

    // ---- per-point adaptive radius outlier test ----------------------------
    std::size_t removed = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const auto & p = pts[i];
      const double rxy = std::sqrt(double(p.x) * p.x + double(p.y) * p.y);
      if (rxy < cfg_.min_range) continue;                       // dense near-field
      if (use_int && p.intensity > cfg_.max_intensity) continue;// strong = real

      double sr = cfg_.multiplier * rxy * ang;
      sr = std::min(std::max(sr, cfg_.min_search_radius), cfg_.max_search_radius);
      const double sr2 = sr * sr;

      int cnt = 0;
      bool ok = false;
      for (int dx = -1; dx <= 1 && !ok; ++dx)
        for (int dy = -1; dy <= 1 && !ok; ++dy)
          for (int dz = -1; dz <= 1 && !ok; ++dz)
          {
            auto it = grid.find(pack(vx[i] + dx, vy[i] + dy, vz[i] + dz));
            if (it == grid.end()) continue;
            for (int j : it->second)
            {
              if (j == static_cast<int>(i)) continue;
              const auto & q = pts[j];
              const double ex = double(p.x) - q.x, ey = double(p.y) - q.y, ez = double(p.z) - q.z;
              if (ex * ex + ey * ey + ez * ez <= sr2 && ++cnt >= knn) { ok = true; break; }
            }
          }
      if (!ok) { keep[i] = 0u; ++removed; }
    }
    return removed;
  }

private:
  static inline int64_t fl(double v) { return static_cast<int64_t>(std::floor(v)); }
  static inline int64_t pack(int64_t ix, int64_t iy, int64_t iz)
  {
    constexpr int64_t M = 0x1FFFFF;                 // 21 bits per axis
    return ((ix & M) << 42) | ((iy & M) << 21) | (iz & M);
  }

  DenoiseConfig cfg_;
};

}  // namespace cop
