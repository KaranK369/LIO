// =============================================================================
//  Patchwork++-style ground segmentation               (Stage A, step 1)
// -----------------------------------------------------------------------------
//  Separates rigid ground from non-ground in the LiDAR *sensor frame* (z up,
//  sensor near the origin, ground below).  Self-contained: only Eigen + STL,
//  no PCL, no learned models -> real-time, CPU-only, Jetson-friendly.
//
//  This is an in-house implementation of the Patchwork / Patchwork++ idea
//  (Lim 2021 / Lee 2022) rather than the upstream library, so the workspace
//  stays dependency-free and the segmenter runs in-process.  It implements the
//  load-bearing parts:
//
//    * Concentric Zone Model (CZM) ......... polar bins whose resolution grows
//                                            toward the sensor (where trunks /
//                                            obstacles matter most).
//    * Region-wise Ground Plane Fitting .... per-bin seed selection (lowest-
//      (R-GPF)                               point representative + seed margin)
//                                            then iterative PCA plane fitting.
//    * Likelihood gating ................... accept a bin's plane as ground only
//                                            if it is upright (near-vertical
//                                            normal), flat (low surface
//                                            variation) and at a plausible
//                                            elevation -> rejects walls, trunks,
//                                            bushes, car roofs, etc.
//
//  Not implemented (documented upstream refinements, rarely needed in forest):
//    * A-GLE  (fully adaptive online elevation/flatness thresholds)
//    * TGR    (temporal ground revert)
//  The gating thresholds below are exposed as parameters so they can be tuned
//  per environment instead.
//
//  Beyond a clean ground/non-ground label, the segmenter exposes
//  groundHeightAt(x,y): a slope-aware ground-height query (evaluates the fitted
//  per-bin plane) used by the lower-trunk band passthrough (Stage A, step 3).
//  This replaces the old min-z-per-grid-cell heuristic, which biased the ground
//  toward the downhill corner of each cell on sloped, off-road terrain.
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace cop
{

struct GroundConfig
{
  // --- CZM extent (sensor frame, metres) ---
  double min_range = 1.0;       // ignore returns closer than this (ego / mount)
  double max_range = 60.0;      // ground fit only meaningful within this radius

  // --- Sensor geometry ---
  // Nominal LiDAR height above ground; the elevation gate is centred here.
  // <= 0 selects AUTO: the height is estimated from the first `auto_height_scans`
  // scans (upright+flat bins only, elevation gate off), then locked and tracked
  // with a slow EMA. Use auto for any platform whose mount height is not known.
  double sensor_height = 0.70;
  int    auto_height_scans = 20;
  double auto_height_min_r = 2.0;    // bins used for the estimate [m]
  double auto_height_max_r = 15.0;
  double auto_height_ema = 0.02;     // post-lock tracking rate (0 = frozen)

  // --- R-GPF (region-wise ground plane fitting) ---
  int    num_iter   = 3;        // plane refit iterations per bin
  int    num_lpr    = 8;        // # lowest points averaged for the seed level
  double th_seeds   = 0.40;     // seed = z < (lpr_z + th_seeds)            [m]
  double th_dist    = 0.15;     // inlier band around the fitted plane      [m]
  int    min_bin_pts = 6;       // bins sparser than this are skipped

  // --- Ground-likelihood gating ---
  // Upright: |n_z| of the unit plane normal. cos(45deg)=0.707, cos(30deg)=0.866.
  double th_uprightness = 0.80;
  // Flatness: surface variation lambda0 / (lambda0+lambda1+lambda2). Low = flat.
  double th_flatness    = 0.02;
  // Elevation: accepted ground centroid z must lie within this band around the
  // nominal ground level (-sensor_height). Generous to tolerate terrain slope.
  double elevation_tol  = 0.50;
};

class GroundSegmenter
{
public:
  explicit GroundSegmenter(const GroundConfig & cfg = GroundConfig()) { configure(cfg); }

  void configure(const GroundConfig & cfg)
  {
    cfg_ = cfg;
    auto_height_ = cfg_.sensor_height <= 0.0;
    height_ = auto_height_ ? 0.0 : cfg_.sensor_height;
    height_locked_ = false;
    height_samples_.clear();
    buildZones();
  }

  const GroundConfig & config() const { return cfg_; }

  // Classify every point in `pts` (any indexable container of points with
  // .x .y .z, e.g. a PCL aligned point vector). Fills `is_ground` (1 = ground)
  // and caches the per-bin plane fit for groundHeightAt().
  template <typename Cloud>
  void segment(const Cloud & pts, std::vector<uint8_t> & is_ground)
  {
    is_ground.assign(pts.size(), 0u);
    point_bins_.assign(pts.size(), -1);

    // --- Bin the points into the CZM ---
    for (auto & b : bins_) b.idx.clear();
    for (size_t i = 0; i < pts.size(); ++i)
    {
      const int b = binIndex(pts[i].x, pts[i].y);
      point_bins_[i] = b;
      if (b >= 0) bins_[b].idx.push_back(static_cast<int>(i));
    }

    // --- Region-wise ground plane fit + gating, per bin ---
    // Auto height: until locked, the elevation gate is disabled (height unknown).
    const bool calibrating = auto_height_ && !height_locked_;
    fallback_ground_z_ = -height_;
    double accepted_z_sum = 0.0;
    int    accepted_bins  = 0;
    populated_bins_ = 0;
    std::vector<double> near_z;           // accepted centroid z of near bins (auto height)

    std::vector<float> zbuf;
    for (auto & bin : bins_)
    {
      bin.valid = false;
      if (static_cast<int>(bin.idx.size()) < cfg_.min_bin_pts) continue;
      ++populated_bins_;

      // Seed: lowest-point representative + margin (robust to a few low outliers).
      zbuf.clear();
      zbuf.reserve(bin.idx.size());
      for (int id : bin.idx) zbuf.push_back(pts[id].z);
      const int nlpr = std::min<int>(cfg_.num_lpr, static_cast<int>(zbuf.size()));
      if (nlpr < static_cast<int>(zbuf.size()))
        std::nth_element(zbuf.begin(), zbuf.begin() + nlpr, zbuf.end());
      else
        std::sort(zbuf.begin(), zbuf.end());
      double lpr = 0.0;
      for (int k = 0; k < nlpr; ++k) lpr += zbuf[k];
      lpr /= std::max(1, nlpr);
      const double seed_z = lpr + cfg_.th_seeds;

      std::vector<int> ground;
      ground.reserve(bin.idx.size());
      for (int id : bin.idx)
        if (pts[id].z < seed_z) ground.push_back(id);
      if (static_cast<int>(ground.size()) < cfg_.min_bin_pts) continue;

      Eigen::Vector3d normal;
      Eigen::Vector3d centroid;
      double evals[3];
      // Iteratively refit the plane and re-collect inliers around it.
      for (int it = 0; it < cfg_.num_iter; ++it)
      {
        if (!fitPlane(pts, ground, normal, centroid, evals)) { ground.clear(); break; }
        const double d = -normal.dot(centroid);
        ground.clear();
        for (int id : bin.idx)
        {
          const double dist = std::abs(normal.x() * pts[id].x +
                                       normal.y() * pts[id].y +
                                       normal.z() * pts[id].z + d);
          if (dist < cfg_.th_dist) ground.push_back(id);
        }
        if (static_cast<int>(ground.size()) < cfg_.min_bin_pts) { ground.clear(); break; }
      }
      if (ground.empty()) continue;
      if (!fitPlane(pts, ground, normal, centroid, evals)) continue;

      // --- Ground-likelihood gating ---
      const double uprightness = std::abs(normal.z());                  // |n_z|
      const double trace = evals[0] + evals[1] + evals[2];
      const double flatness = (trace > 1e-12) ? evals[0] / trace : 1.0; // lambda0/sum
      const double elev_err = std::abs(centroid.z() - (-height_));

      if (uprightness < cfg_.th_uprightness) continue;  // wall / trunk, not ground
      if (flatness    > cfg_.th_flatness)    continue;  // bushy / volumetric
      if (!calibrating && elev_err > cfg_.elevation_tol) continue;  // elevated flat surface
      if (auto_height_ && bin_r_[&bin - bins_.data()] >= cfg_.auto_height_min_r &&
          bin_r_[&bin - bins_.data()] <= cfg_.auto_height_max_r)
        near_z.push_back(centroid.z());

      // Accept: label inliers and cache the plane for height queries.
      for (int id : ground) is_ground[id] = 1u;
      bin.valid = true;
      bin.n = normal.cast<float>();
      bin.d = static_cast<float>(-normal.dot(centroid));
      bin.cz = static_cast<float>(centroid.z());
      accepted_z_sum += centroid.z();
      ++accepted_bins;
    }

    if (accepted_bins > 0) fallback_ground_z_ = accepted_z_sum / accepted_bins;
    accepted_bins_ = accepted_bins;
    if (auto_height_) updateAutoHeight(near_z);
  }

  // Fraction of populated CZM bins accepted as ground this scan. Low values mean
  // the ground model is unreliable (wrong height, no ground in view) and any
  // height-above-ground band must NOT be applied.
  double groundBinFraction() const
  { return populated_bins_ > 0 ? double(accepted_bins_) / populated_bins_ : 0.0; }

  double sensorHeight() const { return height_; }
  bool   heightReady() const { return !auto_height_ || height_locked_; }

  // Slope-aware ground height at (x,y): evaluate the fitted plane of the
  // covering bin; fall back to the nearest valid bin, then to the scene mean.
  float groundHeightAt(float x, float y) const
  {
    return groundHeightFromBin(binIndex(x, y), x, y);
  }

  float groundHeightAtPoint(size_t point_index, float x, float y) const
  {
    if (point_index < point_bins_.size())
      return groundHeightFromBin(point_bins_[point_index], x, y);
    return groundHeightAt(x, y);
  }

private:
  // Auto sensor height: per scan take the MEDIAN accepted near-bin centroid z (the
  // ground dominates upright+flat near bins; roofs/tables are a minority), lock on
  // the median over the first N scans, then track slowly within the elevation gate.
  void updateAutoHeight(std::vector<double> & near_z)
  {
    if (near_z.size() < 3) return;
    std::nth_element(near_z.begin(), near_z.begin() + near_z.size() / 2, near_z.end());
    const double h = -near_z[near_z.size() / 2];
    if (h <= 0.0) return;                                   // ground cannot be above the sensor
    if (!height_locked_)
    {
      height_samples_.push_back(h);
      if (static_cast<int>(height_samples_.size()) >= std::max(1, cfg_.auto_height_scans))
      {
        std::vector<double> v = height_samples_;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        height_ = v[v.size() / 2];
        height_locked_ = true;
      }
    }
    else
    {
      height_ += cfg_.auto_height_ema * (h - height_);
    }
  }

  struct ZoneDef { double r_min, r_max; int rings, sectors; double ring_size, sector_size; int offset; };
  struct Bin { std::vector<int> idx; bool valid = false; Eigen::Vector3f n; float d = 0.f; float cz = 0.f; };
  struct BinAddr { int zone, ring, sector; };

  float groundHeightFromBin(int b, float x, float y) const
  {
    if (b >= 0 && bins_[b].valid) return planeZ(bins_[b], x, y);

    // Search outward in the same sector / neighbouring rings for a valid bin.
    if (b >= 0)
    {
      const BinAddr a = addr_[b];
      for (int dr = 1; dr <= 3; ++dr)
      {
        for (int s = -1; s <= 1; ++s)
        {
          int nb = neighborBin(a, dr, s);
          if (nb >= 0 && bins_[nb].valid) return planeZ(bins_[nb], x, y);
          nb = neighborBin(a, -dr, s);
          if (nb >= 0 && bins_[nb].valid) return planeZ(bins_[nb], x, y);
        }
      }
    }
    return fallback_ground_z_;
  }

  // Patchwork++-style CZM: finer near the sensor, coarser far away.
  void buildZones()
  {
    zones_.clear();
    const double rmin = cfg_.min_range;
    const double rmax = cfg_.max_range;
    // Four zones split geometrically across [min_range, max_range].
    const double b1 = rmin + (rmax - rmin) * 0.10;
    const double b2 = rmin + (rmax - rmin) * 0.25;
    const double b3 = rmin + (rmax - rmin) * 0.50;
    const int rings[4]   = {2, 4, 4, 4};
    const int sectors[4] = {16, 32, 54, 32};
    const double edges[5] = {rmin, b1, b2, b3, rmax};

    int offset = 0;
    for (int z = 0; z < 4; ++z)
    {
      ZoneDef zd;
      zd.r_min = edges[z];
      zd.r_max = edges[z + 1];
      zd.rings = rings[z];
      zd.sectors = sectors[z];
      zd.ring_size = (zd.r_max - zd.r_min) / std::max(1, zd.rings);
      zd.sector_size = 2.0 * M_PI / zd.sectors;
      zd.offset = offset;
      offset += zd.rings * zd.sectors;
      zones_.push_back(zd);
    }
    bins_.assign(offset, Bin());
    addr_.assign(offset, BinAddr{0, 0, 0});
    bin_r_.assign(offset, 0.0);
    for (int z = 0; z < 4; ++z)
      for (int r = 0; r < zones_[z].rings; ++r)
        for (int s = 0; s < zones_[z].sectors; ++s)
        {
          const int b = zones_[z].offset + r * zones_[z].sectors + s;
          addr_[b] = BinAddr{z, r, s};
          bin_r_[b] = zones_[z].r_min + (r + 0.5) * zones_[z].ring_size;
        }
  }

  int binIndex(double x, double y) const
  {
    const double r = std::sqrt(x * x + y * y);
    if (r < cfg_.min_range || r >= cfg_.max_range) return -1;
    for (size_t z = 0; z < zones_.size(); ++z)
    {
      const ZoneDef & zd = zones_[z];
      if (r < zd.r_min || r >= zd.r_max) continue;
      int ring = static_cast<int>((r - zd.r_min) / zd.ring_size);
      ring = std::min(ring, zd.rings - 1);
      double th = std::atan2(y, x);
      if (th < 0) th += 2.0 * M_PI;
      int sector = static_cast<int>(th / zd.sector_size);
      sector = std::min(sector, zd.sectors - 1);
      return zd.offset + ring * zd.sectors + sector;
    }
    return -1;
  }

  int neighborBin(const BinAddr & a, int dring, int dsector) const
  {
    const ZoneDef & zd = zones_[a.zone];
    int ring = a.ring + dring;
    if (ring < 0 || ring >= zd.rings) return -1;
    int sector = (a.sector + dsector % zd.sectors + zd.sectors) % zd.sectors;
    return zd.offset + ring * zd.sectors + sector;
  }

  static float planeZ(const Bin & b, float x, float y)
  {
    if (std::abs(b.n.z()) < 1e-6f) return b.cz;
    return -(b.n.x() * x + b.n.y() * y + b.d) / b.n.z();
  }

  // PCA plane fit: normal = eigenvector of the smallest covariance eigenvalue.
  // Returns eigenvalues ascending in `evals` for the flatness test.
  template <typename Cloud>
  static bool fitPlane(const Cloud & pts, const std::vector<int> & ids,
                       Eigen::Vector3d & normal, Eigen::Vector3d & centroid, double evals[3])
  {
    const int n = static_cast<int>(ids.size());
    if (n < 3) return false;
    centroid.setZero();
    for (int id : ids) centroid += Eigen::Vector3d(pts[id].x, pts[id].y, pts[id].z);
    centroid /= n;
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (int id : ids)
    {
      Eigen::Vector3d v(pts[id].x - centroid.x(), pts[id].y - centroid.y(), pts[id].z - centroid.z());
      cov += v * v.transpose();
    }
    cov /= n;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    evals[0] = es.eigenvalues()[0];
    evals[1] = es.eigenvalues()[1];
    evals[2] = es.eigenvalues()[2];
    normal = es.eigenvectors().col(0);          // smallest eigenvalue
    if (normal.z() < 0) normal = -normal;        // orient upward
    return true;
  }

  GroundConfig cfg_;
  std::vector<ZoneDef> zones_;
  std::vector<Bin> bins_;
  std::vector<BinAddr> addr_;
  std::vector<int> point_bins_;
  std::vector<double> bin_r_;          // radial centre of each bin [m]
  float fallback_ground_z_ = 0.f;
  int populated_bins_ = 0, accepted_bins_ = 0;
  bool auto_height_ = false, height_locked_ = false;
  double height_ = 0.70;
  std::vector<double> height_samples_;
};

}  // namespace cop
