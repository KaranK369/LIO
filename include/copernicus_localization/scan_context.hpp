// =============================================================================
//  Scan Context global descriptor            (Stage D — loop-closure candidate)
// -----------------------------------------------------------------------------
//  Kim & Kim 2018. An egocentric polar bird's-eye descriptor for place
//  recognition. The cloud (in the sensor/body frame) is binned into a
//  num_rings x num_sectors polar grid; each bin stores the MAX height of the
//  points that fall in it. Two products:
//
//    * sc       — the full num_rings x num_sectors matrix (the "image").
//    * ringkey  — per-ring mean across sectors (a num_rings vector). Rotation
//                 INVARIANT (an azimuth change only permutes columns, leaving
//                 row means unchanged) -> used for fast top-K retrieval.
//
//  Matching the full matrix searches over all column (azimuth) shifts and
//  returns the minimum column-cosine distance AND the best shift, which doubles
//  as a relative-yaw estimate (used to initialise ICP verification downstream).
//
//  HEIGHT, not intensity (methodology §3D): intensity is unreliable across
//  snow/sand/water; height-based SC is terrain-general. Heights are encoded
//  positive (z + height_offset, floored at 0) so empty bins (0) are distinct
//  from occupied ground bins and column-cosine stays well-behaved.
//  Self-contained: Eigen + STL.
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

namespace cop
{

struct ScanContextConfig
{
  int   num_rings    = 20;     // radial bins
  int   num_sectors  = 60;     // azimuth bins
  float max_radius   = 80.0f;  // descriptor reach                          [m]
  float height_offset = 2.0f;  // additive so ground/structure encode positive [m]
};

struct ScanContextDesc
{
  Eigen::MatrixXf sc;          // num_rings x num_sectors (max height per bin)
  Eigen::VectorXf ringkey;     // num_rings (row means) — rotation invariant
};

class ScanContext
{
public:
  explicit ScanContext(const ScanContextConfig & cfg = ScanContextConfig()) : cfg_(cfg) {}
  void configure(const ScanContextConfig & cfg) { cfg_ = cfg; }
  const ScanContextConfig & config() const { return cfg_; }

  // Build a descriptor from a BODY/sensor-frame cloud (.x .y .z floats).
  template <typename PointVec>
  ScanContextDesc make(const PointVec & pts) const
  {
    ScanContextDesc d;
    d.sc = Eigen::MatrixXf::Zero(cfg_.num_rings, cfg_.num_sectors);
    const float rstep = cfg_.max_radius / float(cfg_.num_rings);
    for (const auto & p : pts)
    {
      const float r = std::hypot(float(p.x), float(p.y));
      if (r < 1e-3f || r >= cfg_.max_radius) continue;
      int ri = int(r / rstep);
      ri = std::min(std::max(ri, 0), cfg_.num_rings - 1);
      const float th = std::atan2(float(p.y), float(p.x));           // -pi..pi
      int si = int((th + float(M_PI)) / (2.f * float(M_PI)) * cfg_.num_sectors);
      si = std::min(std::max(si, 0), cfg_.num_sectors - 1);
      float h = float(p.z) + cfg_.height_offset;
      if (h < 0.f) h = 0.f;
      if (h > d.sc(ri, si)) d.sc(ri, si) = h;
    }
    d.ringkey = d.sc.rowwise().mean();
    return d;
  }

  static float ringKeyDist(const ScanContextDesc & a, const ScanContextDesc & b)
  {
    return (a.ringkey - b.ringkey).norm();
  }

  // Min column-cosine distance over all azimuth shifts. `best_shift` (sectors)
  // gives the relative yaw = best_shift * 2pi / num_sectors.
  static float distance(const ScanContextDesc & a, const ScanContextDesc & b, int & best_shift)
  {
    const int Ns = int(a.sc.cols());
    float best = 1e9f; best_shift = 0;
    for (int s = 0; s < Ns; ++s)
    {
      float sum = 0.f; int valid = 0;
      for (int c = 0; c < Ns; ++c)
      {
        const Eigen::VectorXf colA = a.sc.col(c);
        const Eigen::VectorXf colB = b.sc.col((c + s) % Ns);
        const float nA = colA.norm(), nB = colB.norm();
        if (nA < 1e-6f || nB < 1e-6f) continue;
        sum += 1.f - colA.dot(colB) / (nA * nB);
        ++valid;
      }
      if (!valid) continue;
      const float d = sum / float(valid);
      if (d < best) { best = d; best_shift = s; }
    }
    return best;
  }

  // Informativeness: a bin is STRUCTURED if its max height rises at least
  // `min_rise` above the median occupied height of its ring. A flat runway or an
  // empty plain gives ~0 structured bins -> every place looks alike, so the
  // descriptor must not be used for retrieval at all.
  static void structure(const ScanContextDesc & d, float min_rise, int & n_struct, double & frac)
  {
    n_struct = 0;
    int occupied = 0;
    std::vector<float> row;
    for (int r = 0; r < d.sc.rows(); ++r)
    {
      row.clear();
      for (int c = 0; c < d.sc.cols(); ++c) if (d.sc(r, c) > 0.f) row.push_back(d.sc(r, c));
      if (row.empty()) continue;
      std::vector<float> tmp = row;
      std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
      const float med = tmp[tmp.size() / 2];
      occupied += int(row.size());
      for (float v : row) if (v - med >= min_rise) ++n_struct;
    }
    frac = occupied ? double(n_struct) / occupied : 0.0;
  }

  float yawFromShift(int shift) const
  {
    return float(shift) * 2.f * float(M_PI) / float(cfg_.num_sectors);
  }

private:
  ScanContextConfig cfg_;
};

}  // namespace cop
