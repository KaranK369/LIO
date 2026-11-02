// =============================================================================
//  Geometric primitive landmark map               (Stage C — multi-terrain)
// -----------------------------------------------------------------------------
//  A persistent, world-frame map of typed geometric primitives extracted at
//  keyframe rate from the registered cloud:
//
//    * POLE  (vertical cylinder) — trees, posts, pillars, columns, fence posts.
//      A tree trunk is just one instance. Carries base position + diameter + height.
//    * PLANE (vertical patch)    — walls, facades, retaining walls, big flat boards.
//      Carries centroid + normal + extent.
//
//  This class owns ASSOCIATION + PERSISTENCE + PROMOTION only (no point cloud
//  ops — those live in the node). Data association is by relative world-frame
//  geometry, never appearance, which is what survives perceptual aliasing
//  (look-alike trees / repeated facades):
//
//    observe*()  -> match a detection to the nearest compatible landmark and
//                   fuse it (running mean), else open a new candidate.
//    endFrame()  -> promote candidates seen in >= promote_hits keyframes to
//                   CONFIRMED, and forget stale unconfirmed candidates so a
//                   one-off spurious detection never reaches the map.
//
//  Each landmark also carries a coarse PERMANENCE prior (methodology §4): thick
//  poles / large planes (trees, buildings) are durable; thin posts less so.
//  Cross-run relocalization (later) trusts high-permanence landmarks only.
//  Self-contained: Eigen + STL.
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace cop
{

enum class LandmarkType : uint8_t { Pole, Plane };

struct Landmark
{
  LandmarkType type;
  int      id        = -1;
  Eigen::Vector3f pos{0, 0, 0};   // pole: base (x,y, ground z); plane: centroid
  // pole
  float    diameter  = 0.f;
  float    height    = 0.f;
  // plane
  Eigen::Vector3f normal{0, 0, 1};
  float    extent    = 0.f;
  // bookkeeping
  int      hits      = 0;          // # keyframes observed
  uint32_t first_seen = 0;
  uint32_t last_seen  = 0;
  bool     confirmed = false;
  uint8_t  permanence = 1;         // 0 transient .. 2 durable (coarse prior)
};

struct LandmarkConfig
{
  float pole_assoc_dist  = 0.6f;   // xy gate to fuse a pole detection        [m]
  float plane_assoc_dist = 2.0f;   // centroid gate to fuse a plane detection [m]
  float plane_normal_tol = 0.25f;  // 1 - |n_det . n_lm| must be below this
  float plane_offset_tol = 0.5f;   // det centroid must lie within this of the plane [m]
  int   promote_hits     = 3;      // keyframes needed to CONFIRM a candidate
  int   forget_frames    = 8;      // drop UNconfirmed candidates unseen this long
};

class LandmarkMap
{
public:
  explicit LandmarkMap(const LandmarkConfig & cfg = LandmarkConfig()) : cfg_(cfg) {}
  void configure(const LandmarkConfig & cfg) { cfg_ = cfg; }

  // Fuse a pole detection (base at (x,y,ground_z), world frame). Returns its id.
  int observePole(const Eigen::Vector3f & base, float diameter, float height, uint32_t frame)
  {
    int best = -1; float bestd = cfg_.pole_assoc_dist;
    for (size_t i = 0; i < lms_.size(); ++i)
    {
      if (lms_[i].type != LandmarkType::Pole) continue;
      const float d = std::hypot(lms_[i].pos.x() - base.x(), lms_[i].pos.y() - base.y());
      if (d < bestd) { bestd = d; best = int(i); }
    }
    if (best < 0)
    {
      Landmark lm; lm.type = LandmarkType::Pole; lm.id = next_id_++;
      lm.pos = base; lm.diameter = diameter; lm.height = height;
      lm.hits = 1; lm.first_seen = lm.last_seen = frame;
      lm.permanence = permanencePole(diameter);
      lms_.push_back(lm);
      return lm.id;
    }
    Landmark & lm = lms_[best];
    if (lm.last_seen != frame) ++lm.hits;             // one hit per keyframe
    const float w = 1.f / float(lm.hits + 1);
    lm.pos      = (1 - w) * lm.pos + w * base;
    lm.diameter = (1 - w) * lm.diameter + w * diameter;
    lm.height   = std::max(lm.height, height);
    lm.permanence = permanencePole(lm.diameter);
    lm.last_seen = frame;
    return lm.id;
  }

  // Fuse a plane detection (centroid + unit normal + extent, world frame).
  int observePlane(const Eigen::Vector3f & centroid, const Eigen::Vector3f & normal,
                   float extent, uint32_t frame)
  {
    Eigen::Vector3f n = normal.normalized();
    int best = -1; float bestd = cfg_.plane_assoc_dist;
    for (size_t i = 0; i < lms_.size(); ++i)
    {
      if (lms_[i].type != LandmarkType::Plane) continue;
      const float align = 1.f - std::fabs(lms_[i].normal.dot(n));
      if (align > cfg_.plane_normal_tol) continue;
      // centroid must be near the existing plane (along its normal) AND nearby
      const float off = std::fabs((centroid - lms_[i].pos).dot(lms_[i].normal));
      if (off > cfg_.plane_offset_tol) continue;
      const float d = (lms_[i].pos - centroid).norm();
      if (d < bestd) { bestd = d; best = int(i); }
    }
    if (best < 0)
    {
      Landmark lm; lm.type = LandmarkType::Plane; lm.id = next_id_++;
      lm.pos = centroid; lm.normal = n; lm.extent = extent;
      lm.hits = 1; lm.first_seen = lm.last_seen = frame;
      lm.permanence = 2;                              // large flat structure = durable
      lms_.push_back(lm);
      return lm.id;
    }
    Landmark & lm = lms_[best];
    if (lm.last_seen != frame) ++lm.hits;
    const float w = 1.f / float(lm.hits + 1);
    if (lm.normal.dot(n) < 0) n = -n;                 // align before averaging
    lm.pos    = (1 - w) * lm.pos + w * centroid;
    lm.normal = ((1 - w) * lm.normal + w * n).normalized();
    lm.extent = std::max(lm.extent, extent);
    lm.last_seen = frame;
    return lm.id;
  }

  // Promote well-supported candidates; forget stale unconfirmed ones.
  void endFrame(uint32_t frame)
  {
    for (auto & lm : lms_)
      if (!lm.confirmed && lm.hits >= cfg_.promote_hits) lm.confirmed = true;

    lms_.erase(std::remove_if(lms_.begin(), lms_.end(),
      [&](const Landmark & lm) {
        return !lm.confirmed &&
               frame - lm.last_seen > static_cast<uint32_t>(cfg_.forget_frames);
      }), lms_.end());
  }

  const std::vector<Landmark> & landmarks() const { return lms_; }

  void counts(int & poles_c, int & poles_total, int & planes_c, int & planes_total) const
  {
    poles_c = poles_total = planes_c = planes_total = 0;
    for (const auto & lm : lms_)
    {
      const bool pole = lm.type == LandmarkType::Pole;
      (pole ? poles_total : planes_total)++;
      if (lm.confirmed) (pole ? poles_c : planes_c)++;
    }
  }

private:
  static uint8_t permanencePole(float diameter)
  {
    if (diameter >= 0.25f) return 2;                  // thick trunk / pillar
    if (diameter >= 0.12f) return 1;                  // sapling / sturdy post
    return 0;                                         // thin post / sign
  }

  LandmarkConfig cfg_;
  std::vector<Landmark> lms_;
  int next_id_ = 0;
};

}  // namespace cop
