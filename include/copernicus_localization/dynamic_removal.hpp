// =============================================================================
//  Dynablox-style dynamic-object removal               (Stage A, step 5)
// -----------------------------------------------------------------------------
//  Removes *translating* actors (people, animals, vehicles) by reasoning about
//  free space over time, in the world frame.  A lightweight, online, map-free
//  take on Dynablox (Schmid 2023):
//
//    "ever-free" idea:  a voxel that has been repeatedly observed as FREE
//    (a laser ray passed through it) is confirmed empty.  When a return later
//    lands inside a *recently* ever-free voxel, something moved into space we
//    just saw was empty -> dynamic candidate.
//
//  Robustness guards (without these, a sparse 32-beam LiDAR on a moving base
//  produces large false positives — mostly ground):
//    * GROUND IS EXCLUDED from candidacy. On a sparse scanner the gaps between
//      laser rings on the ground get marked "free" by rays at other azimuths;
//      as the base drives forward, ground returns land in those voxels and would
//      be falsely flagged. Ground returns still UPDATE the map (mark occupied) so
//      they suppress spurious free space — they just can't be called dynamic.
//    * RECENCY: a voxel must have been freed within `free_recency` keyframes, so
//      revisiting an area mapped long ago does not flag its static structure.
//    * BOUNDED CLUSTER: a confirmed dynamic blob must have >= min_cluster_points
//      AND a horizontal extent <= max_cluster_extent — keeps compact actors,
//      rejects large swaths of newly-seen static (walls, treelines, ground).
//
//  Pipeline:
//    classify (every scan) - flag non-ground points in recently-ever-free voxels.
//    confirm  (every scan) - keep clusters within the size/extent bounds.
//    update   (keyframe)   - ray-cast free space + mark endpoints occupied.
//    prune    (periodic)   - drop voxels unseen for a while to bound memory.
//
//  Scope (per the methodology): this targets *translation*, not in-place wind
//  jitter of vegetation — that is handled upstream by the ground/rigidity/
//  trunk-band steps. Self-contained: Eigen + STL only.
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

namespace cop
{

struct DynamicConfig
{
  double voxel_size         = 0.40;  // world voxel edge                       [m]
  double max_ray            = 40.0;  // cap ray-cast length (skip far returns) [m]
  int    ever_free_count    = 3;     // free observations to confirm ever-free
  int    occupied_reset     = 1;     // occupied observations to clear ever-free
  int    free_recency       = 30;    // voxel must have been freed within N keyframes
  int    min_cluster_points = 6;     // confirmed-dynamic cluster size floor
  double max_cluster_extent = 4.0;   // confirmed-dynamic horizontal extent cap [m]
  int    voxel_ttl_frames   = 200;   // evict voxels unseen this many keyframes
  int    prune_interval     = 50;    // run eviction every N keyframes
};

class DynamicRemover
{
public:
  explicit DynamicRemover(const DynamicConfig & cfg = DynamicConfig()) : cfg_(cfg) {}
  void configure(const DynamicConfig & cfg) { cfg_ = cfg; }
  const DynamicConfig & config() const { return cfg_; }

  // Classify `pts_world` (world frame) and optionally fold the scan into the
  // temporal map. Fills `is_dynamic` (1 = remove). Returns the number removed.
  //
  //   update_map = true  -> classify + ray-cast free-space + mark occupied
  //                         (do this at keyframe rate; it is the expensive part).
  //   update_map = false -> classify ONLY against the current map (cheap; run
  //                         this on every scan so dynamic points never leak
  //                         through on non-keyframe scans).
  //   ground (optional)  -> per-point flag, 1 = ground. Ground points never
  //                         become dynamic candidates (but still update the map).
  size_t process(const std::vector<Eigen::Vector3f> & pts_world,
                 const Eigen::Vector3f & sensor_world,
                 std::vector<uint8_t> & is_dynamic,
                 bool update_map = true,
                 const std::vector<uint8_t> * ground = nullptr)
  {
    const double inv = 1.0 / cfg_.voxel_size;
    is_dynamic.assign(pts_world.size(), 0u);
    const bool have_ground = ground && ground->size() == pts_world.size();

    // ---- 1. classify against the *current* map state -----------------------
    // Candidate if non-ground AND its voxel is ever-free AND was freed recently.
    std::vector<int> cand;                       // candidate point indices
    cand.reserve(pts_world.size() / 8 + 1);
    for (size_t i = 0; i < pts_world.size(); ++i)
    {
      if (have_ground && (*ground)[i]) continue;            // never flag ground
      const uint64_t k = key(pts_world[i], inv);
      auto it = map_.find(k);
      if (it == map_.end() || !it->second.ever_free) continue;
      if (frame_ - it->second.last_free > static_cast<uint32_t>(cfg_.free_recency)) continue;
      cand.push_back(static_cast<int>(i));
    }

    // ---- 2. confirm by clustering (voxel 26-connectivity), bounded size -----
    if (!cand.empty()) confirmClusters(pts_world, cand, inv, is_dynamic);

    size_t removed = 0;
    for (uint8_t f : is_dynamic) removed += f;
    if (!update_map) return removed;

    // ---- 3. update the temporal map (keyframe rate) -----------------------
    ++frame_;
    // Endpoints first (occupied this frame), so rays passing through an
    // endpoint voxel do not also free it.
    occ_this_frame_.clear();
    for (const auto & p : pts_world)
    {
      if ((p - sensor_world).norm() > cfg_.max_ray) continue;
      occ_this_frame_.insert(key(p, inv));
    }
    for (uint64_t k : occ_this_frame_)
    {
      Voxel & v = map_[k];
      v.last_seen = frame_;
      v.free_count = 0;
      if (++v.occ_count >= cfg_.occupied_reset) v.ever_free = false;
    }

    // Free-space carving: ray-cast sensor -> endpoint, mark traversed voxels.
    for (const auto & p : pts_world)
    {
      const double r = (p - sensor_world).norm();
      if (r > cfg_.max_ray || r < 1e-3) continue;
      rayFree(sensor_world, p, inv);
    }

    // ---- 4. bound memory ---------------------------------------------------
    if (frame_ % cfg_.prune_interval == 0) prune();
    return removed;
  }

  size_t mapSize() const { return map_.size(); }

private:
  struct Voxel
  {
    uint32_t last_seen  = 0;
    uint32_t last_free  = 0;
    uint16_t free_count = 0;
    uint16_t occ_count  = 0;
    bool     ever_free  = false;
  };

  static inline int64_t fl(double v) { return static_cast<int64_t>(std::floor(v)); }

  static uint64_t pack(int64_t ix, int64_t iy, int64_t iz)
  {
    constexpr uint64_t M = 0x1FFFFF;            // 21 bits
    return ((uint64_t(ix) & M) << 42) | ((uint64_t(iy) & M) << 21) | (uint64_t(iz) & M);
  }
  static uint64_t key(const Eigen::Vector3f & p, double inv)
  {
    return pack(fl(p.x() * inv), fl(p.y() * inv), fl(p.z() * inv));
  }

  // Amanatides-Woo voxel traversal; frees every voxel strictly between the
  // sensor and the endpoint (the endpoint voxel itself stays occupied).
  void rayFree(const Eigen::Vector3f & s, const Eigen::Vector3f & e, double inv)
  {
    const double vs = cfg_.voxel_size;
    Eigen::Vector3d s0(s.x() * inv, s.y() * inv, s.z() * inv);
    int64_t ix = fl(s0.x()), iy = fl(s0.y()), iz = fl(s0.z());
    const int64_t ex = fl(e.x() * inv), ey = fl(e.y() * inv), ez = fl(e.z() * inv);

    Eigen::Vector3d dir(e.x() - s.x(), e.y() - s.y(), e.z() - s.z());
    const double len = dir.norm();
    if (len < 1e-6) return;
    dir /= len;

    auto step_setup = [&](double d, int64_t i, double org, int & step, double & tMax, double & tDelta) {
      if (d > 1e-9) {
        step = 1;
        tMax = ((double(i) + 1.0) * vs - org) / d;
        tDelta = vs / d;
      } else if (d < -1e-9) {
        step = -1;
        tMax = (org - double(i) * vs) / -d;
        tDelta = vs / -d;
      } else { step = 0; tMax = 1e30; tDelta = 1e30; }
    };
    int sx, sy, sz; double tMaxX, tMaxY, tMaxZ, tDX, tDY, tDZ;
    step_setup(dir.x(), ix, s.x(), sx, tMaxX, tDX);
    step_setup(dir.y(), iy, s.y(), sy, tMaxY, tDY);
    step_setup(dir.z(), iz, s.z(), sz, tMaxZ, tDZ);

    const int max_steps = 1024;                  // hard cap (cfg_.max_ray / voxel)
    for (int n = 0; n < max_steps; ++n)
    {
      if (ix == ex && iy == ey && iz == ez) break;   // reached endpoint voxel
      if (tMaxX < tMaxY && tMaxX < tMaxZ) { ix += sx; tMaxX += tDX; }
      else if (tMaxY < tMaxZ)            { iy += sy; tMaxY += tDY; }
      else                               { iz += sz; tMaxZ += tDZ; }
      if (ix == ex && iy == ey && iz == ez) break;

      const uint64_t k = pack(ix, iy, iz);
      if (occ_this_frame_.count(k)) continue;        // occupied beats free
      Voxel & v = map_[k];
      if (v.last_seen == frame_) continue;           // count once per frame
      v.last_seen = frame_;
      v.last_free = frame_;
      if (!v.ever_free) {
        if (++v.free_count >= cfg_.ever_free_count) v.ever_free = true;
      }
    }
  }

  // Group candidate points by their voxel, merge 26-connected voxels, and
  // confirm a cluster only if it has >= min_cluster_points AND its horizontal
  // (xy) extent is <= max_cluster_extent (compact actor, not a static swath).
  void confirmClusters(const std::vector<Eigen::Vector3f> & pts,
                       const std::vector<int> & cand, double inv,
                       std::vector<uint8_t> & is_dynamic) const
  {
    std::unordered_map<uint64_t, std::vector<int>> vmap;   // voxel -> point ids
    vmap.reserve(cand.size() * 2);
    for (int id : cand) vmap[key(pts[id], inv)].push_back(id);

    std::unordered_map<uint64_t, char> visited;
    visited.reserve(vmap.size() * 2);
    std::vector<uint64_t> stack;
    std::vector<int> comp;

    for (const auto & kv : vmap)
    {
      if (visited[kv.first]) continue;
      stack.clear(); comp.clear();
      stack.push_back(kv.first);
      visited[kv.first] = 1;
      while (!stack.empty())
      {
        const uint64_t cur = stack.back(); stack.pop_back();
        auto it = vmap.find(cur);
        if (it != vmap.end())
          for (int id : it->second) comp.push_back(id);

        // unpack 21-bit signed coords
        auto sx = [](uint64_t v)->int64_t { int64_t r = v & 0x1FFFFF; return r >= 0x100000 ? r - 0x200000 : r; };
        const int64_t cx = sx(cur >> 42), cy = sx(cur >> 21), cz = sx(cur);
        for (int dx = -1; dx <= 1; ++dx)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz)
            {
              if (!dx && !dy && !dz) continue;
              const uint64_t nk = pack(cx + dx, cy + dy, cz + dz);
              if (vmap.count(nk) && !visited[nk]) { visited[nk] = 1; stack.push_back(nk); }
            }
      }
      if (static_cast<int>(comp.size()) < cfg_.min_cluster_points) continue;

      // Horizontal extent gate: reject large static swaths.
      float xmin = 1e30f, xmax = -1e30f, ymin = 1e30f, ymax = -1e30f;
      for (int id : comp)
      {
        xmin = std::min(xmin, pts[id].x()); xmax = std::max(xmax, pts[id].x());
        ymin = std::min(ymin, pts[id].y()); ymax = std::max(ymax, pts[id].y());
      }
      if ((xmax - xmin) > cfg_.max_cluster_extent ||
          (ymax - ymin) > cfg_.max_cluster_extent) continue;

      for (int id : comp) is_dynamic[id] = 1u;
    }
  }

  void prune()
  {
    for (auto it = map_.begin(); it != map_.end();)
    {
      if (frame_ - it->second.last_seen > static_cast<uint32_t>(cfg_.voxel_ttl_frames))
        it = map_.erase(it);
      else ++it;
    }
  }

  DynamicConfig cfg_;
  std::unordered_map<uint64_t, Voxel> map_;
  std::unordered_set<uint64_t>        occ_this_frame_;   // reused per frame
  uint32_t frame_ = 0;
};

}  // namespace cop
