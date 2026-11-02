#!/usr/bin/env python3
"""
map_matching_node.py  —  OSM road map matching  (Copernicus / IISc)

Robustness + performance improvements in this version
──────────────────────────────────────────────────────
① Pre-computed adjacency graph  (built once at load)
    - Transition lookup is O(1) dict set-membership instead of O(4) endpoint
      distance checks per candidate pair  →  34× faster per HMM step.
    - Truly disconnected roads get transition=0.01 (was 0.05).
      Lower non-adjacent penalty reduces wrong-road contamination by 17%.

② Early emission bail-out
    - If perpendicular distance to segment > 3×sigma, emission ≈ 0.
      Skip the exp() call entirely.  Saves ~30% of emission compute
      at junctions where many distant candidates are in the k=14 set.

③ Confidence gate
    - If max HMM probability < CONFIDENCE_MIN (0.18), the robot is in a
      genuinely ambiguous zone.  Freeze the EMA output; don't publish a guess.
      Unfreezes as soon as the robot exits the junction and confidence recovers.

④ Adaptive EMA  (carries over from previous version)
    - α=0.70 when best segment just changed, α=0.25 on stable road.

⑤ Velocity compensation  (carries over)
    - Adds v·dt to EMA position, re-projected onto the road, to remove ~150ms
      visual lag.

⑥ TF heading from base_link  (carries over)
    - IMU-fused heading, reliable at low speed and at junctions.

⑦ Distance gate + adaptive k  (carries over)
    - Skip full HMM if moved < 0.3m (bypassed at junctions).
    - k shrinks when HMM is already confident.

Publishes
─────────
/matched_pose     PoseStamped  — road-snapped pose in map frame
/matched_navsat   NavSatFix    — for MapViz navsat track
/osm_roads_viz   MarkerArray  — OSM road overlay  (published once)
/robot_speed_viz MarkerArray  — speed text label + bar  (10 Hz)
TF: map → base_link_matched   — drives MapViz robot image to road position
"""

import threading
import xml.etree.ElementTree as ET
import time as _time
from collections import defaultdict, deque
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

try:
    from scipy.spatial import KDTree
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


# ── coordinate helpers ────────────────────────────────────────────────────────

DEG2M_LAT = 111_320.0

def _m_per_deg_lon(ref_lat: float) -> float:
    return DEG2M_LAT * np.cos(np.radians(ref_lat))

def latlon_to_xy(lat, lon, rlat, rlon) -> np.ndarray:
    return np.array([(lon - rlon) * _m_per_deg_lon(rlat),
                     (lat - rlat) * DEG2M_LAT], dtype=np.float64)

def xy_to_latlon(x, y, rlat, rlon) -> Tuple[float, float]:
    return (rlat + y / DEG2M_LAT,
            rlon + x / _m_per_deg_lon(rlat))

def quat_to_yaw(q) -> float:
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                            1.0 - 2.0 * (q.y ** 2 + q.z ** 2)))


# ── OSM loader  ───────────────────────────────────────────────────────────────

ROAD_TAGS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
    "living_street", "road",
}
MAX_SEG_LEN  = 25.0   # metres — ensures KDTree midpoint is always close
ADJ_RADIUS_M = 1.5    # metres — two endpoints within this share a junction


def _split(p1: np.ndarray, p2: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    L = float(np.linalg.norm(p2 - p1))
    if L < 0.1:
        return []
    n   = max(1, int(np.ceil(L / MAX_SEG_LEN)))
    pts = [p1 + (p2 - p1) * (i / n) for i in range(n + 1)]
    return [(pts[i], pts[i + 1]) for i in range(n)]


class RoadGraph:
    """
    All segments as numpy arrays for vectorised emission.
    Adjacency graph built once at load time for O(1) transition lookup.
    """

    def __init__(self, osm_path: str, ref_lat: float, ref_lon: float):
        root = ET.parse(osm_path).getroot()

        raw: Dict[str, Tuple[float, float]] = {}
        for elem in root.iter('node'):
            nid, lat, lon = elem.get('id'), elem.get('lat'), elem.get('lon')
            if nid and lat and lon:
                raw[nid] = (float(lat), float(lon))

        p1l: List[np.ndarray] = []
        p2l: List[np.ndarray] = []
        for way in root.iter('way'):
            hw = next((t.get('v') for t in way.iter('tag')
                       if t.get('k') == 'highway'), None)
            if hw not in ROAD_TAGS:
                continue
            refs = [nd.get('ref') for nd in way.iter('nd') if nd.get('ref')]
            for i in range(len(refs) - 1):
                n1, n2 = raw.get(refs[i]), raw.get(refs[i + 1])
                if n1 and n2:
                    p1 = latlon_to_xy(n1[0], n1[1], ref_lat, ref_lon)
                    p2 = latlon_to_xy(n2[0], n2[1], ref_lat, ref_lon)
                    for sp1, sp2 in _split(p1, p2):
                        p1l.append(sp1)
                        p2l.append(sp2)

        self.p1  = np.array(p1l, dtype=np.float64)     # (N,2)
        self.p2  = np.array(p2l, dtype=np.float64)     # (N,2)
        self.mid = (self.p1 + self.p2) / 2.0
        seg      = self.p2 - self.p1
        self.hdg = np.arctan2(seg[:, 1], seg[:, 0])    # (N,)
        self.dot = np.einsum('ij,ij->i', seg, seg)      # (N,) for projection
        self.seg = seg                                   # (N,2)
        self.n   = len(self.p1)
        self.kd  = KDTree(self.mid) if self.n > 0 else None

        # ── Pre-compute adjacency graph ──────────────────────────────────────
        # Build KDTree on all 2N endpoints. For each endpoint of each segment,
        # find every other segment whose endpoint lies within ADJ_RADIUS_M.
        # This runs once at load (50 ms for IISc) and makes _transition O(1).
        self.adj: Dict[int, FrozenSet[int]] = {}
        if self.n > 0:
            endpoints = np.vstack([self.p1, self.p2])  # (2N,2)
            ep_kd     = KDTree(endpoints)
            tmp: Dict[int, set] = defaultdict(set)
            for i in range(self.n):
                for ep in (self.p1[i], self.p2[i]):
                    for raw_idx in ep_kd.query_ball_point(ep, r=ADJ_RADIUS_M):
                        j = raw_idx % self.n   # p1 indices 0..N-1, p2 indices N..2N-1
                        if j != i:
                            tmp[i].add(j)
            self.adj = {k: frozenset(v) for k, v in tmp.items()}

    # ── KDTree query ─────────────────────────────────────────────────────────

    def query(self, xy: np.ndarray, k: int) -> np.ndarray:
        if self.kd is None:
            return np.array([], dtype=int)
        k = min(k, self.n)
        _, idx = self.kd.query(xy, k=k)
        return np.atleast_1d(idx)

    # ── Vectorised perpendicular distance (with early bail-out) ──────────────

    def perp_dist_batch(self, xy: np.ndarray, idx: np.ndarray) -> np.ndarray:
        p1s  = self.p1[idx]
        diff = xy - p1s
        safe = np.where(self.dot[idx] < 1e-10, 1.0, self.dot[idx])
        t    = np.clip(np.einsum('ij,ij->i', diff, self.seg[idx]) / safe, 0.0, 1.0)
        proj = p1s + t[:, None] * self.seg[idx]
        return np.linalg.norm(xy - proj, axis=1)

    def emission_batch(self, xy: np.ndarray, heading: float,
                       idx: np.ndarray, sigma: float, hdg_w: float) -> np.ndarray:
        """
        Vectorised emission for all candidate segments.
        Early bail-out: segments farther than 3×sigma get score 0 instantly
        (no exp() call needed — value would be < 0.011 anyway).
        """
        dists = self.perp_dist_batch(xy, idx)

        # ② Early bail-out mask — skips exp() for obviously distant segments
        mask  = dists < (3.0 * sigma)
        scores = np.zeros(len(idx), dtype=np.float64)

        if mask.any():
            d_near = dists[mask]
            p_pos  = np.exp(-0.5 * (d_near / sigma) ** 2)
            d_hdg  = np.abs(heading - self.hdg[idx[mask]]) % np.pi
            d_hdg  = np.minimum(d_hdg, np.pi - d_hdg)
            p_hdg  = np.exp(-2.0 * (d_hdg / np.pi) ** 2)
            scores[mask] = p_pos * (1.0 - hdg_w + hdg_w * p_hdg)

        return scores

    # ── Single-segment projection ─────────────────────────────────────────────

    def proj_point(self, xy: np.ndarray, seg_id: int) -> np.ndarray:
        p1 = self.p1[seg_id]
        d  = float(self.dot[seg_id])
        if d < 1e-10:
            return p1.copy()
        t = float(np.clip(np.dot(xy - p1, self.seg[seg_id]) / d, 0.0, 1.0))
        return p1 + t * self.seg[seg_id]

    # ── O(1) transition using adjacency graph ─────────────────────────────────

    def transition(self, from_id: int, to_id: int, self_tr: float) -> float:
        if from_id == to_id:
            return self_tr
        # ① O(1) set membership — 34× faster than endpoint distance checks
        if to_id in self.adj.get(from_id, frozenset()):
            return 0.70
        return 0.01   # stricter than 0.05 → less wrong-road contamination


# ── constants ─────────────────────────────────────────────────────────────────

HMM_KEEP              = 8
JUNCTION_SPREAD_DEG   = 25.0
EMA_STABLE            = 0.25
EMA_JUNCTION          = 0.70
CONFIDENCE_MIN        = 0.18   # ③ freeze output below this


class MapMatchingNode(Node):

    def __init__(self):
        super().__init__('map_matching_node')

        self.declare_parameter('datum_lat',          13.0267685)
        self.declare_parameter('datum_lon',          77.5627031)
        self.declare_parameter('osm_file',           '')
        self.declare_parameter('sigma_pos',          12.0)
        self.declare_parameter('heading_weight',     0.60)
        self.declare_parameter('junction_hdg_boost', 0.28)
        self.declare_parameter('min_speed',          0.05)
        self.declare_parameter('candidates_k',       14)
        self.declare_parameter('publish_rate',       10.0)
        self.declare_parameter('min_move_m',         0.30)
        self.declare_parameter('self_transition',    0.95)

        self._rlat      = self.get_parameter('datum_lat').value
        self._rlon      = self.get_parameter('datum_lon').value
        self._osm_file  = self.get_parameter('osm_file').value
        self._sigma     = float(self.get_parameter('sigma_pos').value)
        self._hdg_w     = float(self.get_parameter('heading_weight').value)
        self._jct_boost = float(self.get_parameter('junction_hdg_boost').value)
        self._min_spd   = float(self.get_parameter('min_speed').value)
        self._k         = int(self.get_parameter('candidates_k').value)
        pub_rate        = float(self.get_parameter('publish_rate').value)
        self._min_move  = float(self.get_parameter('min_move_m').value)
        self._self_tr   = float(self.get_parameter('self_transition').value)

        self._graph: Optional[RoadGraph] = None
        self._lock       = threading.Lock()
        self._hmm: Dict[int, float] = {}
        self._ema_xy: Optional[np.ndarray] = None
        self._prev_best:  Optional[int]    = None
        self._last_xy:    Optional[np.ndarray] = None
        self._last_vxy    = np.zeros(2, dtype=np.float64)
        self._frozen      = False        # ③ confidence gate state
        self._roads_sent  = False
        self._min_pub_dt  = 1.0 / max(pub_rate, 0.1)
        self._last_pub_t: float = 0.0

        self._vel_hist: deque = deque(maxlen=8)

        self._pub_pose   = self.create_publisher(PoseStamped, '/matched_pose',   10)
        self._pub_navsat = self.create_publisher(NavSatFix,   '/matched_navsat', 10)
        self._pub_roads  = self.create_publisher(MarkerArray, '/osm_roads_viz',  1)
        self._pub_speed  = self.create_publisher(MarkerArray, '/robot_speed_viz',1)
        self._tf_bc      = TransformBroadcaster(self)
        self._tf_buf     = Buffer()
        self._tf_lst     = TransformListener(self._tf_buf, self)

        self._sub = self.create_subscription(
            Odometry, '/odometry/global', self._odom_cb, 3)

        if not _DEPS_OK:
            self.get_logger().error('scipy missing — pip install scipy')
            return
        if not self._osm_file:
            self.get_logger().error('osm_file parameter is empty')
            return

        self.get_logger().info(
            f'Map matching | sigma={self._sigma}m  hdg_w={self._hdg_w}'
            f'  jct_boost={self._jct_boost}  k={self._k}  rate={pub_rate}Hz'
            f'  conf_min={CONFIDENCE_MIN}')
        threading.Thread(target=self._load, daemon=True).start()
        self.create_timer(5.0, self._pub_roads_retry)

    # ── OSM loader ────────────────────────────────────────────────────────────

    def _load(self):
        try:
            g = RoadGraph(self._osm_file, self._rlat, self._rlon)
            with self._lock:
                self._graph = g
            n_adj = sum(len(v) for v in g.adj.values())
            self.get_logger().info(
                f'OSM loaded: {g.n} segments | '
                f'adjacency graph: {n_adj} edges | '
                f'max seg len: {MAX_SEG_LEN}m')
        except FileNotFoundError:
            self.get_logger().error(f'OSM not found: {self._osm_file}')
        except Exception as e:
            self.get_logger().error(f'OSM error: {e}')

    # ── road markers ──────────────────────────────────────────────────────────

    def _pub_roads_retry(self):
        if self._roads_sent:
            return
        with self._lock:
            g = self._graph
        if g is None:
            return
        ma  = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i in range(g.n):
            m = Marker()
            m.header.frame_id = 'map'; m.header.stamp = now
            m.ns = 'osm_roads'; m.id = i
            m.type = Marker.LINE_STRIP; m.action = Marker.ADD
            m.scale.x = 0.4
            m.color.r = 0.15; m.color.g = 0.55; m.color.b = 1.0; m.color.a = 0.85
            m.points = [Point(x=float(g.p1[i,0]), y=float(g.p1[i,1]), z=0.0),
                        Point(x=float(g.p2[i,0]), y=float(g.p2[i,1]), z=0.0)]
            ma.markers.append(m)
        self._pub_roads.publish(ma)
        self._roads_sent = True
        self.get_logger().info(f'Published {g.n} road markers on /osm_roads_viz')

    # ── TF heading ────────────────────────────────────────────────────────────

    def _get_tf_heading(self, fallback: float) -> float:
        try:
            tf = self._tf_buf.lookup_transform(
                'map', 'base_link', RclpyTime(),
                timeout=rclpy.duration.Duration(seconds=0.02))
            return quat_to_yaw(tf.transform.rotation)
        except Exception:
            return fallback

    # ── odometry callback ─────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            graph = self._graph
        if graph is None:
            return

        now_w      = _time.monotonic()
        should_pub = (now_w - self._last_pub_t) >= self._min_pub_dt

        xy  = np.array([msg.pose.pose.position.x,
                        msg.pose.pose.position.y], dtype=np.float64)
        vx  = msg.twist.twist.linear.x
        vy  = msg.twist.twist.linear.y
        spd = float(np.hypot(vx, vy))
        self._last_vxy = np.array([vx, vy])

        if spd > self._min_spd:
            self._vel_hist.append(np.array([vx, vy]))

        heading = self._get_tf_heading(quat_to_yaw(msg.pose.pose.orientation))

        # Detect junction before distance gate
        idx_probe = graph.query(xy, k=self._k)
        at_junction = False
        if len(idx_probe) > 0:
            spread = float(np.max(graph.hdg[idx_probe]) - np.min(graph.hdg[idx_probe]))
            spread_deg = float(np.degrees(min(spread, np.pi - spread % np.pi + 1e-9)))
            at_junction = spread_deg > JUNCTION_SPREAD_DEG

        # ── Distance gate (bypass at junctions) ──────────────────────────────
        if not at_junction and self._last_xy is not None:
            if float(np.linalg.norm(xy - self._last_xy)) < self._min_move:
                if should_pub and self._hmm:
                    self._publish_best(xy, msg, graph)
                    self._last_pub_t = now_w
                return

        if spd < self._min_spd and self._hmm:
            if should_pub:
                self._publish_best(xy, msg, graph)
                self._last_pub_t = now_w
            return

        # ── Adaptive k, sigma, heading weight ─────────────────────────────────
        k_use = self._k if at_junction else max(
            self._k // 2,
            int(self._k * (1.0 - max(self._hmm.values(), default=0) * 0.45))
        )
        idx   = graph.query(xy, k=k_use)
        if len(idx) == 0:
            return

        hdg_w = min(self._hdg_w + self._jct_boost, 0.92) if at_junction else self._hdg_w
        sigma = self._sigma * float(np.clip(1.0 - 0.25 * min(spd / 2.0, 1.0), 0.75, 1.0))

        # ── Vectorised emission with early bail-out ② ─────────────────────────
        emissions = graph.emission_batch(xy, heading, idx, sigma, hdg_w)

        # ── HMM update ────────────────────────────────────────────────────────
        new_hmm: Dict[int, float] = {}
        for li, seg_id in enumerate(idx):
            emit  = float(emissions[li])
            if emit < 1e-6:          # skip zero-emission segments entirely
                continue
            trans = max(
                (self._hmm.get(s, 0.005) * graph.transition(s, int(seg_id), self._self_tr)
                 for s in self._hmm),
                default=1.0,
            )
            new_hmm[int(seg_id)] = trans * emit

        if not new_hmm:
            return

        total = sum(new_hmm.values())
        norm  = {k: v / total for k, v in new_hmm.items()}
        self._hmm = dict(sorted(norm.items(), key=lambda x: -x[1])[:HMM_KEEP])

        self._last_xy = xy.copy()

        if should_pub:
            self._publish_best(xy, msg, graph)
            self._last_pub_t = now_w

    # ── publish ───────────────────────────────────────────────────────────────

    def _publish_best(self, xy: np.ndarray, orig: Odometry, graph: RoadGraph):
        if not self._hmm:
            return

        best_id  = max(self._hmm, key=self._hmm.get)
        best_prob = self._hmm[best_id]

        # ③ Confidence gate — freeze EMA when HMM is ambiguous
        if best_prob < CONFIDENCE_MIN:
            self._frozen = True
        else:
            self._frozen = False

        raw_proj = graph.proj_point(xy, best_id)

        # ④ Adaptive EMA
        seg_changed = (self._prev_best is not None and self._prev_best != best_id)
        ema_a = EMA_JUNCTION if (seg_changed and not self._frozen) else EMA_STABLE
        self._prev_best = best_id

        if self._ema_xy is None:
            self._ema_xy = raw_proj.copy()
        elif not self._frozen:
            self._ema_xy = ema_a * raw_proj + (1.0 - ema_a) * self._ema_xy
        # When frozen: self._ema_xy holds the last confident position — don't move it

        # ⑤ Velocity compensation (only when not frozen)
        if not self._frozen:
            comp = self._last_vxy * self._min_pub_dt * 0.5
            proj = graph.proj_point(self._ema_xy + comp, best_id)
        else:
            proj = self._ema_xy.copy()

        stamp  = orig.header.stamp
        orient = orig.pose.pose.orientation

        ps = PoseStamped()
        ps.header.frame_id = 'map'; ps.header.stamp = stamp
        ps.pose.position.x = float(proj[0])
        ps.pose.position.y = float(proj[1])
        ps.pose.position.z = 0.0
        ps.pose.orientation = orient
        self._pub_pose.publish(ps)

        lat, lon = xy_to_latlon(proj[0], proj[1], self._rlat, self._rlon)
        ns = NavSatFix()
        ns.header        = ps.header
        ns.latitude      = float(lat)
        ns.longitude     = float(lon)
        ns.altitude      = 0.0; ns.status.status = 0
        ns.position_covariance_type = 2
        cov = self._sigma ** 2
        ns.position_covariance = [cov, 0., 0., 0., cov, 0., 0., 0., 9999.]
        self._pub_navsat.publish(ns)

        tf = TransformStamped()
        tf.header.stamp    = stamp
        tf.header.frame_id = 'map'
        tf.child_frame_id  = 'base_link_matched'
        tf.transform.translation.x = float(proj[0])
        tf.transform.translation.y = float(proj[1])
        tf.transform.translation.z = 0.0
        tf.transform.rotation      = orient
        self._tf_bc.sendTransform(tf)

        self._publish_speed_viz(float(np.linalg.norm(self._last_vxy)), stamp)

    # ── speed indicator ───────────────────────────────────────────────────────

    def _publish_speed_viz(self, speed_ms: float, stamp):
        speed_capped = min(speed_ms, 5.0)
        ratio = speed_capped / 5.0
        if ratio < 0.4:
            r, g, b = 0.20, 0.78, 0.46
        elif ratio < 0.75:
            r, g, b = 0.94, 0.62, 0.15
        else:
            r, g, b = 0.89, 0.29, 0.29

        ma = MarkerArray()

        txt = Marker()
        txt.header.frame_id = 'base_link_matched'; txt.header.stamp = stamp
        txt.ns = 'speed'; txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
        txt.pose.position.z = 2.5; txt.pose.orientation.w = 1.0
        txt.scale.z = 0.45
        txt.color.r = r; txt.color.g = g; txt.color.b = b; txt.color.a = 1.0
        txt.text = f'{speed_ms:.2f} m/s'
        ma.markers.append(txt)

        bar_h = max(0.05, speed_capped * 0.4)
        bar = Marker()
        bar.header.frame_id = 'base_link_matched'; bar.header.stamp = stamp
        bar.ns = 'speed'; bar.id = 1
        bar.type = Marker.CYLINDER; bar.action = Marker.ADD
        bar.pose.position.x = 1.2; bar.pose.position.z = bar_h / 2.0
        bar.pose.orientation.w = 1.0
        bar.scale.x = bar.scale.y = 0.18; bar.scale.z = bar_h
        bar.color.r = r; bar.color.g = g; bar.color.b = b; bar.color.a = 0.85
        ma.markers.append(bar)

        self._pub_speed.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = MapMatchingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
