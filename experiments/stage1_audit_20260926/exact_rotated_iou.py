"""Exact float64 rotated-IoU for the VoD/KITTI evaluation protocol.

Why this module exists
----------------------
The rotated-IoU routine that ships with the View-of-Delft devkit
(``vod/evaluation/rotate_iou_cpu.py``) silently drops polygon vertices when a
corner of one box lies on the boundary of the other.  Concretely, comparing a
box with *itself* gives 1/3 instead of 1:

    box [0, 0, 2, 1, 0.5236] vs itself -> rotate_iou_cpu returns 0.3333

OpenPCDet's numba-CUDA kernel (``rotate_iou_gpu_eval``) is the same RRPN-derived
algorithm and shows the same behaviour.  A prediction set that contains a box
identical to the GT - which is exactly what a "score the GT against itself"
sanity test does - therefore collapses to ~0 AP under the shipped evaluators.

This module replaces only the IoU primitive.  It clips one convex quad by the
other with Sutherland-Hodgman in float64, so a box clipped by itself yields its
own area exactly, and boundary-coincident geometry is handled by the clip
predicate ``cross >= 0`` instead of a float32 parallelogram test.

Box layouts (identical to what the devkit feeds its own routine):

* BEV row: ``[x, y, w, h, angle]`` -- ``w`` is the extent along the box's local
  x axis, ``h`` along its local y axis, ``angle`` in radians.
* 3D row:  ``[x, y, z, l, h, w, ry]`` -- the layout built by
  ``kitti_official_evaluate.calculate_iou_partly``; the BEV footprint is
  ``(x, z, l, w, ry)`` and the vertical extent is ``[y - h, y]`` (camera-frame
  bottom-centre convention, y pointing down).

Run ``python exact_rotated_iou.py`` for the self-test.
"""

import math

import numba
import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------------------
# pure-python reference (used by the self-test, never in the hot path)
# ---------------------------------------------------------------------------

def _quad(x, y, w, h, angle):
    c, s = math.cos(angle), math.sin(angle)
    pts = ((-w / 2.0, -h / 2.0), (w / 2.0, -h / 2.0), (w / 2.0, h / 2.0), (-w / 2.0, h / 2.0))
    return [(c * px - s * py + x, s * px + c * py + y) for px, py in pts]


def _signed_area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _clip_polygon(subject, clipper):
    """Sutherland-Hodgman clip of convex ``subject`` by convex ``clipper``."""
    if _signed_area(clipper) < 0:
        clipper = clipper[::-1]
    out = list(subject)
    n = len(clipper)
    for i in range(n):
        ax, ay = clipper[i]
        bx, by = clipper[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            px, py = inp[j]
            qx, qy = inp[(j + 1) % len(inp)]
            dp = ex * (py - ay) - ey * (px - ax)
            dq = ex * (qy - ay) - ey * (qx - ax)
            if dp >= 0:
                out.append((px, py))
            if (dp > 0 and dq < 0) or (dp < 0 and dq > 0):
                t = dp / (dp - dq)
                out.append((px + t * (qx - px), py + t * (qy - py)))
    return out


def ref_bev_iou(a, b):
    """Reference float64 IoU of two BEV rows ``[x, y, w, h, angle]``."""
    pa = _clip_polygon(_quad(*a), _quad(*b))
    if len(pa) < 3:
        return 0.0
    inter = abs(_signed_area(pa))
    aa, ab = a[2] * a[3], b[2] * b[3]
    denom = aa + ab - inter
    return inter / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# numba kernels (hot path)
# ---------------------------------------------------------------------------

@numba.njit(cache=True, inline='always')
def _quad_nb(x, y, w, h, angle, out):
    c, s = math.cos(angle), math.sin(angle)
    hw, hh = w / 2.0, h / 2.0
    xs = (-hw, hw, hw, -hw)
    ys = (-hh, -hh, hh, hh)
    for i in range(4):
        out[2 * i] = c * xs[i] - s * ys[i] + x
        out[2 * i + 1] = s * xs[i] + c * ys[i] + y


@numba.njit(cache=True, inline='always')
def _signed_area_nb(poly, n):
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += poly[2 * i] * poly[2 * j + 1] - poly[2 * j] * poly[2 * i + 1]
    return s / 2.0


@numba.njit(cache=True)
def _clip_area_nb(subj, clipper):
    """Area of ``subj`` clipped by convex ``clipper``; both 8-float corner arrays."""
    # orientation: ensure the clipper is counter-clockwise
    if _signed_area_nb(clipper, 4) < 0.0:
        tmp = np.empty(8, dtype=numba.float64)
        for i in range(4):
            k = 3 - i
            tmp[2 * i] = clipper[2 * k]
            tmp[2 * i + 1] = clipper[2 * k + 1]
        clipper = tmp

    cur = np.empty(32, dtype=numba.float64)
    nxt = np.empty(32, dtype=numba.float64)
    for i in range(8):
        cur[i] = subj[i]
    n_cur = 4

    for e in range(4):
        if n_cur == 0:
            return 0.0
        ax = clipper[2 * e]
        ay = clipper[2 * e + 1]
        bx = clipper[2 * ((e + 1) % 4)]
        by = clipper[2 * ((e + 1) % 4) + 1]
        ex = bx - ax
        ey = by - ay
        n_nxt = 0
        for j in range(n_cur):
            jn = (j + 1) % n_cur
            px, py = cur[2 * j], cur[2 * j + 1]
            qx, qy = cur[2 * jn], cur[2 * jn + 1]
            dp = ex * (py - ay) - ey * (px - ax)
            dq = ex * (qy - ay) - ey * (qx - ax)
            if dp >= 0.0:
                nxt[2 * n_nxt] = px
                nxt[2 * n_nxt + 1] = py
                n_nxt += 1
            if (dp > 0.0 and dq < 0.0) or (dp < 0.0 and dq > 0.0):
                t = dp / (dp - dq)
                nxt[2 * n_nxt] = px + t * (qx - px)
                nxt[2 * n_nxt + 1] = py + t * (qy - py)
                n_nxt += 1
            if n_nxt > 15:
                return 0.0
        for i in range(2 * n_nxt):
            cur[i] = nxt[i]
        n_cur = n_nxt

    if n_cur < 3:
        return 0.0
    return abs(_signed_area_nb(cur, n_cur))


@numba.njit(cache=True, inline='always')
def _circumradius(w, h):
    return 0.5 * math.sqrt(w * w + h * h)


@numba.njit(cache=True, inline='always')
def _aabb(x, y, w, h, angle, out):
    c = abs(math.cos(angle))
    s = abs(math.sin(angle))
    ex = 0.5 * (w * c + h * s)
    ey = 0.5 * (w * s + h * c)
    out[0] = x - ex
    out[1] = y - ey
    out[2] = x + ex
    out[3] = y + ey


@numba.njit(cache=True)
def rotated_iou(boxes, query_boxes, criterion=-1):
    """Exact rotated IoU; rows are ``[x, y, w, h, angle]``.

    Mirrors ``rotate_iou_eval``: ``out[n, k] = f(boxes[n], query_boxes[k])``.
    ``criterion``: -1 IoU, 0 inter/area(box), 1 inter/area(query), 2 raw inter.
    """
    n_b = boxes.shape[0]
    n_q = query_boxes.shape[0]
    out = np.zeros((n_b, n_q), dtype=np.float64)
    qa = np.empty(8, dtype=np.float64)
    qb = np.empty(8, dtype=np.float64)
    ba = np.empty(4, dtype=np.float64)
    bb = np.empty(4, dtype=np.float64)

    for n in range(n_b):
        xa, ya, wa, ha, ra = boxes[n, 0], boxes[n, 1], boxes[n, 2], boxes[n, 3], boxes[n, 4]
        area_a = wa * ha
        _quad_nb(xa, ya, wa, ha, ra, qa)
        _aabb(xa, ya, wa, ha, ra, ba)
        r_a = _circumradius(wa, ha)
        for k in range(n_q):
            xb, yb, wb, hb, rb = (query_boxes[k, 0], query_boxes[k, 1], query_boxes[k, 2],
                                  query_boxes[k, 3], query_boxes[k, 4])
            if math.hypot(xa - xb, ya - yb) > r_a + _circumradius(wb, hb):
                continue
            _aabb(xb, yb, wb, hb, rb, bb)
            if ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]:
                continue
            _quad_nb(xb, yb, wb, hb, rb, qb)
            inter = _clip_area_nb(qa, qb)
            if criterion == 2:
                out[n, k] = inter
            elif criterion == -1:
                denom = area_a + wb * hb - inter
                out[n, k] = inter / denom if denom > EPS else 0.0
            elif criterion == 0:
                out[n, k] = inter / area_a if area_a > EPS else 0.0
            elif criterion == 1:
                out[n, k] = inter / (wb * hb) if wb * hb > EPS else 0.0
            else:
                out[n, k] = inter
    return out


@numba.njit(cache=True)
def d3_box_overlap(boxes, q_boxes, criterion=-1):
    """Exact stand-in for the devkit ``d3_box_overlap``.

    Rows are ``[x, y, z, l, h, w, ry]`` exactly as built by
    ``calculate_iou_partly`` for ``metric == 2``.  Returns the 3D IoU, or - when
    ``criterion == 2`` - the raw 3D intersection volume, matching the devkit's
    contract (``r_inc`` is fed straight into the height-overlap kernel).
    """
    n_b = boxes.shape[0]
    n_q = q_boxes.shape[0]
    out = np.zeros((n_b, n_q), dtype=np.float64)
    bev_a = np.empty((n_b, 5), dtype=np.float64)
    bev_b = np.empty((n_q, 5), dtype=np.float64)
    for i in range(n_b):
        bev_a[i, 0] = boxes[i, 0]
        bev_a[i, 1] = boxes[i, 2]
        bev_a[i, 2] = boxes[i, 3]
        bev_a[i, 3] = boxes[i, 5]
        bev_a[i, 4] = boxes[i, 6]
    for j in range(n_q):
        bev_b[j, 0] = q_boxes[j, 0]
        bev_b[j, 1] = q_boxes[j, 2]
        bev_b[j, 2] = q_boxes[j, 3]
        bev_b[j, 3] = q_boxes[j, 5]
        bev_b[j, 4] = q_boxes[j, 6]
    r_inc = rotated_iou(bev_a, bev_b, 2)
    for i in range(n_b):
        h_a = boxes[i, 4]
        y_a = boxes[i, 1]
        area_a = boxes[i, 3] * h_a * boxes[i, 5]
        for j in range(n_q):
            if r_inc[i, j] <= 0.0:
                continue
            h_b = q_boxes[j, 4]
            iw = min(y_a, q_boxes[j, 1]) - max(y_a - h_a, q_boxes[j, 1] - h_b)
            if iw <= 0.0:
                continue
            inc = iw * r_inc[i, j]
            area_b = q_boxes[j, 3] * h_b * q_boxes[j, 5]
            if criterion == -1:
                denom = area_a + area_b - inc
            elif criterion == 0:
                denom = area_a
            elif criterion == 1:
                denom = area_b
            else:
                denom = inc
            out[i, j] = inc / denom if denom > EPS else 0.0
    return out


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def self_test(verbose=True):
    """Analytic + cross-implementation checks. Returns the number of failures."""
    rng = np.random.default_rng(0)
    fails = []

    def check(name, got, want, tol=1e-6):
        ok = abs(got - want) <= tol
        if verbose:
            print('  %-52s got %8.5f want %8.5f  %s' % (name, got, want, 'ok' if ok else 'FAIL'))
        if not ok:
            fails.append(name)

    print('[1] identical boxes must give IoU 1 (the shipped CPU routine fails this)')
    for ang in (0.0, 0.5236, 1.0472, -1.5708, 3.0):
        row = np.array([[0.0, 0.0, 2.0, 1.0, ang]])
        got = rotated_iou(row, row, -1)[0, 0]
        check('self IoU, angle=%.4f' % ang, got, 1.0)

    print('[2] analytic overlap cases')
    a = np.array([[0.0, 0.0, 2.0, 2.0, 0.0]])
    b = np.array([[1.0, 0.0, 2.0, 2.0, 0.0]])
    check('half overlap along x', rotated_iou(a, b, -1)[0, 0], 2.0 / 6.0)
    b = np.array([[1.5, 0.0, 2.0, 2.0, 0.0]])
    check('quarter overlap along x', rotated_iou(a, b, -1)[0, 0], 0.5 / 3.5)
    b = np.array([[5.0, 0.0, 2.0, 2.0, 0.0]])
    check('disjoint', rotated_iou(a, b, -1)[0, 0], 0.0)
    # octagon intersection of a 2x2 square and its 45 deg rotation: area = 4 - 2*(2-sqrt2)^2
    b = np.array([[0.0, 0.0, 2.0, 2.0, math.pi / 4.0]])
    oct_area = 4.0 - 2.0 * (2.0 - math.sqrt(2.0)) ** 2
    check('45 deg square on square', rotated_iou(a, b, -1)[0, 0],
          oct_area / (8.0 - oct_area), 1e-9)
    check('45 deg square on square == 1/sqrt(2)', rotated_iou(a, b, -1)[0, 0],
          1.0 / math.sqrt(2.0), 1e-9)

    print('[3] symmetric and self-consistent')
    box_a = np.array([[3.0, 7.0, 4.26, 1.97, -1.2]])
    box_b = np.array([[3.4, 7.2, 4.10, 1.90, -1.05]])
    i_ab = rotated_iou(box_a, box_b, -1)[0, 0]
    i_ba = rotated_iou(box_b, box_a, -1)[0, 0]
    check('symmetry', i_ab, i_ba)
    inter = rotated_iou(box_a, box_b, 2)[0, 0]
    check('inter consistency', inter / (4.26 * 1.97 + 4.10 * 1.90 - inter), i_ab)

    print('[4] numba kernel vs pure-python reference on 400 random pairs')
    worst = 0.0
    for _ in range(400):
        r1 = [rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(0.3, 4.5),
              rng.uniform(0.3, 2.5), rng.uniform(-math.pi, math.pi)]
        r2 = [rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(0.3, 4.5),
              rng.uniform(0.3, 2.5), rng.uniform(-math.pi, math.pi)]
        got = rotated_iou(np.array([r1]), np.array([r2]), -1)[0, 0]
        want = ref_bev_iou(r1, r2)
        worst = max(worst, abs(got - want))
    check('max |numba - reference|', worst, 0.0, 1e-9)

    print('[5] devkit CPU routine vs exact, on random *close* pairs (quantifies the shipped bug)')
    import sys
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, '_devkit_stubs'))
    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    sys.path.insert(0, os.path.join(project, '参考项目', 'SGDet3D-main', 'tools_det3d',
                                    'view-of-delft-dataset'))
    from vod.evaluation.rotate_iou_cpu import rotate_iou_eval
    n_bad = 0
    worst_gap = 0.0
    for _ in range(300):
        r1 = [rng.uniform(-20, 20), rng.uniform(-20, 20), 4.2, 1.9, rng.uniform(-math.pi, math.pi)]
        r2 = [r1[0] + rng.normal(0, 0.05), r1[1] + rng.normal(0, 0.05), 4.2, 1.9,
              r1[4] + rng.normal(0, 0.02)]
        shipped = float(rotate_iou_eval(np.array([r1]), np.array([r2]), -1)[0, 0])
        exact = rotated_iou(np.array([r1]), np.array([r2]), -1)[0, 0]
        gap = abs(shipped - exact)
        worst_gap = max(worst_gap, gap)
        if gap > 0.01:
            n_bad += 1
    if verbose:
        print('  pairs with |shipped-exact| > 0.01: %d/300, worst gap %.4f' % (n_bad, worst_gap))

    print('\n%s (%d failure(s))' % ('SELF-TEST PASSED' if not fails else 'SELF-TEST FAILED', len(fails)))
    return len(fails)


if __name__ == '__main__':
    import sys
    sys.exit(1 if self_test() else 0)
