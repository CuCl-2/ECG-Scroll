"""ECG-Scroll verifiers: objective, rule-based scoring for each task.

All verifiers return a float score in [0,1]. No model-as-judge.
"""
from typing import List, Sequence, Tuple

Interval = Sequence[float]


def _iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def temporal_f1(pred: List[Interval], gt: List[Interval], tau: float = 0.5) -> float:
    """Greedy IoU matching between predicted and ground-truth intervals -> F1."""
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    matched_gt = set()
    tp = 0
    # greedy: for each pred, take best unused gt with IoU>=tau
    for p in pred:
        best_j, best_iou = -1, tau
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            v = _iou(p, g)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched_gt.add(best_j)
            tp += 1
    prec = tp / len(pred)
    rec = tp / len(gt)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def rel_error(pred_value: float, gt_value: float, floor: float = 0.02) -> float:
    """Score = max(0, 1 - |p-g| / max(g, floor))."""
    denom = max(abs(gt_value), floor)
    return max(0.0, 1.0 - abs(pred_value - gt_value) / denom)


def cp_tolerance(pred_t: float, gt_t: float, delta: float = 10.0) -> float:
    """1 if the predicted change-point is within delta seconds of ground truth."""
    return 1.0 if abs(pred_t - gt_t) <= delta else 0.0


def hit_iou(pred: Interval, gt: Interval, tau: float = 0.5) -> float:
    """1 if the single predicted interval overlaps ground truth with IoU>=tau."""
    return 1.0 if _iou(pred, gt) >= tau else 0.0


def hit_tolerance(pred: Interval, gt: Interval, delta: float = 15.0) -> float:
    """1 if the predicted interval localizes a point event within `delta` seconds.

    For a near-instantaneous target (e.g. a single PVC), IoU against a ~1 s ground-truth
    interval is unreachable under a 30 s streaming protocol: even a perfect chunk-level hit
    scores IoU~=1/30. We instead score temporal localization directly: the midpoint of the
    predicted interval must fall within `delta` s of the ground-truth event time (its
    midpoint). With delta = half a chunk (15 s), a 1 means the agent pointed at the correct
    chunk; a padded/whole-record guess misses because its midpoint drifts away. Mirrors the
    tolerance semantics of `cp_tolerance` for change detection."""
    pm = 0.5 * (pred[0] + pred[1])
    gm = 0.5 * (gt[0] + gt[1])
    return 1.0 if abs(pm - gm) <= delta else 0.0


def detection_latency(events: List[Interval], writes: List[dict], tau: float = 0.5):
    """Streaming detection latency (rule-based, no model judge).

    For each ground-truth event [a,b], find the earliest ledger write whose interval matches
    it (IoU>=tau) and whose recorded time t_recorded>=a; latency = t_recorded - a. Events with
    no matching write are `missed`. `writes` are ledger entries carrying `interval` and
    `t_recorded` (stamped by the streaming env). Returns a summary dict."""
    ws = [w for w in writes if "interval" in w and "t_recorded" in w]
    lats, missed = [], 0
    for ev in events:
        hit_t = None
        for w in ws:
            if _iou(w["interval"], ev) >= tau and w["t_recorded"] >= ev[0]:
                t = w["t_recorded"]
                hit_t = t if hit_t is None else min(hit_t, t)
        if hit_t is None:
            missed += 1
        else:
            lats.append(max(0.0, hit_t - ev[0]))
    return {"n_events": len(events), "detected": len(lats), "missed": missed,
            "mean_latency_s": (sum(lats) / len(lats)) if lats else None}


def ischemia_composite(pred: dict, gt: dict, tau: float = 0.5,
                       eps_mv: float = 0.1) -> float:
    """Composite for ischemia localization: mean of lead-match, interval IoU>=tau,
    and ST-magnitude within eps_mv. Expects keys: lead, interval, st_mv."""
    lead_ok = float(pred.get("lead") == gt.get("lead"))
    iou_ok = hit_iou(pred["interval"], gt["interval"], tau)
    st_ok = float(abs(pred.get("st_mv", 0.0) - gt.get("st_mv", 0.0)) <= eps_mv)
    return (lead_ok + iou_ok + st_ok) / 3.0


VERIFIERS = {
    "temporal_f1": lambda pred, gt, p: temporal_f1(pred["intervals"], gt["intervals"],
                                                   p.get("tau", 0.5)),
    "rel_error": lambda pred, gt, p: rel_error(pred["value"], gt["value"],
                                               p.get("floor", 0.02)),
    "cp_tolerance": lambda pred, gt, p: cp_tolerance(pred["change_point"],
                                                     gt["change_point"], p.get("delta", 10.0)),
    "hit_iou": lambda pred, gt, p: hit_iou(pred["interval"], gt["interval"], p.get("tau", 0.5)),
    "hit_tolerance": lambda pred, gt, p: hit_tolerance(pred["interval"], gt["interval"],
                                                       p.get("delta", 15.0)),
    "ischemia": lambda pred, gt, p: ischemia_composite(pred, gt, p.get("tau", 0.5),
                                                       p.get("eps_mv", 0.1)),
}


def score(verifier: dict, pred: dict, gt: dict) -> float:
    """Dispatch: verifier={'type':..,'params':..}; pred/gt are task 'answer' dicts."""
    fn = VERIFIERS[verifier["type"]]
    return float(fn(pred, gt, verifier.get("params", {})))


if __name__ == "__main__":
    ok = 0; tot = 0
    def check(name, got, exp, tol=1e-6):
        global ok, tot; tot += 1
        p = abs(got - exp) <= tol
        ok += p; print(f"  [{'PASS' if p else 'FAIL'}] {name}: got={got:.3f} exp={exp:.3f}")

    # temporal_f1
    check("f1 perfect", temporal_f1([[0,10]], [[0,10]]), 1.0)
    check("f1 empty-empty", temporal_f1([], []), 1.0)
    check("f1 miss", temporal_f1([[0,1]], [[100,110]]), 0.0)
    check("f1 half-precision", temporal_f1([[0,10],[200,201]], [[0,10]]), 2*0.5*1/1.5)
    check("f1 iou-below-tau", temporal_f1([[0,10]], [[5,15]]), 0.0)  # IoU=5/15=0.33<0.5
    # rel_error
    check("rel exact", rel_error(0.2, 0.2), 1.0)
    check("rel off", rel_error(0.25, 0.20), 1 - 0.05/0.20)
    check("rel floor", rel_error(0.01, 0.0, 0.02), 1 - 0.01/0.02)
    check("rel clamp", rel_error(1.0, 0.1), 0.0)
    # cp_tolerance
    check("cp in", cp_tolerance(105, 100, 10), 1.0)
    check("cp out", cp_tolerance(120, 100, 10), 0.0)
    # hit_iou
    check("hit yes", hit_iou([0,10],[0,10]), 1.0)
    check("hit no", hit_iou([0,10],[8,20]), 0.0)
    # hit_tolerance (point-event localization; delta = half a 30s chunk)
    check("tol chunk-hit", hit_tolerance([900, 930], [914.5, 915.5]), 1.0)   # mid 915 vs 915
    check("tol edge-in", hit_tolerance([900, 930], [929.5, 930.5]), 1.0)     # mid 915 vs 930, ==15
    check("tol miss", hit_tolerance([900, 930], [959.5, 960.5]), 0.0)        # mid 915 vs 960, >15
    check("tol wide-guess", hit_tolerance([0, 1806], [1400.5, 1401.5]), 0.0) # mid 903 vs 1401, >15
    # detection_latency
    _ev = [[100, 200], [400, 450]]
    _w = [{"interval": [100, 205], "t_recorded": 130},   # matches ev0, latency 30
          {"interval": [398, 452], "t_recorded": 470}]   # matches ev1, latency 70
    _r = detection_latency(_ev, _w)
    check("lat detected", float(_r["detected"]), 2.0)
    check("lat missed", float(_r["missed"]), 0.0)
    check("lat mean", _r["mean_latency_s"], 50.0)  # (30+70)/2
    _r2 = detection_latency([[0, 10]], [{"interval": [500, 510], "t_recorded": 505}])
    check("lat all-missed", float(_r2["missed"]), 1.0)
    # dispatch
    check("dispatch f1", score({"type":"temporal_f1","params":{"tau":0.5}},
                               {"intervals":[[0,10]]}, {"intervals":[[0,10]]}), 1.0)
    print(f"\n{ok}/{tot} passed")
