import json
import math
import numpy as np

def load_cell_info(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def compute_cell_scores(Q_l, cell_info):
    scores = []

    for cell in cell_info:
        num = 0.0
        den = 0.0

        for edge in cell["edges"]:
            x, y = edge["midpoint_wall"]
            qx, qy = Q_l(x, y)
            qmag = math.sqrt(qx*qx + qy*qy)

            x1, y1 = edge["p1_out"]
            x2, y2 = edge["p2_out"]
            le = math.sqrt((x2-x1)**2 + (y2-y1)**2)

            num += le * qmag
            den += le

        score = num / den if den > 0 else 0.0

        scores.append({
            "cell_id": cell["cell_id"],
            "center_x": cell["center"][0],
            "center_y": cell["center"][1],
            "score": score
        })

    return scores

def summarize_cell_scores(cell_scores, threshold_ratio=0.2):
    vals = np.array([c["score"] for c in cell_scores], dtype=float)

    mean_val = float(vals.mean()) if len(vals) else 0.0
    std_val  = float(vals.std()) if len(vals) else 0.0
    cv_val   = float(std_val / mean_val) if mean_val > 0 else 0.0

    thr = threshold_ratio * mean_val
    dead_frac = float(np.mean(vals < thr)) if len(vals) else 0.0

    return {
        "mean_score": mean_val,
        "std_score": std_val,
        "cv_score": cv_val,
        "dead_threshold": thr,
        "dead_fraction": dead_frac,
    }