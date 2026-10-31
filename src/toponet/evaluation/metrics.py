"""Exposure-weighted metrics and the frozen development selection rule."""


def aggregate(rows):
    exposure = sum(r["elapsed_s"] for r in rows)
    if exposure <= 0:
        raise ValueError("Positive exposure required")
    stays = sum(r["completed_stays"] for r in rows)
    short = sum(r["short_stays"] for r in rows)
    return dict(
        elapsed_s=exposure,
        throughput_mbps=sum(r["user_delivered_mbit"] for r in rows) / exposure,
        handovers=sum(r["handovers"] for r in rows),
        handovers_per_hour=sum(r["handovers"] for r in rows) * 3600 / exposure,
        ping_pongs=sum(r["ping_pongs"] for r in rows),
        weak_fraction=sum(r["weak_signal_duration_s"] for r in rows) / exposure,
        short_stays=short,
        completed_stays=stays,
        short_stay_percent=100 * short / stays if stays else None,
    )


def assessment(metrics, baseline):
    margins = dict(
        throughput_mbps=metrics["throughput_mbps"] - baseline["throughput_mbps"],
        handovers=baseline["handovers"] - metrics["handovers"],
        ping_pongs=baseline["ping_pongs"] - metrics["ping_pongs"],
        weak_s=baseline["weak_fraction"] * baseline["elapsed_s"]
        - metrics["weak_fraction"] * metrics["elapsed_s"],
    )
    feasible = (
        margins["throughput_mbps"] > 1e-9
        and margins["handovers"] >= 0
        and margins["ping_pongs"] >= 0
        and margins["weak_s"] >= -1e-7
    )
    violation = max(0, -margins["throughput_mbps"]) / max(
        baseline["throughput_mbps"], 1e-9
    )
    for key, denom in [
        ("handovers", baseline["handovers"]),
        ("ping_pongs", baseline["ping_pongs"]),
        ("weak_s", baseline["weak_fraction"] * baseline["elapsed_s"]),
    ]:
        violation += max(0, -margins[key]) / max(denom, 1)
    return dict(feasible=feasible, normalized_violation=violation, margins=margins)


def selection_rank(record):
    m, a = record["metrics"], record["assessment"]
    return (
        (0, -m["throughput_mbps"], m["handovers"])
        if a["feasible"]
        else (1, a["normalized_violation"], -m["throughput_mbps"])
    )
