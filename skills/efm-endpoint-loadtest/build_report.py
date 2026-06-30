"""Aggregate the Locust per-run CSVs into an EFM latency comparison HTML report.

Reads results/locust/<endpoint>-u<N>_stats.csv (Locust --csv output), pulls the
Aggregated row, and writes a self-contained results/efm_latency_report.html with
latency (p50/p90/p99 ms) and throughput (req/s) across model x instance x
concurrency, plus a single-stream bar chart. Also prints a short markdown
summary to stdout.
"""

from __future__ import annotations

import csv
import glob
import html
import os
import re

import sys

HERE = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = os.path.join(HERE, "efm_latency_report.html")

MODEL = {"libtorch": "torchlib", "trt-bf16": "trt (bf16)", "trt-fp8": "trt (fp8-gemm)"}
MODEL_ORDER = ["torchlib", "trt (bf16)", "trt (fp8-gemm)"]
INSTANCES = [("g6", "g6 (L4)"), ("g6e", "g6e (L40S)")]
BAR_COLORS = {
    "torchlib": "#4e79a7",
    "trt (bf16)": "#59a14f",
    "trt (fp8-gemm)": "#e15759",
}


def _parse_name(path: str):
    base = os.path.basename(path)[: -len("_stats.csv")]
    m = re.match(r"efm-(libtorch|trt-bf16|trt-fp8)-(g6e?)-u(\d+)$", base)
    return (MODEL[m.group(1)], m.group(2), int(m.group(3))) if m else None


def _agg_row(path: str) -> dict:
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("Name") == "Aggregated":
                return row
    return {}


def load_rows() -> dict:
    rows = {}
    for path in glob.glob(os.path.join(HERE, "locust", "*_stats.csv")):
        key = _parse_name(path)
        r = _agg_row(path) if key else None
        if not r:
            continue
        rows[key] = {
            "reqs": int(r["Request Count"]),
            "fails": int(r["Failure Count"]),
            "p50": float(r["50%"]),
            "p90": float(r["90%"]),
            "p99": float(r["99%"]),
            "rps": float(r["Requests/s"]),
        }
    return rows


def _conc_table(rows: dict, u: int) -> str:
    body = []
    for model in MODEL_ORDER:
        for inst, label in INSTANCES:
            r = rows.get((model, inst, u))
            if not r:
                continue
            cls = ' class="fail"' if r["fails"] else ""
            body.append(
                f"<tr><td>{model}</td><td>{label}</td>"
                f"<td>{r['p50']:.0f}</td><td>{r['p90']:.0f}</td><td>{r['p99']:.0f}</td>"
                f"<td>{r['rps']:.1f}</td><td>{r['reqs']}</td><td{cls}>{r['fails']}</td></tr>"
            )
    return (
        f"<h2>Concurrency = {u} client{'s' if u > 1 else ''}</h2>"
        "<table><thead><tr><th>model</th><th>instance</th><th>p50</th><th>p90</th>"
        "<th>p99</th><th>req/s</th><th>reqs</th><th>fails</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _bar_chart(rows: dict) -> str:
    pts = []
    for model in MODEL_ORDER:
        for inst, label in INSTANCES:
            r = rows.get((model, inst, 1))
            if r:
                pts.append((f"{model} · {inst}", r["p50"], BAR_COLORS[model]))
    if not pts:
        return ""
    mx = max(v for _, v, _ in pts)
    bars = []
    for name, val, color in pts:
        w = max(1.0, val / mx * 100)
        bars.append(
            f'<div class="barrow"><div class="barlabel">{html.escape(name)}</div>'
            f'<div class="bartrack"><div class="bar" style="width:{w:.1f}%;background:{color}">'
            f"<span>{val:.0f} ms</span></div></div></div>"
        )
    return (
        "<h2>Single-stream latency floor (u=1, p50)</h2>"
        '<div class="chart">' + "".join(bars) + "</div>"
    )


def _speedup_table(rows: dict) -> str:
    body = []
    for model in MODEL_ORDER:
        g6, g6e = rows.get((model, "g6", 1)), rows.get((model, "g6e", 1))
        if not g6 or not g6e:
            continue
        spd = g6["p50"] / g6e["p50"] if g6e["p50"] else float("nan")
        body.append(
            f"<tr><td>{model}</td><td>{g6['p50']:.0f}</td>"
            f"<td>{g6e['p50']:.0f}</td><td>{spd:.2f}×</td></tr>"
        )
    if not body:
        return ""
    return (
        "<h2>g6 vs g6e (u=1, p50 ms)</h2><table><thead><tr><th>model</th>"
        "<th>g6 (L4)</th><th>g6e (L40S)</th><th>g6e speedup</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render(rows: dict) -> str:
    users = sorted({u for (_, _, u) in rows})
    css = """
    body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:2rem;color:#1c2733;max-width:900px}
    h1{font-size:1.6rem;margin-bottom:.2rem} h2{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid #e3e8ee;padding-bottom:.3rem}
    .meta{color:#5b6b7b;font-size:.9rem;margin-bottom:1rem}
    table{border-collapse:collapse;width:100%;margin:.5rem 0} th,td{padding:.4rem .7rem;text-align:right;border-bottom:1px solid #eef2f6}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
    thead th{background:#f6f8fa;font-weight:600} tbody tr:hover{background:#f9fbfd}
    td.fail{color:#e15759;font-weight:600}
    .chart{margin:.6rem 0} .barrow{display:flex;align-items:center;margin:.25rem 0}
    .barlabel{width:170px;font-size:.85rem;color:#3a4a5a} .bartrack{flex:1;background:#f0f3f6;border-radius:4px}
    .bar{height:22px;border-radius:4px;display:flex;align-items:center;justify-content:flex-end;min-width:40px}
    .bar span{color:#fff;font-size:.78rem;padding-right:6px;font-weight:600}
    footer{margin-top:2rem;color:#8895a3;font-size:.8rem}
    """
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EFM endpoint latency</title><style>",
        css,
        "</style></head><body>",
        "<h1>EFM embedding endpoint — latency load test</h1>",
        "<div class='meta'>3 serving variants × ml.g6 (L4) / ml.g6e (L40S), single GPU each. "
        "Load = real ~16k-token user sequences (parity sample, 12,877 segments). "
        "Driver = Locust (boto3 <code>invoke_endpoint</code>), closed-loop, 25 s/level. "
        "Latency in ms (server+network round-trip).</div>",
        _bar_chart(rows),
        _speedup_table(rows),
    ]
    parts += [_conc_table(rows, u) for u in users]
    parts.append(
        "<footer>Generated by results/build_report.py from Locust CSVs.</footer>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    rows = load_rows()
    with open(OUT, "w") as f:
        f.write(render(rows))
    print(f"wrote {OUT} ({len(rows)} runs)")
    # terse stdout summary
    for model in MODEL_ORDER:
        g6, g6e = rows.get((model, "g6", 1)), rows.get((model, "g6e", 1))
        if g6 and g6e:
            print(
                f"  u=1 p50  {model:16s}  g6={g6['p50']:.0f}ms  g6e={g6e['p50']:.0f}ms"
            )


if __name__ == "__main__":
    main()
