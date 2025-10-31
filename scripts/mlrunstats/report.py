# mlrunstats/report.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional


def assemble_summary_dict(
    *,
    cfg: Any,                    # dataclass (Config)
    label_a: str,
    vals_a_path: Path,
    sum_a: Any,                  # dataclass (Summary)
    paper: Optional[float] = None,
    p_one_sample: Optional[float] = None,
    label_b: Optional[str] = None,
    vals_b_path: Optional[Path] = None,
    sum_b: Optional[Any] = None,  # dataclass (Summary)
    p_two_sample: Optional[float] = None,
    effects: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable dictionary capturing config, summaries, and tests.
    """
    summary: Dict[str, Any] = {
        "config": asdict(cfg) | {"label_a": label_a, "label_b": label_b},
        "A": {
            "label": label_a,
            "values_file": str(vals_a_path),
            "summary": asdict(sum_a),
        },
        "paper": paper,
    }
    if p_one_sample is not None:
        summary["A_vs_paper"] = {"p_value_greater": float(p_one_sample)}

    if sum_b is not None and label_b is not None and vals_b_path is not None:
        summary["B"] = {
            "label": label_b,
            "values_file": str(vals_b_path),
            "summary": asdict(sum_b),
        }
        summary["A_vs_B"] = {
            "p_value_greater": float(p_two_sample) if p_two_sample is not None else None,
            "effects": effects if effects is not None else None,
        }
    return summary


def write_summary_json(summary_dict: Dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(summary_dict, indent=2))


def render_summary_md(
    *,
    metric_col: str,
    label_a: str,
    sum_a: Any,                       # Summary dataclass
    paper: Optional[float] = None,
    p_one_sample: Optional[float] = None,
    label_b: Optional[str] = None,
    sum_b: Optional[Any] = None,      # Summary dataclass
    p_two_sample: Optional[float] = None,
    effects: Optional[Dict[str, float]] = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Experiment Analysis ({metric_col})\n")

    # A
    lines.append(f"## {label_a}")
    lines.append(f"- n={sum_a.n}, mean={sum_a.mean:.4f}, std={sum_a.std:.4f}, CV={sum_a.cv:.4f}")
    lines.append(f"- 95% CI: [{sum_a.ci_low:.4f}, {sum_a.ci_high:.4f}]")
    if paper is not None and p_one_sample is not None:
        lines.append(f"- vs Paper ({paper:.4f}) — p (greater) = {p_one_sample:.4g}")

    # B (+ compare)
    if label_b is not None and sum_b is not None:
        lines.append(f"\n## {label_b}")
        lines.append(f"- n={sum_b.n}, mean={sum_b.mean:.4f}, std={sum_b.std:.4f}, CV={sum_b.cv:.4f}")
        lines.append(f"- 95% CI: [{sum_b.ci_low:.4f}, {sum_b.ci_high:.4f}]")
        lines.append(f"\n## {label_a} vs {label_b}")
        if p_two_sample is not None:
            lines.append(f"- p (greater) = {p_two_sample:.4g}")
        if effects:
            if "cohens_d" in effects:
                lines.append(f"- Cohen's d = {effects['cohens_d']:.4f}")
            if "hedges_g" in effects:
                lines.append(f"- Hedges' g = {effects['hedges_g']:.4f}")
            if effects.get("cliffs_delta") is not None:
                lines.append(f"- Cliff's delta = {effects['cliffs_delta']:.4f}")

    return "\n".join(lines)


def write_summary_md(md_text: str, out_path: Path) -> None:
    out_path.write_text(md_text)
