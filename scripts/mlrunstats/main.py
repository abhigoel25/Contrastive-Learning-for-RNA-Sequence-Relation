# mlrunstats/cli.py  (updated)
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List
import yaml
import data, stats
import viz
import report
import time

trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")



def resolve_exp_dir(name_or_path: str, results_root: str) -> str:
    """
    Accepts either:
      - a bare experiment name like 'exprmnt_YYYY_MM_DD__hh_mm_ss', or
      - a path to .../results/<exp>/weights/checkpoints, or
      - a path to .../results/<exp>

    Returns a path to .../results/<exp>/weights/checkpoints.
    """
    p = Path(name_or_path).expanduser().resolve()
    # Already the checkpoints dir?
    if p.name == "checkpoints" and p.parent.name == "weights":
        return str(p)
    # If points to the experiment root, append weights/checkpoints
    if p.exists() and p.is_dir():
        maybe = p / "weights" / "checkpoints"
        if maybe.exists():
            return str(maybe)
    # Otherwise treat as experiment name under results_root
    return str(Path(results_root).expanduser().resolve() / name_or_path / "weights" / "checkpoints")


def smart_label_for_exp_dir(exp_dir: str) -> str:
    """
    Prefer the experiment folder name (…/results/<exp>/weights/checkpoints → '<exp>').
    Falls back to basename if structure is different.
    """
    p = Path(exp_dir)
    try:
        if p.name == "checkpoints" and p.parent.name == "weights":
            return p.parent.parent.name  # the '<exp>' segment
    except Exception:
        pass
    return p.name


@dataclass
class Config:
    exp_a: str
    running_platform: str
    label_a: str | None = None
    exp_b: Optional[str] = None
    label_b: Optional[str] = None
    paper: Optional[float] = None

    runs_glob: str = "run_*"
    spearman_file: str = "tsplice_spearman_by_tissue.tsv"

    metric_col: str = "spearman_psi"
    count_col: Optional[str] = None
    weighted: bool = False

    out_dir: str = "analysis_out"
    no_plots: bool = False

    alpha: float = 0.05
    n_boot: int = 10000
    n_perm: int = 10000
    seed: int = 42
    tissue: Optional[str] = None   


def run_analysis(cfg: Config) -> None:
    random.Random(cfg.seed)

    # timestamp suffix
    trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")

    # stats dir right under the experiment, with timestamp
    out = Path("/".join(cfg.exp_a.split("/")[:-2] + [f"stats{trimester}"]))
    out.mkdir(parents=True, exist_ok=True)

    # labels
    label_a = cfg.label_a or Path(cfg.exp_a).name
    label_b = cfg.label_b or (Path(cfg.exp_b).name if cfg.exp_b else None)

    # 1) collect per-run scalars (per-tissue if cfg.tissue is set)
    vals_a = data.collect_values(
        exp_dir=cfg.exp_a, runs_glob=cfg.runs_glob, spearman_file=cfg.spearman_file,
        metric_col=cfg.metric_col, count_col=cfg.count_col, weighted=cfg.weighted,
        tissue=cfg.tissue,
    )
    vals_a_path = out / f"{label_a}_values{trimester}.tsv"
    vals_a_path.write_text("\n".join(map(str, vals_a)))

    vals_b = None
    vals_b_path = None
    if cfg.exp_b:
        vals_b = data.collect_values(
            exp_dir=cfg.exp_b, runs_glob=cfg.runs_glob, spearman_file=cfg.spearman_file,
            metric_col=cfg.metric_col, count_col=cfg.count_col, weighted=cfg.weighted,
            tissue=cfg.tissue,
        )
        vals_b_path = out / f"{label_b}_values{trimester}.tsv"
        vals_b_path.write_text("\n".join(map(str, vals_b)))

    # 2) summaries
    sum_a = stats.summarize(vals_a, alpha=cfg.alpha, n_boot=cfg.n_boot)
    sum_b = stats.summarize(vals_b, alpha=cfg.alpha, n_boot=cfg.n_boot) if vals_b is not None else None

    # 3) tests
    p_one_sample = stats.one_sample_greater(vals_a, cfg.paper, n_boot=cfg.n_boot) if cfg.paper is not None else None
    p_two_sample = stats.two_sample_greater(vals_a, vals_b, n_perm=cfg.n_perm) if vals_b is not None else None
    effects     = stats.effect_sizes(vals_a, vals_b) if vals_b is not None else None

    # 4) reports (JSON + Markdown) with timestamped filenames
    summary_dict = report.assemble_summary_dict(
        cfg=cfg,
        label_a=label_a, vals_a_path=vals_a_path, sum_a=sum_a,
        paper=cfg.paper, p_one_sample=p_one_sample,
        label_b=label_b, vals_b_path=vals_b_path, sum_b=sum_b,
        p_two_sample=p_two_sample, effects=effects,
    )
    report.write_summary_json(summary_dict, out / f"summary{trimester}.json")

    metric_title = f"{cfg.metric_col}" + (f" @ {cfg.tissue}" if cfg.tissue else "")
    md_text = report.render_summary_md(
        metric_col=metric_title,
        label_a=label_a, sum_a=sum_a,
        paper=cfg.paper, p_one_sample=p_one_sample,
        label_b=label_b if vals_b is not None else None,
        sum_b=sum_b if vals_b is not None else None,
        p_two_sample=p_two_sample if vals_b is not None else None,
        effects=effects if vals_b is not None else None,
    )
    report.write_summary_md(md_text, out / f"summary{trimester}.md")

    # 5) plots (timestamped filenames)
    try:
        viz.hist(vals_a, f"{label_a} ({metric_title})", str(out / f"{label_a}_hist{trimester}.png"))
        if vals_b is not None:
            viz.hist(vals_b, f"{label_b} ({metric_title})", str(out / f"{label_b}_hist{trimester}.png"))
            viz.box(vals_a, vals_b, [label_a, label_b], f"Boxplot: {metric_title}", str(out / f"boxplot{trimester}.png"))
            viz.violin(vals_a, vals_b, [label_a, label_b], f"Violin: {metric_title}", str(out / f"violin{trimester}.png"))
    except Exception:
        pass
    

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze many ML runs and test significance vs paper and/or another experiment.")

    # New: optional config file (YAML or JSON)
    # '/gpfs/commons/home/atalukder/Contrastive_Learning/code/ML_model/scripts/mlrunstats/config.yaml'
    p.add_argument("--config", default='/mnt/home/at3836/Contrastive_Learning/code/ML_model/scripts/mlrunstats/config.yaml', 
                   help="Path to YAML or JSON config (optional)")

    p.add_argument("--exp-a", help="Path to experiment A (folder containing run_* subfolders)")
    p.add_argument("--label-a", default=None, help="Label for experiment A (default: basename of --exp-a)")

    p.add_argument("--exp-b", default=None, help="Path to experiment B (optional)")
    p.add_argument("--label-b", default=None, help="Label for experiment B (default: basename of --exp-b)")

    p.add_argument("--paper", type=float, default=None, help="Paper's reported single-number baseline (optional)")

    p.add_argument("--runs-glob", default="run_*", help="Glob for runs inside an experiment directory (default: run_*)")
    p.add_argument("--spearman-file", default="tsplice_spearman_by_tissue.tsv", help="Per-run TSV filename")

    p.add_argument("--metric-col", default="spearman_psi", help="Metric column to aggregate (e.g., spearman_psi or spearman_delta)")
    p.add_argument("--count-col", default=None, help="Optional count column to use as weights (e.g., n_valid_psi)")
    p.add_argument("--weighted", action="store_true", help="If set, use weighted mean with --count-col")

    p.add_argument("--out", dest="out_dir", default="analysis_out", help="Output directory (default: analysis_out)")
    p.add_argument("--no-plots", action="store_true", help="Disable plotting even if viz is available")

    p.add_argument("--alpha", type=float, default=0.05, help="Alpha for confidence intervals (default: 0.05)")
    p.add_argument("--n-boot", type=int, default=10000, help="Bootstrap iterations for CI/p-values (default: 10000)")
    p.add_argument("--n-perm", type=int, default=10000, help="Permutation iterations for two-sample test (default: 10000)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument("--tissue", default='Retina - Eye',
               help="Analyze a single tissue (case/space-insensitive, e.g., 'Adrenal Gland').")
    p.add_argument("--running_platform", default='NYGC',
               help="Running in NYGC or EMPRAI")


    return p


def _load_config_file(path: Path) -> dict:
    text = path.read_text()
    # Try YAML first if available; fall back to JSON
    # if _YAML_AVAILABLE:
    #     try:
    #         return yaml.safe_load(text) or {}
    #     except Exception:
    #         pass
    # # JSON fallback
    # return json.loads(text)

    try:
        return yaml.safe_load(text) or {}
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    cfg_dict: dict = {}

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"--config file not found: {cfg_path}")
        cfg_dict = _load_config_file(cfg_path)    

    if cfg_dict.get("running_platform") == 'NYGC':
        DEFAULT_RESULTS_ROOT = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/results"
    elif cfg_dict.get("running_platform") == 'EMPRAI':
        DEFAULT_RESULTS_ROOT = "/mnt/home/at3836//Contrastive_Learning/files/results"

    results_root = cfg_dict.get("results_root", DEFAULT_RESULTS_ROOT)
    if "exp_a" not in cfg_dict or not cfg_dict["exp_a"]:
        raise ValueError("Missing required parameter: exp_a")

    cfg_dict["exp_a"] = resolve_exp_dir(cfg_dict["exp_a"], results_root)
    if cfg_dict.get("exp_b"):
        cfg_dict["exp_b"] = resolve_exp_dir(cfg_dict["exp_b"], results_root)

    # Defaults for labels if not provided (use experiment folder name)
    if not cfg_dict.get("label_a"):
        cfg_dict["label_a"] = smart_label_for_exp_dir(cfg_dict["exp_a"])
    if cfg_dict.get("exp_b") and not cfg_dict.get("label_b"):
        cfg_dict["label_b"] = smart_label_for_exp_dir(cfg_dict["exp_b"])

    cfg = Config(**cfg_dict)
    run_analysis(cfg)


if __name__ == "__main__":
    main()
