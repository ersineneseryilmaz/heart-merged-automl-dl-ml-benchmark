\
"""Budget compliance analysis for repeated lightweight runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LIGHT_BUDGET_SECONDS


def compute_budget_compliance(
    selected_runs: pd.DataFrame,
    budget_s: float = LIGHT_BUDGET_SECONDS,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Compute budget compliance per framework."""
    rows = []
    for framework, g in selected_runs.groupby("analysis_framework"):
        runtime = pd.to_numeric(g["runtime_s"], errors="coerce")
        overhead = runtime - budget_s
        compliant = runtime <= budget_s
        rows.append(
            {
                "framework": framework,
                "regime": "Lightweight",
                "n_runs": int(len(g)),
                "compliant_runs": int(compliant.sum()),
                "runtime_mean_s": runtime.mean(),
                "runtime_sd_s": runtime.std(ddof=1),
                "overhead_mean_s": overhead.mean(),
                "max_overhead_s": overhead.max(),
                "runtime_to_budget_ratio_mean": (runtime / budget_s).mean(),
                "compliance_rate_pct": compliant.mean() * 100,
            }
        )

    out = pd.DataFrame(rows)
    if output_csv is not None:
        out.to_csv(output_csv, index=False, encoding="utf-8")
    return out
