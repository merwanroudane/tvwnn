"""
Publication-quality tables.

Two renderers for every table: a fixed-width console version with rules, and a
``booktabs`` LaTeX version ready to paste into a manuscript.  Layout follows
the conventions of Tables 1 and 2 of the paper -- the lowest value in each
block in bold, significance markers from the Diebold-Mariano test, and a
footer carrying the legend and the variance/benchmark used.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
        https://github.com/merwanroudane/tvwnn
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "forecast_accuracy_table",
    "render_console",
    "render_latex",
    "model_summary_table",
    "dm_matrix",
]

_STAR_LEGEND = (
    "Significance of the Diebold-Mariano test against the {bench} benchmark: "
    "* p<0.10, ** p<0.05, *** p<0.01."
)


def forecast_accuracy_table(
    long: pd.DataFrame,
    value: str = "relative_mse",
    benchmark: str = "AR",
    model_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Pivot the long frame from :func:`tvwnn.evaluate.mse_table` into Table 1 shape.

    Parameters
    ----------
    long : DataFrame
        Columns ``model``, ``P``, ``h``, ``mse``, ``relative_mse``, ``stars``.
    value : {"relative_mse", "mse"}
    benchmark : str
    model_order : sequence of str or None
        Row order within each ``P`` block.  Defaults to the paper's:
        AR, TVP, ARX, FFNN, RNN, LSTM, TVNN.

    Returns
    -------
    DataFrame indexed by ``(P, model)`` with one column per horizon, holding
    formatted strings such as ``"0.779***"``.
    """
    if model_order is None:
        model_order = ["AR", "TVP", "ARX", "FFNN", "RNN", "LSTM", "TVNN"]

    wide_val = long.pivot_table(index=["P", "model"], columns="h", values=value)
    wide_star = long.pivot_table(
        index=["P", "model"], columns="h", values="stars", aggfunc="first"
    )

    out = pd.DataFrame(index=wide_val.index, columns=wide_val.columns, dtype=object)
    for idx in wide_val.index:
        for col in wide_val.columns:
            v = wide_val.loc[idx, col]
            if not np.isfinite(v):
                out.loc[idx, col] = ""
                continue
            star = wide_star.loc[idx, col] if idx in wide_star.index else ""
            out.loc[idx, col] = f"{v:.3f}{star if isinstance(star, str) else ''}"

    order_key = {m: i for i, m in enumerate(model_order)}
    out = out.reset_index()
    out["_k"] = out["model"].map(lambda m: order_key.get(m, len(order_key)))
    out = out.sort_values(["P", "_k", "model"]).drop(columns="_k")
    out = out.set_index(["P", "model"])
    out.columns = [f"h = {c}" for c in out.columns]
    out.attrs["benchmark"] = benchmark
    return out


def _fmt_cell(x: object, width: int) -> str:
    return f"{str(x):>{width}}"


def render_console(
    table: pd.DataFrame,
    title: str = "",
    note: str | None = None,
    bold_min_within: str | None = "P",
    float_fmt: str = "{:.3f}",
    index_labels: Sequence[str] | None = None,
) -> str:
    """Render a DataFrame as a fixed-width table with rules.

    Parameters
    ----------
    table : DataFrame
        Values may be numbers or pre-formatted strings.
    title : str
    note : str or None
        Footer text.  ``None`` and a ``benchmark`` attribute on the table
        produce the standard significance legend.
    bold_min_within : str or None
        Index level within which to mark the smallest value with a leading
        ``*`` flag column.  Console output has no bold, so the minimum in each
        block is marked with a trailing ``<`` instead.
    index_labels : sequence of str or None
        Header names for the index levels.

    Returns
    -------
    str
    """
    df = table.copy()
    idx_names = list(df.index.names) if index_labels is None else list(index_labels)
    idx_frame = df.index.to_frame(index=False)
    idx_frame.columns = idx_names

    # Mark the block minimum, the way the paper bolds it.
    marks = pd.DataFrame("", index=df.index, columns=df.columns)
    if bold_min_within is not None and bold_min_within in (df.index.names or []):
        numeric = df.map(_to_float)
        for _, block in numeric.groupby(level=bold_min_within):
            for col in df.columns:
                vals = block[col].dropna()
                if vals.empty:
                    continue
                best = vals.idxmin()
                marks.loc[best, col] = "<"

    body = df.map(lambda v: v if isinstance(v, str) else _safe_fmt(v, float_fmt))
    body = body.astype(str) + marks.astype(str)

    headers = idx_names + list(df.columns)
    rows = [
        list(idx_frame.iloc[i].astype(str)) + list(body.iloc[i].astype(str))
        for i in range(len(df))
    ]
    widths = [
        max(len(str(headers[j])), *(len(r[j]) for r in rows)) + 2
        for j in range(len(headers))
    ]

    lines: list[str] = []
    total = sum(widths)
    if title:
        lines.append(title)
    lines.append("=" * total)
    lines.append("".join(_fmt_cell(h, w) for h, w in zip(headers, widths)))
    lines.append("-" * total)

    prev_block = object()
    for i, r in enumerate(rows):
        if bold_min_within in idx_names and i > 0:
            key = r[idx_names.index(bold_min_within)]
            if key != prev_block:
                lines.append("-" * total)
        if bold_min_within in idx_names:
            prev_block = r[idx_names.index(bold_min_within)]
        lines.append("".join(_fmt_cell(c, w) for c, w in zip(r, widths)))
    lines.append("=" * total)

    footer = note
    if footer is None and table.attrs.get("benchmark"):
        footer = _STAR_LEGEND.format(bench=table.attrs["benchmark"])
    if footer:
        lines.append(_wrap(f"Note: {footer}", total))
    if bold_min_within is not None:
        lines.append(_wrap("'<' marks the lowest value in each block.", total))
    return "\n".join(lines)


def render_latex(
    table: pd.DataFrame,
    caption: str = "",
    label: str = "tab:tvnn",
    note: str | None = None,
    bold_min_within: str | None = "P",
    float_fmt: str = "{:.3f}",
) -> str:
    """Render the same table as a ``booktabs`` LaTeX tabular.

    Requires ``\\usepackage{booktabs}`` and, for the footer,
    ``\\usepackage{threeparttable}`` (or drop the note).
    """
    df = table.copy()
    idx_names = [n if n else "" for n in (df.index.names or [])]
    idx_frame = df.index.to_frame(index=False)

    numeric = df.map(_to_float)
    bold = pd.DataFrame(False, index=df.index, columns=df.columns)
    if bold_min_within is not None and bold_min_within in (df.index.names or []):
        for _, block in numeric.groupby(level=bold_min_within):
            for col in df.columns:
                vals = block[col].dropna()
                if not vals.empty:
                    bold.loc[vals.idxmin(), col] = True

    ncol = len(idx_names) + len(df.columns)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(idx_names) + "r" * len(df.columns) + "}",
        "\\toprule",
        " & ".join(list(idx_names) + [str(c) for c in df.columns]) + " \\\\",
        "\\midrule",
    ]
    prev = object()
    for i in range(len(df)):
        cells = list(idx_frame.iloc[i].astype(str))
        if bold_min_within in idx_names and i > 0 and cells[idx_names.index(bold_min_within)] != prev:
            lines.append("\\midrule")
        if bold_min_within in idx_names:
            prev = cells[idx_names.index(bold_min_within)]
        for col in df.columns:
            raw = df.iloc[i][col]
            txt = raw if isinstance(raw, str) else _safe_fmt(raw, float_fmt)
            txt = txt.replace("*", "$^{*}$") if "*" in str(txt) else txt
            if bold.iloc[i][col]:
                txt = f"\\textbf{{{txt}}}"
            cells.append(str(txt))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    footer = note
    if footer is None and table.attrs.get("benchmark"):
        footer = _STAR_LEGEND.format(bench=table.attrs["benchmark"])
    if footer:
        lines += [
            f"\\begin{{minipage}}{{\\linewidth}}\\footnotesize",
            f"\\textit{{Note:}} {footer}",
            "\\end{minipage}",
        ]
    lines.append("\\end{table}")
    return "\n".join(lines)


def model_summary_table(summary: dict, title: str = "TVNN estimates") -> str:
    """Console block for the fitted variance components of one TVNN."""
    Omega = np.atleast_2d(summary["Omega"])
    lines = [title, "=" * max(len(title), 46)]
    lines.append(f"{'Basis functions d':<30}{summary['d']:>16d}")
    lines.append(f"{'Hidden layers':<30}{str(summary['hidden_sizes']):>16}")
    lines.append(f"{'Network objective':<30}{summary['loss']:>16}")
    lines.append(f"{'Filter states used':<30}{summary['state']:>16}")
    lines.append("-" * 46)
    lines.append(f"{'sigma^2 (obs. variance)':<30}{summary['sigma2']:>16.6f}")
    for i in range(Omega.shape[0]):
        for j in range(i, Omega.shape[1]):
            lab = f"Omega[{i + 1},{j + 1}]"
            lines.append(f"{lab:<30}{Omega[i, j]:>16.6e}")
    lines.append(f"{'trace(Omega)':<30}{summary['omega_trace']:>16.6e}")
    ratio = summary["omega_trace"] / max(summary["sigma2"], 1e-16)
    lines.append(f"{'signal-to-noise tr(Omega)/s^2':<30}{ratio:>16.6e}")
    lines.append("-" * 46)
    lines.append(f"{'Final log-likelihood':<30}{summary['final_loglik']:>16.4f}")
    lines.append(f"{'Total epochs (N x M)':<30}{summary['total_epochs']:>16d}")
    lines.append(f"{'Learning rate':<30}{summary['learning_rate']:>16.4f}")
    lines.append(f"{'L2 penalty alpha':<30}{summary['l2']:>16.4f}")
    lines.append("=" * 46)
    lines.append(
        _wrap(
            "Note: a larger trace(Omega)/sigma^2 means more parameter drift. "
            "Omega is identified only up to a rotation of the learned basis, "
            "so its individual entries are not comparable across runs.",
            70,
        )
    )
    return "\n".join(lines)


def dm_matrix(errors: pd.DataFrame, h: int = 1, **kwargs) -> pd.DataFrame:
    """All pairwise Diebold-Mariano p-values.

    Entry ``(i, j)`` tests row model ``i`` against column model ``j``; a small
    value with a negative statistic says ``i`` beats ``j``.  The paper tests
    only against the AR benchmark, so this fills in the TVNN-versus-FFNN
    comparison its headline claim actually rests on.
    """
    from .evaluate import dm_test

    names = list(errors.columns)
    out = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i in names:
        for j in names:
            if i == j:
                continue
            res = dm_test(errors[i].to_numpy(), errors[j].to_numpy(), h=h, **kwargs)
            out.loc[i, j] = res.p_value
    return out


# ------------------------------------------------------------------- helpers


def _to_float(v: object) -> float:
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    s = str(v).replace("*", "").replace("<", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _safe_fmt(v: object, fmt: str) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "" if not np.isfinite(f) else fmt.format(f)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, max(width, 40)))
