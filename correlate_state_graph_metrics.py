"""Partial Spearman state-graph correlations in the pooled PD cohort."""

from submission_config import DATA_DIR, RESULTS_DIR



import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, t
from statsmodels.stats.multitest import multipletests


DATA_PATH = DATA_DIR / 'cross_sectional_metrics.csv'
OUTPUT_DIR = RESULTS_DIR / 'state_graph_correlations'
RESULT_PATH = OUTPUT_DIR / 'state_graph_correlations.csv'

X_COL = "Minor 2 duration"

COVARIATES = ["Age", "Sex", "Education", "mean FD", "Duration", "HY", "MoCA", "LEDD"]

METRICS = [
    ("Eglob_AUC_minor2", "Eglob AUC", "Eglob_AUC"),
    (
        "Pallidum_R_efficiency_minor2",
        "Right pallidum efficiency",
        "Pallidum_R_efficiency",
    ),
    (
        "Putamen_L_efficiency_minor2",
        "Left putamen efficiency",
        "Putamen_L_efficiency",
    ),
]

COLORS = {
    "PD-FOG": "#48C0AA",
    "PD-nonFOG": "#456990",
}
FIT_COLOR = "#555555"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def residualize(values, covariates):
    design = np.column_stack([np.ones(len(covariates)), covariates])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _covariate_matrix(dataframe, covariate_cols):
    parts = []
    for column in covariate_cols:
        if column == "Sex":
            encoded = pd.get_dummies(
                dataframe[column].astype(str), prefix="Sex", drop_first=True,
                dtype=float,
            )
            if not encoded.empty:
                parts.append(encoded)
        else:
            parts.append(pd.DataFrame({
                column: pd.to_numeric(dataframe[column], errors="coerce")
            }, index=dataframe.index))
    return pd.concat(parts, axis=1).astype(float)


def partial_spearman(dataframe, x_col, y_col, covariate_cols):
    columns = [x_col, y_col, *covariate_cols]
    data = dataframe[columns].copy()
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    for column in covariate_cols:
        if column != "Sex":
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    if len(data) < len(covariate_cols) + 4:
        return np.nan, np.nan, len(data)
    ranked = data[[x_col, y_col]].rank(method="average")
    design = _covariate_matrix(data, covariate_cols)
    covariates = design.to_numpy(dtype=float)
    x_residual = residualize(ranked[x_col].to_numpy(dtype=float), covariates)
    y_residual = residualize(ranked[y_col].to_numpy(dtype=float), covariates)
    r_value = pearsonr(x_residual, y_residual).statistic

    degrees_freedom = len(data) - design.shape[1] - 2
    statistic = r_value * np.sqrt(degrees_freedom / (1 - r_value**2))
    p_value = 2 * t.sf(abs(statistic), degrees_freedom)
    return float(r_value), float(p_value), len(data)


def format_p(p_value):
    if pd.isna(p_value):
        return r"$\mathit{p}$ = NA"
    if p_value < 0.001:
        return r"$\mathit{p}$ < 0.001"
    return rf"$\mathit{{p}}$ = {p_value:.3f}"


def draw_plot(dataframe, y_col, y_label, r_value, p_value, output_stem):
    plot_data = dataframe[["GroupLabel", X_COL, y_col]].apply(
        lambda column: pd.to_numeric(column, errors="coerce")
        if column.name != "GroupLabel" else column
    ).dropna()

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    x_values = plot_data[X_COL].to_numpy(dtype=float)
    y_values = plot_data[y_col].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x_values)), x_values])
    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    fitted = design @ coefficients
    dof = len(x_values) - 2
    residual_scale = np.sqrt(np.sum((y_values - fitted) ** 2) / dof)
    x_grid = np.linspace(x_values.min(), x_values.max(), 300)
    y_grid = coefficients[0] + coefficients[1] * x_grid
    sxx = np.sum((x_values - x_values.mean()) ** 2)
    se = residual_scale * np.sqrt(
        1 / len(x_values) + (x_grid - x_values.mean()) ** 2 / sxx
    )
    critical = t.ppf(0.975, dof)
    ax.fill_between(
        x_grid, y_grid - critical * se, y_grid + critical * se,
        color="#BDBDBD", alpha=0.35, linewidth=0, zorder=1,
    )
    ax.plot(x_grid, y_grid, color=FIT_COLOR, linewidth=1.7, zorder=2)
    for group_label in ["PD-FOG", "PD-nonFOG"]:
        group_data = plot_data.loc[plot_data["GroupLabel"] == group_label]
        ax.scatter(
            group_data[X_COL],
            group_data[y_col],
            s=42,
            facecolor=COLORS[group_label],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.90,
            zorder=3,
        )

    ax.text(
        0.04,
        0.05,
        f"r = {r_value:.2f}\n{format_p(p_value)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
    )
    ax.set_xlabel("Duration of minor state #2(TR)", labelpad=6)
    ax.set_ylabel(y_label, labelpad=7)
    ax.tick_params(axis="both", width=1.0, length=4)
    sns.despine(ax=ax)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", label=label,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.7, markersize=7,
        )
        for label, color in COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.4,
    )

    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.96)
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    tiff_path = OUTPUT_DIR / f"{output_stem}.tiff"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        tiff_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {tiff_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    pd_data = df.loc[df["group"].isin(["FOG", "noFOG"])].copy()
    pd_data["GroupLabel"] = pd_data["group"].map({
        "FOG": "PD-FOG",
        "noFOG": "PD-nonFOG",
    })
    pd_data["GroupCode"] = pd_data["group"].map({"FOG": 1, "noFOG": 2})

    required = {
        X_COL, *COVARIATES,
        *(metric for metric, _, _ in METRICS),
    }
    missing = sorted(required.difference(pd_data.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    results = []
    for metric, y_label, file_label in METRICS:
        r_value, p_value, n_value = partial_spearman(
            pd_data, X_COL, metric, COVARIATES
        )
        results.append({
            "analysis": "primary",
            "X_variable": X_COL,
            "Y_variable": metric,
            "r": r_value,
            "p": p_value,
            "n": n_value,
            "Covariates": "; ".join(COVARIATES),
        })
        draw_plot(
            pd_data,
            metric,
            y_label,
            r_value,
            p_value,
            f"Partial_correlation_minor2_{file_label}",
        )

    adjusted = multipletests(
        [row["p"] for row in results], method="fdr_bh"
    )[1]
    for row, adjusted_p in zip(results, adjusted):
        row["p_FDR"] = float(adjusted_p)
        row["FDR_method"] = "Benjamini-Hochberg"
        row["FDR_scope"] = "BH-FDR across 3 graph metrics"
        row["n_covariates"] = len(COVARIATES)
    pd.DataFrame(results).to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(results).to_string(index=False))
    print("PD group counts:", pd_data["GroupLabel"].value_counts().to_dict())
    print(f"Saved: {RESULT_PATH}")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", label=label,
               markerfacecolor=color, markeredgecolor="white",
               markeredgewidth=0.7, markersize=8)
        for label, color in COLORS.items()
    ]
    legend_fig, legend_ax = plt.subplots(figsize=(5.5, 1.2))
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_handles, loc="center", ncol=2,
                     frameon=False, fontsize=12)
    legend_fig.savefig(
        OUTPUT_DIR / "Partial_correlation_minor2_graph_metrics_legend.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(legend_fig)


if __name__ == "__main__":
    main()
