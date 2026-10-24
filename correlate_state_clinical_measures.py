"""Partial Spearman clinical correlations with primary and UPDRS III sensitivity models."""

from submission_config import DATA_DIR, RESULTS_DIR



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, t
from statsmodels.stats.multitest import multipletests


DATA_PATH = DATA_DIR / 'cross_sectional_metrics.csv'
OUTPUT_DIR = RESULTS_DIR / 'state_clinical_correlations'
RESULT_PATH = OUTPUT_DIR / 'state_clinical_correlations.csv'

ELA = [
    ("Major 2 Size", "Size of major state #2"),
    ("Minor 2 Size", "Size of minor state #2"),
    ("Major 2 duration", "Duration of major state #2"),
    ("Minor 2 duration", "Duration of minor state #2"),
    ("Direct transition", "Direct transition"),
]
FOG = [
    ("5m TUG", "5m TUG (s)"),
    ("SS180", "SS180 (s)"),
    ("FoGQ", "FOGQ score"),
]
BASE_COVARIATES = ["Sex", "Age", "Education", "MoCA"]
MODEL_COVARIATES = {
    "exploratory": BASE_COVARIATES,
    "sensitivity": BASE_COVARIATES + ["UPDRS III"],
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def residualize(values, covariates):
    design = np.column_stack([np.ones(len(covariates)), covariates])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def partial_spearman(dataframe, x_col, y_col, covariate_cols):
    columns = [x_col, y_col, *covariate_cols]
    data = dataframe[columns].apply(pd.to_numeric, errors="coerce").dropna()
    ranked = data.rank(method="average")
    covariates = ranked[covariate_cols].to_numpy(dtype=float)
    x_residual = residualize(ranked[x_col].to_numpy(dtype=float), covariates)
    y_residual = residualize(ranked[y_col].to_numpy(dtype=float), covariates)
    r_value = float(pearsonr(x_residual, y_residual).statistic)
    degrees_freedom = len(data) - len(covariate_cols) - 2
    if degrees_freedom <= 0 or np.isclose(abs(r_value), 1.0):
        p_value = np.nan
    else:
        statistic = r_value * np.sqrt(degrees_freedom / (1 - r_value**2))
        p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
    return r_value, p_value, len(data), degrees_freedom


def add_bh(rows):
    p_values = np.asarray([row["p_raw"] for row in rows], dtype=float)
    valid = np.isfinite(p_values)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    if valid.any():
        q_values[valid] = multipletests(p_values[valid], method="fdr_bh")[1]
    for row, q_value in zip(rows, q_values):
        row["q_BH_FDR_15"] = float(q_value) if np.isfinite(q_value) else np.nan
        row["significant_q<0.05"] = bool(
            np.isfinite(q_value) and q_value < 0.05
        )


def significance(q_value):
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def draw_heatmap(rows, filename, title):
    ela_labels = [label for _, label in ELA]
    fog_labels = [label for _, label in FOG]
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.8, 6.2), dpi=300)
    ax.set_axis_off()
    heat_ax = ax.inset_axes([0.27, 0.08, 0.46, 0.67])
    cbar_ax = ax.inset_axes([0.76, 0.26, 0.025, 0.45])
    x_index = {label: i for i, label in enumerate(fog_labels)}
    y_index = {label: i for i, label in enumerate(ela_labels)}
    x_positions = [x_index[row["FOG_label"]] for row in rows]
    y_positions = [y_index[row["ELA_label"]] for row in rows]
    correlations = np.asarray([row["r"] for row in rows], dtype=float)
    dots = heat_ax.scatter(
        x_positions, y_positions,
        s=220 + np.abs(correlations) * 700,
        c=correlations, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
        edgecolors="none", alpha=0.9, zorder=2,
    )
    for x_pos, y_pos, q_value in zip(
        x_positions, y_positions,
        [row["q_BH_FDR_15"] for row in rows],
    ):
        stars = significance(q_value)
        if stars:
            heat_ax.annotate(
                stars, xy=(x_pos, y_pos), xytext=(0, 0),
                textcoords="offset points", ha="center", va="center",
                fontsize=18, fontweight="bold", color="black", zorder=3,
            )
    heat_ax.set_xticks(range(len(fog_labels)))
    heat_ax.set_xticklabels(fog_labels, rotation=45, ha="left", fontsize=17)
    heat_ax.xaxis.tick_top()
    heat_ax.set_yticks(range(len(ela_labels)))
    heat_ax.set_yticklabels(ela_labels, fontsize=17)
    heat_ax.set_xlim(-0.55, len(fog_labels) - 0.45)
    heat_ax.set_ylim(len(ela_labels) - 0.5, -0.5)
    heat_ax.set_xticks(np.arange(-0.5, len(fog_labels), 1), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, len(ela_labels), 1), minor=True)
    heat_ax.grid(False, which="major")
    heat_ax.grid(True, which="minor", color="#E6E6E6", linewidth=0.6,
                 alpha=0.85)
    heat_ax.set_axisbelow(True)
    heat_ax.spines["top"].set_visible(True)
    heat_ax.spines["left"].set_visible(True)
    heat_ax.spines["bottom"].set_visible(False)
    heat_ax.spines["right"].set_visible(False)
    heat_ax.tick_params(axis="x", bottom=False, top=True, pad=1, length=3)
    heat_ax.tick_params(axis="y", length=0, pad=3)
    heat_ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(dots, cax=cbar_ax)
    colorbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    colorbar.set_label("Partial Spearman Correlation (r)", fontsize=20)
    colorbar.ax.tick_params(labelsize=17, length=3)
    fig.suptitle(title, fontsize=20, y=0.995)
    fig.savefig(OUTPUT_DIR / filename, dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_line(ax, dataframe, y_col, y_label, r_value, p_value):
    plot_data = dataframe[["Minor 2 duration", y_col]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    use_log_y = y_col in {"5m TUG", "SS180"}
    if use_log_y:
        plot_data = plot_data.loc[plot_data[y_col] > 0].copy()
    plot_y_col = y_col
    if use_log_y:
        plot_y_col = "_log10_y"
        plot_data[plot_y_col] = np.log10(plot_data[y_col])
    x_values = plot_data["Minor 2 duration"].to_numpy(dtype=float)
    y_values = plot_data[plot_y_col].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x_values)), x_values])
    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    fitted = design @ coefficients
    df_line = len(x_values) - 2
    residual_scale = np.sqrt(np.sum((y_values - fitted) ** 2) / df_line)
    x_grid = np.linspace(x_values.min(), x_values.max(), 300)
    y_grid = coefficients[0] + coefficients[1] * x_grid
    centered = x_grid - x_values.mean()
    sum_squares = np.sum((x_values - x_values.mean()) ** 2)
    standard_error = residual_scale * np.sqrt(
        1 / len(x_values) + centered**2 / sum_squares
    )
    critical = t.ppf(0.975, df_line)
    ax.fill_between(
        x_grid, y_grid - critical * standard_error,
        y_grid + critical * standard_error,
        color="#BDBDBD", alpha=0.35, linewidth=0, zorder=1,
    )
    ax.plot(x_grid, y_grid, color="#555555", linewidth=1.7, zorder=2)
    ax.scatter(
        x_values, y_values, s=42, facecolors="#48C0AA",
        edgecolors="white", linewidths=0.7, alpha=0.9, zorder=3,
    )
    p_text = "NA" if pd.isna(p_value) else (
        "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    )
    ax.text(
        0.04, 0.95, f"r = {r_value:.2f}\n" +
        r"$\mathit{p}$" + f" = {p_text}",
        transform=ax.transAxes, ha="left", va="top", fontsize=17,
    )
    ax.set_xlabel("Duration of minor state #2 (TR)", labelpad=6)
    ax.set_ylabel(y_label, labelpad=7)
    if use_log_y:
        tick_min = int(np.floor(plot_data[plot_y_col].min()))
        tick_max = int(np.ceil(plot_data[plot_y_col].max()))
        ticks = np.arange(tick_min, tick_max + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([rf"$10^{{{value}}}$" for value in ticks])
        ax.set_ylim(tick_min, tick_max)
    ax.tick_params(axis="both", width=1.0, length=4)
    sns.despine(ax=ax)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    fog = df.loc[df["group"].astype(str).str.strip().eq("FOG")].copy()
    all_covariates = {
        column
        for covariates in MODEL_COVARIATES.values()
        for column in covariates
    }
    required = {
        *(column for column, _ in ELA),
        *(column for column, _ in FOG),
        *all_covariates,
    }
    missing = sorted(required.difference(fog.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    all_rows = []
    fog_labels = {column: label for column, label in FOG}
    for model, covariates in MODEL_COVARIATES.items():
        rows = []
        for ela_col, ela_label in ELA:
            for fog_col, fog_label in FOG:
                r_value, p_value, n_value, df_value = partial_spearman(
                    fog, ela_col, fog_col, covariates
                )
                rows.append({
                    "analysis": model,
                    "ELA_variable": ela_col,
                    "ELA_label": ela_label,
                    "FOG_variable": fog_col,
                    "FOG_label": fog_label,
                    "r": r_value,
                    "p_raw": p_value,
                    "q_BH_FDR_15": np.nan,
                    "significant_q<0.05": False,
                    "n": n_value,
                    "df": df_value,
                    "covariates": ", ".join(covariates),
                })

        add_bh(rows)
        all_rows.extend(rows)

        covariate_tag = model
        covariate_label = ", ".join(covariates)
        draw_heatmap(
            rows,
            f"state_clinical_{covariate_tag}_BH_FDR_heatmap.png",
            f"Partial Spearman correlations adjusted for {covariate_label}",
        )

        for row in rows:
            if row["ELA_variable"] != "Minor 2 duration":
                continue
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            draw_line(
                ax, fog, row["FOG_variable"], fog_labels[row["FOG_variable"]],
                row["r"], row["p_raw"],
            )
            fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.96)
            stem = (
                f"Partial_correlation_{row['FOG_variable']}_"
                f"{covariate_tag}"
            )
            fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300,
                        bbox_inches="tight", facecolor="white")
            plt.close(fig)

        model_frame = pd.DataFrame(rows)
        model_frame.to_csv(
            OUTPUT_DIR / f"state_clinical_{covariate_tag}_BH_FDR_results.csv",
            index=False, encoding="utf-8-sig",
        )

    result_frame = pd.DataFrame(all_rows)
    result_frame.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(
        f"FOG n = {len(fog)}; models = {len(MODEL_COVARIATES)}; "
        "comparisons/model = 15 (BH-FDR applied separately per model)"
    )
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
