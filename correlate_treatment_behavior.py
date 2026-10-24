"""Partial Spearman correlations of treatment-related state changes and behavioral improvement."""

from submission_config import DATA_DIR, RESULTS_DIR



import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests


DATA_PATH = DATA_DIR / 'treatment_metrics.csv'
OUTPUT_DIR = RESULTS_DIR / 'treatment_behavior_correlations'

X_COL = "chang minor 2 duration"
X_LABEL = "ΔDuration of Minor state #2"

Y_CONFIG = [
    ("5mTUGrate", "5m TUG Improvement Rate (%)"),
    ("FOGQ rate", "FOGQ Improvement Rate (%)"),
    ("SS180rate", "SS 180 Improvement Rate (%)"),
]


COVARIATES = ["Sex", "Age", "Education", "MoCA"]
SENSITIVITY_COVARIATES = COVARIATES + ["preUPDRSIII"]
FDR_SCOPE = "Benjamini-Hochberg FDR across 3 clinical outcomes (5m TUG, FOGQ, SS180), separately per model"


COLORS = {
    "Responders": "#D89000",
    "No-responders": "#8064A9",
    "Fit line": "#555555",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def p_text(p_value):
    if pd.isna(p_value):
        return r"$\mathit{p}$ = NA"
    if p_value < 0.001:
        return r"$\mathit{p}$ < 0.001"
    return rf"$\mathit{{p}}$ = {p_value:.3f}"


def _covariate_matrix(dataframe, covariates):
    parts = []
    for column in covariates:
        if column == "Sex":
            encoded = pd.get_dummies(
                dataframe[column].astype(str),
                prefix="Sex",
                drop_first=True,
                dtype=float,
            )
            if not encoded.empty:
                parts.append(encoded)
        else:
            parts.append(pd.DataFrame({
                column: pd.to_numeric(dataframe[column], errors="coerce")
            }, index=dataframe.index))
    return pd.concat(parts, axis=1).astype(float)


def calculate_partial_spearman(dataframe, x_col, y_col, covariates):
    subset = dataframe[[x_col, y_col, *covariates]].copy()
    subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
    subset[y_col] = pd.to_numeric(subset[y_col], errors="coerce")
    for column in covariates:
        if column != "Sex":
            subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.dropna()
    n = len(subset)
    if n < 4:
        return np.nan, np.nan, n
    ranked = subset[[x_col, y_col]].rank(method="average")
    design = _covariate_matrix(subset, covariates)
    x_model = LinearRegression().fit(design, ranked[x_col].to_numpy())
    y_model = LinearRegression().fit(design, ranked[y_col].to_numpy())
    x_res = ranked[x_col].to_numpy() - x_model.predict(design)
    y_res = ranked[y_col].to_numpy() - y_model.predict(design)
    rho = float(np.corrcoef(x_res, y_res)[0, 1])
    dof = n - design.shape[1] - 2
    if dof <= 0 or abs(rho) >= 1:
        return rho, np.nan, n
    t_stat = rho * np.sqrt(dof / (1 - rho**2))
    p_value = float(2 * t.sf(abs(t_stat), dof))
    return rho, p_value, n


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = {
        "group", X_COL, *(column for column, _ in Y_CONFIG),
        *SENSITIVITY_COVARIATES,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df["Group_label"] = pd.to_numeric(df["group"], errors="coerce").map({
        1: "Responders",
        2: "No-responders",
    })
    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
    for y_col, _ in Y_CONFIG:

        df[y_col] = pd.to_numeric(df[y_col], errors="coerce") * 100

    partial_results = {
        y_col: calculate_partial_spearman(df, X_COL, y_col, COVARIATES)
        for y_col, _ in Y_CONFIG
    }
    results_df = pd.DataFrame([
        {
            "analysis": "primary",
            "Variable": y_col,
            "partial_r": partial_results[y_col][0],
            "p": partial_results[y_col][1],
            "n": partial_results[y_col][2],
            "covariates": ", ".join(COVARIATES),
        }
        for y_col, _ in Y_CONFIG
    ])
    valid_p = results_df["p"].notna()
    results_df["p_FDR"] = np.nan
    if valid_p.any():
        results_df.loc[valid_p, "p_FDR"] = multipletests(
            results_df.loc[valid_p, "p"].to_numpy(),
            method="fdr_bh",
        )[1]
    results_df["FDR_scope"] = FDR_SCOPE
    sensitivity_results = []
    for y_col, _ in Y_CONFIG:
        rho, p_value, n_value = calculate_partial_spearman(
            df, X_COL, y_col, SENSITIVITY_COVARIATES
        )
        sensitivity_results.append({
            "analysis": "UPDRS III sensitivity",
            "Variable": y_col,
            "partial_r": rho,
            "p": p_value,
            "n": n_value,
            "covariates": ", ".join(SENSITIVITY_COVARIATES),
        })
    sensitivity_df = pd.DataFrame(sensitivity_results)
    valid_sensitivity = sensitivity_df["p"].notna()
    sensitivity_df["p_FDR"] = np.nan
    if valid_sensitivity.any():
        sensitivity_df.loc[valid_sensitivity, "p_FDR"] = multipletests(
            sensitivity_df.loc[valid_sensitivity, "p"].to_numpy(),
            method="fdr_bh",
        )[1]
    sensitivity_df["FDR_scope"] = (
        FDR_SCOPE + "; additionally adjusted for preUPDRSIII"
    )
    table_df = pd.concat([results_df, sensitivity_df], ignore_index=True)
    table_df.to_csv(
        OUTPUT_DIR / "Correlation_minor2_behavior_partial_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for y_col, y_label in Y_CONFIG:
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        plot_df = df[["Group_label", X_COL, y_col]].dropna()
        if len(plot_df) < 3:
            raise ValueError(f"Too few valid observations for {y_col}")

        x_values = plot_df[X_COL].to_numpy(dtype=float)
        y_values = plot_df[y_col].to_numpy(dtype=float)
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
        ax.plot(
            x_grid, y_grid, color=COLORS["Fit line"],
            linewidth=1.8, zorder=1,
        )
        sns.scatterplot(
            data=plot_df,
            x=X_COL,
            y=y_col,
            hue="Group_label",
            hue_order=["Responders", "No-responders"],
            palette={
                "Responders": COLORS["Responders"],
                "No-responders": COLORS["No-responders"],
            },
            s=115,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            legend=False,
            zorder=2,
            ax=ax,
        )

        rho, p_value, _ = partial_results[y_col]
        raw_p = results_df.loc[
            results_df["Variable"] == y_col, "p"
        ].iloc[0]
        ax.text(
            0.05,
            0.05,
            f"r = {rho:.2f}\n{p_text(raw_p)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=17,
        )

        ax.set_xlabel(X_LABEL, labelpad=9)
        ax.set_ylabel(y_label, labelpad=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="both", length=5, width=1.2)
        sns.despine(ax=ax)

        stem = {
            "5mTUGrate": "Correlation_minor2_5mTUG_rate",
            "SS180rate": "Correlation_minor2_SS180_rate",
            "FOGQ rate": "Correlation_minor2_FOGQ_rate",
        }[y_col]
        fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.97)
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", label="Responders",
            markerfacecolor=COLORS["Responders"], markeredgecolor="white",
            markeredgewidth=0.8, markersize=15,
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", label="No-responders",
            markerfacecolor=COLORS["No-responders"], markeredgecolor="white",
            markeredgewidth=0.8, markersize=15,
        ),
    ]
    legend_fig, legend_ax = plt.subplots(figsize=(6.0, 1.3))
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_handles, loc="center", ncol=2,
                     frameon=False, fontsize=16)
    legend_fig.savefig(
        OUTPUT_DIR / "Correlation_minor2_behavior_legend.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(legend_fig)


if __name__ == "__main__":
    main()
