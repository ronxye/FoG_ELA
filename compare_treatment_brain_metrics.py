"""Paired treatment comparisons with mean changes and 95% confidence intervals."""

from submission_config import DATA_DIR, RESULTS_DIR



import matplotlib
import matplotlib.collections
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import t as student_t


DATA_PATH = DATA_DIR / 'treatment_metrics.csv'
OUTPUT_DIR = RESULTS_DIR / 'treatment_brain_comparison'

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.linewidth": 1.2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


GROUP_STYLES = {
    "All": {
        "label": "All participants",
        "pre": "#48C0AA",
        "post": "#A7DDD3",
    },
    "Response": {
        "label": "Responders",
        "pre": "#D89000",
        "post": "#F2D38A",
    },
    "No-response": {
        "label": "No-responders",
        "pre": "#8064A9",
        "post": "#C9BADE",
    },
}

INDICATORS = [
    {
        "pre": "Pre Minor 2 duration",
        "post": "Post Minor 2 duration",
        "title": "Minor 2 duration",
        "ylabel": "Duration of Minor state #2 (TR)",
        "file": "Minor_2_duration",
    },
    {
        "pre": "Pre_Eglob",
        "post": "post_Eglob",
        "title": "Eglob",
        "ylabel": "Global efficiency",
        "file": "Eglob",
    },
    {
        "pre": "Pre_Putamen_L_efficiency",
        "post": "post_L_putamen_efficiency",
        "title": "Putamen L efficiency",
        "ylabel": "Left putamen efficiency",
        "file": "Putamen_L_efficiency",
    },
]


COVARIATES = []
SENSITIVITY_COVARIATES = []


def get_shared_y_settings(data, config):
    all_values = data[[config["pre"], config["post"]]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy().ravel()
    all_values = all_values[np.isfinite(all_values)]
    if len(all_values) == 0:
        raise ValueError(f"No numeric values found for {config['title']}")

    raw_min = float(np.min(all_values))
    raw_max = float(np.max(all_values))
    span = raw_max - raw_min
    padding = span * 0.13 if span > 0 else max(abs(raw_max) * 0.1, 0.1)

    return {
        "raw_max": raw_max,
        "padding": padding,
        "ylim": (raw_min - padding, raw_max + padding * 2.4),
    }


def mean_and_ci(values, confidence=0.95):
    numeric_values = np.asarray(values, dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    sample_size = len(numeric_values)
    mean_value = float(np.mean(numeric_values))
    standard_error = np.std(numeric_values, ddof=1) / np.sqrt(sample_size)
    critical_value = student_t.ppf(
        (1 + confidence) / 2, sample_size - 1
    )
    margin = critical_value * standard_error
    return mean_value, mean_value - margin, mean_value + margin


def _covariate_matrix(frame, covariates):
    parts = []
    for column in covariates:
        if column == "Sex":
            encoded = pd.get_dummies(
                frame[column].astype(str), prefix="Sex", drop_first=True,
                dtype=float,
            )
            if not encoded.empty:
                parts.append(encoded)
        else:
            parts.append(pd.DataFrame({
                column: pd.to_numeric(frame[column], errors="coerce")
            }, index=frame.index))
    return pd.concat(parts, axis=1).astype(float)


def adjusted_change_test(data, config, covariates):
    columns = [config["pre"], config["post"], *covariates]
    work = data[columns].copy()
    work[config["pre"]] = pd.to_numeric(work[config["pre"]], errors="coerce")
    work[config["post"]] = pd.to_numeric(work[config["post"]], errors="coerce")
    for column in covariates:
        if column != "Sex":
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna()
    if len(work) < len(covariates) + 4:
        return np.nan, len(work)
    change = work[config["post"]] - work[config["pre"]]
    if covariates:
        design_cov = _covariate_matrix(work, covariates)

        for column in design_cov.columns:
            if not column.startswith("Sex_"):
                design_cov[column] = design_cov[column] - design_cov[column].mean()
    else:
        design_cov = pd.DataFrame(index=work.index)
    design = sm.add_constant(design_cov, has_constant="add")
    model = sm.OLS(change.astype(float), design).fit()
    return float(model.pvalues.get("const", np.nan)), len(work)


def paired_summary(data, config):
    pair = data[[config["pre"], config["post"]]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    pre = pair[config["pre"]].to_numpy(dtype=float)
    post = pair[config["post"]].to_numpy(dtype=float)
    change = post - pre
    n = len(pair)
    summary = {"n_pairs": n, "change_direction": "Post - Pre"}
    for prefix, values in [("Pre", pre), ("Post", post), ("Delta", change)]:
        if n >= 2:
            mean, ci_low, ci_high = mean_and_ci(values)
            sd = float(np.std(values, ddof=1))
        else:
            mean = float(np.mean(values)) if n else np.nan
            sd = ci_low = ci_high = np.nan
        summary.update({
            prefix + "_mean": mean,
            prefix + "_SD": sd,
            prefix + "_CI95_low": ci_low,
            prefix + "_CI95_high": ci_high,
        })
    if n >= 2:
        model = sm.OLS(change, np.ones((n, 1))).fit()
        summary["paired_t"] = float(model.tvalues[0])
        summary["paired_df"] = float(model.df_resid)
        summary["paired_raw_p"] = float(model.pvalues[0])
        change_sd = summary["Delta_SD"]
        summary["Cohen_dz"] = summary["Delta_mean"] / change_sd if change_sd > 0 else np.nan
    else:
        summary.update({"paired_t": np.nan, "paired_df": np.nan,
                        "paired_raw_p": np.nan, "Cohen_dz": np.nan})
    summary["figure_annotation"] = (
        f"Δ mean = {summary['Delta_mean']:.3f}; "
        f"95% CI [{summary['Delta_CI95_low']:.3f}, {summary['Delta_CI95_high']:.3f}]"
    )
    return summary


def paired_plot(data, config, cohort_key, shared_y, p_value):
    style = GROUP_STYLES[cohort_key]
    pair = data[[config["pre"], config["post"]]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(pair) < 2:
        print(f"Skipped {config['title']} - {style['label']}: fewer than 2 pairs")
        return

    values = pair.rename(columns={
        config["pre"]: "Pre",
        config["post"]: "Post",
    })

    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = ""
    long_data = values.reset_index(drop=True).reset_index(names="Subject").melt(
        id_vars="Subject",
        var_name="Timepoint",
        value_name="Value",
    )

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    palette = {"Pre": style["pre"], "Post": style["post"]}
    sns.violinplot(
        data=long_data,
        x="Timepoint",
        y="Value",
        order=["Pre", "Post"],
        palette=palette,
        hue="Timepoint",
        hue_order=["Pre", "Post"],
        legend=False,
        inner=None,
        linewidth=0,
        width=0.62,
        cut=0,
        saturation=1,
        ax=ax,
    )
    for collection in ax.collections:
        if isinstance(collection, matplotlib.collections.PolyCollection):
            collection.set_alpha(0.30)

    rng = np.random.default_rng(42)
    jitter = 0.065
    x_pre = rng.uniform(-jitter, jitter, len(values))
    x_post = 1 + rng.uniform(-jitter, jitter, len(values))

    for i, (_, row) in enumerate(values.iterrows()):
        ax.plot(
            [x_pre[i], x_post[i]],
            [row["Pre"], row["Post"]],
            color="#8A8A8A",
            alpha=0.38,
            linewidth=0.9,
            zorder=1,
        )

    sns.boxplot(
        data=long_data,
        x="Timepoint",
        y="Value",
        order=["Pre", "Post"],
        width=0.14,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.5},
        ax=ax,
        zorder=2,
    )
    ax.scatter(
        x_pre, values["Pre"], s=43, color=style["pre"], alpha=0.95,
        edgecolor="white", linewidth=0.7, zorder=3,
    )
    ax.scatter(
        x_post, values["Post"], s=43, color=style["post"], alpha=0.95,
        edgecolor="white", linewidth=0.7, zorder=3,
    )

    pre_mean, pre_ci_low, pre_ci_high = mean_and_ci(values["Pre"])
    post_mean, post_ci_low, post_ci_high = mean_and_ci(values["Post"])
    delta_values = values["Post"] - values["Pre"]
    delta_mean, delta_ci_low, delta_ci_high = mean_and_ci(delta_values)
    ax.errorbar(
        [0, 1],
        [pre_mean, post_mean],
        yerr=[
            [pre_mean - pre_ci_low, post_mean - post_ci_low],
            [pre_ci_high - pre_mean, post_ci_high - post_mean],
        ],
        fmt="D",
        markersize=6,
        markerfacecolor="black",
        markeredgecolor="white",
        ecolor="black",
        elinewidth=1.4,
        capsize=4,
        capthick=1.4,
        zorder=5,
    )
    padding = shared_y["padding"]
    ax.set_ylim(*shared_y["ylim"])

    if significance:
        bracket_y = shared_y["raw_max"] + padding * 0.65
        bracket_height = padding * 0.22
        ax.plot(
            [0, 0, 1, 1],
            [bracket_y, bracket_y + bracket_height,
             bracket_y + bracket_height, bracket_y],
            color="black", linewidth=1.3, clip_on=False,
        )
        ax.text(
            0.5,
            bracket_y + bracket_height + padding * 0.05,
            significance,
            ha="center",
            va="bottom",
            fontsize=23,
            fontweight="bold",
        )
    ax.set_xlabel("")
    ax.set_ylabel(config["ylabel"], labelpad=10)
    ax.set_title(f"{style['label']} (n = {len(values)})", pad=12)
    ax.tick_params(axis="both", length=5, width=1.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    ax.text(
        0.50,
        0.04,
        f"Δ mean = {delta_mean:.3f}\n95% CI [{delta_ci_low:.3f}, {delta_ci_high:.3f}]",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=18,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
            "pad": 2.5,
        },
    )

    output_path = OUTPUT_DIR / f"{config['file']}_{cohort_key}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path} (raw P = {p_value:.4g}, {significance or 'not significant'})")


def save_standalone_legends():
    handles = []
    for cohort_key in ["All", "Response", "No-response"]:
        style = GROUP_STYLES[cohort_key]
        handles.extend([
            mpatches.Patch(
                facecolor=style["pre"],
                edgecolor="none",
                label=f"{style['label']} — Pre",
            ),
            mpatches.Patch(
                facecolor=style["post"],
                edgecolor="none",
                label=f"{style['label']} — Post",
            ),
        ])

    fig, ax = plt.subplots(figsize=(10.8, 2.5))
    ax.axis("off")
    legend = ax.legend(
        handles=handles,
        ncol=3,
        loc="center",
        frameon=False,
        fontsize=20,
        handlelength=1.8,
        handleheight=1.1,
        columnspacing=1.8,
        labelspacing=0.9,
    )
    for text_item in legend.get_texts():
        text_item.set_fontfamily("Arial")

    output_path = OUTPUT_DIR / "Color_legend_horizontal.png"
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    print(f"Saved horizontal legend: {output_path}")

    fig, ax = plt.subplots(figsize=(4.8, 6.8))
    ax.axis("off")
    legend = ax.legend(
        handles=handles,
        ncol=1,
        loc="center",
        frameon=False,
        fontsize=20,
        handlelength=1.8,
        handleheight=1.1,
        labelspacing=1.1,
    )
    for text_item in legend.get_texts():
        text_item.set_fontfamily("Arial")

    output_path = OUTPUT_DIR / "Color_legend_vertical.png"
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    print(f"Saved vertical legend: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required = {"group", *SENSITIVITY_COVARIATES}
    for item in INDICATORS:
        required.update([item["pre"], item["post"]])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    group_numeric = pd.to_numeric(df["group"], errors="coerce")
    cohorts = {
        "All": df,
        "Response": df.loc[group_numeric.eq(1)],
        "No-response": df.loc[group_numeric.eq(2)],
    }
    primary_stats = {}
    sensitivity_stats = {}
    for config in INDICATORS:
        for cohort_key, cohort_data in cohorts.items():
            primary_stats[(config["file"], cohort_key)] = adjusted_change_test(
                cohort_data, config, COVARIATES
            )
            sensitivity_stats[(config["file"], cohort_key)] = adjusted_change_test(
                cohort_data, config, SENSITIVITY_COVARIATES
            )

    primary_rows = []
    sensitivity_rows = []
    for config in INDICATORS:
        for cohort_key in cohorts:
            raw_p, n_value = primary_stats[(config["file"], cohort_key)]
            primary_rows.append({
                "analysis": "primary",
                "indicator": config["title"],
                "file": config["file"],
                "cohort": GROUP_STYLES[cohort_key]["label"],
                "raw_p": raw_p,
                "n": n_value,
                "covariates": ", ".join(COVARIATES),
            })
            sens_p, sens_n = sensitivity_stats[(config["file"], cohort_key)]
            sensitivity_rows.append({
                "analysis": "unadjusted",
                "indicator": config["title"],
                "file": config["file"],
                "cohort": GROUP_STYLES[cohort_key]["label"],
                "raw_p": sens_p,
                "n": sens_n,
                "covariates": ", ".join(SENSITIVITY_COVARIATES),
            })

    output_rows = primary_rows if COVARIATES == SENSITIVITY_COVARIATES else primary_rows + sensitivity_rows
    summaries = {
        (config["file"], GROUP_STYLES[cohort_key]["label"]): paired_summary(cohort_data, config)
        for config in INDICATORS
        for cohort_key, cohort_data in cohorts.items()
    }
    for row in output_rows:
        row.update(summaries[(row["file"], row["cohort"])])
        row["multiplicity_correction"] = "None"
        row["p_value_type"] = "Raw two-sided p value"
        row["CI_method"] = "Two-sided Student t confidence interval for the paired mean change"
    table = pd.DataFrame(output_rows)
    front_columns = [
        "analysis", "indicator", "cohort", "n", "n_pairs",
        "Pre_mean", "Pre_SD", "Post_mean", "Post_SD",
        "Delta_mean", "Delta_SD", "Delta_CI95_low", "Delta_CI95_high",
        "paired_t", "paired_df", "raw_p", "Cohen_dz", "figure_annotation",
    ]
    table = table[front_columns + [c for c in table.columns if c not in front_columns]]
    table.to_csv(
        OUTPUT_DIR / "ELA_pre_post_unadjusted_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for config in INDICATORS:
        shared_y = get_shared_y_settings(df, config)
        for cohort_key, cohort_data in cohorts.items():
            raw_p, _ = primary_stats[(config["file"], cohort_key)]
            paired_plot(cohort_data, config, cohort_key, shared_y, raw_p)

    save_standalone_legends()

    print(f"Finished: 9 figures and 2 standalone legends saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
