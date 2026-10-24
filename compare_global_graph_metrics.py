"""ANCOVA and raincloud plots for global graph metrics."""

from submission_config import DATA_DIR, RESULTS_DIR


import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


DATA_PATH = DATA_DIR / 'cross_sectional_metrics.csv'
OUTPUT_DIR = RESULTS_DIR / 'global_graph_comparison'
RESULTS_PATH = OUTPUT_DIR / "ANCOVA_posthoc_results.csv"

COMMON_COVARIATES = ["Age", "Sex", "Education", "mean FD"]
ELA_PD_COVARIATES = COMMON_COVARIATES + ["Duration", "LEDD", "HY"]
GLOBAL_PD_COVARIATES = COMMON_COVARIATES + ["Duration","LEDD", "HY"]
ALL_COVARIATES = list(dict.fromkeys(
    ELA_PD_COVARIATES + GLOBAL_PD_COVARIATES
))

INDICATORS = [
    "Eglob_AUC_minor2",
    "Eloc_AUC_minor2",
    "Modularity_minor2",
    "Eglob major 1",
    "Eloc major 1",
    "Modularity major 1",
    "Eglob major2",
    "Eloc major2",
    "Modularity major2",
    "Eglob minor1",
    "Eloc minor1",
    "Modularity minor1",
]


ELA_INDICATORS = []


GLOBAL_INDICATORS = [
    "Eglob major 1",
    "Eloc major 1",
    "Modularity major 1",
    "Eglob major2",
    "Eloc major2",
    "Modularity major2",
    "Eglob minor1",
    "Eloc minor1",
    "Modularity minor1",
    "Eglob_AUC_minor2",
    "Eloc_AUC_minor2",
    "Modularity_minor2",
]

GROUPS = [
    ("FOG", "PD-FOG", "#48C0AA"),
    ("noFOG", "PD-nonFOG", "#456990"),
    ("HC", "HC", "#EF767A"),
]


SHARED_Y_FAMILIES = [
    ["Major 1 Size", "Major 2 Size"],
    ["Minor 1 Size", "Minor 2 Size"],
    ["Major 1 duration", "Major 2 duration"],
    ["Minor 1 duration", "Minor 2 duration"],
    ["Direct transition", "Indirect transition"],
    [
        "Eglob major 1", "Eglob major2", "Eglob minor1",
        "Eglob_AUC_minor2",
    ],
    [
        "Eloc major 1", "Eloc major2", "Eloc minor1",
        "Eloc_AUC_minor2",
    ],
    [
        "Modularity major 1", "Modularity major2", "Modularity minor1",
        "Modularity_minor2",
    ],
]


Y_AXIS_LOWER_LIMITS = {

    "Major 1 Size": 15.0,
    "Major 2 Size": 15.0,
    "Major 1 duration": 1.0,
    "Major 2 duration": 1.0,
    "Minor 1 duration": 0.5,
    "Minor 2 duration": 0.5,
    "Minor 1 Size": 0.5,
    "Minor 2 Size": 0.5,
    **{
        indicator: 0.03
        for indicator in INDICATORS
        if indicator.startswith(("Eglob", "Eloc"))
    },
    **{
        indicator: 0.06
        for indicator in INDICATORS
        if indicator.startswith("Modularity")
    },
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _numeric_columns(frame, columns):
    converted = frame.copy()
    for column in columns:
        if column != "Sex":
            converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def _covariate_matrix(frame, covariates):
    continuous = [column for column in covariates if column != "Sex"]
    parts = [frame[continuous].astype(float)] if continuous else []
    if "Sex" in covariates:
        sex = pd.get_dummies(
            frame["Sex"].astype(str), prefix="Sex", drop_first=True, dtype=float
        )
        if not sex.empty:
            parts.append(sex)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)


def _partial_eta_squared(statistic, df_num, df_denom, stat_type="F"):
    if not np.isfinite(statistic) or not np.isfinite(df_denom):
        return np.nan
    if stat_type == "F":
        numerator = statistic * df_num
        denominator = numerator + df_denom
    else:
        numerator = statistic ** 2
        denominator = numerator + df_denom
    return float(numerator / denominator) if denominator > 0 else np.nan


def _partial_omega_squared(f_value, df_num, df_denom):
    if not np.isfinite(f_value) or not np.isfinite(df_denom):
        return np.nan
    denominator = f_value * df_num + df_denom + 1
    return float(max(0.0, (f_value - 1) * df_num / denominator)) if denominator > 0 else np.nan


def _contrast_summary(estimate, t_value, raw_p, df, ci_low, ci_high, n,
                      covariates, comparison_model):
    return {
        "t": float(t_value),
        "df_num": 1.0,
        "df_denom": float(df),
        "raw_p": float(raw_p),
        "adjusted_p": min(float(raw_p) * 3, 1.0),
        "estimate": float(estimate),
        "se": abs(float(estimate) / float(t_value)) if t_value != 0 else np.nan,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "partial_eta_squared": _partial_eta_squared(t_value, 1, df, "t"),
        "n": int(n),
        "covariates": ", ".join(covariates),
        "comparison_model": comparison_model,
    }


def ancova_for_indicator(df, indicator):
    columns = ["group", indicator, *ALL_COVARIATES]
    work = _numeric_columns(df[columns], ALL_COVARIATES)
    work[indicator] = pd.to_numeric(work[indicator], errors="coerce")

    omnibus_data = work[["group", indicator, *COMMON_COVARIATES]].dropna().copy()
    omnibus_data = omnibus_data[omnibus_data["group"].isin([g[0] for g in GROUPS])]
    group_dummies = pd.get_dummies(
        pd.Categorical(
            omnibus_data["group"], categories=["HC", "noFOG", "FOG"]
        ),
        prefix="group",
        drop_first=True,
        dtype=float,
    )
    group_dummies.index = omnibus_data.index
    covariates = _covariate_matrix(omnibus_data, COMMON_COVARIATES)
    design = sm.add_constant(pd.concat([group_dummies, covariates], axis=1))
    model = sm.OLS(omnibus_data[indicator].astype(float), design).fit()
    restriction = np.zeros((len(group_dummies.columns), len(design.columns)))
    for row, column in enumerate(group_dummies.columns):
        restriction[row, design.columns.get_loc(column)] = 1.0
    omnibus_test = model.f_test(restriction)
    omnibus_p = float(np.asarray(omnibus_test.pvalue).squeeze())
    omnibus_f = float(np.asarray(omnibus_test.fvalue).squeeze())
    omnibus_df_num = float(np.asarray(omnibus_test.df_num).squeeze())
    omnibus_df_denom = float(model.df_resid)

    pairs = [(0, 1), (1, 2), (0, 2)]
    posthoc = []
    full_model_contrasts = {

        (1, 2): {"group_noFOG": 1.0},

        (0, 2): {"group_FOG": 1.0},
    }
    for left, right in pairs:
        left_group = GROUPS[left][0]
        right_group = GROUPS[right][0]
        if (left, right) in full_model_contrasts:
            contrast = np.zeros(len(design.columns), dtype=float)
            for column, weight in full_model_contrasts[(left, right)].items():
                contrast[design.columns.get_loc(column)] = weight
            contrast_test = model.t_test(contrast)
            t_value = float(np.asarray(contrast_test.tvalue).squeeze())
            raw_p = float(np.asarray(contrast_test.pvalue).squeeze())
            estimate = float(np.asarray(contrast_test.effect).squeeze())
            ci_low, ci_high = np.asarray(contrast_test.conf_int()).reshape(-1)
            pair_df = omnibus_df_denom
            comparison_n = int(len(omnibus_data))
            covariate_names = COMMON_COVARIATES
            comparison_model = "Three-group estimated marginal means"
        else:

            covariate_names = (
                GLOBAL_PD_COVARIATES
                if indicator in GLOBAL_INDICATORS
                else ELA_PD_COVARIATES
            )
            pair_data = work.loc[
                work["group"].isin([left_group, right_group]),
                ["group", indicator, *covariate_names],
            ].dropna().copy()
            pair_data["group_difference"] = (
                pair_data["group"] == right_group
            ).astype(float)
            pair_covariates = _covariate_matrix(pair_data, covariate_names)
            pair_design = sm.add_constant(pd.concat(
                [pair_data[["group_difference"]], pair_covariates], axis=1
            ))
            pair_model = sm.OLS(
                pair_data[indicator].astype(float), pair_design
            ).fit()

            t_value = -float(pair_model.tvalues["group_difference"])
            raw_p = float(pair_model.pvalues["group_difference"])
            estimate = -float(pair_model.params["group_difference"])
            pair_df = float(pair_model.df_resid)
            ci = np.asarray(pair_model.conf_int().loc["group_difference"], dtype=float)
            ci_low, ci_high = -ci[1], -ci[0]
            comparison_n = int(len(pair_data))
            comparison_model = "PD subgroup ANCOVA"

        summary = _contrast_summary(
            estimate, t_value, raw_p, pair_df, ci_low, ci_high,
            comparison_n, covariate_names, comparison_model,
        )
        summary.update({"left": left, "right": right})
        posthoc.append(summary)

    return {
        "omnibus_f": omnibus_f,
        "omnibus_p": omnibus_p,
        "omnibus_df_num": omnibus_df_num,
        "omnibus_df_denom": omnibus_df_denom,
        "omnibus_partial_eta_squared": _partial_eta_squared(
            omnibus_f, omnibus_df_num, omnibus_df_denom, "F"
        ),
        "omnibus_partial_omega_squared": _partial_omega_squared(
            omnibus_f, omnibus_df_num, omnibus_df_denom
        ),
        "omnibus_n": int(len(omnibus_data)),
        "posthoc": posthoc,
    }


def apply_posthoc_bonferroni(analyses, indicators, result_key):
    comparisons = [
        (indicator, result)
        for indicator in indicators
        for result in analyses[indicator]["posthoc"]
    ]
    raw_p = np.asarray([result["raw_p"] for _, result in comparisons], dtype=float)
    valid = np.isfinite(raw_p)
    adjusted = np.full(raw_p.shape, np.nan)
    if valid.any():
        adjusted[valid] = multipletests(raw_p[valid], method="bonferroni")[1]
    for (_, result), adjusted_p in zip(comparisons, adjusted):
        result[result_key] = float(adjusted_p)


def draw_half_violin(ax, values, x_center, color, width=0.35):
    if len(values) < 2 or np.isclose(np.std(values, ddof=1), 0):
        return
    value_span = values.max() - values.min()
    bandwidth = value_span / 15.0
    sample_std = np.std(values, ddof=1)
    kde = gaussian_kde(values, bw_method=bandwidth / sample_std)

    tail_padding = bandwidth * 3.0
    y_grid = np.linspace(
        values.min() - tail_padding,
        values.max() + tail_padding,
        240,
    )
    density = kde(y_grid)

    density[0] = 0.0
    density[-1] = 0.0
    density_max = density.max()
    if density_max <= 0:
        return
    density = density / density_max * width
    baseline = x_center + 0.05
    ax.fill_betweenx(
        y_grid,
        baseline,
        baseline + density,
        facecolor=color,
        edgecolor="none",
        alpha=0.40,
        zorder=0,
    )


def draw_box(ax, values, x_center):
    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    lower_candidates = values[values >= q1 - 1.5 * iqr]
    upper_candidates = values[values <= q3 + 1.5 * iqr]
    lower = lower_candidates.min() if len(lower_candidates) else values.min()
    upper = upper_candidates.max() if len(upper_candidates) else values.max()

    box_x = x_center + 0.05
    box_width = 0.06

    ax.plot([box_x, box_x], [lower, upper], color="black", linewidth=0.8, zorder=2)
    ax.add_patch(plt.Rectangle(
        (box_x - box_width / 2, q1),
        box_width,
        q3 - q1,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    ))
    ax.plot(
        [box_x - box_width / 2, box_x + box_width / 2],
        [median, median],
        color="black",
        linewidth=1.5,
        zorder=3,
    )


def make_y_settings(raw_min, raw_max, comparison_levels, lower_limit=0.0):
    data_span = raw_max - raw_min
    if data_span <= 0:
        data_span = max(abs(raw_max), 1.0)
    return {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "span": data_span,
        "ylim": (
            lower_limit,
            raw_max + data_span * (0.16 + 0.15 * comparison_levels),
        ),
    }


def add_significance(ax, comparisons, y_settings):
    data_span = y_settings["span"]
    first_y = y_settings["raw_max"] + data_span * 0.08
    level_gap = data_span * 0.14
    bracket_height = data_span * 0.025

    for level, (left, right, p_value) in enumerate(comparisons):
        y = first_y + level * level_gap
        x_left = left + 1
        x_right = right + 1
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y - bracket_height, y, y, y - bracket_height],
            color="black",
            linewidth=0.8,
            clip_on=False,
        )
        ax.text(
            (left + right) / 2 + 1,
            y + data_span * 0.012,
            significance_stars(p_value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(*y_settings["ylim"])


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def _group_descriptives(df, indicator):
    summaries = {}
    for code, label, _ in GROUPS:
        values = pd.to_numeric(
            df.loc[df["group"] == code, indicator], errors="coerce"
        ).dropna()
        summaries[code] = {
            "n": int(values.size),
            "mean": float(values.mean()) if values.size else np.nan,
            "sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
            "label": label,
        }
    return summaries


def plot_indicator(df, indicator, seed, y_settings, comparisons):
    fig, ax = plt.subplots(figsize=(4, 3))
    rng = np.random.default_rng(seed)
    group_values = []

    for position, (source_group, _, color) in enumerate(GROUPS, start=1):
        values = pd.to_numeric(
            df.loc[df["group"] == source_group, indicator], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        group_values.append(values)
        if len(values) == 0:
            continue

        draw_half_violin(ax, values, position, color)
        jitter = rng.uniform(0.05, 0.30, size=len(values))
        ax.scatter(
            position - jitter,
            values,
            s=18,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.70,
            zorder=3,
        )
        draw_box(ax, values, position)

    all_values = np.concatenate([values for values in group_values if len(values)])
    add_significance(ax, comparisons, y_settings)

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.set_xlim(0.35, 3.65)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([display for _, display, _ in GROUPS])
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=4, width=0.8)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.96)

    output_path = OUTPUT_DIR / f"{safe_filename(indicator)}.png"
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path} ({len(comparisons)} significant comparison(s))")


def save_group_legend():
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", label=display,
               markerfacecolor=color, markeredgecolor="white",
               markeredgewidth=0.7, markersize=8)
        for _, display, color in GROUPS
    ]
    fig, ax = plt.subplots(figsize=(6.0, 1.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=11)
    fig.savefig(
        OUTPUT_DIR / "Cross_sectional_group_legend.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    required = {"group", *INDICATORS, *ALL_COVARIATES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    analyses = {
        indicator: ancova_for_indicator(df, indicator)
        for indicator in INDICATORS
    }
    apply_posthoc_bonferroni(
        analyses, ELA_INDICATORS, "bonf_p_across_30_ela"
    )
    apply_posthoc_bonferroni(
        analyses, GLOBAL_INDICATORS, "bonf_p_across_36_global"
    )

    for indicator, analysis in analyses.items():
        if indicator in ELA_INDICATORS:
            analysis["omnibus_adjusted_p"] = min(
                analysis["omnibus_p"] * len(ELA_INDICATORS), 1.0
            )
            analysis["omnibus_correction"] = "Bonferroni across 10 ELA indicators"
        elif indicator in GLOBAL_INDICATORS:
            analysis["omnibus_adjusted_p"] = min(
                analysis["omnibus_p"] * len(GLOBAL_INDICATORS), 1.0
            )
            analysis["omnibus_correction"] = (
                "Bonferroni across 12 global network indicators"
            )
        else:
            analysis["omnibus_adjusted_p"] = analysis["omnibus_p"]
            analysis["omnibus_correction"] = "None"

        analysis["significant"] = []
        if analysis["omnibus_adjusted_p"] < 0.05:
            for result in analysis["posthoc"]:
                figure_p = (
                    result["bonf_p_across_30_ela"]
                    if indicator in ELA_INDICATORS
                    else (
                        result["bonf_p_across_36_global"]
                        if indicator in GLOBAL_INDICATORS
                        else result["adjusted_p"]
                    )
                )
                result["figure_adjusted_p"] = figure_p
                if figure_p < 0.05:
                    analysis["significant"].append(
                        (result["left"], result["right"], figure_p)
                    )

    result_rows = []
    for indicator, analysis in analyses.items():
        desc = _group_descriptives(df, indicator)
        group_columns = {}
        for group_code, _, _ in GROUPS:
            group_columns[f"n_{group_code}"] = desc[group_code]["n"]
            group_columns[f"mean_{group_code}"] = desc[group_code]["mean"]
            group_columns[f"sd_{group_code}"] = desc[group_code]["sd"]
        result_rows.append({
            "indicator": indicator,
            "test": "Three-group ANCOVA main effect",
            "comparison": "PD-FOG vs PD-nonFOG vs HC",
            "F": analysis["omnibus_f"],
            "df_num": analysis["omnibus_df_num"],
            "df_denom": analysis["omnibus_df_denom"],
            "t": np.nan,
            "raw_p": analysis["omnibus_p"],
            "Bonferroni_p": analysis["omnibus_adjusted_p"],
            "Bonferroni_p_across_30_ELA_posthocs": np.nan,
            "Bonferroni_p_across_36_global_posthocs": np.nan,
            "partial_eta_squared": analysis["omnibus_partial_eta_squared"],
            "partial_omega_squared": analysis["omnibus_partial_omega_squared"],
            "estimate": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": analysis["omnibus_n"],
            **group_columns,
            "covariates": ", ".join(COMMON_COVARIATES),
            "multiple_comparison_scope": analysis["omnibus_correction"],
            "comparison_model": "Three-group ANCOVA",
        })
        for posthoc in analysis["posthoc"]:
            result_rows.append({
                "indicator": indicator,
                "test": "Post-hoc ANCOVA",
                "comparison": (
                    f"{GROUPS[posthoc['left']][1]} vs "
                    f"{GROUPS[posthoc['right']][1]}"
                ),
                "F": np.nan,
                "df_num": posthoc["df_num"],
                "df_denom": posthoc["df_denom"],
                "t": posthoc["t"],
                "raw_p": posthoc["raw_p"],
                "Bonferroni_p": (
                    np.nan
                    if indicator in ELA_INDICATORS or indicator in GLOBAL_INDICATORS
                    else posthoc["adjusted_p"]
                ),
                "Bonferroni_p_across_30_ELA_posthocs": (
                    posthoc["bonf_p_across_30_ela"]
                    if indicator in ELA_INDICATORS else np.nan
                ),
                "Bonferroni_p_across_36_global_posthocs": (
                    posthoc["bonf_p_across_36_global"]
                    if indicator in GLOBAL_INDICATORS else np.nan
                ),
                "partial_eta_squared": posthoc["partial_eta_squared"],
                "partial_omega_squared": np.nan,
                "estimate": posthoc["estimate"],
                "se": posthoc["se"],
                "ci_low": posthoc["ci_low"],
                "ci_high": posthoc["ci_high"],
                "n": posthoc["n"],
                **group_columns,
                "covariates": posthoc["covariates"],
                "multiple_comparison_scope": (
                    "Bonferroni across 30 ELA pairwise tests"
                    if indicator in ELA_INDICATORS
                    else (
                        "Bonferroni across 36 global network pairwise tests"
                        if indicator in GLOBAL_INDICATORS
                        else "Bonferroni across 3 pairwise group comparisons"
                    )
                ),
                "comparison_model": posthoc["comparison_model"],
            })
    pd.DataFrame(result_rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    summaries = {}
    for indicator in INDICATORS:
        group_values = [
            pd.to_numeric(
                df.loc[df["group"] == source_group, indicator], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            for source_group, _, _ in GROUPS
        ]
        all_values = np.concatenate([values for values in group_values if len(values)])
        summaries[indicator] = {
            "raw_min": float(all_values.min()),
            "raw_max": float(all_values.max()),
            "levels": len(analyses[indicator]["significant"]),
        }

    y_settings = {
        indicator: make_y_settings(
            summary["raw_min"],
            summary["raw_max"],
            summary["levels"],
            lower_limit=Y_AXIS_LOWER_LIMITS.get(indicator, 0.0),
        )
        for indicator, summary in summaries.items()
    }
    for family in SHARED_Y_FAMILIES:
        family = [name for name in family if name in summaries]
        if len(family) < 2:
            continue

        if len({Y_AXIS_LOWER_LIMITS.get(name, 0.0) for name in family}) > 1:
            continue
        family_min = min(summaries[name]["raw_min"] for name in family)
        family_max = max(summaries[name]["raw_max"] for name in family)
        family_levels = max(summaries[name]["levels"] for name in family)
        family_lower_limit = max(
            Y_AXIS_LOWER_LIMITS.get(name, 0.0) for name in family
        )
        shared = make_y_settings(
            family_min,
            family_max,
            family_levels,
            lower_limit=family_lower_limit,
        )
        for name in family:
            y_settings[name] = shared

    for index, indicator in enumerate(INDICATORS):
        plot_indicator(
            df,
            indicator,
            seed=2026 + index,
            y_settings=y_settings[indicator],
            comparisons=analyses[indicator]["significant"],
        )
    save_group_legend()
    print(f"Finished: {len(INDICATORS)} figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
