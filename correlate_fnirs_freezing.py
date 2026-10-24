"""Target-state fNIRS correlations using anonymized participant IDs."""

from submission_config import DATA_DIR, RESULTS_DIR
import itertools
import os
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, spearmanr, t
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


FONT_FAMILY = 'Arial'
FONT_SIZE = 20
PANEL_SIZE = (6.5, 5.5)
PANEL_DPI = 300

plt.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

POINT_COLOR = '#48C0AA'
FIT_COLOR = '#555555'
CI_COLOR = '#D9D9D9'
HEATMAP_CMAP = 'RdBu_r'

file_path = DATA_DIR / 'fnirs_clinical.xlsx'
sheet_name = 'Sheet1'
output_dir = str(RESULTS_DIR / 'fnirs_correlations')
os.makedirs(output_dir, exist_ok=True)

print(f'Reading {file_path} [{sheet_name}]')
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
except Exception as exc:
    print(f'Cannot read input: {exc}')
    raise SystemExit


def read_current_metrics(path):
    for encoding in ['utf-8-sig', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Cannot decode current MATLAB metrics: {path}')


def update_metrics_by_identity(clinical):
    analysis_root = RESULTS_DIR / 'fnirs_state_dynamics'
    metrics = read_current_metrics(analysis_root / 'master_minor_state2_4segments_summary.csv')
    definitions = read_current_metrics(analysis_root / 'target_state_patterns.csv')
    if len(definitions) != 14:
        raise ValueError('Expected the 14-pattern target-state definition.')
    if not (metrics['Actual_Pre_Window_s'].eq(10).all() and metrics['Actual_Post_Window_s'].eq(10).all()):
        raise ValueError('Expected complete 10-s pre/post windows.')
    clinical = clinical.copy()
    for frame in (clinical, metrics):
        if 'Subject_ID' not in frame:
            raise ValueError('Both inputs require an anonymized Subject_ID column.')
        frame['Subject_ID'] = frame['Subject_ID'].astype(str).str.strip()
        if frame['Subject_ID'].duplicated().any():
            raise ValueError('Subject_ID must be unique in each input.')
    if not metrics['Subject_ID'].isin(clinical['Subject_ID']).all():
        raise ValueError('Some fNIRS Subject_ID values are absent from the clinical input.')
    clinical = clinical.loc[clinical['Subject_ID'].isin(metrics['Subject_ID'])].copy()
    clinical = clinical.set_index('Subject_ID')
    current = metrics.set_index('Subject_ID').reindex(clinical.index)
    columns = [c for c in metrics if c.startswith(('Occupancy_', 'DwellingTime_', 'Mean_'))]
    for column in columns + ['Valid_Trials']:
        clinical[column] = current[column]
    return clinical.reset_index()


df = update_metrics_by_identity(df)


primary_covariates = [
    'sex',
    'age',
    'education',
    'MoCA-pre',
]


PRIMARY_APPLY_FDR = False
analysis_1_pairs = [
    ('Occupancy_Turning_Pct', 'TF%'),
    ('DwellingTime_Turning_s', 'TF%'),
]
analysis_2_metrics = [
    'DwellingTime_Turning_s',
    'TF%',
    'Mean_Turning_Duration_s',
    'Mean_Freezing_Duration_s',
]
analysis_2_pairs = list(itertools.combinations(analysis_2_metrics, 2))

all_metrics = [
    'Occupancy_Turning_Pct',
    'DwellingTime_Turning_s',
    'TF%',
    'Mean_Turning_Duration_s',
    'Mean_Freezing_Duration_s',
]
all_needed = list(dict.fromkeys(
    all_metrics + primary_covariates
))
missing = [column for column in all_needed if column not in df.columns]
if missing:
    print(f'Missing input columns: {missing}')
    raise SystemExit

for column in all_needed:
    df[column] = pd.to_numeric(df[column], errors='coerce')

label_dict = {
    'Occupancy_Turning_Pct': 'Occupancy of Minor state #2 (%)',
    'DwellingTime_Turning_s': 'Dwell time of Minor state #2 (s)',
    'TF%': 'TF%',
    'Mean_Turning_Duration_s': 'Turning time (s)',
    'Mean_Freezing_Duration_s': 'Freezing time (s)',
}
file_label_dict = {
    'Occupancy_Turning_Pct': 'Occupancy_Turning_Pct',
    'DwellingTime_Turning_s': 'DwellingTime_Turning_s',
    'TF%': 'TF_percent',
    'Mean_Turning_Duration_s': 'Mean_Turning_Duration_s',
    'Mean_Freezing_Duration_s': 'Mean_Freezing_Duration_s',
}
matrix_label_dict = {
    'Occupancy_Turning_Pct': 'Occupancy of\nMinor state #2 (%)',
    'DwellingTime_Turning_s': 'Dwell time of\nMinor state #2 (s)',
    'TF%': 'TF%',
    'Mean_Turning_Duration_s': 'Turning\ntime (s)',
    'Mean_Freezing_Duration_s': 'Freezing\ntime (s)',
}
x_matrix_label_dict = dict(matrix_label_dict)
x_matrix_label_dict['Occupancy_Turning_Pct'] = 'Occupancy\nof Minor\nstate #2 (%)'
x_matrix_label_dict['DwellingTime_Turning_s'] = 'Dwell time\nof Minor\nstate #2 (s)'
a2_x_label_dict = dict(x_matrix_label_dict)
a2_x_label_dict['DwellingTime_Turning_s'] = 'Dwell\ntime of\nMinor state\n#2 (s)'


def get_residuals(y, x):
    model = LinearRegression().fit(x, y)
    return y - model.predict(x)


def calculate_partial_spearman(dataframe, x_col, y_col, covariate_cols):
    columns = [x_col, y_col] + list(covariate_cols)
    data = dataframe[columns].dropna()
    n = len(data)
    k = len(covariate_cols)
    dof = n - k - 2
    if n < 3 or dof <= 0:
        return np.nan, np.nan, n, dof

    ranked = data.rank(method='average')
    x = ranked[x_col].to_numpy(dtype=float)
    y = ranked[y_col].to_numpy(dtype=float)
    covariates = ranked[list(covariate_cols)].to_numpy(dtype=float)

    x_residual = get_residuals(x, covariates)
    y_residual = get_residuals(y, covariates)
    if np.std(x_residual) == 0 or np.std(y_residual) == 0:
        return np.nan, np.nan, n, dof

    r = float(np.corrcoef(x_residual, y_residual)[0, 1])
    r_for_p = np.clip(r, -1.0 + 1e-15, 1.0 - 1e-15)
    t_stat = r_for_p * np.sqrt(dof / (1.0 - r_for_p ** 2))
    p_value = float(2.0 * t.sf(abs(t_stat), dof))
    return r, p_value, n, dof


def calculate_unadjusted_spearman(dataframe, x_col, y_col):
    data = dataframe[[x_col, y_col]].dropna()
    if len(data) < 3:
        return np.nan, np.nan
    r, p_value = spearmanr(data[x_col], data[y_col])
    return float(r), float(p_value)


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return q_values

    valid_indices = np.flatnonzero(valid)
    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    ranks = np.arange(1, len(sorted_p) + 1, dtype=float)
    sorted_q = sorted_p * len(sorted_p) / ranks
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)

    corrected = np.empty(len(sorted_q), dtype=float)
    corrected[order] = sorted_q
    q_values[valid_indices] = corrected
    return q_values


def build_results(dataframe, analysis_set, pairs, model_name, covariates, apply_fdr=True):
    rows = []
    for x_col, y_col in pairs:
        r_unadjusted, p_unadjusted = calculate_unadjusted_spearman(
            dataframe, x_col, y_col
        )
        r_partial, p_raw, n, dof = calculate_partial_spearman(
            dataframe, x_col, y_col, covariates
        )
        rows.append({
            'analysis_set': analysis_set,
            'model': model_name,
            'Variable_1': x_col,
            'Variable_2': y_col,
            'Variable_1_label': label_dict[x_col],
            'Variable_2_label': label_dict[y_col],
            'n': n,
            'n_covariates': len(covariates),
            'degrees_of_freedom': dof,
            'covariates': ', '.join(covariates),
            'r_unadjusted_spearman': r_unadjusted,
            'p_unadjusted_spearman': p_unadjusted,
            'r_partial_spearman': r_partial,
            'p_raw': p_raw,
        })

    results = pd.DataFrame(rows)
    if apply_fdr:
        results['q_BH_FDR'] = benjamini_hochberg(
            results['p_raw'].to_numpy(dtype=float)
        )
        results['FDR_method'] = 'Benjamini-Hochberg'
        results['FDR_scope'] = (
            f'{len(pairs)} pairwise comparisons within '
            f'{analysis_set} / {model_name}'
        )
    else:
        results['q_BH_FDR'] = np.nan
        results['FDR_method'] = 'None'
        results['FDR_scope'] = 'No multiplicity correction; report raw two-sided p values'
    return results


primary_results_1 = build_results(
    df,
    'Analysis 1: Minor state #2 occupancy/dwell time vs TF%',
    analysis_1_pairs,
    'Primary partial Spearman',
    primary_covariates,
    apply_fdr=PRIMARY_APPLY_FDR,
)
primary_results_2 = build_results(
    df,
    'Analysis 2: four-metric pairwise correlations',
    analysis_2_pairs,
    'Primary partial Spearman',
    primary_covariates,
)

all_results = pd.concat([
    primary_results_1,
    primary_results_2,
], ignore_index=True)

print('\nAnalysis 1 - primary model:')
print(primary_results_1[['Variable_1', 'Variable_2', 'n', 'r_partial_spearman', 'p_raw', 'q_BH_FDR']].round(4))
print('\nAnalysis 2 - primary model:')
print(primary_results_2[['Variable_1', 'Variable_2', 'n', 'r_partial_spearman', 'p_raw', 'q_BH_FDR']].round(4))


all_results.to_csv(Path(output_dir) / 'fnirs_partial_correlations.csv', index=False, encoding='utf-8-sig')


def raw_p_text(p_value):
    if pd.isna(p_value):
        return r'$\mathit{p}$ = NA'
    if p_value < 0.001:
        return r'$\mathit{p}$ < 0.001'
    return rf'$\mathit{{p}}$ = {p_value:.3f}'


def save_panel(fig, output_path):
    fig.set_size_inches(*PANEL_SIZE)
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, dpi=PANEL_DPI, bbox_inches=None)
    plt.close(fig)


def plot_style_scatter(
    dataframe,
    x_col,
    y_col,
    partial_r,
    raw_p,
    output_path,
):

    data_clean = dataframe[[x_col, y_col]].dropna()
    x = data_clean[x_col].to_numpy(dtype=float)
    y = data_clean[y_col].to_numpy(dtype=float)
    n = len(x)

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    slope, intercept, _, _, _ = linregress(x, y)
    x_sort = np.linspace(x.min(), x.max(), 100)
    y_pred = slope * x_sort + intercept

    residual = y - (slope * x + intercept)
    mse = np.sum(residual ** 2) / max(n - 2, 1)
    x_mean = np.mean(x)
    denominator = np.sum((x - x_mean) ** 2)
    if denominator > 0 and n > 2:
        se_line = np.sqrt(
            mse * (
                1 / n + (x_sort - x_mean) ** 2 / denominator
            )
        )
        ci_upper = y_pred + 1.96 * se_line
        ci_lower = y_pred - 1.96 * se_line
        ax.fill_between(
            x_sort,
            ci_lower,
            ci_upper,
            color=CI_COLOR,
            alpha=0.75,
            edgecolor='none',
        )

    ax.plot(x_sort, y_pred, color=FIT_COLOR, linewidth=2.0)
    ax.scatter(
        x,
        y,
        facecolors=POINT_COLOR,
        edgecolors='white',
        s=70,
        linewidths=0.8,
        alpha=0.90,
        zorder=3,
    )

    ax.text(
        0.06,
        0.98,
        f'r = {partial_r:.2f}',
        transform=ax.transAxes,
        fontsize=FONT_SIZE,
        verticalalignment='top',
    )
    ax.text(
        0.06,
        0.90,
        raw_p_text(raw_p),
        transform=ax.transAxes,
        fontsize=FONT_SIZE,
        verticalalignment='top',
    )

    ax.set_xlabel(label_dict[x_col].replace(' of Minor state', ' of\nMinor state'),
                  fontsize=FONT_SIZE)
    ax.set_ylabel(label_dict[y_col].replace(' of Minor state', ' of\nMinor state'),
                  fontsize=FONT_SIZE)
    ax.tick_params(
        direction='out',
        length=5,
        width=1,
        colors='black',
        labelsize=FONT_SIZE,
    )
    sns.despine(ax=ax, top=True, right=True)
    save_panel(fig, output_path)


def create_scatter_set(dataframe, results, prefix):
    for _, row in results.iterrows():
        x_col = row['Variable_1']
        y_col = row['Variable_2']
        output_path = os.path.join(
            output_dir,
            f'{prefix}_Scatter_{file_label_dict[x_col]}_vs_'
            f'{file_label_dict[y_col]}.png',
        )
        plot_style_scatter(
            dataframe,
            x_col,
            y_col,
            partial_r=row['r_partial_spearman'],
            raw_p=row['p_raw'],
            output_path=output_path,
        )
        print(f'Figure saved to: {output_path}')


create_scatter_set(df, primary_results_1, 'A1_Primary')
create_scatter_set(df, primary_results_2, 'A2_Primary')


def get_sig_label(q_value):
    if pd.isna(q_value):
        return ''
    if q_value < 0.001:
        return '***'
    if q_value < 0.01:
        return '**'
    if q_value < 0.05:
        return '*'
    return ''


def draw_centered_star(ax, x, y, q_value):
    sig = get_sig_label(q_value)
    if sig:
        ax.annotate(
            sig,
            xy=(x, y),
            xycoords='data',
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='center',
            fontsize=FONT_SIZE,
            color='black',
            fontweight='bold',
            zorder=4,
        )


def draw_colorbar(fig, ax):
    color_scale = plt.cm.ScalarMappable(
        cmap=HEATMAP_CMAP,
        norm=plt.Normalize(vmin=-1.0, vmax=1.0),
    )
    color_scale.set_array(np.array([-1.0, 1.0]))
    cbar = fig.colorbar(
        color_scale,
        ax=ax,
        orientation='vertical',
        fraction=0.06,
        pad=0.08,
        ticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    cbar.set_label('Partial Spearman correlation (r)', fontsize=FONT_SIZE,
                   labelpad=12)
    cbar.ax.tick_params(labelsize=FONT_SIZE, pad=7)


def plot_analysis_1_heatmap(results, output_path):
    result_lookup = {
        (row['Variable_1'], row['Variable_2']): row
        for _, row in results.iterrows()
    }
    x_variables = ['Occupancy_Turning_Pct', 'DwellingTime_Turning_s']
    y_variable = 'TF%'

    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    ax.set_xlim(-0.5, len(x_variables) - 0.5)
    ax.set_ylim(0.5, -0.5)
    ax.set_aspect('equal')

    for x_idx in range(len(x_variables)):
        ax.add_patch(plt.Rectangle(
            (x_idx - 0.5, -0.5),
            1,
            1,
            fill=False,
            edgecolor='#E0E0E0',
            linewidth=0.8,
            zorder=0,
        ))
        x_col = x_variables[x_idx]
        row = result_lookup[(x_col, y_variable)]
        r_value = row['r_partial_spearman']
        ax.scatter(
            x_idx,
            0,
            s=280 + abs(r_value) * 2200,
            c=[r_value],
            cmap=HEATMAP_CMAP,
            vmin=-1.0,
            vmax=1.0,
            alpha=0.90,
            edgecolors='white',
            linewidths=1.0,
            zorder=3,
        )
        draw_centered_star(ax, x_idx, 0, row['p_raw'])

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(range(len(x_variables)))
    ax.set_xticklabels(
        [x_matrix_label_dict[col] for col in x_variables],
        rotation=0,
        ha='center',
        fontsize=FONT_SIZE,
    )
    ax.set_yticks([0])
    ax.set_yticklabels([matrix_label_dict[y_variable]], fontsize=FONT_SIZE)
    ax.tick_params(axis='x', pad=14)
    ax.tick_params(axis='y', pad=12)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    draw_colorbar(fig, ax)
    save_panel(fig, output_path)
    print(f'Analysis 1 heatmap saved to: {output_path}')


def plot_analysis_2_heatmap(results, output_path):
    result_lookup = {
        (row['Variable_1'], row['Variable_2']): row
        for _, row in results.iterrows()
    }
    n_metrics = len(analysis_2_metrics)
    fig, ax = plt.subplots(figsize=PANEL_SIZE)

    ax.set_xlim(-0.5, n_metrics - 1.5)
    ax.set_ylim(n_metrics - 1.5, -0.5)
    ax.set_aspect('equal')

    for row_idx in range(n_metrics):
        for col_idx in range(row_idx):
            ax.add_patch(plt.Rectangle(
                (col_idx - 0.5, row_idx - 1.5),
                1,
                1,
                fill=False,
                edgecolor='#E0E0E0',
                linewidth=0.8,
                zorder=0,
            ))
            x_col = analysis_2_metrics[col_idx]
            y_col = analysis_2_metrics[row_idx]
            row = result_lookup[(x_col, y_col)]
            r_value = row['r_partial_spearman']
            ax.scatter(
                col_idx,
                row_idx - 1,
                s=260 + abs(r_value) * 2200,
                c=[r_value],
                cmap=HEATMAP_CMAP,
                vmin=-1.0,
                vmax=1.0,
                alpha=0.90,
                edgecolors='white',
                linewidths=1.0,
                zorder=3,
            )
            draw_centered_star(ax, col_idx, row_idx - 1, row['q_BH_FDR'])

    ax.set_xticks(range(n_metrics - 1))
    ax.set_yticks(range(n_metrics - 1))
    ax.set_xticklabels(
        [a2_x_label_dict[col] for col in analysis_2_metrics[:-1]],
        rotation=0,
        ha='center',
        va='top',
        multialignment='center',
        fontsize=FONT_SIZE,
    )
    ax.set_yticklabels(
        [matrix_label_dict[col] for col in analysis_2_metrics[1:]],
        fontsize=FONT_SIZE,
    )
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    draw_colorbar(fig, ax)
    save_panel(fig, output_path)
    print(f'Analysis 2 heatmap saved to: {output_path}')


plot_analysis_1_heatmap(
    primary_results_1,
    os.path.join(output_dir, 'A1_Partial_Spearman_dot_heatmap_primary.png'),
)
plot_analysis_2_heatmap(
    primary_results_2,
    os.path.join(output_dir, 'A2_Partial_Spearman_dot_heatmap_primary.png'),
)

legend_fig, legend_ax = plt.subplots(figsize=(6.5, 1.2))
legend_ax.axis('off')
legend_ax.legend(handles=[
    Line2D(
        [0], [0], marker='o', linestyle='none', label='Data point',
        markerfacecolor=POINT_COLOR, markeredgecolor='white',
        markeredgewidth=0.7, markersize=8,
    ),
    Line2D(
        [0], [0], color=FIT_COLOR, linewidth=2.0, label='Fitted line',
    ),
], loc='center', ncol=2, frameon=False, fontsize=FONT_SIZE)
legend_fig.savefig(
    os.path.join(output_dir, 'Partial_Spearman_legend.png'),
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
)
plt.close(legend_fig)

print('Correlation analyses and figures completed.')
