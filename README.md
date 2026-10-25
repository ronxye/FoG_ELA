# Altered brain-state residence and network topology associated with dopa-unresponsive freezing of gait in Parkinson’s disease

# Brain-state dynamics analysis code

This package contains the analysis and plotting code supplied with the manuscript. Primary and sensitivity analyses are retained. Statistical CSV outputs and figures are retained; manuscript text, formatted supplementary tables, notebook outputs, local user paths, and participant-name lookup rules are excluded.

## Requirements

- Python with the packages in `requirements.txt`.
- MATLAB with Statistics and Machine Learning Toolbox for MATLAB analyses.
- Preprocessed, de-identified input data. No participant data or fitted models are bundled.

Install the Python packages with `python -m pip install -r requirements.txt`.

## Analyses and execution

Run each Python script as `python <script_name>.py`. Run MATLAB scripts from their containing folder or add this folder to the MATLAB path. The two original Live Scripts are supplied as ordinary `.m` files containing their executable code, without embedded outputs or metadata.

| Script | Analysis |
| --- | --- |
| `calculate_state_dwell_time.py` | Per-participant dwell times from basin-label sequences, in TR units |
| `calculate_state_transitions.py` | Direct and minor-state-mediated transitions per TR |
| `compare_ela_metrics.py` | ELA group ANCOVA, post-hoc contrasts, UPDRS III sensitivity analysis, and plots |
| `compare_global_graph_metrics.py` | Global graph group ANCOVA, post-hoc contrasts, and plots |
| `correlate_state_clinical_measures.py` | PD-FOG partial Spearman correlations and UPDRS III sensitivity analysis |
| `correlate_state_graph_metrics.py` | Pooled-PD partial Spearman state–graph correlations |
| `compare_treatment_brain_metrics.py` | Paired pre/post comparisons in all participants, responders, and non-responders |
| `correlate_treatment_behavior.py` | Treatment-change correlations with three behavioral outcomes and UPDRS III sensitivity analysis |
| `analyze_fnirs_state_dynamics.m` | Rest-baseline binarization, 14 target patterns, complete 10-s windows, four-phase comparisons, and plots |
| `correlate_fnirs_freezing.py` | Correlations of fNIRS state metrics with video-derived freezing measures |
| `plot_state_probability_fit.m` | Empirical/model probability agreement and KL-based model accuracy |
| `plot_group_energy_comparison.m` | Pairwise Pearson correlations of group energy vectors and a 3D projection plot |

Run `analyze_fnirs_state_dynamics.m` before `correlate_fnirs_freezing.py`; the latter reads its current participant-level CSV and target-pattern definitions from `results/fnirs_state_dynamics/`. The `.nirs` files must contain the preprocessed `procResult.dc` HbO data and, for task files, the four-column marker matrix `s`. Channels, network order, 11-Hz sampling, 14 target patterns, first-three-trial policy, complete 10-s window rule, segment-boundary truncation, zero dwell time for no-state segments, and equal trial weighting are retained from the source analysis.

## Statistical settings

- Three-group ELA/global ANCOVA: age, sex, education, mean FD. PD-only contrasts additionally include disease duration, LEDD, and HY. ELA sensitivity models additionally include UPDRS III.
- ELA Bonferroni families: 10 omnibus tests, 30 pairwise contrasts, and a separate 10-test UPDRS III sensitivity family.
- Global graph Bonferroni families: 12 omnibus tests and 36 pairwise contrasts.
- State–clinical correlations: sex, age, education, MoCA; sensitivity additionally includes UPDRS III. BH-FDR across 15 comparisons separately per model. The five source ELA metrics exclude indirect transition.
- Pooled-PD state–graph correlations: age, sex, education, mean FD, disease duration, HY, MoCA, LEDD; BH-FDR across three metrics.
- Treatment–behavior correlations: sex, age, education, MoCA; sensitivity additionally includes baseline UPDRS III. BH-FDR across three outcomes separately per model. Plot annotations retain raw p values.
- Treatment brain pre/post comparisons: paired tests without covariates or multiplicity correction; mean changes and 95% confidence intervals are retained.
- fNIRS phase comparisons: uncorrected Friedman omnibus tests and Bonferroni correction across six phase pairs separately per endpoint.
- fNIRS correlations: sex, age, education, baseline MoCA; two primary occupancy/dwell-time–TF% tests are uncorrected, and six exploratory correlations use BH-FDR. No UPDRS III sensitivity model is added.

## State labels and model inputs

The two sequence scripts retain the mapping executed in the supplied code: 1 → Minor state 1; 2 and 4 → Major state 1; 3 and 5 → Major state 2; 6 → Minor state 2. Basin labels depend on the fitted landscape; align this mapping with the basin assignments for the input group before processing another landscape. The contradictory example/comment mapping in the original duration script is not included. Changing this mapping changes the analysis.

Probability MAT files provide `empirical_probabilities` or `model_probabilities`. Group model MAT files provide an energy vector `E` in a consistent state order across groups. Their correlations and probability-fit statistics are uncorrected, as in the supplied code.
