% Target-state occupancy and dwell time across four fNIRS phases.
clear; clc; close all;

FS = 11;
WINDOW_SECONDS = 10;
ANALYSIS_NAMES = {'Comprehensive14_strict10s'};
MAX_TRIALS = 3;
REQUIRED_SECONDS = max(WINDOW_SECONDS);

networks.SMN = [14,15,17,28,30,32,33,36,38,44,46,47,49,53,54,55,56,57];
networks.DMN = [3,6,10,21,22,23,29,39,58];
networks.VN = [42,61];
networks.FPN = [1,5,9,11,13,18,19,26,27,34,35,37,51];
networks.SAN = [7,20,24,25];
networks.ATN = [4,12,16,31,40,41];
target_states = [
     1 -1 -1  1 -1  1;
     1 -1 -1  1  1  1;
     1  1 -1 -1  1  1;
     1 -1 -1  1 -1 -1;
     1 -1 -1  1  1 -1;
     1  1 -1  1  1 -1;
    -1 -1 -1  1 -1  1;
     1  1 -1  1 -1  1;
    -1 -1 -1  1  1  1;
     1  1 -1  1  1  1;
    -1 -1  1  1 -1  1;
     1 -1 -1 -1  1  1;
    -1 -1  1  1  1  1;
     1 -1  1  1  1  1
];

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
cfg = submission_paths();
task_dir = fullfile(cfg.data,'fnirs_task');
rest_dir = fullfile(cfg.data,'fnirs_rest');
output_root = fullfile(cfg.results,'fnirs_state_dynamics');
if ~exist(output_root,'dir'), mkdir(output_root); end
task_files = dir(fullfile(task_dir, '*.nirs'));
rest_files = dir(fullfile(rest_dir, '*.nirs'));
if isempty(task_files) || isempty(rest_files)
    error('Task or resting-state directory contains no .nirs files.');
end
task_names = {task_files.name};
rest_names = {rest_files.name};
if numel(task_names) ~= numel(rest_names) || ~all(ismember(task_names,rest_names))
    error('Task and rest inputs require identical anonymized filenames, e.g. P001.nirs.');
end
[~, rest_order] = ismember(task_names,rest_names);
rest_files = rest_files(rest_order);
n_files = numel(task_files);

pattern_tiers=[repmat({'Strict'},3,1);repmat({'Practical'},7,1);repmat({'Ambiguous'},4,1)];
pattern_table=[table((1:14)',pattern_tiers,'VariableNames',{'Pattern','Tier'}), ...
    array2table(target_states,'VariableNames',{'SMN','DMN','VN','FPN','SAN','ATN'})];
writetable(pattern_table,fullfile(output_root,'target_state_patterns.csv'));

file_pairs = table({task_files.name}', {rest_files.name}', ...
    'VariableNames', {'Task_File','Rest_File'});
writetable(file_pairs, fullfile(output_root, 'task_rest_file_pairs.csv'));

subject_list = {};
rest_values = [];
duration_values = [];
trial_counts = [];
selected_trial_labels = {};
included_trial_labels = {};
threshold_values = [];
task_occ = cell(1, numel(WINDOW_SECONDS));
task_dt = cell(1, numel(WINDOW_SECONDS));
audit_cells = cell(0, 18);
excluded_cells = cell(0, 2);
segment_cells = cell(0,12);

for f = 1:n_files
    [~, sub_id] = fileparts(task_files(f).name);
    selected_indices = 1:MAX_TRIALS;
    fprintf('Processing %d/%d: %s\n', f, n_files, sub_id);
    try
        rest_ts = load_network_hbo(fullfile(rest_dir, rest_files(f).name), networks);
        if any(~isfinite(rest_ts(:)))
            error('Resting network time series contains non-finite values.');
        end
        thresholds = mean(rest_ts, 1);
        rest_state = identify_state(rest_ts, thresholds, target_states);
        [rest_occ, rest_dt, rest_bout_count] = calc_metrics(rest_state, FS);
        [occ, dt, durations, trial_audit, n_valid, included_indices, segment_rows] = process_task_file( ...
            fullfile(task_dir, task_files(f).name), networks, thresholds, ...
            target_states, FS, WINDOW_SECONDS, REQUIRED_SECONDS, selected_indices);
        for a = 1:size(trial_audit, 1)
            audit_cells(end+1, :) = [{sub_id}, trial_audit(a,:)];
        end
        if n_valid == 0
            excluded_cells(end+1,:) = {sub_id,'No complete trial with full 10-s pre/post windows'};
            continue;
        end
        segment_cells(end+1,:) = {sub_id,0,'Resting',1,numel(rest_state), ...
            numel(rest_state)/FS,rest_occ*100,rest_dt,rest_bout_count, ...
            rest_state(1)==1,rest_state(end)==1,rest_bout_count==0};
        for a=1:size(segment_rows,1)
            segment_cells(end+1,:)=[{sub_id},segment_rows(a,:)];
        end
        subject_list{end+1,1} = sub_id;
        rest_values(end+1,:) = [rest_occ,rest_dt];
        duration_values(end+1,:) = durations;
        trial_counts(end+1,1) = n_valid;
        selected_trial_labels{end+1,1} = mat2str(selected_indices);
        included_trial_labels{end+1,1} = mat2str(included_indices);
        threshold_values(end+1,:) = thresholds;
        for w = 1:numel(WINDOW_SECONDS)
            task_occ{w}(end+1,:) = occ(w,:);
            task_dt{w}(end+1,:) = dt(w,:);
        end
    catch ME
        warning('Subject excluded: %s. %s', sub_id, ME.message);
        excluded_cells(end+1,:) = {sub_id,ME.message};
    end
end

audit_names = {'Subject_ID','Trial','Task_Start_Frame','Turn_Start_Frame', ...
    'Turn_End_Frame','Task_End_Frame','Available_Pre_s','Turning_s', ...
    'Available_Post_s','Selected_by_policy','Passes_10s_Criteria', ...
    'Included_10s','Included_Original_Window_10s', ...
    'Pre_10s_Start_Frame','Pre_10s_End_Frame','Post_10s_Start_Frame', ...
    'Post_10s_End_Frame','Reason'};
trial_audit = cell2table(audit_cells, 'VariableNames', audit_names);
segment_audit = cell2table(segment_cells,'VariableNames', ...
    {'Subject_ID','Trial','Phase','Start_Frame','End_Frame','Segment_Duration_s', ...
    'Occupancy_Pct','Window_Dwell_s','Window_Bout_Count','Starts_In_State', ...
    'Ends_In_State','No_State_Run_Zero_Dwell'});
excluded_subjects = cell2table(excluded_cells, ...
    'VariableNames', {'Subject_ID','Reason'});
writetable(trial_audit, fullfile(output_root, 'trial_window_audit.csv'));
writetable(excluded_subjects, fullfile(output_root, 'excluded_subjects.csv'));
writetable(segment_audit,fullfile(output_root,'segment_trial_metrics.csv'));
if numel(subject_list) < 2
    error('Fewer than two eligible subjects. Review trial_window_audit.csv.');
end

all_subject_data = table();
all_descriptive = table();
all_omnibus = table();
all_posthoc = table();
results = struct();
for w = 1:numel(WINDOW_SECONDS)
    analysis = ANALYSIS_NAMES{w};
    output_dir = output_root;
    if ~exist(output_dir,'dir'), mkdir(output_dir); end
    group_occupancy = [rest_values(:,1),task_occ{w}] * 100;
    group_dwelling = [rest_values(:,2),task_dt{w}];
    numerical_data = [duration_values,group_occupancy,group_dwelling];
    numeric_names = {'Mean_Turning_Duration_s','Mean_Before_Turning_Duration_s', ...
        'Mean_After_Turning_Duration_s','Occupancy_Resting_Pct', ...
        'Occupancy_Before_Turning_Pct','Occupancy_Turning_Pct', ...
        'Occupancy_After_Turning_Pct','DwellingTime_Resting_s', ...
        'DwellingTime_Before_Turning_s','DwellingTime_Turning_s', ...
        'DwellingTime_After_Turning_s'};
    subject_table = [table(subject_list,'VariableNames',{'Subject_ID'}), ...
        array2table(numerical_data,'VariableNames',numeric_names)];
    subject_table.Valid_Trials = trial_counts;
    subject_table.Selected_Trial_Indices = selected_trial_labels;
    subject_table.Included_Trial_Indices = included_trial_labels;
    subject_table.Window_s = repmat(WINDOW_SECONDS(w),numel(subject_list),1);
    subject_table.Actual_Pre_Window_s = subject_table.Window_s;
    subject_table.Actual_Post_Window_s = subject_table.Window_s;

    summary_table = [table({'Group_Mean';'Group_SD'},'VariableNames',{'Subject_ID'}), ...
        array2table([mean(numerical_data,1);std(numerical_data,0,1)], ...
        'VariableNames',numeric_names)];
    descriptive = [describe_segments(group_occupancy,'Occupancy','%'); ...
        describe_segments(group_dwelling,'Dwell_time','s')];
    [omni_occ,pair_occ] = perform_stats_4seg(group_occupancy,'Occupancy');
    [omni_dt,pair_dt] = perform_stats_4seg(group_dwelling,'Dwell_time');
    omnibus = [omni_occ;omni_dt];
    posthoc = [pair_occ;pair_dt];

    writetable(subject_table,fullfile(output_dir,'master_minor_state2_4segments_summary.csv'));
    writetable(summary_table,fullfile(output_dir,'group_summary.csv'));
    writetable(descriptive,fullfile(output_dir,'descriptive_statistics.csv'));
    writetable(omnibus,fullfile(output_dir,'friedman_results.csv'));
    writetable(posthoc,fullfile(output_dir,'posthoc_results.csv'));
    plot_4seg_line_chart_nature(group_occupancy,pair_occ.P_Bonferroni, ...
        'Occupancy of target brain state (%)',fullfile(output_dir,'Minor_State2_Occupancy_Rate_4Seg_Trend.png'));
    plot_4seg_line_chart_nature(group_dwelling,pair_dt.P_Bonferroni, ...
        'Dwell time of target brain state (s)',fullfile(output_dir,'Minor_State2_Dwelling_Time_4Seg_Trend.png'));

    results.(analysis).subject_data = subject_table;
    results.(analysis).descriptive = descriptive;
    results.(analysis).omnibus = omnibus;
    results.(analysis).posthoc = posthoc;
    if w == 1
        all_subject_data = with_analysis(subject_table,analysis);
        all_descriptive = with_analysis(descriptive,analysis);
        all_omnibus = with_analysis(omnibus,analysis);
        all_posthoc = with_analysis(posthoc,analysis);
    else
        all_subject_data = [all_subject_data;with_analysis(subject_table,analysis)];
        all_descriptive = [all_descriptive;with_analysis(descriptive,analysis)];
        all_omnibus = [all_omnibus;with_analysis(omnibus,analysis)];
        all_posthoc = [all_posthoc;with_analysis(posthoc,analysis)];
    end
    fprintf('%s: N=%d; valid trials=%d\n',analysis,numel(subject_list),sum(trial_counts));
    disp(omnibus);
end
save(fullfile(output_root,'comprehensive14_strict10s_analysis.mat'), ...
    'results','FS','WINDOW_SECONDS','ANALYSIS_NAMES','MAX_TRIALS', ...
    'REQUIRED_SECONDS','trial_audit','excluded_subjects','file_pairs', ...
    'threshold_values','networks','target_states','segment_audit');
fprintf('Results saved to: %s\n',output_root);

function [net_ts,mat_data] = load_network_hbo(file_path,networks)
    mat_data = load(file_path,'-mat');
    if ~isfield(mat_data,'procResult'), error('Missing procResult.'); end
    pRes = mat_data.procResult;
    if iscell(pRes), pRes = pRes{1}; end
    if ~((isstruct(pRes) && isfield(pRes,'dc')) || (isobject(pRes) && isprop(pRes,'dc')))
        error('Missing procResult.dc.');
    end
    dc = pRes.dc;
    if iscell(dc), dc = dc{1}; end
    if ~isnumeric(dc), dc = dc.data; end
    if ndims(dc) == 3, hbo = squeeze(dc(:,1,:)); else, hbo = dc; end
    if isempty(hbo) || ~isnumeric(hbo), error('No numeric HbO data.'); end
    names = fieldnames(networks);
    net_ts = zeros(size(hbo,1),numel(names));
    for k = 1:numel(names)
        channels = networks.(names{k});
        channels(channels > size(hbo,2)) = [];
        if isempty(channels), error('No available channels for %s.',names{k}); end
        net_ts(:,k) = mean(hbo(:,channels),2);
    end
end

function state = identify_state(net_ts,thresholds,target_states)
    binary = 2*double(bsxfun(@gt,net_ts,thresholds))-1;
    state = double(ismember(binary,target_states,'rows'));
    state(any(~isfinite(net_ts),2)) = NaN;
end

function [occ,dt,durations,audit,n_valid,included_indices,segment_rows] = process_task_file( ...
        file_path,networks,thresholds,target_states,fs,windows,required_s,selected_indices)
    [net_ts,mat_data] = load_network_hbo(file_path,networks);
    state = identify_state(net_ts,thresholds,target_states);
    if ~isfield(mat_data,'s'), error('Missing event-marker matrix s.'); end
    s = mat_data.s;
    if iscell(s), s = s{1}; end
    if size(s,2) < 4 || size(s,1) ~= numel(state)
        error('Event markers must have four columns and match the HbO sample count.');
    end
    onsets = cell(1,4);
    for k = 1:4
        onsets{k} = find(diff([false;s(:,k)>0]) == 1);
    end
    audit = cell(0,17);
    trial_occ = cell(1,numel(windows));
    trial_dt = cell(1,numel(windows));
    trial_durations = [];
    required_frames = round(required_s*fs);
    n_valid = 0;
    included_indices = [];
    segment_rows=cell(0,11);
    for tr = 1:numel(onsets{1})
        a = onsets{1}(tr);
        next_start = numel(state)+1;
        if tr < numel(onsets{1}), next_start = onsets{1}(tr+1); end
        b = NaN; c = NaN; d = NaN;
        markers_ok = true;
        for k = 2:4
            events = onsets{k}(onsets{k}>a & onsets{k}<next_start);
            if numel(events) ~= 1, markers_ok = false; break; end
            if k == 2, b = events; elseif k == 3, c = events; else, d = events; end
        end
        pre = (b-a)/fs; turn = (c-b)/fs; post = (d-c)/fs;
        eligible = false;
        selected = ismember(tr,selected_indices);
        criteria_passed = false;
        reason = 'Included in original window-restricted 10-s analysis';
        indices = [NaN,NaN,NaN,NaN];
        if ~markers_ok || ~(a<b && b<c && c<d)
            reason = 'Missing, repeated, or out-of-order markers';
        elseif (b-a)<required_frames || (d-c)<required_frames
            reason = 'Less than 10 s available before or after turning';
        elseif any(~isfinite(state(b-required_frames:c+required_frames-1)))
            reason = 'Non-finite network signal in required analysis segments';
        else
            criteria_passed = true;
            indices = [b-required_frames,b-1,c,c+required_frames-1];
        end
        if ~selected
            reason = ['Not selected by trial policy; ',reason];
            if criteria_passed, reason = 'Not selected by trial policy; passes 10-s criteria'; end
        elseif criteria_passed
            eligible = true;
            n_valid = n_valid+1;
            included_indices(end+1) = tr;
            trial_durations(end+1,:) = [turn,pre,post];
            for w = 1:numel(windows)
                frames = round(windows(w)*fs);
                segments = {state(b-frames:b-1),state(b:c-1),state(c:c+frames-1)};
                this_occ = zeros(1,3); this_dt = zeros(1,3);
                for seg = 1:3
                    [this_occ(seg),this_dt(seg),bout_count] = calc_metrics(segments{seg},fs);
                    phases={'Pre_turn','Turning','Post_turn'};
                    starts=[b-frames,b,c];stops=[b-1,c-1,c+frames-1];
                    v=segments{seg};
                    segment_rows(end+1,:)={tr,phases{seg},starts(seg),stops(seg), ...
                        numel(v)/fs,this_occ(seg)*100,this_dt(seg),bout_count, ...
                        v(1)==1,v(end)==1,bout_count==0};
                end
                trial_occ{w}(end+1,:) = this_occ;
                trial_dt{w}(end+1,:) = this_dt;
            end
        end
        audit(end+1,:) = {tr,a,b,c,d,pre,turn,post,selected,criteria_passed,eligible,eligible, ...
            indices(1),indices(2),indices(3),indices(4),reason};
    end
    occ = NaN(numel(windows),3); dt = occ; durations = NaN(1,3);
    if n_valid > 0
        for w = 1:numel(windows)
            occ(w,:) = mean(trial_occ{w},1);
            dt(w,:) = mean(trial_dt{w},1);
        end
        durations = mean(trial_durations,1);
    end
end

function [occ,dt,n_bouts] = calc_metrics(sig,fs)
    sig = sig(:);
    n_bouts=NaN;
    if isempty(sig) || any(~isfinite(sig)), occ = NaN; dt = NaN; return; end
    occ = mean(sig==1);
    delta = diff([0;sig;0]);
    lengths = find(delta==-1)-find(delta==1);
    n_bouts=numel(lengths);
    if isempty(lengths), dt = 0; else, dt = mean(lengths)/fs; end
end

function t = describe_segments(data,metric,unit)
    phases = {'Resting';'Pre_turn';'Turning';'Post_turn'};
    t = table(repmat({metric},4,1),phases,repmat({unit},4,1), ...
        repmat(size(data,1),4,1),mean(data,1)',std(data,0,1)', ...
        median(data,1)',prctile(data,25,1)',prctile(data,75,1)', ...
        'VariableNames',{'Metric','Phase','Unit','N','Mean','SD','Median','Q1','Q3'});
end

function [omnibus,posthoc] = perform_stats_4seg(data,metric)
    complete = all(isfinite(data),2);
    data = data(complete,:);
    n = size(data,1);
    p = NaN; chi2 = NaN;
    if n >= 2
        [p,tbl] = friedman(data,1,'off');
        chi2 = tbl{2,5};
    end
    omnibus = table({metric},{'Friedman'},n,chi2,3,p,chi2/(n*3), ...
        'VariableNames',{'Metric','Test','N','Chi_square','DF','P_raw','Kendalls_W'});
    pairs = [1 2;2 3;3 4;1 3;1 4;2 4];
    phases = {'Resting','Pre_turn','Turning','Post_turn'};
    rows = cell(6,10);
    for k = 1:6
        x = data(:,pairs(k,1)); y = data(:,pairs(k,2));
        raw = NaN; signed_rank = NaN; z = NaN;
        if n >= 2
            if all(x==y)
                raw = 1; signed_rank = 0;
            else
                [raw,~,stats] = signrank(x,y);
                signed_rank = stats.signedrank;
                if isfield(stats,'zval'), z = stats.zval; end
            end
        end
        adjusted = NaN;
        if isfinite(raw), adjusted = min(raw*6,1); end
        rows(k,:) = {metric,phases{pairs(k,1)},phases{pairs(k,2)},n, ...
            sum(x~=y),signed_rank,z,raw,adjusted,6};
    end
    posthoc = cell2table(rows,'VariableNames',{'Metric','Phase_1','Phase_2', ...
        'N_pairs','N_nonzero','Signed_rank','Z','P_raw','P_Bonferroni','Correction_family_size'});
end

function t = with_analysis(t,analysis)
    t = [table(repmat({analysis},height(t),1),'VariableNames',{'Analysis'}),t];
end

function plot_4seg_line_chart_nature(data_matrix, p_pairs_adj, y_label_str, save_path)
    fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [1, 1, 4, 3]);
    ax = axes(fig);
    hold(ax, 'on');

    x_coords = 1:4;
    num_subs = size(data_matrix, 1);

    c_individual = [0.65, 0.68, 0.72];
    c_mean       = [0.80, 0.18, 0.18];
    c_dark       = [0.15, 0.15, 0.15];

    for i = 1:num_subs
        plot(ax, x_coords, data_matrix(i, :), '-o', ...
            'Color', [c_individual, 0.20], ...
            'LineWidth', 0.5, ...
            'MarkerSize', 3.0, ...
            'MarkerFaceColor', c_individual, ...
            'MarkerEdgeColor', 'none');
    end

    grp_mean = mean(data_matrix, 1, 'omitnan');
    grp_se   = std(data_matrix, 0, 1, 'omitnan') / sqrt(num_subs);

    errorbar(ax, x_coords, grp_mean, grp_se, '-', ...
        'Color', c_mean, ...
        'LineWidth', 2.0, ...
        'CapSize', 3.5);

    plot(ax, x_coords, grp_mean, 'o', ...
        'MarkerSize', 6.0, ...
        'MarkerFaceColor', c_mean, ...
        'MarkerEdgeColor', 'w', ...
        'LineWidth', 1.0);

    set(ax, 'XTick', 1:4, ...
        'XTickLabel', {'Resting', 'Pre-turn', 'Turning', 'Post-turn'}, ...
        'FontName', 'Helvetica', ...
        'FontSize', 10, ...
        'LineWidth', 0.8, ...
        'XColor', c_dark, ...
        'YColor', c_dark, ...
        'Box', 'off', ...
        'TickDir', 'out', ...
        'YGrid', 'off');

    xlim(ax, [0.5, 4.5]);
    ylabel(ax, y_label_str, 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', c_dark);

    yl_data = [min(data_matrix(:)), max(data_matrix(:))];
    y_range = yl_data(2) - max(0, yl_data(1));
    y_high  = yl_data(2);

    h1 = y_high + y_range * 0.10;
    h2 = y_high + y_range * 0.24;
    h3 = y_high + y_range * 0.38;

    add_nature_sig_bracket(ax, 1, 3, h1, p_pairs_adj(4), y_range);
    add_nature_sig_bracket(ax, 2, 3, h2, p_pairs_adj(2), y_range);
    add_nature_sig_bracket(ax, 3, 4, h3, p_pairs_adj(3), y_range);

    ylim(ax, [min(0, yl_data(1)), y_high + y_range * 0.52]);

    set(fig, 'PaperUnits', 'inches', 'PaperPosition', [0, 0, 4, 3], 'PaperPositionMode', 'manual');
    print(fig, save_path, '-dpng', '-r300');
    close(fig);
end

function add_nature_sig_bracket(ax, x1, x2, y_pos, p_adj, y_range)
    if isnan(p_adj) || p_adj >= 0.05, return; end

    if p_adj < 0.001
        sig_str = '***';
    elseif p_adj < 0.01
        sig_str = '**';
    elseif p_adj < 0.05
        sig_str = '*';
    else
        sig_str = 'n.s.';
    end

    tick_h = y_range * 0.025;

    plot(ax, [x1, x1, x2, x2], [y_pos - tick_h, y_pos, y_pos, y_pos - tick_h], '-', ...
        'LineWidth', 0.75, 'Color', [0.2 0.2 0.2]);

    y_offset = -0.005 * y_range;

    f_size = 11;
    f_style = 'normal';
    f_weight = 'bold';

    text(ax, (x1 + x2)/2, y_pos + y_offset, sig_str, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontName', 'Helvetica', ...
        'FontSize', f_size, ...
        'FontAngle', f_style, ...
        'FontWeight', f_weight, ...
        'Color', [0.2 0.2 0.2]);
end
