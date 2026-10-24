% Empirical and modeled state probabilities.
clear;
clc;
close all;
addpath(fileparts(mfilename('fullpath')));
cfg = submission_paths();
input_dir = fullfile(cfg.data,'energy_model');
output_dir = fullfile(cfg.results,'energy_model');
if ~exist(output_dir,'dir'), mkdir(output_dir); end

empirical_data_file = fullfile(input_dir,'empirical_probabilities_result.mat');
if ~exist(empirical_data_file, 'file')
    error('Required input is missing or incompatible; see README.md.');
end
load(empirical_data_file, 'empirical_probabilities');

model_data_file = fullfile(input_dir,'model_probabilities_result.mat');
if ~exist(model_data_file, 'file')
    error('Required input is missing or incompatible; see README.md.');
end
load(model_data_file, 'model_probabilities');

valid_indices = (empirical_probabilities > 0) & (model_probabilities > 0);

x_data = empirical_probabilities(valid_indices);
y_data = model_probabilities(valid_indices);

greenishBlueColor = [239/255, 118/255, 122/255];

figure('Color', 'white', 'Position', [100 100 600 500]);

scatter(x_data, y_data, 30, 'o', ...
    'MarkerEdgeColor', greenishBlueColor, ...
    'LineWidth', 1.5);

hold on;

set(gca, 'XScale', 'log', 'YScale', 'log');

min_limit = 1e-4;
max_data_val = max([x_data; y_data]);
max_limit = ceil(log10(max_data_val));
max_limit = 10^(max_limit);
if max_limit < 1e-2
    max_limit = 1e-2;
end

axis_limits_fixed = [min_limit, max_limit];
xlim(axis_limits_fixed);
ylim(axis_limits_fixed);

plot(axis_limits_fixed, axis_limits_fixed, 'k-', 'LineWidth', 1.5);
hold off;

axis square;
grid off;
box off;

set(gca, ...
    'FontSize', 16, ...
    'FontName', 'Arial', ...
    'LineWidth', 1);

desired_ticks = [1e-4, 1e-3, 1e-2];
set(gca, 'XTick', desired_ticks, 'YTick', desired_ticks);
set(gca, 'XTickLabel', {'$10^{-4}$', '$10^{-3}$', '$10^{-2}$'});
set(gca, 'YTickLabel', {'$10^{-4}$', '$10^{-3}$', '$10^{-2}$'});
set(gca, 'TickLabelInterpreter', 'latex');

[r, p] = corr(x_data, y_data, 'Type', 'Pearson');

p_emp_valid_norm = x_data / sum(x_data);
p_model_valid_norm = y_data / sum(y_data);
epsilon = 1e-10;
p_emp_valid_norm(p_emp_valid_norm == 0) = epsilon;
p_model_valid_norm(p_model_valid_norm == 0) = epsilon;
D_KL = sum(p_emp_valid_norm .* log(p_emp_valid_norm ./ p_model_valid_norm));
S_emp = -sum(p_emp_valid_norm .* log(p_emp_valid_norm));
accuracy = 1 - (D_KL / S_emp);

fprintf('  Pearson Correlation (r) = %.3f\n', r);
fprintf('  p-value = %g\n', p);
fprintf('  Model Accuracy (KL-based) = %.4f (%.2f%%)\n\n', accuracy, accuracy*100);

if p < 0.001
    p_string = 'p < 0.001';
else
    p_string = sprintf('p = %.3f', p);
end

text_to_display = {
    sprintf('r = %.3f, %s', r, p_string),
    sprintf('Accuracy = %.1f%%', accuracy * 100)
};

x_lim = get(gca, 'XLim');
y_lim = get(gca, 'YLim');
x_text_pos = x_lim(1) * 1.5;
y_text_pos = y_lim(2) * 0.7;

text(x_text_pos, y_text_pos, text_to_display, ...
    'FontSize', 16, ...
    'FontName', 'Arial', ...
    'EdgeColor', 'k', ...
    'BackgroundColor', 'w', ...
    'VerticalAlignment', 'top', ...
    'HorizontalAlignment', 'left');

print(gcf,fullfile(output_dir,'state_probability_fit.png'),'-dpng','-r300');
savefig(gcf,fullfile(output_dir,'state_probability_fit.fig'));
fit_statistics = table(r,p,D_KL,S_emp,accuracy);
writetable(fit_statistics,fullfile(output_dir,'state_probability_fit_statistics.csv'));
