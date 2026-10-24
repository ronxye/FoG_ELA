% Pairwise comparisons of group energy vectors.
clear;
clc;
close all;
addpath(fileparts(mfilename('fullpath')));
cfg = submission_paths();
input_dir = fullfile(cfg.data,'energy_model');
output_dir = fullfile(cfg.results,'energy_model');
if ~exist(output_dir,'dir'), mkdir(output_dir); end

color_HC           = [239/255, 118/255, 122/255];
color_PD_noFOG     = [69/255, 105/255, 144/255];
color_PD_FOG       = [72/255, 192/255, 170/255];

if ~exist(fullfile(input_dir,'model_probabilities_HC.mat'), 'file')
    error('Required input is missing or incompatible; see README.md.');
end
load(fullfile(input_dir,'model_probabilities_HC.mat'), 'E');
E_HC = E;

if ~exist(fullfile(input_dir,'model_probabilities_PD_noFOG.mat'), 'file')
    error('Required input is missing or incompatible; see README.md.');
end
load(fullfile(input_dir,'model_probabilities_PD_noFOG.mat'), 'E');
E_PD_noFOG = E;

if ~exist(fullfile(input_dir,'model_probabilities_PD_FOG.mat'), 'file')
    error('Required input is missing or incompatible; see README.md.');
end
load(fullfile(input_dir,'model_probabilities_PD_FOG.mat'), 'E');
E_PD_FOG = E;

if length(E_HC) ~= length(E_PD_noFOG) || length(E_HC) ~= length(E_PD_FOG)
    error('Required input is missing or incompatible; see README.md.');
end

[r_HC_noFOG, p_HC_noFOG] = corr(E_HC, E_PD_noFOG);
[r_HC_FOG, p_HC_FOG] = corr(E_HC, E_PD_FOG);
[r_noFOG_FOG, p_noFOG_FOG] = corr(E_PD_noFOG, E_PD_FOG);

fprintf('  HC vs PD_noFOG:    r = %.3f, p = %g\n', r_HC_noFOG, p_HC_noFOG);
fprintf('  HC vs PD_FOG:      r = %.3f, p = %g\n', r_HC_FOG, p_HC_FOG);
fprintf('  PD_noFOG vs PD_FOG: r = %.3f, p = %g\n\n', r_noFOG_FOG, p_noFOG_FOG);

figure('Color', 'white');
hold on;

z_min = min(E_PD_FOG) - 0.5;
x_min = min(E_HC) - 0.5;
y_max = max(E_PD_noFOG) + 0.5;

h1 = plot3(E_HC, E_PD_noFOG, ones(size(E_HC))*z_min, 'o', ...
    'MarkerEdgeColor', color_PD_noFOG, 'MarkerSize', 5);
p_fit1 = polyfit(E_HC, E_PD_noFOG, 1);
y_fit1 = polyval(p_fit1, sort(E_HC));
plot3(sort(E_HC), y_fit1, ones(size(E_HC))*z_min, 'k--', 'LineWidth', 1);

h2 = plot3(ones(size(E_PD_noFOG))*x_min, E_PD_noFOG, E_PD_FOG, 'o', ...
    'MarkerEdgeColor', color_PD_FOG, 'MarkerSize', 5);
p_fit2 = polyfit(E_PD_noFOG, E_PD_FOG, 1);
z_fit2 = polyval(p_fit2, sort(E_PD_noFOG));
plot3(ones(size(E_PD_noFOG))*x_min, sort(E_PD_noFOG), z_fit2, 'k--', 'LineWidth', 1);

h3 = plot3(E_HC, ones(size(E_HC))*y_max, E_PD_FOG, 'o', ...
    'MarkerEdgeColor', color_HC, 'MarkerSize', 5);
p_fit3 = polyfit(E_HC, E_PD_FOG, 1);
z_fit3 = polyval(p_fit3, sort(E_HC));
plot3(sort(E_HC), ones(size(E_HC))*y_max, z_fit3, 'k--', 'LineWidth', 1);

hold off;

grid on;
box on;
set(gca, 'FontSize', 12, 'LineWidth', 1);
view(135, 30);
xlabel('Energy value of HC', 'FontSize', 14);
ylabel('Energy value of PD-noFOG', 'FontSize', 14);
zlabel('Energy value of PD-FOG', 'FontSize', 14);

p_str_HC_noFOG = 'p < 0.001'; if p_HC_noFOG >= 0.0001, p_str_HC_noFOG = sprintf('p = %.4f', p_HC_noFOG); end
p_str_HC_FOG = 'p < 0.001'; if p_HC_FOG >= 0.0001, p_str_HC_FOG = sprintf('p = %.4f', p_HC_FOG); end
p_str_noFOG_FOG = 'p < 0.001'; if p_noFOG_FOG >= 0.0001, p_str_noFOG_FOG = sprintf('p = %.4f', p_noFOG_FOG); end

text_HC_noFOG = sprintf('r = %.3f\n%s', r_HC_noFOG, p_str_HC_noFOG);
text_HC_FOG = sprintf('r = %.3f\n%s', r_HC_FOG, p_str_HC_FOG);
text_noFOG_FOG = sprintf('r = %.3f\n%s', r_noFOG_FOG, p_str_noFOG_FOG);

text(min(E_HC), max(E_PD_noFOG)*0.8, z_min, text_HC_noFOG, 'FontSize', 12, 'FontWeight','normal');

text(x_min, min(E_PD_noFOG), max(E_PD_FOG)*0.9, text_noFOG_FOG, 'FontSize', 12, 'FontWeight','normal');

text(min(E_HC), y_max, max(E_PD_FOG)*0.9, text_HC_FOG, 'FontSize', 12, 'FontWeight','normal');

print(gcf,fullfile(output_dir,'group_energy_comparison.png'),'-dpng','-r300');
savefig(gcf,fullfile(output_dir,'group_energy_comparison.fig'));
Comparison = {'HC vs PD-nonFOG';'HC vs PD-FOG';'PD-nonFOG vs PD-FOG'};
r = [r_HC_noFOG;r_HC_FOG;r_noFOG_FOG];
p = [p_HC_noFOG;p_HC_FOG;p_noFOG_FOG];
writetable(table(Comparison,r,p),fullfile(output_dir,'group_energy_correlations.csv'));
