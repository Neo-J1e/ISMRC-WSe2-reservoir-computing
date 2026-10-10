%% Empirical device model: reproducibility audit
% This script is a read-only reconstruction of the model implemented in
% single_device_model_no_mask_diff_tao.mlx. It never writes to data/.
% Derived tables and figures are written to outputs/.

clear; clc; close all;
rng(20260907, 'twister');

package_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(package_dir, 'data');
output_dir = fullfile(package_dir, 'outputs');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

parameter_file = fullfile(data_dir, 'single_device_model_data.mat');
single_pulse_file = fullfile(data_dir, 'fig2_c.mat');
classification_input_file = fullfile(data_dir, 'wave_classification_laser.txt');
classification_measured_file = fullfile(data_dir, 'chip1-classification-20241222.mat');

required_files = {parameter_file, single_pulse_file, classification_input_file, ...
    classification_measured_file};
for i = 1:numel(required_files)
    assert(isfile(required_files{i}), 'Missing required file: %s', required_files{i});
end

P = load(parameter_file);

%% Complete numerical parameter tables
currents_mA = P.I_laser(:);
current_table = table(currents_mA, ...
    P.p_a_all(:), P.p_b_all(:), P.pd_a_all(:), P.pd_b_all(:), ...
    P.n_a_all(:), P.n_b_all(:), P.nd_a_all(:), P.nd_b_all(:), ...
    'VariableNames', {'current_mA','Aplus_fitpoint_V','tau_plus_fitpoint_s', ...
    'decay_amp_plus_V','tau_off_plus_s','Aminus_fitpoint_V', ...
    'tau_minus_fitpoint_s','decay_amp_minus_V','tau_off_minus_s'});
writetable(current_table, fullfile(output_dir, 'per_current_parameter_table.csv'));

models = {'A_plus'; 'tau_plus'; 'A_minus'; 'tau_minus'};
fit_objects = {P.p_fitresult_a; P.p_fitresult_b; P.n_fitresult_a; P.n_fitresult_b};
fit_y = {P.p_a_all(:); P.p_b_all(:); P.n_a_all(:); P.n_b_all(:)};
functional_forms = {'a*exp(b*I)+c*exp(d*I)'; 'a*exp(b*I)'; ...
    'a*exp(b*I)+c*exp(d*I)'; 'a*exp(b*I)'};

coefficient_rows = {};
fit_summary_rows = {};
fit_residual_rows = {};
for m = 1:numel(models)
    fit_object = fit_objects{m};
    coefficient_names = coeffnames(fit_object);
    coefficient_values = coeffvalues(fit_object);
    confidence_interval = confint(fit_object, 0.95);
    for c = 1:numel(coefficient_names)
        coefficient_rows(end+1,:) = {models{m}, functional_forms{m}, ...
            coefficient_names{c}, coefficient_values(c), ...
            confidence_interval(1,c), confidence_interval(2,c)}; %#ok<SAGROW>
    end

    observed = fit_y{m};
    predicted = feval(fit_object, currents_mA);
    residual = observed - predicted;
    n = numel(observed);
    p = numel(coefficient_names);
    sse = sum(residual.^2);
    sst = sum((observed - mean(observed)).^2);
    r_squared = 1 - sse/sst;
    adjusted_r_squared = 1 - (1-r_squared)*(n-1)/(n-p-1);
    rmse = sqrt(sse/(n-p));
    mae = mean(abs(residual));
    fit_summary_rows(end+1,:) = {models{m}, functional_forms{m}, n, p, ...
        sse, r_squared, adjusted_r_squared, rmse, mae, mean(residual)}; %#ok<SAGROW>
    for j = 1:n
        fit_residual_rows(end+1,:) = {models{m}, currents_mA(j), ...
            observed(j), predicted(j), residual(j)}; %#ok<SAGROW>
    end
end

coefficient_table = cell2table(coefficient_rows, 'VariableNames', ...
    {'model','functional_form','coefficient','estimate','ci95_low','ci95_high'});
writetable(coefficient_table, fullfile(output_dir, 'functional_coefficients_and_ci.csv'));

fit_summary_table = cell2table(fit_summary_rows, 'VariableNames', ...
    {'model','functional_form','n','n_coefficients','sse','r_squared', ...
    'adjusted_r_squared','rmse','mae','mean_residual'});
writetable(fit_summary_table, fullfile(output_dir, 'functional_fit_goodness.csv'));

fit_residual_table = cell2table(fit_residual_rows, 'VariableNames', ...
    {'model','current_mA','observed','predicted','residual_observed_minus_predicted'});
writetable(fit_residual_table, fullfile(output_dir, 'functional_fit_residuals.csv'));

off_state_table = table(P.pd_b_mean, P.nd_b_mean, ...
    mean(P.pd_b_all), std(P.pd_b_all), mean(P.nd_b_all), std(P.nd_b_all), ...
    'VariableNames', {'tau_off_plus_used_s','tau_off_minus_used_s', ...
    'tau_off_plus_mean_s','tau_off_plus_sd_s','tau_off_minus_mean_s','tau_off_minus_sd_s'});
writetable(off_state_table, fullfile(output_dir, 'off_state_parameters.csv'));

%% First-stage transient fits: diagnostics and confidence intervals
S = load(single_pulse_file);
all_data = S.all_data;
first_stage_rows = {};
first_stage_residual_rows = {};
for polarity_index = 1:2
    if polarity_index == 1
        polarity = 'positive';
        traces = all_data.p_voc;
        response_start = P.p_param_value_all;
        decay_start = P.pd_param_value_all;
    else
        polarity = 'negative';
        traces = all_data.n_voc;
        response_start = P.n_param_value_all;
        decay_start = P.nd_param_value_all;
    end

    for current_index = 1:numel(currents_mA)
        source_trace = traces{current_index + 1};

        response_raw = source_trace(2003:3000);
        response_fit_target = smooth(response_raw);
        response_time_s = (1:numel(response_fit_target))' * 20e-3;
        response_type = fittype('a*(1-exp(-x/b))+c', ...
            'coefficients', {'a','b','c'}, 'independent', 'x');
        response_model = fit(response_time_s, response_fit_target, response_type, ...
            'Lower', [-10,-10,0], 'Upper', [10,10,0], ...
            'StartPoint', response_start(current_index,:));
        [first_stage_rows, first_stage_residual_rows] = add_first_stage_fit( ...
            first_stage_rows, first_stage_residual_rows, polarity, currents_mA(current_index), ...
            'photoresponse', 'a*(1-exp(-t/b))+c', response_time_s, ...
            response_fit_target, response_model);

        decay_raw = source_trace(3001:6000);
        decay_fit_target = smooth(decay_raw);
        decay_time_s = (1:numel(decay_fit_target))' * 20e-3;
        decay_type = fittype('a*exp(-x/b)', ...
            'coefficients', {'a','b'}, 'independent', 'x');
        decay_model = fit(decay_time_s, decay_fit_target, decay_type, ...
            'StartPoint', decay_start(current_index,:));
        [first_stage_rows, first_stage_residual_rows] = add_first_stage_fit( ...
            first_stage_rows, first_stage_residual_rows, polarity, currents_mA(current_index), ...
            'dark_decay', 'a*exp(-t/b)', decay_time_s, decay_fit_target, decay_model);
    end
end

first_stage_table = cell2table(first_stage_rows, 'VariableNames', ...
    {'polarity','current_mA','segment','functional_form','n','coefficient', ...
    'estimate','ci95_low','ci95_high','sse','r_squared','adjusted_r_squared', ...
    'rmse','mae','mean_residual'});
writetable(first_stage_table, fullfile(output_dir, 'single_pulse_fit_parameters_and_diagnostics.csv'));

first_stage_residual_table = cell2table(first_stage_residual_rows, 'VariableNames', ...
    {'polarity','current_mA','segment','time_s','observed_smoothed_V', ...
    'predicted_V','residual_observed_minus_predicted_V'});
writetable(first_stage_residual_table, fullfile(output_dir, 'single_pulse_fit_residuals.csv'));

%% Exact model constants and channel coefficients
model_constants = table(0.2, 40, 37, 6, 1/0.3, 0.3, 20260907, ...
    'VariableNames', {'state_step_s','activation_threshold_mA','input_offset_mA', ...
    'input_scale_mA','switch_target_gain','switch_denominator', ...
    'audit_random_seed'});
writetable(model_constants, fullfile(output_dir, 'model_constants.csv'));

channel_delays = (-7:2)';
channel_taps = [0.01,0.03,0.04,-0.05,0.091,-0.10,0.18,1.00,-0.12,0.08]';
channel_table = table(channel_delays, channel_taps, ...
    'VariableNames', {'relative_symbol_delay','linear_coefficient'});
writetable(channel_table, fullfile(output_dir, 'channel_coefficients.csv'));
nonlinear_channel_table = table(1, 0.03, -0.011, ...
    'VariableNames', {'q_coefficient','q_squared_coefficient','q_cubed_coefficient'});
writetable(nonlinear_channel_table, fullfile(output_dir, 'channel_nonlinearity.csv'));

mask_table = build_mask_table();
writetable(mask_table, fullfile(output_dir, 'mask_definitions.csv'));

%% Out-of-fit-domain continuous-sequence validation against measured states
% The single-pulse parameters were fitted using fig2_c.mat. The continuous
% classification record is a separate acquisition from the single-pulse fits.
classification_input = readmatrix(classification_input_file);
classification_measured = load(classification_measured_file);
[classification_prediction, masks] = simulate_smrc_sequence( ...
    classification_input, P, 0.2, 0.0);
[classification_metrics, classification_residuals] = validation_metrics( ...
    'classification_20241222', classification_measured.data_all, ...
    classification_prediction, masks);
writetable(classification_metrics, fullfile(output_dir, ...
    'classification_continuous_validation_metrics.csv'));
writetable(classification_residuals, fullfile(output_dir, ...
    'classification_continuous_validation_residuals.csv'));
make_validation_figure(classification_measured.data_all, classification_prediction, ...
    masks, 'Classification continuous-sequence validation', ...
    fullfile(output_dir, 'classification_continuous_validation.png'));

aggregate = aggregate_validation(classification_metrics);
writetable(aggregate, fullfile(output_dir, 'continuous_validation_aggregate.csv'));

fprintf('Audit complete. Derived files are in: %s\n', output_dir);
disp(fit_summary_table);
disp(aggregate);

%% Local functions
function [summary_rows, residual_rows] = add_first_stage_fit( ...
    summary_rows, residual_rows, polarity, current_mA, segment, ...
    functional_form, time_s, observed, fit_object)
    predicted = feval(fit_object, time_s);
    residual = observed - predicted;
    names = coeffnames(fit_object);
    values = coeffvalues(fit_object);
    ci = confint(fit_object, 0.95);
    n = numel(observed);
    p = numel(names);
    sse = sum(residual.^2);
    sst = sum((observed - mean(observed)).^2);
    r2 = 1 - sse/sst;
    adjusted_r2 = 1 - (1-r2)*(n-1)/(n-p-1);
    rmse = sqrt(sse/(n-p));
    mae = mean(abs(residual));
    mean_residual = mean(residual);
    for c = 1:numel(names)
        summary_rows(end+1,:) = {polarity, current_mA, segment, ...
            functional_form, n, names{c}, values(c), ci(1,c), ci(2,c), ...
            sse, r2, adjusted_r2, rmse, mae, mean_residual}; %#ok<AGROW>
    end
    for j = 1:n
        residual_rows(end+1,:) = {polarity, current_mA, segment, ...
            time_s(j), observed(j), predicted(j), residual(j)}; %#ok<AGROW>
    end
end

function mask_table = build_mask_table()
    names = strings(14,1);
    bits = zeros(14,4);
    polarities = zeros(14,4);
    reversals = zeros(14,1);
    for i = 1:14
        names(i) = string(dec2bin(i,4));
        bits(i,:) = double(char(names(i))) - double('0');
        polarities(i,:) = bits(i,:);
        polarities(i, polarities(i,:)==0) = -1;
        reversals(i) = sum(diff(polarities(i,:))~=0);
    end
    mask_table = table((1:14)', names, bits(:,1), bits(:,2), bits(:,3), bits(:,4), ...
        polarities(:,1), polarities(:,2), polarities(:,3), polarities(:,4), reversals, ...
        'VariableNames', {'mask_index','mask_word','bit1','bit2','bit3','bit4', ...
        'polarity1','polarity2','polarity3','polarity4','within_word_reversals'});
end

function [curve_v, mask] = simulate_smrc_sequence(input_signal, P, time_step, noise_sigma)
    input_drive = 37 + input_signal(:)*6;
    mask = zeros(14,4);
    for i = 1:14
        mask(i,:) = double(dec2bin(i,4)) - double('0');
    end
    mask(mask==0) = -1;

    n_masks = size(mask,1);
    mask_length = size(mask,2);
    n_inputs = numel(input_drive);
    curve_v = zeros(n_masks, n_inputs*mask_length);
    previous_state = zeros(n_masks,1);
    previous_polarity = [];

    for i = 1:n_inputs
        for j = 1:mask_length
            polarity = mask(:,j);
            [a, b] = model_ab(input_drive(i), polarity, P);
            if ~isempty(previous_polarity)
                zero_a = (a == 0);
                polarity_change = (polarity - previous_polarity)/2;
                changed = (polarity_change ~= 0);
                apply_switch_term = changed & zero_a;
                a(apply_switch_term) = polarity_change(apply_switch_term)/0.3;
            end
            if noise_sigma > 0
                a = a + noise_sigma*randn(size(a));
                b = b + noise_sigma*randn(size(b));
            end
            next_state = (a - previous_state).*(1-exp(-time_step./b)) + previous_state;
            curve_v(:,4*(i-1)+j) = next_state;
            previous_state = next_state;
            previous_polarity = polarity;
        end
    end
end

function [a, b] = model_ab(laser_current, polarity, P)
    a = zeros(size(polarity));
    b = ones(size(polarity));
    positive = polarity > 0;
    negative = polarity < 0;
    b(positive) = P.pd_b_mean;
    b(negative) = P.nd_b_mean;
    if laser_current >= 40
        a(positive) = feval(P.p_fitresult_a, laser_current);
        a(negative) = feval(P.n_fitresult_a, laser_current);
        b(positive) = feval(P.p_fitresult_b, laser_current);
        b(negative) = feval(P.n_fitresult_b, laser_current);
    end
end

function [metric_table, residual_table] = validation_metrics(dataset, measured, predicted, masks)
    assert(isequal(size(measured), size(predicted)), ...
        'Measured and predicted arrays must have identical size.');
    n_masks = size(measured,1);
    metric_rows = cell(n_masks,16);
    residual_dataset = strings(numel(measured),1);
    residual_mask_index = zeros(numel(measured),1);
    residual_mask_word = strings(numel(measured),1);
    residual_sample_index = zeros(numel(measured),1);
    residual_measured = zeros(numel(measured),1);
    residual_predicted = zeros(numel(measured),1);
    residual_value = zeros(numel(measured),1);
    cursor = 1;
    for i = 1:n_masks
        y = measured(i,:)';
        yhat = predicted(i,:)';
        e = y - yhat;
        rmse = sqrt(mean(e.^2));
        data_range = max(y)-min(y);
        nrmse_range = rmse/data_range;
        mae = mean(abs(e));
        bias = mean(e);
        cc = corrcoef(y,yhat);
        correlation = cc(1,2);
        sse = sum(e.^2);
        sst = sum((y-mean(y)).^2);
        r_squared_prediction = 1-sse/sst;
        measured_amplitude = data_range;
        predicted_amplitude = max(yhat)-min(yhat);
        amplitude_error = predicted_amplitude-measured_amplitude;
        peak_positive_error = max(yhat)-max(y);
        peak_negative_error = min(yhat)-min(y);
        mask_word = string(sprintf('%d%d%d%d', masks(i,:)>0));
        reversals = sum(diff(masks(i,:))~=0);
        metric_rows(i,:) = {dataset, i, mask_word, reversals, numel(y), ...
            rmse, nrmse_range, mae, bias, correlation, r_squared_prediction, ...
            measured_amplitude, predicted_amplitude, amplitude_error, ...
            peak_positive_error, peak_negative_error};

        count = numel(y);
        range = cursor:(cursor+count-1);
        residual_dataset(range) = string(dataset);
        residual_mask_index(range) = i;
        residual_mask_word(range) = mask_word;
        residual_sample_index(range) = (1:count)';
        residual_measured(range) = y;
        residual_predicted(range) = yhat;
        residual_value(range) = e;
        cursor = cursor + count;
    end
    metric_table = cell2table(metric_rows, 'VariableNames', ...
        {'dataset','mask_index','mask_word','within_word_reversals','n_samples', ...
        'rmse_V','nrmse_by_measured_range','mae_V','mean_error_measured_minus_predicted_V', ...
        'pearson_r','prediction_r_squared','measured_peak_to_peak_V', ...
        'predicted_peak_to_peak_V','peak_to_peak_error_predicted_minus_measured_V', ...
        'positive_peak_error_predicted_minus_measured_V', ...
        'negative_peak_error_predicted_minus_measured_V'});
    residual_table = table(residual_dataset, residual_mask_index, residual_mask_word, ...
        residual_sample_index, residual_measured, residual_predicted, residual_value, ...
        'VariableNames', {'dataset','mask_index','mask_word','sample_index', ...
        'measured_V','predicted_V','residual_measured_minus_predicted_V'});
end

function aggregate = aggregate_validation(metric_table)
    aggregate = table(string(metric_table.dataset{1}), height(metric_table), ...
        sum(metric_table.n_samples), mean(metric_table.rmse_V), std(metric_table.rmse_V), ...
        mean(metric_table.nrmse_by_measured_range), std(metric_table.nrmse_by_measured_range), ...
        mean(metric_table.mae_V), std(metric_table.mae_V), ...
        mean(metric_table.mean_error_measured_minus_predicted_V), ...
        std(metric_table.mean_error_measured_minus_predicted_V), ...
        mean(metric_table.pearson_r), std(metric_table.pearson_r), ...
        mean(metric_table.prediction_r_squared), std(metric_table.prediction_r_squared), ...
        'VariableNames', {'dataset','n_masks','n_total_samples','mean_rmse_V','sd_rmse_V', ...
        'mean_nrmse_by_range','sd_nrmse_by_range','mean_mae_V','sd_mae_V', ...
        'mean_bias_V','sd_bias_V','mean_pearson_r','sd_pearson_r', ...
        'mean_prediction_r_squared','sd_prediction_r_squared'});
end

function make_validation_figure(measured, predicted, masks, title_text, output_file)
    representative = [1,3,5,10];
    fig = figure('Visible','off','Color','w','Position',[100 100 1400 900]);
    tiledlayout(4,2,'Padding','compact','TileSpacing','compact');
    sample_range = 1:min(600,size(measured,2));
    for row = 1:numel(representative)
        i = representative(row);
        word = sprintf('%d%d%d%d', masks(i,:)>0);
        nexttile;
        plot(sample_range, measured(i,sample_range), 'k-', 'LineWidth', 1.0); hold on;
        plot(sample_range, predicted(i,sample_range), 'Color',[0 0.447 0.741], 'LineWidth', 1.0);
        ylabel('State (V)'); title(sprintf('Mask %s',word));
        if row == 1, legend('Measured','Model','Location','best'); end
        if row == numel(representative), xlabel('Sample index'); end
        grid on;
        nexttile;
        residual = measured(i,:) - predicted(i,:);
        histogram(residual,50,'Normalization','pdf','FaceColor',[0.85 0.33 0.10]);
        xlabel('Measured - model (V)'); ylabel('Density');
        title(sprintf('Residuals: mean %.3g V, RMSE %.3g V', ...
            mean(residual),sqrt(mean(residual.^2)))); grid on;
    end
    sgtitle(title_text);
    exportgraphics(fig, output_file, 'Resolution', 220);
    close(fig);
end
