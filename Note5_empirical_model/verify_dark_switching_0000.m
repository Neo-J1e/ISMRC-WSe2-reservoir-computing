% Read-only check of measured gate-switching transients under optical code 0000.
% Source data and all pre-existing files remain unchanged.

script_dir = fileparts(mfilename('fullpath'));
source_candidates = { ...
    fullfile(script_dir, 'model_reproducibility_package', 'data', 'Voc_all_400ms_no_wait.mat'), ...
    fullfile(script_dir, 'data', 'Voc_all_400ms_no_wait.mat')};
source_file = '';
for c = 1:numel(source_candidates)
    if isfile(source_candidates{c})
        source_file = source_candidates{c};
        break;
    end
end
assert(~isempty(source_file), 'The packaged Voc_all_400ms_no_wait.mat is missing.');
S = load(source_file);
V = S.Voc_all;
ids = find(strcmp(V.series, '0000'));

jump = [];
nonjump = [];
gate_step = [];
for z = 1:numel(ids)
    k = ids(z);
    voc = V.data{k};
    vg = V.Vg{k};
    event = find(diff(vg < 0) ~= 0) + 1;
    other = setdiff(2:numel(voc), event);
    this_jump = voc(event) - voc(event-1);
    this_nonjump = voc(other) - voc(other-1);
    this_gate_step = sign(vg(event) - vg(event-1));
    jump = [jump; this_jump(:)]; %#ok<AGROW>
    nonjump = [nonjump; this_nonjump(:)]; %#ok<AGROW>
    gate_step = [gate_step; this_gate_step(:)]; %#ok<AGROW>
end

fprintf('mask traces: %d\n', numel(ids));
fprintf('gate reversals: %d\n', numel(jump));
fprintf('median abs reversal jump: %.6f V\n', median(abs(jump)));
fprintf('IQR abs reversal jump: %.6f V\n', iqr(abs(jump)));
fprintf('non-reversal adjacent steps: %d\n', numel(nonjump));
fprintf('median abs non-reversal change: %.6f V\n', median(abs(nonjump)));
fprintf('95th percentile abs non-reversal change: %.6f V\n', prctile(abs(nonjump),95));
fprintf('median sample interval: %.3f s\n', median(diff(V.time{ids(1)})));
fprintf('jump and gate-step signs agree: %d/%d\n', ...
    sum(sign(jump)==gate_step), numel(jump));

% Expected: 16 traces; 128 reversals; 0.102087 V median; 0.006275 V IQR;
% 18,416 non-reversal steps; 0.000075 V median; 0.000675 V 95th percentile;
% 0.040 s sample interval; 127/128 sign agreement.
