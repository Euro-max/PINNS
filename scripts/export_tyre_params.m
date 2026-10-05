% Export the Magic Formula parameter set used by the high-fidelity plant (docs/PLAN_HIGH_FIDELITY.md)
% from the MathWorks Vehicle Dynamics Blockset.  Needs MATLAB + Simulink + Vehicle Dynamics Blockset.
%
% The set is read from the "Combined Slip Wheel 2DOF" block (library vehdynlibtire) with
% tireType = "Light passenger car 205/60R15": selecting the tyre type makes the block load the
% corresponding MF parameters into its mask, which are then written as NAME = value lines.
%
% Output: %USERPROFILE%\pinc_tyre\mf_205_60R15_params.txt and tire_types.txt.  Copy them to
% data/tyre/ in this repository (gitignored: MathWorks data is read locally, not redistributed).
%
% From WSL:  "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch \
%            "cd('\\wsl.localhost\Ubuntu\home\elgondy_wsl\repo\PINNS\scripts'); export_tyre_params"

out = fullfile(getenv('USERPROFILE'), 'pinc_tyre');
if ~exist(out, 'dir'); mkdir(out); end
tyre = '205/60R15';

new_system('pinc_tmp');
cleanup = onCleanup(@() close_system('pinc_tmp', 0));
blk = 'pinc_tmp/w';
add_block('vehdynlibtire/Combined Slip Wheel 2DOF', blk);

P = get_param(blk, 'MaskObject').getParameter('tireType');
labels = cellfun(@(id) message(id).getString, P.TypeOptions, 'UniformOutput', false);
fid = fopen(fullfile(out, 'tire_types.txt'), 'w'); fprintf(fid, '%s\n', labels{:}); fclose(fid);
pick = labels(contains(labels, tyre));
if isempty(pick); error('%s not offered: %s', tyre, strjoin(labels, ' | ')); end
set_param(blk, 'tireType', pick{1});

mo = get_param(blk, 'MaskObject');
fid = fopen(fullfile(out, 'mf_205_60R15_params.txt'), 'w');
fprintf(fid, '%% %s, block vehdynlibtire/Combined Slip Wheel 2DOF, tireType = %s\n', version, pick{1});
for j = 1:numel(mo.Parameters)
    fprintf(fid, '%s = %s\n', mo.Parameters(j).Name, strrep(mo.Parameters(j).Value, newline, ' '));
end
fclose(fid);
fprintf('OK %s -> %s\n', pick{1}, out);
