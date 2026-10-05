% Export the Vehicle Dynamics Blockset tyre parameter sets used by the high-fidelity plant
% (docs/PLAN_HIGH_FIDELITY.md) to JSON.  Run once in MATLAB (Windows side) with the Vehicle
% Dynamics Blockset installed.  Output: %USERPROFILE%\pinc_tyre\*.json plus export_log.txt.
%
% The parameter sets are MathWorks data: they are read locally (data/tyre/, gitignored) and
% are not committed to this repository.  Re-run this script to regenerate them.

out = fullfile(getenv('USERPROFILE'), 'pinc_tyre');
if ~exist(out, 'dir'); mkdir(out); end
log = fopen(fullfile(out, 'export_log.txt'), 'w');
fprintf(log, '%s\n%s\n', version, char(datetime('now')));

names = {'mf52.tm20560R15', 'mf62.tm20560R15', 'fiala.tm20560R15', ...
         'dugoff.tm20560R15nom', 'dugoff.tm20560R15ext'};
for k = 1:numel(names)
    fn = ['vdyntire.internal.models.' names{k}];
    try
        fprintf(log, '%s: nargin=%d nargout=%d\n', fn, nargin(fn), nargout(fn));
    catch err
        fprintf(log, '%s: nargin/nargout unavailable (%s)\n', fn, err.message);
    end
    try
        p = feval(fn);
        txt = jsonencode(p, PrettyPrint=true);
        fid = fopen(fullfile(out, [strrep(names{k}, '.', '_') '.json']), 'w');
        fprintf(fid, '%s', txt);
        fclose(fid);
        msg = sprintf('OK   %s (%s)', fn, class(p));
    catch err
        msg = sprintf('FAIL %s: %s', fn, err.message);
    end
    disp(msg);
    fprintf(log, '%s\n', msg);
end
fclose(log);
fprintf('Wrote to %s\n', out);
