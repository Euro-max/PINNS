function vdbs_train_data(mode, nworkers)
% E30 training data on the Blockset vehicle (pinc_vdbs14).  For every drive of train_inputs.mat (written by
% experiments/e30_vdbs_retrain.py --part inputs): drive from free rolling at v0 with the drive commands, stop at
% each of the four sampled times (saving the operating point), and from there hold one input for k*dt, reading
% the state at the end.  States in ISO axes, 16 columns as in E29 (vdbs_extract).
%   mode 'pilot'  first 12 train drives (48 trajectories), reports the time per drive and the projected total
%   mode 'full'   all train and val drives, in blocks of 100 saved as they finish (a rerun resumes)
%   mode 'check'  re-simulates the first 5 train drives serially -> train_check.mat
root = 'C:\Users\elgondy\AppData\Local\Temp\claude_vdbs'; cd(root);
ex = fullfile(root,'e30'); I = load(fullfile(ex,'train_inputs.mat'));
if nargin < 2, nworkers = 4; end
paths = {fullfile(root,'vehconfig'), fullfile(root,'common'), fullfile(root,'pv14')};
for k = 1:numel(paths), addpath(genpath(paths{k})); end
switch mode
  case 'check'
    load_system('pinc_vdbs14'); S1 = nan(5, 4, 16);
    for i = 1:5, [~, S1(i,:,:)] = one_drive(I, 'train', i); end
    save(fullfile(ex,'train_check.mat'), 'S1'); return
  case 'pilot'
    sets = {'train'}; limit = 12;
  otherwise
    sets = {'train','val'}; limit = inf;
end
p = gcp('nocreate'); if isempty(p), p = parpool('Processes', nworkers); end
spmd
  cd(root); for k = 1:numel(paths), addpath(genpath(paths{k})); end
  c = fullfile(tempdir, sprintf('pinc_cache_%d', spmdIndex)); if ~exist(c,'dir'), mkdir(c); end
  Simulink.fileGenControl('set', 'CacheFolder', c, 'CodeGenFolder', c, 'createDir', true);
  load_system('pinc_vdbs14');
end
for s = 1:numel(sets)
  nm = sets{s}; nd = min(numel(I.([nm '_v0'])), limit); blk = 100;
  for b0 = 1:blk:nd
    f = fullfile(ex, sprintf('%s_block_%04d.mat', nm, b0));
    if exist(f, 'file') && ~strcmp(mode,'pilot'), continue; end
    ids = b0:min(b0 + blk - 1, nd); A0 = nan(numel(ids), 4, 16); A1 = A0; t0 = tic;
    parfor j = 1:numel(ids)
      [A0(j,:,:), A1(j,:,:)] = one_drive(I, nm, ids(j));
    end
    el = toc(t0);
    if strcmp(mode,'pilot')
      fprintf('pilot: %d drives (%d trajectories) in %.0f s on %d workers: %.1f s per drive; projected total for %d drives %.1f h\n', ...
              numel(ids), 4*numel(ids), el, nworkers, el/numel(ids), numel(I.train_v0) + numel(I.val_v0), ...
              el/numel(ids)*(numel(I.train_v0) + numel(I.val_v0))/3600);
      S0 = A0; S1 = A1; save(fullfile(ex,'pilot.mat'), 'S0', 'S1'); return
    end
    S0 = A0; S1 = A1; save(f, 'S0', 'S1', 'ids');
    fprintf('%s drives %d-%d done in %.0f s (%d failed)\n', nm, ids(1), ids(end), el, sum(any(isnan(A1(:,:,1)),2)));
  end
  % gather the blocks
  S0 = nan(numel(I.([nm '_v0'])), 4, 16); S1 = S0;
  for b0 = 1:blk:nd
    B = load(fullfile(ex, sprintf('%s_block_%04d.mat', nm, b0))); S0(B.ids,:,:) = B.S0; S1(B.ids,:,:) = B.S1;
  end
  R.([nm '_S0']) = S0; R.([nm '_S1']) = S1; %#ok<STRNU>
end
save(fullfile(ex,'train_data.mat'), '-struct', 'R');
fprintf('all done\n');
end

function [S0, S1] = one_drive(I, nm, i)
% states at the four sampled times of drive i (S0) and after the held input from each (S1); NaN on failure
mdl = 'pinc_vdbs14'; T = 0.1; Rw = 0.316843923235597;
S0 = nan(4, 16); S1 = nan(4, 16);
v0 = I.([nm '_v0'])(i); F0 = I.([nm '_F0'])(i); cmd = squeeze(I.([nm '_cmd'])(i,:,:));
pick = I.([nm '_pick'])(i,:); u = squeeze(I.([nm '_u'])(i,:,:)); k = I.([nm '_k'])(i,:); dt = I.([nm '_dt']);
K = size(cmd, 1); ucmd = [T*(0:K-1)' cmd; K*T cmd(end,:)];
try
  V = get_param(mdl,'ModelWorkspace').getVariable('VEH'); V.InitialLongVel = v0; op = [];
  for j = 1:4
    tj = pick(j)*T;
    in = Simulink.SimulationInput(mdl).setVariable('ucmd', ucmd, 'Workspace', mdl);
    if isempty(op)
      in = in.setVariable('VEH', V, 'Workspace', mdl).setVariable('omega0', v0/Rw, 'Workspace', mdl) ...
             .setVariable('F0', F0, 'Workspace', mdl).setVariable('d0', 0, 'Workspace', mdl);
    else
      in = in.setInitialState(op);
    end
    in = in.setModelParameter('StopTime', sprintf('%.10g', tj), 'OutputOption','SpecifiedOutputTimes', ...
           'OutputTimes', sprintf('%.10g', tj), 'SaveFinalState','on','SaveOperatingPoint','on','FinalStateName','xFinal');
    out = sim(in); op = out.xFinal; S0(j,:) = lastrow(vdbs_extract(out));
    % one held input from the sampled state, read at t = k dt
    te = tj + k(j)*dt; uj = u(j,:);
    in = Simulink.SimulationInput(mdl).setVariable('ucmd', [tj uj; te uj], 'Workspace', mdl).setInitialState(op);
    in = in.setModelParameter('StopTime', sprintf('%.12g', te), 'OutputOption','SpecifiedOutputTimes', 'OutputTimes', sprintf('%.12g', te));
    S1(j,:) = lastrow(vdbs_extract(sim(in)));
  end
catch e
  fprintf('%s drive %d failed: %s\n', nm, i, e.message);
end
end

function z = lastrow(S)
  v = @(x) x(end); om = S.omega; re = S.Re;
  if size(om,2) ~= 4, om = reshape(om, 4, []).'; end
  if size(re,2) ~= 4, re = reshape(re, 4, []).'; end
  z = [v(S.vx) v(S.vy) v(S.r) v(S.psi) v(S.X) v(S.Y) om(end,:) v(S.Fact(:)) v(S.Dact(:)) re(end,:)];
end
