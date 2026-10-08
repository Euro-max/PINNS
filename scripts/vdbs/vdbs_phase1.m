% Phase 1 of E29: warm-up drives of the Blockset vehicle; saves the operating point and the state at the end of
% each drive.  Input inputs1.mat: v0 (n), F0 (n), tw (n, steps), cmd (n, K, 2).  Output: ops/op_<i>.mat, ics.mat.
root = 'C:\Users\elgondy\AppData\Local\Temp\claude_vdbs'; cd(root);
addpath(genpath(fullfile(root,'vehconfig'))); addpath(genpath(fullfile(root,'common'))); addpath(genpath(fullfile(root,'pv14')));
mdl = 'pinc_vdbs14'; load_system(mdl); mw = get_param(mdl,'ModelWorkspace');
I = load(fullfile(root,'e29','inputs1.mat')); n = numel(I.v0); T = 0.1; Rw = 0.316843923235597;
if ~exist(fullfile(root,'e29','ops'),'dir'), mkdir(fullfile(root,'e29','ops')); end
V0 = mw.getVariable('VEH'); IC = nan(n, 16); t0 = tic;
for i = 1:n
  f = fullfile(root,'e29','ops',sprintf('op_%03d.mat',i));
  K = double(I.tw(i)); t = T*(0:K-1)'; u = [t squeeze(I.cmd(i,1:K,:)); K*T squeeze(I.cmd(i,K,:))'];
  V = V0; V.InitialLongVel = I.v0(i);
  in = Simulink.SimulationInput(mdl).setVariable('VEH', V, 'Workspace', mdl).setVariable('omega0', I.v0(i)/Rw, 'Workspace', mdl);
  in = in.setVariable('ucmd', u, 'Workspace', mdl).setVariable('F0', I.F0(i), 'Workspace', mdl).setVariable('d0', 0, 'Workspace', mdl);
  in = in.setModelParameter('StopTime', sprintf('%.10g', K*T), 'OutputOption','SpecifiedOutputTimes','OutputTimes', sprintf('%.10g', K*T), ...
        'SaveFinalState','on','SaveOperatingPoint','on','FinalStateName','xFinal');
  try
    out = sim(in); S = vdbs_extract(out); op = out.xFinal; save(f, 'op');
    IC(i,:) = lastrow(S);
  catch e
    fprintf('drive %d failed: %s\n', i, e.message);
    if i <= 2, fprintf('  sizes: vx %s omega %s Fact %s Re %s\n', mat2str(size(S.vx)), mat2str(size(S.omega)), mat2str(size(S.Fact)), mat2str(size(S.Re))); end
  end
  if mod(i,10) == 0, fprintf('%d/%d drives, %.0f s\n', i, n, toc(t0)); end
end
save(fullfile(root,'e29','ics.mat'), 'IC');
fprintf('phase 1 done in %.0f s\n', toc(t0));

function z = lastrow(S)
  % state at the last output time, whatever the layout of the logged arrays
  v = @(x) x(end);
  w = @(x) x(end - 3:end);
  om = S.omega; re = S.Re;
  if size(om,2) ~= 4, om = reshape(om, 4, []).'; end
  if size(re,2) ~= 4, re = reshape(re, 4, []).'; end
  z = [v(S.vx) v(S.vy) v(S.r) v(S.psi) v(S.X) v(S.Y) om(end,:) v(S.Fact(:)) v(S.Dact(:)) re(end,:)];
  if numel(z) ~= 16
    error('state has %d entries: vx %s omega %s Fact %s Re %s', numel(z), mat2str(size(S.vx)), mat2str(size(S.omega)), mat2str(size(S.Fact)), mat2str(size(S.Re)));
  end
end
