% Phase 2 of E29: from each saved operating point, run the input sequences of seqs.mat (n, n_seq, n_steps, 2) and
% record the state after every control period.  Output truth.mat: X (n, n_seq, n_steps, 16) in ISO axes.
root = 'C:\Users\elgondy\AppData\Local\Temp\claude_vdbs'; cd(root);
addpath(genpath(fullfile(root,'vehconfig'))); addpath(genpath(fullfile(root,'common'))); addpath(genpath(fullfile(root,'pv14')));
mdl = 'pinc_vdbs14'; load_system(mdl);
I = load(fullfile(root,'e29','inputs1.mat')); Q = load(fullfile(root,'e29','seqs.mat')); T = 0.1;
[n, ns, K, ~] = size(Q.u); X = nan(n, ns, K, 16); t0 = tic;
for i = 1:n
  f = fullfile(root,'e29','ops',sprintf('op_%03d.mat',i)); if ~exist(f,'file'), continue; end
  L = load(f); tw = double(I.tw(i))*T;
  for j = 1:ns
    u = [tw + T*(0:K-1)' squeeze(Q.u(i,j,:,:)); tw + K*T squeeze(Q.u(i,j,K,:))'];
    in = Simulink.SimulationInput(mdl).setVariable('ucmd', u, 'Workspace', mdl).setInitialState(L.op);
    in = in.setModelParameter('StopTime', sprintf('%.10g', tw + K*T), 'OutputOption','SpecifiedOutputTimes', ...
          'OutputTimes', ['[' sprintf('%.10g ', tw + T*(1:K)) ']']);
    try
      S = vdbs_extract(sim(in));
      Z = [S.vx S.vy S.r S.psi S.X S.Y S.omega S.Fact S.Dact S.Re];
      if i == 1 && j == 1, fprintf('first run: %d rows, t(1) %.3f, t(end) %.3f (start %.3f)\n', size(Z,1), S.t(1), S.t(end), tw); end
      X(i,j,:,:) = Z(end - K + 1:end, :);       % a run from a saved state also reports its start time
    catch e
      fprintf('ic %d seq %d failed: %s\n', i, j, e.message);
    end
  end
  if mod(i,5) == 0, fprintf('%d/%d initial states, %.0f s\n', i, n, toc(t0)); save(fullfile(root,'e29','truth.mat'), 'X'); end
end
save(fullfile(root,'e29','truth.mat'), 'X');
fprintf('phase 2 done in %.0f s\n', toc(t0));
