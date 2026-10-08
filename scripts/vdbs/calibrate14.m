% M0 rule on the Blockset 14-DOF vehicle: scale lam_Kya (LAM(6,:)) so that the axle cornering stiffness in gentle
% steady turning equals the prior's 50 kN/rad.  Axle stiffness from the steady-state force balance of two turns
% (steer 0.01 and 0.02 rad at 15 m/s): Fy_f = m ay b/L, Fy_r = m ay a/L, alpha_f = d - (vy + a r)/vx,
% alpha_r = -(vy - b r)/vx; C = dFy/dalpha between the two turns (removes ply-steer offsets).
root = 'C:\Users\elgondy\AppData\Local\Temp\claude_vdbs'; cd(root);
addpath(genpath(fullfile(root,'vehconfig'))); addpath(genpath(fullfile(root,'common'))); addpath(genpath(fullfile(root,'pv14')));
mdl = 'pinc_vdbs14'; load_system(mdl); mw = get_param(mdl,'ModelWorkspace'); fid = fopen('calibrate14.txt','w');
m = 1500; a = 1.4; b = 1.4; L = 2.8; target = 50000; v0 = 15; Rw = 0.316843923235597;
V = mw.getVariable('VEH'); V.InitialLongVel = v0; lam = 0.4730809365160366;
for it = 1:5
  LAM = ones(27,4); LAM(6,:) = lam; Fy = zeros(2,2); al = zeros(2,2);
  for j = 1:2
    d = 0.01*j; u = [0 250 d; 6 250 d];
    in = Simulink.SimulationInput(mdl).setVariable('VEH', V, 'Workspace', mdl).setVariable('omega0', v0/Rw, 'Workspace', mdl).setVariable('LAM', LAM, 'Workspace', mdl);
    in = in.setVariable('ucmd', u, 'Workspace', mdl).setVariable('F0', 250, 'Workspace', mdl).setVariable('d0', d, 'Workspace', mdl);
    in = in.setModelParameter('StopTime','6','OutputOption','SpecifiedOutputTimes','OutputTimes','[5.8 6]');
    S = vdbs_extract(sim(in)); vx = S.vx(end); vy = S.vy(end); r = S.r(end); ay = vx*r;   % vdbs_extract returns ISO axes
    Fy(j,:) = [m*ay*b/L, m*ay*a/L]; al(j,:) = [d - (vy + a*r)/vx, -(vy - b*r)/vx];
    fprintf(fid,'  lam %.4f d %.3f: vx %.3f vy %.4f r %.4f\n', lam, d, vx, vy, r);
  end
  C = (Fy(2,:) - Fy(1,:))./(al(2,:) - al(1,:));
  fprintf(fid,'iter %d: lam %.4f -> Cf %.0f, Cr %.0f N/rad per axle\n', it, lam, C(1), C(2));
  if abs(mean(C)/target - 1) < 0.01, break; end
  lam = lam*target/mean(C);
end
LAM = ones(27,4); LAM(6,:) = lam; mw.assignin('LAM', LAM); save_system(mdl);
fprintf(fid,'final lam_Kya %.4f\n', lam); fclose(fid);
