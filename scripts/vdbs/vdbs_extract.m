function S = vdbs_extract(out)
% States of pinc_vdbs14 at the output times in our (ISO) axes: vx, vy, r, psi, X, Y, omega (FL FR RL RR), F_act,
% delta_act, Re (4).  The Blockset body is SAE (y right, yaw positive to the right): vy, r, psi and Y change sign.
y = out.yout; veh = y{1}.Values; whl = y{2}.Values;
S.t = veh.BdyFrm.Cg.Vel.xdot.Time;
S.vx = veh.BdyFrm.Cg.Vel.xdot.Data; S.vy = -veh.BdyFrm.Cg.Vel.ydot.Data; S.r = -veh.BdyFrm.Cg.AngVel.r.Data;
S.psi = -veh.InertFrm.Cg.Ang.psi.Data; S.X = veh.InertFrm.Cg.Disp.X.Data; S.Y = -veh.InertFrm.Cg.Disp.Y.Data;
S.omega = squeeze(whl.TireFrame.Omega.Data); S.Re = squeeze(whl.TireFrame.Re.Data);
if size(S.omega,1) ~= numel(S.t), S.omega = S.omega'; S.Re = S.Re'; end
S.Fact = y{3}.Values.Data; S.Dact = y{4}.Values.Data;
end
