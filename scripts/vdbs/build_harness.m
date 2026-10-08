% Build pinc_vdbs14: the Vehicle Dynamics Blockset 14-DOF passenger vehicle with our vehicle parameters,
% our tyre coefficients (M0 calibration), first-order actuators and our drive/brake split.
root = 'C:\Users\elgondy\AppData\Local\Temp\claude_vdbs';
addpath(genpath(fullfile(root,'vehconfig'))); addpath(genpath(fullfile(root,'common'))); addpath(genpath(fullfile(root,'pv14')));
cd(root);
if bdIsLoaded('pinc_vdbs14'), close_system('pinc_vdbs14',0); end
load_system('PassVeh14DOF'); set_param('PassVeh14DOF','SimulationMode','normal');
save_system('PassVeh14DOF', fullfile(root,'pinc_vdbs14.slx')); close_system('PassVeh14DOF',0);
mdl = 'pinc_vdbs14'; load_system(mdl);
% ---- vehicle parameters (model workspace)
mw = get_param(mdl,'ModelWorkspace'); V = mw.getVariable('VEH');
V.Mass = 1500; V.FrontAxlePositionfromCG = 1.4; V.RearAxlePositionfromCG = 1.4; V.WheelBase = 2.8;
V.HeightCG = 0.23316; V.TrackWidth = 1.6; V.YawMomentInertia = 2500; V.FrontalArea = 2.2; V.DragCoefficient = 0.3;
V.InitialLatPosition = 0; V.InitialLongVel = 15;
mw.assignin('VEH', V);
mw.assignin('tauF', 0.15); mw.assignin('tauD', 0.10); mw.assignin('Rw', 0.316843923235597);
mw.assignin('gamF', 1.0); mw.assignin('betF', 0.6); mw.assignin('F0', 0); mw.assignin('d0', 0); mw.assignin('omega0', 15/0.316843923235597);
mw.assignin('ucmd', [0 0 0; 100 0 0]);
% ---- tyre: the active block (MF Tires Vector) takes its MF coefficients from the tireType preset; the preset
% "Mid-size passenger car 235/45R18" is the set our plant uses.  M0 stiffness through the solver's scale-factor
% input: lambda .* LAM, with LAM(6,:) (lam_Kya) calibrated by calibrate14.m
wb = find_system(mdl,'MatchFilter',@Simulink.match.allVariants,'LookUnderMasks','all','FollowLinks','on','MaskType','Combined Slip Wheel 2DOF');
for k = 1:numel(wb), set_param(wb{k},'tireType','Mid-size passenger car 235/45R18','omegao','omega0'); end
vec = [mdl '/Wheels and Tires/VDBS/Tires/MF Tires Vector']; blk = [vec '/Combined Slip Wheel 2DOF'];
pc = get_param(blk,'PortHandles'); lh = get_param(pc.Inport(10),'Line'); srcp = get_param(lh,'SrcPortHandle'); delete_line(lh);
add_block('simulink/Math Operations/Product',[vec '/lamScale'],'Inputs','2','Multiplication','Element-wise(.*)');
add_block('simulink/Sources/Constant',[vec '/lamM0'],'Value','LAM');
pp = get_param([vec '/lamScale'],'PortHandles'); pk = get_param([vec '/lamM0'],'PortHandles');
add_line(vec, srcp, pp.Inport(1)); add_line(vec, pk.Outport, pp.Inport(2)); add_line(vec, pp.Outport, pc.Inport(10));
LAM = ones(27,4); LAM(6,:) = 0.4730809365160366; mw.assignin('LAM', LAM);
% ---- replace the root inports by our actuation
ins = {'WhlAng','AxlTrq','BrkPrs','WindXYZ','Ground','Friction','DCM'}; dst = struct();
for k = 1:numel(ins)
  b = [mdl '/' ins{k}]; lh = get_param(b,'LineHandles'); d = get_param(lh.Outport,'DstPortHandle');
  pos = get_param(b,'Position'); delete_line(lh.Outport); delete_block(b); dst.(ins{k}) = struct('ports', d, 'pos', pos);
end
add_block('simulink/Sources/From Workspace', [mdl '/u_in'], 'VariableName','ucmd', 'Interpolate','off', 'OutputAfterFinalValue','Holding final value', 'SampleTime','0', 'Position',[-600 100 -520 130]);
add_block('simulink/Signal Routing/Demux', [mdl '/u_dmx'], 'Outputs','2', 'Position',[-500 95 -495 135]);
add_line(mdl,'u_in/1','u_dmx/1');
lag = {'F','tauF','F0';'D','tauD','d0'};
for k = 1:2
  n = lag{k,1}; y = 60*k;
  add_block('simulink/Math Operations/Sum', [mdl '/s' n], 'Inputs','+-', 'Position',[-460 50+y -440 70+y]);
  add_block('simulink/Math Operations/Gain', [mdl '/g' n], 'Gain',['1/' lag{k,2}], 'Position',[-420 50+y -390 70+y]);
  add_block('simulink/Continuous/Integrator', [mdl '/i' n], 'InitialCondition', lag{k,3}, 'Position',[-370 50+y -340 70+y]);
  add_line(mdl, sprintf('u_dmx/%d',k), ['s' n '/1']); add_line(mdl, ['s' n '/1'], ['g' n '/1']); add_line(mdl, ['g' n '/1'], ['i' n '/1']);
  add_line(mdl, ['i' n '/1'], ['s' n '/2'], 'autorouting','on');
end
add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/split'], 'Position',[-300 80 -200 160]);
rt = sfroot; ch = rt.find('-isa','Stateflow.EMChart','Path',[mdl '/split']);
ch.Script = sprintf(['function [T, ang] = split(F, d, Rw, gamF, betF)\n' ...
  's = 0.5*(1 + tanh(F/50)); front = s*gamF + (1 - s)*betF;\n' ...
  'T = [0.5*front*F*Rw, 0.5*front*F*Rw, 0.5*(1 - front)*F*Rw, 0.5*(1 - front)*F*Rw];\n' ...
  'ang = [-d, -d, 0, 0];  %% Blockset axes are SAE: a left turn is a negative wheel angle\n']);
for nm = {'Rw','gamF','betF'}
  dd = ch.find('-isa','Stateflow.Data','Name',nm{1});
  if isempty(dd), dd = Stateflow.Data(ch); dd.Name = nm{1}; end
  dd.Scope = 'Parameter';
end
add_line(mdl,'iF/1','split/1'); add_line(mdl,'iD/1','split/2');
con = {'BrkPrs','zeros(1,4)';'WindXYZ','zeros(1,3)';'Ground','zeros(1,4)';'Friction','ones(1,4)';'DCM','repmat(eye(3),[1 1 4])'};
for k = 1:size(con,1)
  add_block('simulink/Sources/Constant', [mdl '/c' con{k,1}], 'Value', con{k,2}, 'Position', dst.(con{k,1}).pos);
  ph = get_param([mdl '/c' con{k,1}],'PortHandles');
  for p = dst.(con{k,1}).ports', add_line(mdl, ph.Outport, p); end
end
ph = get_param([mdl '/split'],'PortHandles');
for p = dst.AxlTrq.ports', add_line(mdl, ph.Outport(1), p); end
for p = dst.WhlAng.ports', add_line(mdl, ph.Outport(2), p); end
% actuator states as root outputs for logging
add_block('simulink/Sinks/Out1', [mdl '/Fact'], 'Position',[-300 200 -270 215]); add_line(mdl,'iF/1','Fact/1');
add_block('simulink/Sinks/Out1', [mdl '/Dact'], 'Position',[-300 240 -270 255]); add_line(mdl,'iD/1','Dact/1');
set_param(mdl,'SaveOutput','on','OutputSaveName','yout','SaveFormat','Dataset','SaveTime','on');
save_system(mdl);
fprintf('solver %s, %s, step %s\n', get_param(mdl,'SolverType'), get_param(mdl,'Solver'), get_param(mdl,'FixedStep'));
disp('built');
