% Reference tyre forces from the MathWorks Vehicle Dynamics Blockset Magic Formula solver
% (vdyntire.internal.solvers.vdyncsmtire, as called by the Combined Slip Wheel 2DOF block) for the
% "Mid-size passenger car 235/45R18" set (the block default, as in export_tyre_params.m), on a grid of slip, slip angle and vertical load at 20 m/s.
% Used by tests/test_plant_hf.py to cross-check pinc/tyre_mf.py.  The solver's argument list is read
% from the block's own MATLAB Function chart.  Scale factors: all 1 except lam_muV = 0 (no slip-speed
% friction decay; ScaleFctrs(4) per the block documentation).  Turn slip off, zero camber, nominal
% pressure, vertical load given externally.
%
% Output: %USERPROFILE%\pinc_tyre\mf_reference.csv, columns
%   Fz_in, kappa, alpha (solver convention), Fz, Fx, Fy (SAE axes, y to the right), effective radius.
% Copy it to data/tyre/ (gitignored, MathWorks data is not redistributed).
out = fullfile(getenv('USERPROFILE'),'pinc_tyre');
new_system('pinc_ref'); c = onCleanup(@() close_system('pinc_ref',0));
blk = 'pinc_ref/w'; add_block('vehdynlibtire/Combined Slip Wheel 2DOF', blk);
set_param(blk,'tireType','Mid-size passenger car 235/45R18');
set_param('pinc_ref','SimulationCommand','update');
wsv = get_param(blk,'MaskWSVariables'); W = containers.Map({wsv.Name}, {wsv.Value});
load_system('vehdynlibtirecommon'); rt = sfroot;
ch = rt.find('-isa','Stateflow.EMChart'); src = '';
for i=1:numel(ch), if contains(ch(i).Path,'Magic Tire Const Input/Magic Tire Const Input'), src = ch(i).Script; end, end
i0 = strfind(src, 'vdyncsmtire('); body = src(i0+numel('vdyncsmtire('):end);
body = regexprep(body, '%[^\n]*', ''); body = body(1:find(body==')',1)-1);
names = strtrim(strsplit(regexprep(body,'\.\.\.|\s',''), ','));
Vx = 20; R0 = 0.31;
[K, A, Z] = ndgrid(linspace(-0.3,0.3,13), linspace(-0.3,0.3,13), [2000 3679 6000]);
n = numel(K); Omega = (1 + K(:)')*Vx/R0; Vy = -Vx*tan(A(:)');
sf = ones(27,1); sf(4) = 0; layouts = {sf};
res = nan(n,6); good = 0;
for L = 1:numel(layouts)
  try
    for j = 1:n
      special = struct('Omega',Omega(j),'Vx',Vx,'Vy',Vy(j),'psidot',0,'Gamma',0,'TirePrs',W('NOMPRES'), ...
          'ScaleFactors',layouts{L},'rhoz',0,'plySteer',1,'turnslip',0,'Fx_ext',0,'Fy_ext',0,'Fz_ext',Z(j),'vertType',1);
      args = cell(1,numel(names));
      for i=1:numel(names)
          nm = names{i};
          if ~isnan(str2double(nm)), args{i} = str2double(nm);
          elseif isfield(special, nm), args{i} = special.(nm);
          elseif isKey(W, nm), args{i} = W(nm);
          else, error('no value for argument %s', nm); end
      end
      [fx,fy,fz,~,~,~,re,kp,al] = vdyntire.internal.solvers.vdyncsmtire(args{:});
      res(j,:) = [kp al fz fx fy re];
    end
    good = L; fprintf('layout %d (%s) works\n', L, mat2str(size(layouts{L}))); break
  catch e
    fprintf('layout %d (%s): %s\n', L, mat2str(size(layouts{L})), e.message);
  end
end
Kap = res(:,1); Alp = res(:,2); FzT = res(:,3); Fx = res(:,4); Fy = res(:,5); Re = res(:,6);
T = [Z(:) Kap(:) Alp(:) FzT(:) Fx(:) Fy(:) Re(:)];
writematrix(T, fullfile(out,'mf_reference.csv'));
fprintf('wrote %d points; Fz range %g..%g, kappa %g..%g, alpha %g..%g\n', n, min(FzT), max(FzT), min(Kap), max(Kap), min(Alp), max(Alp));
