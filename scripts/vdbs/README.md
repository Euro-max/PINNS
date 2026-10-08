# Blockset 14-DOF vehicle (E29)

MATLAB scripts that build and run the independent plant of E29: the Vehicle Dynamics Blockset 14-DOF passenger
vehicle reference (`PassVeh14DOF`, R2025b) with our vehicle parameters, actuators and drive split. The Simulink
model is built locally from the MathWorks reference and is not part of the repository.

1. Unzip `PassVeh14DOF.zip`, `Common.zip` and `VehicleConfig.zip` from
   `<matlabroot>/toolbox/vdynblks/vdynsolution/` into the working folder (`root` in the scripts).
2. `build_harness` builds `pinc_vdbs14.slx`: our mass, axle positions, CG height, track, yaw inertia and drag;
   first-order actuators (0.15 s, 0.10 s) and our drive/brake split; the tyre preset "Mid-size passenger car
   235/45R18" (the coefficient set of our plant). The active tyre block reads its coefficients from the preset,
   not from its mask, so stiffness is changed through the solver's scale-factor input (lam_Kya, element 6).
3. `calibrate14` scales lam_Kya by the M0 rule: axle cornering stiffness 50 kN/rad in gentle steady turning
   (two turns at 15 m/s, force balance; result 0.318: front 46.9, rear 53.2 kN/rad).
4. `test_harness` runs a steering step for a first comparison with our plant.
5. E29: `python -m experiments.e29_vdbs_transfer --part inputs1`, then `vdbs_phase1` (warm-up drives, saved
   operating points), `--part inputs2`, `vdbs_phase2` (the E9 input sequences from each saved state),
   `--part eval`.

The Blockset body uses SAE axes (y right, yaw positive to the right); the harness turns our steer angle into a
negative wheel angle for a left turn, and `vdbs_extract` returns our ISO axes. Wheel order: FL, FR, RL, RR.
