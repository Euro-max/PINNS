# E30 data (Blockset vehicle)

{
 "sets": {
  "train": {
   "drives": 1438,
   "drives_kept": 1437,
   "trajectories": 5000
  },
  "val": {
   "drives": 288,
   "drives_kept": 287,
   "trajectories": 1000
  }
 },
 "checks": {
  "resim_max_abs_diff": 0.0,
  "resim_trajectories": 20,
  "small_t_max": 0.001,
  "small_t_max_scaled_change": 0.00709720275451503,
  "prior_qs_one_step_body": 0.012126543044513518,
  "prior_qs_one_step_body_by_ay": {
   "0-1": {
    "n": 2792,
    "rms": 0.010566617486574151
   },
   "1-2": {
    "n": 1139,
    "rms": 0.012750509505736548
   },
   "2-4": {
    "n": 877,
    "rms": 0.01485358523571532
   },
   "4-6": {
    "n": 178,
    "rms": 0.015467204934618006
   },
   "6-inf": {
    "n": 14,
    "rms": 0.0127617775761381
   }
  },
  "prior_full_one_step_body": 0.012165201219215408,
  "prior_full_one_step_body_by_ay": {
   "0-1": {
    "n": 2792,
    "rms": 0.010602543385351972
   },
   "1-2": {
    "n": 1139,
    "rms": 0.012784999006359687
   },
   "2-4": {
    "n": 877,
    "rms": 0.01489683760687373
   },
   "4-6": {
    "n": 178,
    "rms": 0.01553683141020704
   },
   "6-inf": {
    "n": 14,
    "rms": 0.012917878108587223
   }
  }
 }
}
