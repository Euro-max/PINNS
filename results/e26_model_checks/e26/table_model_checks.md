# E26 checks of the double-track plant

1. Tyre model against the MathWorks solver (507 points): max |Fx error| 0.0703 N (|Fx| up to 5942 N), max |Fy error| 3.66e-07 N (|Fy| up to 5134 N)

2. Double-track (M0) against single-track, gentle driving (|a_y| <= 2 m/s^2, reached 1.42; 40 drives of 4 s): NRMSE body 0.0009, vx 0.0002, vy 0.0014, r 0.0006

3. Wheel-speed time constant at free rolling and static load

| speed [m/s] | time constant [ms] |
|---|---|
| 5 | 0.71 |
| 10 | 1.43 |
| 20 | 2.86 |
| 30 | 4.29 |
