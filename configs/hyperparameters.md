# Hyperparameters and Configuration

All hyperparameters, reward weights, environment settings, and filter gains used in the paper.

## Environment Parameters

| Parameter | Symbol | Training Env | Test Env |
|---|---|---|---|
| World size | $w \times h$ | 1000 × 1000 | 100 × 100 |
| Static obstacles | $N_{\text{sta}}$ | 20 | 40 |
| Dynamic obstacles | $N_{\text{dyn}}$ | 5 | 10 |
| Obstacle radius range | $[r_{\min}, r_{\max}]$ | [1.0, 5.0] | [0.5, 1.0] |
| Dynamic obstacle max speed | — | 5.0 | 2.0 |
| Max episode steps | — | 1000 | 300 |
| Agent radius | — | 0.06 | 0.06 |

## Agent Parameters

| Parameter | Symbol | Value |
|---|---|---|
| Max velocity | $v_{\max}^A$ | 2.0 |
| Max velocity change per step | $\Delta v_{\max}$ | 0.5 |
| Max heading change per step | $\Delta\theta_{\max}$ | $\pi/6$ |
| LiDAR rays (training) | — | 9 |
| LiDAR arc (training) | — | 120° (forward) |
| LiDAR sensing range | $r_{\text{cap}}$ | 15.0 |
| Success distance threshold | — | 5.0 (train), 0.5 (test) |

## TD3 / MPTD3 Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | TD3 (via Stable-Baselines3 2.7.0) |
| Policy | MlpPolicy |
| Learning rate | 1e-3 |
| Buffer size | 1,000,000 |
| Batch size | 256 |
| Discount factor $\gamma$ | 0.99 |
| Soft update coefficient $\tau$ | 0.005 |
| Policy delay | 2 |
| Target policy noise | 0.2 |
| Target noise clip | 0.5 |
| Gradient clipping | 0.5 |
| Training timesteps | 1,000,000 |
| Target success fraction $\eta$ | Configured in MPTD3 |
| Action noise | NormalActionNoise (σ = 0.1) |

## Reward Weights

| Parameter | Symbol | Value |
|---|---|---|
| Distance shaping weight | $k_D$ | 0.05 |
| Alignment shaping weight | $k_\Theta$ | 0.05 |
| Obstacle clearance weight | $k_{\text{obs}}$ | 0.01 |
| Velocity reward weight | $k_V$ | 0.01 |
| Velocity threshold | $v_c$ | 0.5 |
| Success reward | $k_T$ | 10.0 |
| Collision penalty | $k_C$ | 10.0 |

## CLF-CBF-QP Filter Gains

| Parameter | Symbol | Value |
|---|---|---|
| CLF convergence rate | $k_{\text{clf}}$ ($\gamma_{\text{clf}}$ in code) | 0.5 |
| CBF gain (obstacle) | $k_{\text{cbf}}$ ($\alpha_{\text{cbf}}$ in code) | 1.0 |
| Boundary CBF gain | $\sigma_{\text{bcbf}}$ ($\alpha_{\text{bcbf}}$ in code) | 1.0 |
| Slack penalty | $\lambda_{\text{slack}}$ | 1e5 |
| Safety margin (obstacles) | $d_{\text{safe}}$ | 1.0 |
| Safety margin (boundary) | $m$ | 5.0 |
| CBF activation distance | — | 2.0 |
| RL decision frequency | — | 1 Hz |
| QP filter frequency | — | 10 Hz |
| Physics simulation frequency | — | 100 Hz |
| LiDAR rays (filter) | — | 24 (360°) |
| EMA smoothing factor | $\gamma_{\text{smooth}}$ | 0.3 |
| $\dot{R}$ clamp range | — | [-0.5, 0.5] |
| QP solver | — | OSQP (via CVXPY) |

## Sensor Parameters

| Parameter | Value |
|---|---|
| Training LiDAR | 9 rays, 120° forward arc |
| Filter LiDAR | 24 rays, 360° coverage |
| GPS | Noiseless (assumed) |
| Sensor model | Noiseless (ideal) |

## Physics / Simulation

| Parameter | Value |
|---|---|
| Simulator | PyBullet 3.21 |
| Physics timestep | 0.01 s |
| Dynamics model | Single-integrator (holonomic) |
| RL timestep ($\Delta t$) | 1.0 s |
| Altitude | Fixed at $z = 1.0$ |
