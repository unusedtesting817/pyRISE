import pyrise as pr
import yaml
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path

# Define the Japan DSGE model with regime switching
japan_model_spec = """
name: Japan DSGE with Regime Switching
parameters:
    beta: 0.99
    sigma: 2.0
    chi: 1.0
    alpha: 0.33
    delta: 0.025
    kappa: 0.1
    psi_pi:
        0: 1.5 # Normal regime
        1: 3.0 # Hawkish regime
    psi_y: 0.1
    rho_a: 0.9
    rho_m: 0.5
    sigma_a: 0.01
    sigma_m: 0.005
    transition_matrix:
        - [0.95, 0.05]
        - [0.1, 0.9]
equations:
    states: [a, k, r_lag]
    controls: [y, c, i, pi, r]
    shocks: [eps_a, eps_m]
    equilibrium:
        - y = a * k(-1)^alpha
        - y = c + i
        - i = k - (1-delta)*k(-1)
        - 1/c = beta * (1/c(+1)) * (1+r-pi(+1))
        - pi = beta * pi(+1) + kappa * (y - y_bar)
        - r = rho_r * r(-1) + (1-rho_r) * (psi_pi * pi + psi_y * (y - y_bar)) + eps_m
"""

# Write the model spec to a temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    f.write(japan_model_spec)
    temp_file = f.name

try:
    # Load the model
    model = pr.load_model(temp_file)

    # Solve the model
    solution = pr.solve(model)

    # Compute and plot IRFs
    shock = "eps_m"
    irfs_regime0 = pr.simulate.irf(solution, shock, regime=0)
    irfs_regime1 = pr.simulate.irf(solution, shock, regime=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    irfs_regime0[["y", "c", "i", "pi"]].plot(ax=axes[0, 0], title="Regime 0 (Normal)")
    irfs_regime1[["y", "c", "i", "pi"]].plot(ax=axes[0, 1], title="Regime 1 (Hawkish)")
    irfs_regime0[["r"]].plot(ax=axes[1, 0], title="Interest Rate (Regime 0)")
    irfs_regime1[["r"]].plot(ax=axes[1, 1], title="Interest Rate (Regime 1)")

    plt.tight_layout()
    plt.show()

finally:
    Path(temp_file).unlink()
