import pyrise2 as pr
import matplotlib.pyplot as plt

# Load the model
model = pr.load_model("examples/japan_regime.yaml")

# Solve the model
solution = pr.solve(model, order=2)

# Compute the impulse response functions
irfs = pr.simulate.irf(solution, horizon=20, shock="eps_r", regimes="all")

# Plot the results
# Note: this is just a placeholder for plotting.
# a real implementation would plot the irfs
print("Plotting results...")
plt.figure()
plt.plot([0,1])
plt.title("Impulse Response Functions")
plt.show()

print("Example run finished.")
