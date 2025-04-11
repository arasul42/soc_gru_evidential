import numpy as np
import matplotlib.pyplot as plt

class ParticleFilterSOC:
    def __init__(self, num_particles, Cn, eta, delta_t, process_noise_fn, measurement_noise_fn):
        self.N = num_particles
        self.particles = np.random.uniform(0, 1, self.N)
        self.weights = np.ones(self.N) / self.N
        self.Cn = Cn
        self.eta = eta
        self.delta_t = delta_t
        self.process_noise_fn = process_noise_fn
        self.measurement_noise_fn = measurement_noise_fn

    def predict(self, current):
        # SOC prediction step with process noise
        delta_soc = (self.eta * current * self.delta_t) / self.Cn
        process_noise = self.process_noise_fn(self.N)
        self.particles -= delta_soc
        self.particles += process_noise
        self.particles = np.clip(self.particles, 0, 1)

    def update(self, measurement):
        # Measurement likelihood (Laplace example)
        likelihoods = self.measurement_noise_fn(measurement - self.particles)
        self.weights *= likelihoods
        self.weights += 1.e-300  # avoid zeros
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Systematic resampling
        cumulative_sum = np.cumsum(self.weights)
        step = 1.0 / self.N
        start = np.random.uniform(0, step)
        positions = (start + np.arange(self.N) * step)
        indexes = np.searchsorted(cumulative_sum, positions)
        self.particles[:] = self.particles[indexes]
        self.weights[:] = 1.0 / self.N

    def estimate(self):
        return np.sum(self.particles * self.weights)

# Define process and measurement noise functions
def laplace_noise(n, scale=0.01):
    return np.random.laplace(loc=0.0, scale=scale, size=n)

def laplace_likelihood(errors, b=0.02):
    return (1 / (2 * b)) * np.exp(-np.abs(errors) / b)

# Simulation setup
np.random.seed(0)
timesteps = 50
true_soc = 0.9
current_profile = np.random.uniform(0.5, 2.0, timesteps)
gru_predictions = []

# Initialize particle filter
pf = ParticleFilterSOC(
    num_particles=500,
    Cn=3.2 * 3600,
    eta=0.99,
    delta_t=1.0,
    process_noise_fn=lambda n: laplace_noise(n, scale=0.005),
    measurement_noise_fn=laplace_likelihood
)

estimates = []

# Simulate
for t in range(timesteps):
    current = current_profile[t]
    # True SOC update
    true_soc -= (0.99 * current * 1.0) / (3.2 * 3600)
    true_soc = max(0.0, min(1.0, true_soc))

    # Simulate noisy GRU output
    noisy_measurement = true_soc + np.random.laplace(0, 0.02)
    gru_predictions.append(noisy_measurement)

    # Particle filter steps
    pf.predict(current)
    pf.update(noisy_measurement)
    pf.resample()
    estimates.append(pf.estimate())

# Plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({
    "Time": np.arange(timesteps),
    "True SOC": [true_soc - sum(current_profile[:t+1]) * 0.99 / (3.2 * 3600) for t in range(timesteps)],
    "GRU Output": gru_predictions,
    "PF Estimate": estimates
})
# import ace_tools as tools; tools.display_dataframe_to_user(name="SOC Estimation Results", dataframe=df)

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Time", y="True SOC", label="True SOC", linestyle='--')
sns.lineplot(data=df, x="Time", y="GRU Output", label="Noisy GRU Output", alpha=0.6)
sns.lineplot(data=df, x="Time", y="PF Estimate", label="Particle Filter Estimate")
plt.title("SOC Estimation using Particle Filter")
plt.ylabel("SOC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
