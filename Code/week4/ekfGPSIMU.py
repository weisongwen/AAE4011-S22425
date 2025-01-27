import numpy as np
import matplotlib.pyplot as plt

# Define the state transition function
def f(state, dt):
    x, y, vx, vy = state
    return np.array([x + vx * dt, y + vy * dt, vx, vy])

# Define the observation function for GPS
def h_gps(state):
    x, y, vx, vy = state
    return np.array([x, y])

# Define the observation function for IMU
def h_imu(state):
    x, y, vx, vy = state
    return np.array([vx, vy])

# Define the Extended Kalman Filter
class EKF:
    def __init__(self, state_dim, obs_dim_gps, obs_dim_imu):
        self.state_dim = state_dim
        self.obs_dim_gps = obs_dim_gps
        self.obs_dim_imu = obs_dim_imu
        self.state = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        self.Q = np.eye(state_dim) * 0.1  # Process noise covariance
        self.R_gps = np.eye(obs_dim_gps) * 0.5  # GPS measurement noise covariance
        self.R_imu = np.eye(obs_dim_imu) * 0.1  # IMU measurement noise covariance

    def predict(self, dt):
        F = np.eye(self.state_dim)
        F[0, 2] = dt
        F[1, 3] = dt
        self.state = f(self.state, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, z):
        H = np.zeros((self.obs_dim_gps, self.state_dim))
        H[0, 0] = 1
        H[1, 1] = 1
        y = z - h_gps(self.state)
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def update_imu(self, z):
        H = np.zeros((self.obs_dim_imu, self.state_dim))
        H[0, 2] = 1
        H[1, 3] = 1
        y = z - h_imu(self.state)
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

# Simulate GPS and IMU data
np.random.seed(42)
dt = 0.1
time_steps = 100

true_states = []
gps_measurements = []
imu_measurements = []

state = np.array([0, 0, 1, 1])
for t in range(time_steps):
    true_states.append(state.copy())
    gps_measurements.append(h_gps(state) + np.random.randn(2) * 0.5)
    imu_measurements.append(h_imu(state) + np.random.randn(2) * 0.1)
    state = f(state, dt)

true_states = np.array(true_states)
gps_measurements = np.array(gps_measurements)
imu_measurements = np.array(imu_measurements)

# Apply EKF to fuse GPS and IMU data
ekf = EKF(state_dim=4, obs_dim_gps=2, obs_dim_imu=2)
estimated_states = []

for t in range(time_steps):
    ekf.predict(dt)
    ekf.update_gps(gps_measurements[t])
    ekf.update_imu(imu_measurements[t])
    estimated_states.append(ekf.state.copy())

estimated_states = np.array(estimated_states)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(true_states[:, 0], true_states[:, 1], label='True Position')
plt.scatter(gps_measurements[:, 0], gps_measurements[:, 1], label='GPS Measurements', color='r', s=10)
plt.plot(estimated_states[:, 0], estimated_states[:, 1], label='Estimated Position (EKF)', linestyle='--')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.title('GPS/IMU Fusion using EKF')
plt.show()
