import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        """
        Initialize the PID controller.

        :param Kp: Proportional gain
        :param Ki: Integral gain
        :param Kd: Derivative gain
        :param setpoint: Desired setpoint
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, measured_value, dt):
        """
        Compute the control signal.

        :param measured_value: The current measured value
        :param dt: Time step (delta time)
        :return: Control signal
        """
        error = self.setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        # PID control signal
        control_signal = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)

        self.previous_error = error

        return control_signal

# Drone simulation parameters
initial_altitude = 0.0  # Starting altitude
target_altitude = 10.0  # Desired altitude
mass = 1.0  # Mass of the drone (kg)
gravity = 9.81  # Gravity (m/s^2)
dt = 0.1  # Time step (s)
simulation_time = 50.0  # Total simulation time (s)

# PID gains (tuned for the drone)
Kp = 2.0
Ki = 0.1
Kd = 0.5

# Initialize PID controller
pid = PIDController(Kp, Ki, Kd, target_altitude)

# Simulation variables
time_steps = np.arange(0, simulation_time, dt)
altitude = initial_altitude
velocity = 0.0
altitudes = []
control_signals = []

# Create figure and axis for animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_xlim(0, simulation_time)
ax1.set_ylim(0, target_altitude + 2)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Altitude (m)")
ax1.set_title("Drone Altitude Control with PID")
ax1.grid()

ax2.set_xlim(0, simulation_time)
ax2.set_ylim(0, 20)  # Adjust based on expected thrust range
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Thrust (N)")
ax2.set_title("Control Signal (Thrust) Over Time")
ax2.grid()

# Initialize lines for animation
line1, = ax1.plot([], [], label="Altitude (m)")
line2, = ax2.plot([], [], label="Thrust (N)", color="orange")
target_line = ax1.axhline(y=target_altitude, color="r", linestyle="--", label="Target Altitude")
ax1.legend()
ax2.legend()

# Function to initialize the animation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# Function to update the animation
def animate(i):
    global altitude, velocity

    # Compute control signal (thrust)
    control_signal = pid.compute(altitude, dt)

    # Simulate drone dynamics (simple physics model)
    thrust = control_signal
    net_force = thrust - mass * gravity  # Net force = thrust - gravity
    acceleration = net_force / mass  # F = ma
    velocity += acceleration * dt
    altitude += velocity * dt

    # Store results for plotting
    altitudes.append(altitude)
    control_signals.append(control_signal)

    # Update plots
    line1.set_data(time_steps[:i+1], altitudes)
    line2.set_data(time_steps[:i+1], control_signals)
    return line1, line2

# Create animation
ani = FuncAnimation(fig, animate, frames=len(time_steps), init_func=init, blit=True, interval=dt*1000, repeat=False)

# Show the animation
plt.tight_layout()
plt.show()