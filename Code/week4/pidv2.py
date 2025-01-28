import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
simulation_time = 100.0  # Total simulation time (s)

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

# Create figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)  # Horizontal range
ax.set_ylim(0, target_altitude + 2)  # Vertical range
ax.set_xlabel("X Position")
ax.set_ylabel("Altitude (m)")
ax.set_title("Drone Altitude Control with PID")
ax.grid()

# Load drone logo (replace 'drone.png' with your image file)
drone_image = plt.imread("drone.jpg")  # Ensure the image file is in the same directory
imagebox = OffsetImage(drone_image, zoom=0.1)  # Adjust zoom for image size
drone_logo = AnnotationBbox(imagebox, (5, initial_altitude), frameon=False)  # Initial position
ax.add_artist(drone_logo)

# Function to initialize the animation
def init():
    drone_logo.xy = (5, initial_altitude)  # Reset drone position
    return drone_logo,

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

    # Update drone logo position
    drone_logo.xy = (5, altitude)  # Move drone vertically (x position fixed)

    return drone_logo,

# Create animation
ani = FuncAnimation(fig, animate, frames=len(time_steps), init_func=init, blit=True, interval=dt*1000, repeat=False)

# Show the animation
plt.tight_layout()
plt.show()