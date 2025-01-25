import math
import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x):
    """Computes the logistic function for a given input x."""
    return 1 / (1 + math.exp(-x))

# Generate a range of x values
x_values = np.linspace(-10, 10, 400)
# Compute the logistic function for each x value
y_values = [logistic_function(x) for x in x_values]

# Plot the logistic function
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='Logistic Function', color='blue')
plt.title('Logistic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.show()