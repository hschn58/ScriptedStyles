import numpy as np
import matplotlib.pyplot as plt

def superformula(m, n1, n2, n3, points=5000):
    phi = np.linspace(0, 2 * np.pi, points)
    r = (np.abs(np.cos(m * phi / 4) / 1) ** n2 + np.abs(np.sin(m * phi / 4) / 1) ** n3) ** (-1 / n1)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

# Parameters for the Superformula
m = 5
n1 = 0.2
n2 = 1.7
n3 = 1.7

# Generate points
x, y = superformula(m, n1, n2, n3)

# Plotting
plt.figure(figsize=(10, 10), facecolor='black')
plt.plot(x, y, color='cyan', linewidth=2)
plt.fill(x, y, color='magenta', alpha=0.6)
plt.axis('equal')
plt.axis('off')
plt.show()
