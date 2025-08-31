from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def superformula_3d(m, n1, n2, n3, points=500):
    theta = np.linspace(0, np.pi, points)
    phi = np.linspace(0, 2 * np.pi, points)
    theta, phi = np.meshgrid(theta, phi)

    r1 = superformula_radius(theta, m, n1, n2, n3)
    r2 = superformula_radius(phi, m, n1, n2, n3)

    x = r1 * np.sin(theta) * np.cos(phi)
    y = r1 * np.sin(theta) * np.sin(phi)
    z = r2 * np.cos(theta)
    return x, y, z

def superformula_radius(angle, m, n1, n2, n3):
    a = b = 1
    term1 = np.abs(np.cos(m * angle / 4) / a) ** n2
    term2 = np.abs(np.sin(m * angle / 4) / b) ** n3
    r = (term1 + term2) ** (-1 / n1)
    return r

x, y, z = superformula_3d(m=7, n1=0.2, n2=1.7, n3=1.7)

fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.axis('off')
plt.show()
