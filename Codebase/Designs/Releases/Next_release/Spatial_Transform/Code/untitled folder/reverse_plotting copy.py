import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# You will need Shapely for the union:
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def base_shape(num_points=200):
    x = np.linspace(-2, 2, num_points)
    y = np.exp(-x**2)
    return x, y, -y

def transform(x, y, offset_x=0.0, offset_y=0.0, scale_x=1.0, scale_y=1.0, theta=0.0):
    x_s = x * scale_x
    y_s = y * scale_y
    # Rotate
    x_r = x_s * np.cos(theta) - y_s * np.sin(theta)
    y_r = x_s * np.sin(theta) + y_s * np.cos(theta)
    # Translate
    x_t = x_r + offset_x
    y_t = y_r + offset_y
    return x_t, y_t

# --------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))

x_base, y_top_base, y_bot_base = base_shape(num_points=300)

# For demonstration, freq=1
freq = 1
R = 10
n_shapes = 2000
partition = n_shapes // freq  # with freq=1, partition=n_shapes

# We'll store each shape (as a Shapely polygon) here:
polygons_for_union = []

for i in range(partition):
    # offset/scale/rotation
    offset_x = R * np.cos(2*np.pi*(i/n_shapes))
    offset_y = R * np.sin(2*np.pi*(i/n_shapes))
    scale_x  = 0.5*np.sin((freq+0.25)*2*np.pi*(i/partition)) + 1
    scale_y  = 0.5*np.sin((freq+0.25)*2*np.pi*(i/partition)) + 1
    rotation_deg = (np.pi/9) * np.sin(2*np.pi*(i/partition))

    # Top curve
    x_top, y_top = transform(x_base, y_top_base,
                             offset_x, offset_y, scale_x, scale_y, rotation_deg)
    # Bottom curve
    x_bot, y_bot = transform(x_base, y_bot_base,
                             offset_x, offset_y, scale_x, scale_y, rotation_deg)

    # Ensure left->right ordering for fill_between
    # (purely for consistency, but not absolutely required for union)
    if x_top[0] > x_top[-1]:
        x_top, y_top = x_top[::-1], y_top[::-1]
        x_bot, y_bot = x_bot[::-1], y_bot[::-1]

    # Create the polygon as if we walked around the shape:
    #   top boundary left->right, bottom boundary right->left
    # This ensures a closed loop in order.
    top_coords = np.column_stack((x_top, y_top))
    bot_coords = np.column_stack((x_bot[::-1], y_bot[::-1]))

    # Combine them to make a single list of boundary coordinates
    boundary_coords = np.vstack((top_coords, bot_coords))

    # Convert boundary_coords -> Shapely Polygon
    shape_poly = Polygon(boundary_coords)
    polygons_for_union.append(shape_poly)

# Union all polygons into one
merged_poly = unary_union(polygons_for_union)

# The result could be one Polygon or multiple disjoint pieces (MultiPolygon).
# In typical usage, it should be one continuous boundary. But we handle both cases.
if isinstance(merged_poly, MultiPolygon):
    # If multiple pieces, fill each one
    for subpoly in merged_poly:
        x_, y_ = subpoly.exterior.xy
        ax.fill(x_, y_, color='mediumorchid')  # or any color
else:
    # Single polygon
    x_, y_ = merged_poly.exterior.xy
    ax.fill(x_, y_, color='mediumorchid')

# Some optional styling
ax.set_aspect('equal', 'datalim')
plt.axis('off')
plt.show()
