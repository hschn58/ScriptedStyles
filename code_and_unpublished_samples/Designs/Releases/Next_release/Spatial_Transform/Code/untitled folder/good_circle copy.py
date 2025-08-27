import numpy as np
import matplotlib.pyplot as plt

def base_shape(num_points=200):
    """
    Returns x, y_top, y_bottom for a lobed shape.
    In this example, we use a Gaussian-like shape: y = exp(-x^2).
    """
    x = np.linspace(-2, 2, num_points)
    y = np.exp(-x**2)
    return x, y, -y

def transform(x, y, 
              offset_x=0.0, 
              offset_y=0.0, 
              scale_x=1.0, 
              scale_y=1.0, 
              theta=0.0):
    """
    Applies translation, scaling, and rotation to an array of coordinates.
    x, y should be 1D arrays of the same length.
    """
    # Scale
    x_s = x * scale_x
    y_s = y * scale_y
    # Rotate
    x_r = x_s * np.cos(theta) - y_s * np.sin(theta)
    y_r = x_s * np.sin(theta) + y_s * np.cos(theta)
    # Translate
    x_t = x_r + offset_x
    y_t = y_r + offset_y
    return x_t, y_t

# --- Main plotting routine ---
fig, ax = plt.subplots(figsize=(8, 4))

# Get our base shape
x_base, y_top_base, y_bot_base = base_shape(num_points=300)

# Number of shapes to plot
n_shapes=2000

# Use a color map across all shapes
import matplotlib.cm as cm

my_cmap = cm.get_cmap('plasma')
colors = plt.cm.plasma(np.linspace(0, 1, n_shapes))

# Track the overall min/max for x and y so we can set the axis limits
min_x, max_x = float('inf'), float('-inf')
min_y, max_y = float('inf'), float('-inf')

R=10
freq=5
inc=2*np.pi/freq
partition=n_shapes//freq

lss=[]

offset_x, offset_y, scale_x, scale_y, rotation_deg = 0, 0, 1, 1, 0
globals = [offset_x, offset_y, scale_x, scale_y, rotation_deg]

def offx(offset_x, R, iter, n_shapes):
    return offset_x + R*np.cos(2*np.pi*(iter/n_shapes)+inc*j)*(1-(iter/n_shapes))-1

def offy(offset_y, R, iter, n_shapes):
    return offset_y-0.25 + R*np.sin(2*np.pi*(iter/n_shapes)+inc*j)*(1-(iter/n_shapes))

def scalx(scale_x, freq, iter, partition, n_shapes):
    return scale_x + abs(np.sin((abs(freq*0.1))*2*np.pi*(iter/partition))*np.sin((abs(freq*3))*2*np.pi*(iter/n_shapes))+1)

def scaly(scale_y, freq, iter, partition, n_shapes):
    return scale_y + abs(np.cos((abs(freq*0.1))*2*np.pi*(iter/partition))*np.sin((abs(freq*3))*2*np.pi*(iter/n_shapes))+1)

def rot(rotation_deg, iter, n_shapes):
    return rotation_deg + (np.pi/9)*np.sin(2*np.pi*(iter/n_shapes))


# offset_x = R*np.cos(2*np.pi*(iter/num1)+inc*j)*(1-(iter/num1))-1
# offset_y = R*np.sin(2*np.pi*(iter/num1)+inc*j)*(1-(iter/num1))
# scale_x  = abs(np.sin((abs(freq*0.1))*2*np.pi*(iter/num1))*np.sin((abs(freq*3))*2*np.pi*(iter/num1))+1)
# scale_y  = abs(np.cos((abs(freq*0.1))*2*np.pi*(iter/num1))*np.sin((abs(freq*3))*2*np.pi*(iter/num1))+1)

for i in range(partition):

    #first transform set
    for j in range(freq):
        # Define how the offset, scaling, and rotation vary with i
        offset_x = R*np.cos(2*np.pi*(i/n_shapes)+inc*j)
        offset_y = R*np.sin(2*np.pi*(i/n_shapes)+inc*j)
        scale_x  = 0.5*np.sin((freq+0.25)*2*np.pi*(i/partition))+1
        scale_y  = 0.5*np.sin((freq+0.25)*2*np.pi*(i/partition))+1
        rotation_deg = (np.pi/9)*np.sin(2*np.pi*(i/partition))

        # Apply transform to top and bottom curves
        x_top, y_top = transform(x_base, y_top_base,
                                offset_x, offset_y,
                                scale_x, scale_y,
                                rotation_deg)
        x_bot, y_bot = transform(x_base, y_bot_base,
                                offset_x, offset_y,
                                scale_x, scale_y,
                                rotation_deg)
        
        # If the shape got flipped in x, reorder so fill_between works properly
        if x_top[0] > x_top[-1]:
            x_top, y_top = x_top[::-1], y_top[::-1]
            x_bot, y_bot = x_bot[::-1], y_bot[::-1]

        # Fill between the top and bottom curves
        ax.fill_between(x_top, y_top, y_bot, color=my_cmap(np.sin(10*np.pi*(i/partition))), alpha=1, linewidth=0)


        #int((int((n_shapes-1)*np.sin((i/tau)*np.pi))+1)*0.5)
        # Track min/max of all x and y
        min_x = min(min_x, x_top.min(), x_bot.min())
        max_x = max(max_x, x_top.max(), x_bot.max())
        min_y = min(min_y, y_top.min(), y_bot.min())
        max_y = max(max_y, y_top.max(), y_bot.max())

        
        if i == partition-1:
            for iter in range(n_shapes):
                
                ###second transform

                offset_x0 = offx(offset_x, R, iter, n_shapes)
                offset_y0 = offy(offset_y, R, iter, n_shapes)
                scale_x0  = scalx(scale_x, freq, iter, partition, n_shapes)*0.85  
                scale_y0  = scaly(scale_y, freq, iter, partition, n_shapes)*0.85
                rotation_deg0 = rot(rotation_deg, iter, n_shapes)
                # offset_x = R*np.cos(2*np.pi*(i/n_shapes)+inc*j) 
                # offset_y = R*np.sin(2*np.pi*(i/n_shapes)+inc*j)
                # scale_x  = 0.5*np.sin((freq+0.25)*2*np.pi*(i/partition))+1
                # scale_y  = 0.5*np.sin((freq+0.25)*2*np.pi*(i/partition))+1
                # rotation_deg = (np.pi/9)*np.sin(2*np.pi*(i/partition))

                ###

                        # Apply transform to top and bottom curves
                x_top, y_top = transform(x_base, y_top_base,
                                        offset_x0, offset_y0,
                                        scale_x0, scale_y0,
                                        rotation_deg0)
                x_bot, y_bot = transform(x_base, y_bot_base,
                                        offset_x0, offset_y0,
                                        scale_x0, scale_y0,
                                        rotation_deg0)
                
                # If the shape got flipped in x, reorder so fill_between works properly
                if x_top[0] > x_top[-1]:
                    x_top, y_top = x_top[::-1], y_top[::-1]
                    x_bot, y_bot = x_bot[::-1], y_bot[::-1]

                ax.fill_between(x_top*(1-(iter/n_shapes)), y_top*(1-(iter/n_shapes)), y_bot*(1-(iter/n_shapes)), color=my_cmap(np.sin(30*np.pi*(iter/n_shapes))), alpha=1, linewidth=0, zorder=2)


# Now set the axis limits based on min/max valuesnp.sin(30*np.pi*(iter/num1))
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_aspect('equal', 'datalim')
plt.axis('off')

plt.show()
