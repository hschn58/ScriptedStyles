import numpy as np
import matplotlib.pyplot as plt

# Set parameters
num_waves = 100
vertical_lines = 5
base_amplitude = 1.0
base_phase = 0.5  # in radians
dpi = 300

cmapp = "plasma_r"

# Define decrements so amplitude and phase decrease gradually
amp_decrement = 1 / num_waves
phase_decrement = 1 / num_waves

# Define x positions for vertical lines.
x_start = 0  
separation = np.pi/2
x_positions = [x_start + i * separation for i in range(vertical_lines)]

# Determine a square domain:
# Assume maximum horizontal deviation is ~base_amplitude, so let the x-domain be:
x_dom_min = x_positions[0] - base_amplitude
x_dom_max = x_positions[-1] + base_amplitude

# Now choose y to cover the same interval, making a square coordinate system.
y = np.linspace(x_dom_min, x_dom_max, 1000)

# Create a square canvas.
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')

# Set up two colormaps (here using the reversed plasma)


cmap1 = plt.get_cmap(cmapp)
cmap2 = plt.get_cmap(cmapp)

colormap = cmapp
cut = 1

# Draw the strokes.
for i in range(num_waves, -1, -1):
    amplitude = base_amplitude - i * amp_decrement
    phase = base_phase - i * phase_decrement
    
    for x_base in x_positions:
        # Alternate extra phase shift based on the wave index.
        if i % 2 == 0:
            add = 0
            color = cmap1(cut*(i / num_waves) + (1-cut)*0.5)
        else:
            add = np.pi/4
            color = cmap2(cut*(i / num_waves) + (1-cut)*0.5)
        
        # Vertical propagation: x oscillates around x_base, y runs over our square domain.
        x_curve      = x_base + amplitude * np.sin(y )
        x_curve_neg  = x_base - amplitude * np.sin(y )
        x_curve_sh   = x_base + amplitude * np.sin(y )
        x_curve_sh_n = x_base - amplitude * np.sin(y ) 
        
        #positives
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)

        #negatives

        x_curve      = x_base + amplitude * np.sin(y + np.pi)
        x_curve_neg  = x_base - amplitude * np.sin(y + np.pi)
        x_curve_sh   = x_base + amplitude * np.sin(y + np.pi)
        x_curve_sh_n = x_base - amplitude * np.sin(y + np.pi)
                
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)


        #amplitude is 1/2

        # Vertical propagation: x oscillates around x_base, y runs over our square domain.
        param = 2
        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y )
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y ) 
        
        #positives
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)

        #negatives

        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
                
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)


        # Vertical propagation: x oscillates around x_base, y runs over our square domain.
        param = 3
        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y )
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y ) 
        
        #positives
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)

        #negatives

        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
                
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)


 # Vertical propagation: x oscillates around x_base, y runs over our square domain.
        param = 4
        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y )
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y ) 
        
        #positives
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)

        #negatives

        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
                
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)

        param = 4
        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y )
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y ) 
        
        #positives
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)

        #negatives

        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
                
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)


        param = 5
        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y )
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y ) 
        
        #positives
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)

        #negatives

        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
                
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)


        param = 6
        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y )
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y )
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y ) 
        
        #positives
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)

        #negatives

        x_curve      = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_neg  = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh   = x_base + amplitude*(1/param) * np.sin(param*y + np.pi)
        x_curve_sh_n = x_base - amplitude*(1/param) * np.sin(param*y + np.pi)
                
        ax.plot(x_curve,      y, color=color, linewidth=0.1)
        ax.plot(x_curve_neg,  y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh,   y, color=color, linewidth=0.1)
        ax.plot(x_curve_sh_n, y, color=color, linewidth=0.1)
        
        # Horizontal propagation: now the roles are swapped.
        # x runs over our square domain and y oscillates around x_base.
        ax.plot(y, x_curve,      color=color, linewidth=0.1)
        ax.plot(y, x_curve_neg,  color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh,   color=color, linewidth=0.1)
        ax.plot(y, x_curve_sh_n, color=color, linewidth=0.1)








        

#ax.set_title("Abstract Vertical Lines on a Square Canvas")
#ax.set_xlabel("X")
#ax.set_ylabel("Y")
ax.axis("off")
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)


ax = plt.gca()
current_xlim = ax.get_xlim()
current_ylim = ax.get_ylim()

# print(f"Current x limits: {current_xlim}")
# print(f"Current y limits: {current_ylim}")

delta=0.40
ax.set_xlim(current_xlim[0]+delta, current_xlim[1]-delta)
ax.set_ylim(current_ylim[0]+delta, current_ylim[1]-delta)


from ScriptedStyles.Designs.Releases.ismethods.check import unique_filename
import os


dirname = os.path.dirname(__file__)

# Define the path to the images folder
images_dir = os.path.join(dirname, 'images')

# Check if the folder exists; if not, create it.
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Create the filename in the images folder.
#"name "filename = os.path.basename(__file__).split('.')[0] + '.jpg'
code_name = os.path.basename(__file__).split('.')[0] + f' {colormap}' + '.jpg'

filename = unique_filename(os.path.join(images_dir, code_name))
plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.0)


