import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
import matplotlib

matplotlib.use('Agg')

#plasma = colormaps['plasma']


















from ScriptedStyles.Designs.Releases.ismethods.check import unique_filename
import os

dirname = os.path.dirname(__file__)

# Define the path to the images folder
images_dir = os.path.join(dirname, 'images')

# Check if the folder exists; if not, create it.
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Create the filename in the images folder.
filename = unique_filename(os.path.join(images_dir, 'fun_quilt.jpg'))
plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.0)


import pandas as pd
df = pd.read_csv('/Users/henryschnieders/Documents/ScriptedStyles/Automation/Printify/Data/product_information.csv')
df['local_path'].iloc[0] = filename