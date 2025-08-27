from func import get_iter
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib 

matplotlib.use('Agg')
#
#add color scaling-assign color to each line based on its final value on the right side.
#
rang = 10000
prob=0.5
num=10000

y_vals=[]

for n in range(num):
    y_vals+=[get_iter(rang,prob,n)]

fig=plt.figure()

x_vals=np.linspace(1,rang, rang)

colormap=plt.get_cmap('binary')

for n in range(num):
    plt.plot(x_vals,y_vals[n], color=colormap(random.random()), linewidth=0.2*1000/rang)

#plt.style.use('dark_background')
plt.axis('off')

plt.subplots_adjust(left=0.15)

plt.savefig('/Users/henryschnieders/Desktop/proj/Designs/Releases/June 2024/Random Walks/Results/No sum/binary_02_4.png',transparent=True, dpi=5400)
#plt.show()