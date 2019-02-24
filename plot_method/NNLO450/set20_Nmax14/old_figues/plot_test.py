import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt


fig = plt.figure()
plt.subplot(121)
rect = plt.Rectangle((0.1,0.1), 0.5, 0.3, fill=False, edgecolor = 'red',linewidth=1)
ax = plt.gca()
ax.add_patch(rect)
plot_path = 'test.eps'
plt.savefig(plot_path)
plt.show()
