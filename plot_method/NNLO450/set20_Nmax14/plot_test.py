import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt


fig = plt.figure()
#plt.subplot(121)
#rect = plt.Rectangle((0.1,0.1), 0.5, 0.3, fill=False, edgecolor = 'red',linewidth=1)
#ax = plt.gca()
#ax.add_patch(rect)
x_list= np.linspace(65,75,1000)
y_list= np.zeros(1000)
for loop  in range(0,1000):
    y_list[loop] = 0.001079*pow(x_list[loop],4)-0.2974*pow(x_list[loop],3)+30.7*pow(x_list[loop],2)-1406*x_list[loop]+24130

l = plt.plot(x_list,y_list)


print(x_list)


plot_path = 'test.eps'
plt.savefig(plot_path)
plt.show()
