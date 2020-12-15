import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.arange(-3, 3, 0.001)

plt.plot(x, norm.pdf(x))

# Multiple plots on graph
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

# Save it to a file

plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.savefig('MyPlot.png', format = 'png')

# Adjust the axes

axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -3, -2, -1, 0, 1 ,2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

# Add a grid

axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -3, -2, -1, 0, 1 ,2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axes.grid()
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()

# Change line types and Color
axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -3, -2, -1, 0, 1 ,2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axes.grid()
plt.plot(x, norm.pdf(x), 'b-')
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r-')
plt.show()

# labeling axes and adding a legend

axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -3, -2, -1, 0, 1 ,2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axes.grid()
plt.xlabel('Greebles')
plt.ylabel('Probability')
plt.plot(x, norm.pdf(x), 'b-')
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:')
plt.legend(['Sneetches', 'Gacks'], loc=6 )
plt.show()

# PIE Chart

plt.rcdefaults()

values =[30, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
explode = [0, 0, 0.2, 0, 0]
labels = ['India', 'United State', 'Russia', 'China', 'Europe']
plt.pie(values, colors= colors, labels = labels, explode= explode)
plt.title('Student location')
plt.show()

# Bar Chart

values = [30, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
plt.bar(range(0,5), values, color = colors)
plt.show()

# Scatter plot

from pylab import randn
X = randn(500)
Y = randn(500)
plt.scatter(X,Y)
plt.show()

# Histogram

incomes = np.random.normal(27000, 15000, 10000)
plt.hist(incomes, 50)
plt.show()

# Box and Wiskers plot

uniformSkewed = np.random.rand(100) * 100 - 40
high_outliners = np.random.rand(10) * 50 + 100
low_outliners = np.random.rand(10)* -50 - 100
data = np.concatenate((uniformSkewed, high_outliners, low_outliners))
plt.boxplot(data)
plt.show()
