#!/usr/bin/env python
# coding: utf-8

# # What is Matplotlib?

# Matplotlib is a low level graph plotting library in python that serves as a visualization utility.

# # Import Matplotlib and Checking Matplotlib Version

# In[ ]:


import matplotlib

print(matplotlib.__version__)


# # Pyplot

# Most of the Matplotlib utilities lies under the pyplot submodule, and are usually imported under the plt alias

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,6])
y = np.array([0,250])

plt.plot(x,y)
plt.show()


# # Matplotlib Plotting

# Plotting x and y points
# 
# The plot() function is used to draw points (markers) in a diagram.
# 
# By default, the plot() function draws a line from point to point.
# 
# The function takes parameters for specifying points in the diagram.
# 
# Parameter 1 is an array containing the points on the x-axis.
# 
# Parameter 2 is an array containing the points on the y-axis.
# 
# If we need to plot a line from (3, 8) to (10, 15), we have to pass two arrays [1, 8] and [3, 10] to the plot function.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([3,10])
y = np.array([8,15])

plt.plot(x,y)
plt.show()


# # Plotting Without Line

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([3,10])
y = np.array([8,15])

plt.plot(x,y,'o')
plt.show()


# # Multiple Points

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 6, 9,12])
y = np.array([3, 7, 1, 15,5])

plt.plot(x,y)
plt.show()


# # Default X-Points

# The x-points in the example above is [0, 1, 2, 3, 4, 5]

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([3, 8, 1, 10, 5, 7])

plt.plot(y)
plt.show()


# # Matplotlib Markers

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 3, 7, 9])
y = np.array([4, 8, 10, 13])

plt.plot(x,y, marker = 'o')
plt.show()


# # Format Strings fmt

# You can use also use the shortcut string notation parameter to specify the marker.
# 
# This parameter is also called fmt, and is written with this syntax:
# 
# marker | line | color

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([3, 8, 1, 10])

plt.plot(y,'o:r') # plt.plot(y,'o:')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([3, 8, 1, 10])

plt.plot(y,'o-.r') # plt.plot(y,'o-.')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([3, 8, 1, 10])

plt.plot(y,'o--') # plt.plot(y,'o--r')
plt.show()


# # Marker Size

# The keyword argument "markersize" or the shorter version, "ms" to set the size of the markers

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([3, 8, 1, 10])

plt.plot(y,marker = 'o',ms = 20)
plt.show()


# # Marker Color

# The keyword argument "markeredgecolor" or the shorter "mec" to set the color of the edge of the markers

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r')
plt.show()


# # Marker Face Color

# The keyword argument "markerfacecolor" or the shorter "mfc" to set the color inside the edge of the markers

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o', ms = 20, mfc = 'r')
plt.show()


# All

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r', mfc = 'r')
plt.show()


# # Matplotlib Line

# The keyword argument linestyle, or shorter ls, to change the style of the plotted line

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([3, 8, 1, 10])

plt.plot(y,ls = 'dotted')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([3, 8, 1, 10])

plt.plot(y,ls = 'dashed')
plt.show()


# # Line Width

# The keyword argument linewidth or the shorter lw to change the width of the line.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, linewidth = '20.5')
plt.show()


# # Multiple Lines

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([3, 8, 1, 10])
y = np.array([6, 2, 7, 11])

plt.plot(x)
plt.plot(y)

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)
plt.show()


# # Matplotlib Labels and Title

# Create Labels and Titles

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)
plt.show()


# # Set Font Properties for Title and Labels

# Use the fontdict parameter in xlabel(), ylabel(), and title() to set font properties for the title and labels.|

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Sports Watch Data", fontdict = font1, loc = 'left')
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)

plt.plot(x, y)
plt.show()


# # Matplotlib Adding Grid Lines

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid()

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid(axis = 'x')

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid(axis = 'y')

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid(color = 'green', linestyle = '-.', linewidth = 0.5)

plt.show()


# # Matplotlib Subplot

# The subplot() Function
# 
# The subplot() function takes three arguments that describes the layout of the figure.
# 
# The layout is organized in rows and columns, which are represented by the first and second argument.
# 
# The third argument represents the index of the current plot.
# 
# 
# plt.subplot(1, 2, 1)
# #the figure has 1 row, 2 columns, and this plot is the first plot.
# 
# plt.subplot(1, 2, 2)
# #the figure has 1 row, 2 columns, and this plot is the second plot.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# plot 1:

x = np.array([0,1,2,3])
y = np.array([3,8,1,10])

plt.subplot(1,2,1)
plt.plot(x,y)
plt.title("SALES")

# plot 2:

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1,2,2)
plt.plot(x,y)
plt.title("INCOME")


plt.suptitle("MY SHOP")
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# plot 1:

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 1, 1)
plt.plot(x,y)

# plot 2:

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 1, 2)
plt.plot(x,y)

plt.suptitle("MY SHOP")
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 1)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 2)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 3)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 4)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 5)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 6)
plt.plot(x,y)

plt.show()


# # Matplotlib Scatter

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)
plt.show()


# ### Compare Plots

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

#day one, the age and speed of 13 cars:

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y)

#day two, the age and speed of 15 cars:

x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y)

plt.show()


# ### Colors

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y, color = 'red')

x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y, color = 'blue')

plt.show()


# ### Color Each Dot

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])

plt.scatter(x, y, c=colors) # use the color argument for this, only the c argument.

plt.show()


# ### Color Map

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

plt.scatter(x,y,c = colors, cmap = 'viridis')

plt.colorbar()

plt.show()


# # Size

# You can change the 'size' of the dots with the 's' argument.
# 
# Just like colors, make sure the array for sizes has the same length as the arrays for the x- and y-axis:

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])

plt.scatter(x, y, s=sizes)

plt.show()


# # Alpha

# You can adjust the transparency of the dots with the 'alpha' argument.
# 
# Just like colors, make sure the array for sizes has the same length as the arrays for the x- and y-axis

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])

plt.scatter(x, y, s=sizes, alpha=0.5)

plt.show()


# # Combine Color Size and Alpha

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(100, size=(100))
y = np.random.randint(100, size=(100))
colors = np.random.randint(100, size=(100))
sizes = 10 * np.random.randint(100, size=(100))

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')

plt.colorbar()

plt.show()


# # Matplotlib Bars

# Creating Bars

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)
plt.show()


# Horizontal Bars

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x,y)
plt.show()


# # Bar Color

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y,color = 'red')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x,y,color = 'hotpink')
plt.show()


# # Bar Width

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y,color = 'red',width = 0.3)
plt.show()


# # Bar Height

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x,y, height = 0.3,color = 'red')
plt.show()


# # Matplotlib Histograms

# Create Histogram
# 
# In Matplotlib, we use the hist() function to create histograms.
# 
# The hist() function will use an array of numbers to create a histogram, the array is sent into the function as an argument.
# 
# For simplicity we use NumPy to randomly generate an array with 250 values, where the values will concentrate around 170, and the standard deviation is 10.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(170,10,250)

plt.hist(x,color = 'red')
plt.show()


# # Matplotlib Pie Charts

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,10,15])

plt.pie(y)
plt.show()


# Labels

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,10,15])
my_labels = ["Apples","Mangoes","Bananas","Dates",]

plt.pie(y, labels = my_labels)
plt.show()


# Start Angle

# As mentioned the default start angle is at the x-axis, but you can change the start angle by specifying a "startangle" parameter.
# 
# The "startangle" parameter is defined with an angle in degrees, default angle is 0:

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,10,15])
my_labels = ["Apples","Mangoes","Bananas","Dates"]

plt.pie(y, labels = my_labels,startangle = 90)
plt.show()


# # Explode

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,10,15])
my_labels = ["Apples","Mangoes","Bananas","Dates"]
my_explode = [0.2, 0, 0, 0]

plt.pie(y, labels = my_labels, explode = my_explode)
plt.show()


# # Shadow

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,10,15])
my_labels = ["Apples","Mangoes","Bananas","Dates"]
my_explode = [0.2, 0, 0, 0]

plt.pie(y, labels = my_labels, explode = my_explode ,shadow = True)
plt.show()


# # Color

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,10,15])
my_labels = ["Apples","Mangoes","Bananas","Dates"]
my_explode = [0.2, 0, 0, 0]
my_color = ['b','m','k','y']

plt.pie(y, labels = my_labels, explode = my_explode ,shadow = True,colors = my_color)
plt.show()


# # Legend

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,10,15])
my_labels = ["Apples","Mangoes","Bananas","Dates"]
my_explode = [0.2, 0, 0, 0]
my_color = ['b','m','k','y']

plt.pie(y, labels = my_labels, explode = my_explode ,shadow = True,colors = my_color)
plt.legend()
plt.show()


# # Legend With Header

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([35,25,10,15])
my_labels = ["Apples","Mangoes","Bananas","Dates"]
my_explode = [0.2, 0, 0, 0]
my_color = ['b','m','k','y']

plt.pie(y, labels = my_labels, explode = my_explode ,shadow = True,colors = my_color)
plt.legend(title = "Four Fruites")
plt.show()

