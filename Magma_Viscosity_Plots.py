# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 18:04:13 2025

@author: Jason

"""

import numpy as np
import matplotlib.pyplot as plt

viscosity = [10.57**10, 7.72**10, 6.63**10, 5.87**10, 5.28**10, 4.79**10, 4.36**10]
wt_percent = [0, 1, 2, 3, 4, 5, 6]

plt.scatter(wt_percent, viscosity)
plt.yscale("log")   # log10 y-axis
plt.xlabel("$H_{2}O$ Weight %")
plt.ylabel("Viscosity [Pa s]")
plt.title("Rhyolite Viscosity vs. $H_{2}O$ Weight %")
plt.grid()
plt.show()




viscosity = [5.48**10, 3.90**10, 2.75**10, 1.88**10, 1.19**10, 0.64**10]
Temperature = [800, 900, 1000, 1100, 1200, 1300]

plt.scatter(Temperature, viscosity)
plt.yscale("log")   # log10 y-axis
plt.xlabel("Temperature [$°C$]")
plt.ylabel("Viscosity [Pa s]")
plt.title("Basalt (0.5% $H_{2}O$) Viscosity vs. Temperature [$°C$]")
plt.grid()
plt.show()