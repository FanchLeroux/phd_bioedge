import numpy as np
import matplotlib.pyplot as plt # type: ignore

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.sin(1.2*x + np.pi)

plt.figure(1)
plt.plot(x,y1,'r')
plt.plot(x,y2,'g')
plt.show()
