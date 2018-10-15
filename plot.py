import numpy as np
from matplotlib import pyplot as plt
x = np.arange(0, 3 * np.pi,0.1)
siny=np.sin(x)
cosy=np.cos(x)
plt.plot(x,siny,label='sin')
plt.plot(x,cosy,label='cos')
plt.title('sin and cos')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.legend(loc=4)
plt.axis([0,9,-3,3])
plt.show()
