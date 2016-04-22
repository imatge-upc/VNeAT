from curve_fit_v2 import GLM
import numpy as np
from matplotlib.pyplot import plot, show

x = np.linspace(-4, 4, 100)
xdata = np.array([x**i for i in range(4)])

params = np.random.uniform(0, 10, 4)
params2 = np.random.normal(0, 10, 4)

y = np.array(sum(xdata[i]*params[i] for i in range(xdata.shape[0])))
y2 = np.array(sum(xdata[i]*params2[i] for i in range(len(xdata))))
plot(x, y, 'bs', x, y2, 'rs')
show()

scale = 200.
ydata = np.array([y + np.random.normal(0, 0.1*scale, y.shape), y2 + np.random.normal(0, scale, y2.shape)])

plot(x, y, 'bs', x, ydata[0], 'g^')
show()

plot(x, y2, 'bs', x, ydata[1], 'g^')
show()

glm = GLM(xdata.T, ydata.T)
glm.orthonormalize()
glm.optimize()

prediction = GLM.predict(glm.xdata, glm.opt_params)
plot(x, y, 'cs', x, ydata[0], 'y^', x, prediction[:, 0], 'mo')
plot(x, y2, 'bs', x, ydata[1], 'g^', x, prediction[:, 1], 'ro')
show()


# import pdb
# pdb.pm() # For after the exception
# consult the following website for further information: https://pymotw.com/2/pdb/


