import numpy as np
from matplotlib.pyplot import plot, show

from curve_fit import GLM

params = np.random.uniform(0, 10, 4)
x = np.linspace(0, 4, 100)

xdata = []
for i in range(4):
    xdata.append(x ** i)

xdata = np.array(xdata)

y = np.array(sum(xdata[i] * params[i] for i in range(xdata.shape[0])))
plot(x, y, 'bs')
show()

ydata = y + np.random.normal(0, 20, y.shape)

plot(x, y, 'bs', x, ydata, 'g^')
show()

glm = GLM(xdata, ydata)
glm.optimize()

plot(x, y, 'bs', x, ydata, 'g^', x, glm.pred_function(glm.xdata, *glm.opt_params), 'ro')
show()

perr = np.sqrt(np.diag(glm.opt_params_cov))
pcorr = np.array([[glm.opt_params_cov[i][j] / (perr[i] * perr[j]) for j in range(len(perr))] for i in range(len(perr))])


# import pdb
# pdb.pm() # For after the exception
# consult the following website for further information: https://pymotw.com/2/pdb/
