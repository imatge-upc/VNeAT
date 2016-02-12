from GLM import GLM, PolyGLM as PGLM
import numpy as np
from matplotlib.pyplot import plot, show

x = np.linspace(-4, 4, 100)
xdata = np.array([x**i for i in xrange(4)])

params = np.random.uniform(-5, 5, 4)
params2 = np.random.normal(0, 10, 4)

y = np.array(sum(xdata[i]*params[i] for i in xrange(xdata.shape[0])))
y2 = np.array(sum(xdata[i]*params2[i] for i in xrange(len(xdata))))
plot(x, y, 'bs', x, y2, 'rs')
show()

scale1 = y.max() - y.min()
scale2 = y2.max() - y2.min()
ydata = np.array([y + np.random.normal(0, 0.15*scale1, y.shape), y2 + np.random.normal(0, 0.15*scale2, y2.shape)])

plot(x, y, 'bs', x, ydata[0], 'g^')
show()

plot(x, y2, 'bs', x, ydata[1], 'g^')
show()

glm = GLM(xdata[1:].T)
glm.orthonormalize_all()
glm.fit(ydata.T)

corrected_data = glm.correct(ydata.T)
prediction = glm.predict()

yc = y - float(sum(y))/len(y)
y2c = y2 - float(sum(y2))/len(y2)

plot(x, yc, 'cs', x, corrected_data[:, 0], 'y^', x, prediction[:, 0], 'mo')
show()
plot(x, y2c, 'bs', x, corrected_data[:, 1], 'g^', x, prediction[:, 1], 'ro')
show()

# PolyGLM

pxdata = np.array([x]).T
pglm = PGLM(pxdata, degrees = [3])
pglm.orthonormalize_all()
pglm.fit(ydata.T)

corrected_pdata = pglm.correct(ydata.T)
pprediction = pglm.predict()

plot(x, yc, 'cs', x, corrected_pdata[:, 0], 'y^', x, pprediction[:, 0], 'mo')
show()
plot(x, y2c, 'bs', x, corrected_pdata[:, 1], 'g^', x, pprediction[:, 1], 'ro')
show()

print 'Mean energy of the difference (GLM - PGLM) in corrected data:', sum((corrected_pdata - corrected_data)**2)/len(corrected_data)
print 'Mean energy of the difference (GLM - PGLM) in predicted data:', sum((prediction - pprediction)**2)/len(prediction)

# perr = np.sqrt(np.diag(glm.opt_params_cov))
# pcorr = np.array([[glm.opt_params_cov[i][j]/(perr[i]*perr[j]) for j in range(len(perr))] for i in range(len(perr))])


# import pdb
# pdb.pm() # For after the exception
# consult the following website for further information: https://pymotw.com/2/pdb/


