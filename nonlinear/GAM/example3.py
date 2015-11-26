import excelIO as eIO
from curve_fit import GAM
from curve_fit import Smoother

example = 2  # 1,2 or 3

import numpy as np
import numpy.random as R
import matplotlib
import matplotlib.pyplot as plt

from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)


standardize = lambda x: (x - x.mean()) / x.std()
nobs = 150
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()
y= 0.5*R.standard_normal(nobs)#np.zeros(nobs)#

f1 = lambda x1: (1 + x1 + x1**2 )
f2 = lambda x2: (1 + x2 - x2**2)
f3 = lambda x2: (1 - x3 - x3**2)


z = standardize(f1(x1)) + standardize(f2(x2)) + standardize(f3(x3))
z = standardize(z)

y += z
d = np.array([x1,x2,x3]).T

#Estimation
m=GAM(y)
m.basisFunctions.set_polySmoother(x1,2)
m.basisFunctions.set_polySmoother(x2,2)
m.basisFunctions.set_polySmoother(x3,2)
m.backfitting_algorithm()
y_pred=m.prediction()

# m = AdditiveModel(d)
# m.fit(y)
# x = np.linspace(-2,2,50)
# y_pred=m.results.predict(d)
# print(y-y_pred)


#Plots
plt.figure()
plt.plot(x1, f1(x1), 'r')
plt.plot(x2, f2(x2), 'b')
plt.plot(x3, f3(x3), 'k')

plt.figure()
plt.plot(y, '.')
plt.plot(z, 'b-', label='true')
plt.plot(y_pred, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

plt.figure()
plt.plot(x1,y-standardize(f2(x2))-standardize(f3(x3)), 'k.')
plt.plot(x1, standardize(m.AM.smoothers[0](x1)), 'r-', label='AdditiveModel')
plt.plot(x1, standardize(f1(x1)),label='true', linewidth=2)

#print m.AM.smoothers[0](x1)

plt.figure()
plt.plot(x2,y-standardize(f1(x1))-standardize(f3(x3)), 'k.')
plt.plot(x2,   standardize(m.AM.smoothers[1](x2)), 'r-', label='AdditiveModel')
plt.plot(x2, standardize(f2(x2)), label='true', linewidth=2)

plt.figure()
plt.plot(x3,y-standardize(f1(x1))-standardize(f2(x2)), 'k.')
plt.plot(x2,   standardize(m.AM.smoothers[1](x2)), 'r-', label='AdditiveModel')
plt.plot(x2, standardize(f2(x2)), label='true', linewidth=2)


plt.show()