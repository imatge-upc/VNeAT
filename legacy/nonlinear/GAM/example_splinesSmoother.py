import excelIO as eIO
from curve_fit import GAM
from curve_fit import Smoother
import numpy as np
import numpy.random as R
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import IPython.display as display

from statsmodels.sandbox.gam import AdditiveModel
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM

standardize = lambda x: x#(x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())# / x.std()
nknots = 10
nobs = 500
t1 = np.linspace(-2,2,nobs)
t1.sort()
f1 = lambda x: np.cos(2*np.pi*x)

# t2 = R.standard_normal(nobs)
# t2.sort()
# f2 = lambda x2: (1 + t2 - t2**2)
# x2 = R.uniform(-1,1,nobs)
# x2.sort()
# t2_min=min(x2)
# t2_max=max(x2)
# t2 = np.linspace(t2_min,t2_max,nobs)
# f2 = interpolate.interp1d(x2, s2, kind='cubic')
t2=t1
s2=np.sin(2*np.pi*t2**2)
f2=lambda x: np.sin(2*np.pi*x)


y= 0.1*R.standard_normal(nobs)#np.zeros(nobs)#

z = f1(t1) + f2(t2)

y += z
# d = np.array([x1,x2,x3]).T
d = np.array([t1,t2]).T


#Estimation
m=GAM(y)
m.basisFunctions.set_splinesSmoother(t1,3,s=30)
m.basisFunctions.set_splinesSmoother(t2,3,s=5)
m.backfitting_algorithm()
y_pred=m.prediction()

#Plots
# plt.figure()
# plt.plot(x1, f1(x1), 'r')
# plt.plot(x2, f2(x2), 'b')
# plt.plot(x3, f3(x3), 'k')

plt.figure()
plt.plot(t1,y, '.')
plt.plot(t1,z, 'b-', label='true')
plt.plot(t1,y_pred, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

# print(m.results.offset)
# print(m.results.alpha)
# plt.figure()
# plt.plot(t1,y-m.results.alpha, 'k.')
# plt.plot(t1, standardize(m.results.smoothers[0](t1))+m.results.offset[0], 'r-', label='AdditiveModel')
# plt.plot(t1, standarize(f1(t1)),label='true', linewidth=2)


plt.figure()
plt.plot(t1,y-standarize(f2(t2))-m.results.alpha, 'k.')
plt.plot(t1, m.results.smoothers[0](t1)+m.results.offset[0], 'r-', label='AdditiveModel')
plt.plot(t1, standarize(f1(t1)),label='true', linewidth=2)


plt.figure()
plt.plot(t2,y-standarize(f1(t1))-m.results.alpha, 'k.')
plt.plot(t2,y - (m.results.smoothers[0](t1)+m.results.offset[0])-m.results.alpha,'g.')
plt.plot(t2, m.results.smoothers[1](t2)+m.results.offset[1], 'r-', label='AdditiveModel')
plt.plot(t2, standarize(f2(t2)), label='true', linewidth=2)
#
#
# # lex3 = min(x3)
# # rex3 = max(x3)
# # npoints3 = 100
# # var3_axis = np.linspace(lex3, rex3, npoints3)
#
# plt.figure()
# plt.plot(x3,y-standarize(f1(x1))-standarize(f2(x2))-m.results.alpha, 'k.')
# plt.plot(x3, standardize(m.results.smoothers[2](x3))+m.results.offset[2], 'r-', label='AdditiveModel')
# plt.plot(x3, standarize(f3(x3)), label='true', linewidth=2)

plt.show()