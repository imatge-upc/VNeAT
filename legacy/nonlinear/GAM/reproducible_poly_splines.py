import matplotlib.pyplot as plt
import numpy as np
import numpy.random as R

from curve_fit import GAM

standardize = lambda x: x  # (x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())  # / x.std()
nknots = 10
nobs = 500
t1 = np.linspace(-2, 2, nobs)
t1.sort()
f1 = lambda x1: (1 + t1 - t1 ** 2)
# x1 = R.uniform(-1,1,nknots)
# x1.sort()
# t1_min=min(x1)
# t1_max=max(x1)
# t1 = np.linspace(t1_min,t1_max,nobs)
# s1=np.cos(2*np.pi*x1)
# f1 = interpolate.interp1d(x1, s1, kind='cubic')
# f1=lambda x: np.cos(2*np.pi*x)


# t2 = R.standard_normal(nobs)
# t2.sort()
# f2 = lambda x2: (1 + t2 - t2**2)
# x2 = R.uniform(-1,1,nobs)
# x2.sort()
# t2_min=min(x2)
# t2_max=max(x2)
# t2 = np.linspace(t2_min,t2_max,nobs)
t2 = t1
s2 = np.sin(2 * np.pi * t2 ** 2)
f2 = lambda x: np.sin(2 * np.pi * x)
# f2 = interpolate.interp1d(x2, s2, kind='cubic')

y = 0.5 * R.standard_normal(nobs)  # np.zeros(nobs)#

z = f1(t1) + f2(t2)

y += z
# d = np.array([x1,x2,x3]).T
d = np.array([t1, t2]).T

# Estimation
m = GAM(y)
m.basisFunctions.set_polySmoother(t1, 2)
m.basisFunctions.set_splinesSmoother(t2, 3, s=130)
m.backfitting_algorithm()
y_pred = m.prediction()

# Plots
plt.figure()
plt.plot(t1, y, '.')
plt.plot(t1, z, 'b-', label='true')
plt.plot(t1, y_pred, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

plt.figure()
plt.plot(t1, y - standarize(f2(t2)) - m.results.alpha, 'k.')
plt.plot(t1, m.results.smoothers[0](t1) + m.results.offset[0], 'r-', label='AdditiveModel')
plt.plot(t1, standarize(f1(t1)), label='true', linewidth=2)

plt.figure()
plt.plot(t2, y - standarize(f1(t1)) - m.results.alpha, 'k.')
plt.plot(t2, y - (m.results.smoothers[0](t1) + m.results.offset[0]) - m.results.alpha, 'g.')
plt.plot(t2, m.results.smoothers[1](t2) + m.results.offset[1], 'r-', label='AdditiveModel')
plt.plot(t2, standarize(f2(t2)), label='true', linewidth=2)

plt.show()
