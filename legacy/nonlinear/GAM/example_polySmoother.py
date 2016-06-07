import matplotlib.pyplot as plt
import numpy as np
import numpy.random as R

from curve_fit import GAM

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)

standardize = lambda x: x  # (x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())  # / x.std()
nobs = 150
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()
y = 0 * R.standard_normal(nobs)  # np.zeros(nobs)#

f1 = lambda x1: (1 + x1)
f2 = lambda x2: (1 + x2 - x2 ** 2)
f3 = lambda x3: (1 - x3 - x3 ** 2)

z = standardize(f1(x1)) + standardize(f2(x2)) + standardize(f3(x3))
z = standardize(z)

y += z
d = np.array([x1, x2, x3]).T

# Estimation
m = GAM(y)
m.basisFunctions.set_polySmoother(x1, 1)
m.basisFunctions.set_polySmoother(x2, 2)
m.basisFunctions.set_polySmoother(x3, 2)
m.backfitting_algorithm()
y_pred = m.prediction()

# Plots
# plt.figure()
# plt.plot(x1, f1(x1), 'r')
# plt.plot(x2, f2(x2), 'b')
# plt.plot(x3, f3(x3), 'k')

plt.figure()
plt.plot(y, '.')
plt.plot(z, 'b-', label='true')
plt.plot(y_pred, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

print(m.results.offset)
print(m.results.alpha)
# plt.figure()
# plt.plot(x1,y, 'k.')
# plt.plot(x1, standardize(m.results.smoothers[0](x1))+m.results.offset[0], 'r-', label='AdditiveModel')
# plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)


plt.figure()
plt.plot(x1, y - standarize(f2(x2)) - standarize(f3(x3)) - m.results.alpha, 'k.')
plt.plot(x1, standardize(m.results.smoothers[0](x1)) + m.results.offset[0], 'r-', label='AdditiveModel')
plt.plot(x1, standarize(f1(x1)), label='true', linewidth=2)

plt.figure()
plt.plot(x2, y - standarize(f1(x1)) - standarize(f3(x3)) - m.results.alpha, 'k.')
plt.plot(x2, standardize(m.results.smoothers[1](x2)) + m.results.offset[1], 'r-', label='AdditiveModel')
plt.plot(x2, standarize(f2(x2)), label='true', linewidth=2)

# lex3 = min(x3)
# rex3 = max(x3)
# npoints3 = 100
# var3_axis = np.linspace(lex3, rex3, npoints3)

plt.figure()
plt.plot(x3, y - standarize(f1(x1)) - standarize(f2(x2)) - m.results.alpha, 'k.')
plt.plot(x3, standardize(m.results.smoothers[2](x3)) + m.results.offset[2], 'r-', label='AdditiveModel')
plt.plot(x3, standarize(f3(x3)), label='true', linewidth=2)

plt.show()
