'''original example for checking how far GAM works
Note: uncomment plt.show() to display graphs
'''

example = 2  # 1,2 or 3

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as R
from statsmodels.sandbox.gam import AdditiveModel

standardize = lambda x: (x - x.mean()) / x.std()
nobs = 150
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
y = 0.5 * R.standard_normal(nobs)  # np.zeros(nobs)#

f1 = lambda x1: (1 + x1 + x1 ** 2)
f2 = lambda x2: (1 + x2 - x2 ** 2)

z = standardize(f1(x1)) + standardize(f2(x2))
z = standardize(z)  # * 2 # 0.1

y += z
d = np.array([x1, x2]).T

# Estimation
m = AdditiveModel(d)
m.fit(y)
x = np.linspace(-2, 2, 50)
y_pred = m.results.predict(d);
print(y - y_pred)

# Plots
plt.figure()
plt.plot(x1, f1(x1), 'r')
plt.plot(x2, f2(x2), 'b')

plt.figure()
plt.plot(y, '.')
plt.plot(z, 'b-', label='true')
plt.plot(y_pred, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

plt.figure()
plt.plot(x1, y - standardize(f2(x2)), 'k.')
plt.plot(x1, standardize(m.smoothers[0](x1)), 'r-', label='AdditiveModel')
plt.plot(x1, standardize(f1(x1)), label='true', linewidth=2)

plt.figure()
plt.plot(x2, y - standardize(f1(x1)), 'k.')
plt.plot(x2, standardize(m.smoothers[1](x2)), 'r-', label='AdditiveModel')
plt.plot(x2, standardize(f2(x2)), label='true', linewidth=2)

plt.show()




##     pylab.figure(num=1)
##     pylab.plot(x1, standardize(m.smoothers[0](x1)), 'b')
##     pylab.plot(x1, standardize(f1(x1)), linewidth=2)
##     pylab.figure(num=2)
##     pylab.plot(x2, standardize(m.smoothers[1](x2)), 'b')
##     pylab.plot(x2, standardize(f2(x2)), linewidth=2)
##     pylab.show()
