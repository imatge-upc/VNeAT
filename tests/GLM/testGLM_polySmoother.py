import sys

sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
from tests.GLM import PolyGLM
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt

standardize = lambda x: x  # (x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())  # / x.std()
nobs = 300
R.seed(42)

x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()
y = R.standard_normal(nobs)  # np.zeros(nobs)#
f1 = lambda x1: (1 + x1)
f2 = lambda x2: (1 + x2 + x2 ** 2)
f3 = lambda x3: (1 - x3 + x3 ** 2)

z = standardize(f1(x1))  # + standardize(f2(x2)) #+ standardize(f3(x3))
z = standardize(z)

y += z

glm = PolyGLM(np.array([x1]).T, degrees=[1])
# glm.orthogonalize_all()
glm.fit(y)
y_pred_r = glm.predict()

plt.figure()
plt.plot(glm.correct(y), 'k.')
plt.plot(glm.correct(z), 'b-', label='true')
plt.plot(y_pred_r, 'r-', label='GLM')
plt.legend()
plt.title('glm.AdditiveModel')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x1, standardize(glm.correct(y)), 'k.')
plt.plot(x1, standardize(np.dot(x1, glm.regression_parameters[0])), 'r-', label='GLM')
plt.plot(x1, standarize(f1(x1)), 'b-', label='true', linewidth=2)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x2, standardize(glm.correct(y) - f1(x1)), 'k.')
plt.plot(x2,
         standardize(np.dot(np.array([np.squeeze(x2) ** i for i in np.arange(1, 3)]).T, glm.regression_parameters[1:])),
         'r-', label='GLM')
plt.plot(x2, standardize(f2(x2)), 'b-', label='true', linewidth=2)
plt.legend()
plt.show()

# plt.figure()
# plt.plot(x1,y-standarize(f2(x2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
# plt.plot(x1, standardize(glm.__predict__()), 'r-', label='AdditiveModel')
# plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)
