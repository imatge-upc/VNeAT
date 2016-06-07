import sys

sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
from tests.GAM import GAM, SmootherSet, PolynomialSmoother
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
f2 = lambda x2: (1 - x2 + x2 ** 2)
f3 = lambda x3: (1 - x3 + x3 ** 2)

z = standardize(f1(x1))  # + standardize(f2(x2)) #+ standardize(f3(x3))
z = standardize(z)

y += z

# corrector=np.array((x1,)).T
# regressor=np.array((x2,x3)).T

regressor_smoother = SmootherSet()
corrector_smoother = SmootherSet()

regressor_smoother.append(PolynomialSmoother(x1, order=1))
# regressor_smoother.append(PolynomialSmoother(x2,order=2))
# regressor_smoother.append(PolynomialSmoother(x3,order=2))

gam = GAM(corrector_smoothers=corrector_smoother, regressor_smoothers=regressor_smoother)
# gam.orthogonalize_all()
gam.fit(y, maxiter=600, rtol=1e-10)
y_pred_r = gam.predict()

plt.figure()
plt.plot(x1, gam.correct(y), 'k.')
plt.plot(x1, gam.correct(z), 'b-', label='true')
plt.plot(x1, y_pred_r, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

reg_params = gam.regression_parameters
indx_smthr = 0
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x1, gam.correct(y), 'k.')
plt.plot(x1, gam.predict(gam.regressors[:, 0][..., None],
                         reg_params[indx_smthr:indx_smthr + 2 + reg_params[indx_smthr + 1]]),
         'r-', label='AdditiveModel')
plt.plot(x1, standarize(f1(x1)), 'b-', label='true', linewidth=2)
plt.legend()

indx_smthr = 2 + reg_params[indx_smthr + 1]
plt.subplot(2, 1, 2)
plt.plot(x2, gam.correct(y) - gam.alpha - f1(x1), 'k.')
plt.plot(x2, gam.predict(gam.regressors[:, 1][..., None],
                         reg_params[indx_smthr:indx_smthr + 2 + reg_params[indx_smthr + 1]]),
         'r-', label='AdditiveModel')
plt.plot(x2, standarize(f2(x2)), 'b-', label='true', linewidth=2)
plt.legend()
plt.show()

# plt.figure()
# plt.plot(x1,y-standarize(f2(x2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
# plt.plot(x1, standardize(gam.__predict__()), 'r-', label='AdditiveModel')
# plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)
