import sys
sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
sys.path.insert(1, '/Users/acasamitjana/Repositories/neuroimatge/nonlinear2')
from GAM import GAM, SmootherSet, SplinesSmoother, PolynomialSmoother
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt


standardize = lambda x: x#(x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())# / x.std()
nobs = 150
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()
y= 0.4*R.standard_normal(nobs)#np.zeros(nobs)#
f1 = lambda x1: (1 + x1 )
f2 = lambda x2: (1 + x2 - x2**2)
f3 = lambda x3: (1 - x3 + x3**2)

z = standardize(f1(x1)) + standardize(f2(x2)) #+ standardize(f3(x3))
z = standardize(z)

y += z




regressor_smoother=SmootherSet()
corrector_smoother=SmootherSet()
regressor_smoother.append(PolynomialSmoother(x1,order=1))
regressor_smoother.append(SplinesSmoother(x2,order=2,smoothing_factor=30))
# regressor_smoother.append(PolynomialSmoother(x3,order=2))

gam=GAM(corrector_smoothers = corrector_smoother,regressor_smoothers=regressor_smoother)
gam.fit(y)
y_pred_r=gam.predict()


plt.figure()
plt.plot(gam.correct(y), 'k.')
plt.plot(gam.correct(z), 'b-', label='true')
plt.plot(y_pred_r, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

reg_params=gam.regression_parameters
indx_smthr = 0

plt.figure()
plt.subplot(2,1,1)
plt.plot(x1,standarize(y-gam.alpha-f2(x2)),'k.')
plt.plot(x1, standarize(gam.predict(gam.regressors[:,0][...,None],reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),
         'r-', label='AdditiveModel')
plt.plot(x1, standarize(f1(x1)),'b-',label='true', linewidth=2)

indx_smthr = 2+reg_params[indx_smthr+1]
plt.subplot(2,1,2)
plt.plot(x2,standarize(y-gam.alpha-f1(x1)),'k.')
plt.plot(x2, standarize(gam.predict(gam.regressors[:,1][...,None],reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),
         'r-', label='AdditiveModel')
plt.plot(x2, standarize(f2(x2)),'b-',label='true', linewidth=2)

plt.show()

# plt.figure()
# plt.plot(x1,y-standarize(f2(x2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
# plt.plot(x1, standardize(gam.__predict__()), 'r-', label='AdditiveModel')
# plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)
