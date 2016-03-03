import sys
sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
from GAM import GAM, SmootherSet, SplinesSmoother, PolynomialSmoother
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt


standardize = lambda x: x#(x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())# / x.std()
nknots = 10
nobs = 500
t1 = R.standard_normal(nobs)
t1.sort()
f1 = lambda x2: 0.1*(x2 + x2**2)

t2=t1
s2=R.standard_normal(nobs)
f2=lambda x: np.sin(np.pi*x)


y= 0*R.standard_normal(nobs)#np.zeros(nobs)#

z = f1(t1) + f2(t2)

y += z

regressor_smoother=SmootherSet()
corrector_smoother=SmootherSet()
regressor_smoother.append(PolynomialSmoother(t1,order=2))
regressor_smoother.append(SplinesSmoother(t2,order=3,smoothing_factor=1))
# regressor_smoother.append(PolynomialSmoother(x3,order=2))

gam=GAM(corrector_smoothers = corrector_smoother,regressor_smoothers=regressor_smoother)
# gam.orthogonalize_all()
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
plt.plot(t1,standarize(y-gam.alpha-f2(t2)),'k.')
plt.plot(t1, standarize(gam.predict(gam.regressors[:,0][...,None],reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),
         'r-', label='AdditiveModel')
plt.plot(t1, standarize(f1(t1)),'b-',label='true', linewidth=2)
plt.legend()


plt.subplot(2,1,2)
plt.plot(t2, standarize(y-gam.alpha-f1(t1)),'k.')
plt.plot(t2, standarize(y-gam.predict(gam.regressors[:,0][...,None],
                                                reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),'g.')
indx_smthr = 2+reg_params[indx_smthr+1]
plt.plot(t2, standarize(gam.predict(gam.regressors[:,1][...,None],reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),
         'r-', label='AdditiveModel')
plt.plot(t2, standarize(f2(t2)),'b-',label='true', linewidth=2)
plt.legend()

plt.show()
a=1

# plt.figure()
# plt.plot(t1,y-standarize(f2(t2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
# plt.plot(t1, standardize(gam.__predict__()), 'r-', label='AdditiveModel')
# plt.plot(t1, standarize(f1(t1)),label='true', linewidth=2)
