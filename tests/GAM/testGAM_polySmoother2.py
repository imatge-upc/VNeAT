import sys
sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
from Fitters.GAM import GAM, SmootherSet, SplinesSmoother, PolynomialSmoother
from Fitters import CurveFitting
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)


standarize = lambda x: x#(x - x.mean()) / x.std()
standardize = lambda x: (x - x.mean())# / x.std()
nobs = 129
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()
y= np.zeros(nobs)#R.standard_normal(nobs)#
f1 = lambda x1: (1 + x1 + x1**2)
f2 = lambda x2: (1 + x2 - x2**2)
f3 = lambda x3: (1 - x3 + x3**2)

z = standardize(f1(x1)) + standardize(f2(x2)) #+ standardize(f3(x3))
z = standardize(z)

y += z

# corrector=np.array((x1,)).T
# regressor=np.array((x2,x3)).T

regressor_smoother=SmootherSet()
corrector_smoother=SmootherSet()
regressor_smoother.append(SplinesSmoother(x1,order=5,smoothing_factor=0.01))
regressor_smoother.append(SplinesSmoother(x2,order=2,smoothing_factor=0.1))
# regressor_smoother.append(PolynomialSmoother(x3,order=2))

gam=GAM(corrector_smoothers = corrector_smoother,predictor_smoothers=regressor_smoother, intercept = CurveFitting.CorrectionIntercept)
gam.fit(y)
y_pred_r=gam.predict()


plt.figure()
plt.plot(gam.correct(y), '.')
plt.plot(gam.correct(z), 'b-', label='true')
plt.plot(y_pred_r, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')


plt.figure()
plt.subplot(2,1,1)
plt.plot(x1,gam.correct(y)-f2(x2),'k.')
plt.plot(x1, gam.predict(gam.predictors[:,0][...,None],gam._crvfitter_prediction_parameters.T[0].T), 'r*', label='AdditiveModel')
plt.plot(x1, standardize(f1(x1)),'b-',label='true', linewidth=2)
plt.title(gam.df_model())

plt.subplot(2,1,2)
plt.plot(x2,gam.correct(y)-f1(x1),'k.')
plt.plot(x2, gam.predict(gam.prediction_parameters[:,1][...,None],gam._crvfitter_prediction_parameters.T[1].T), 'r-', label='AdditiveModel')
plt.plot(x2, standardize(f2(x2)),'b-',label='true', linewidth=2)

plt.show()

# plt.figure()
# plt.plot(x1,y-standarize(f2(x2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
# plt.plot(x1, standardize(gam.__predict__()), 'r-', label='AdditiveModel')
# plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)
