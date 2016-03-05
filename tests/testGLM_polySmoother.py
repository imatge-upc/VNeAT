import sys
sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
from GLM import GLM, PolyGLM
from GAM import GAM, SmootherSet, SplinesSmoother, PolynomialSmoother
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)


standardize = lambda x: x#(x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())# / x.std()
nobs = 300
R.seed(seed=42)
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()
y= 0*R.standard_normal(nobs)#np.zeros(nobs)#
f1 = lambda x1: (x1 + x1**2)
f2 = lambda x2: (x2 - x2**2)
f3 = lambda x3: (1 - x3 + x3**2)

z = standardize(f1(x1)) #+ standardize(f2(x2)) #+ standardize(f3(x3))
z = standardize(z)

y += z

# corrector=np.array((x1,)).T
# regressor=np.array((x2,x3)).T

regressor_smoother=SmootherSet()
corrector_smoother=SmootherSet()
regressor_smoother.append(PolynomialSmoother(x1,order=2))
# regressor_smoother.append(PolynomialSmoother(x2,order=2))
# corrector_smoother.append(PolynomialSmoother(x3,order=2))
glm1=PolyGLM(np.zeros((nobs,1)),homogeneous = True)
glm2=PolyGLM(x1[:,None],degrees = [2])
y_corr = glm1.fit(y)
glm2.fit(glm1.correct(y))
y_pred_r=glm2.predict()


plt.figure()
plt.plot(x1,glm1.correct(y), 'k.')
plt.plot(x1,glm1.correct(z), 'b-', label='true')
plt.plot(x1,y_pred_r, 'r-', label='AdditiveModel')
plt.legend()
plt.title('glm.AdditiveModel')


reg_params=glm2.regression_parameters
indx_smthr = 0
plt.figure()
plt.subplot(2,1,1)
plt.plot(x1,standarize(y-glm2.alpha),'k.')
plt.plot(x1, standarize(glm2.predict(glm2.regressors[:,0][...,None],reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),
         'r-', label='AdditiveModel')
plt.plot(x1, standarize(f1(x1)),'b-',label='true', linewidth=2)

indx_smthr = 2+reg_params[indx_smthr+1]
plt.subplot(2,1,2)
plt.plot(x2,standarize(y-glm2.alpha-f1(x1)),'k.')
plt.plot(x2, standarize(glm2.predict(glm2.regressors[:,1][...,None],reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),
         'r-', label='AdditiveModel')
plt.plot(x2, standarize(f2(x2)),'b-',label='true', linewidth=2)

plt.show()

# plt.figure()
# plt.plot(x1,y-standarize(f2(x2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
# plt.plot(x1, standardize(glm.__predict__()), 'r-', label='AdditiveModel')
# plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)
