import sys
sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
from GAM import GAM, SmootherSet, SplinesSmoother, PolynomialSmoother
from GLM import GLM, PolyGLM
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt


standardize = lambda x: x#(x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())# / x.std()
nobs = 300
R.seed(42)
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()
y= 0.5*R.standard_normal(nobs)#np.zeros(nobs)#
f1 = lambda x1: (1 + x1 )
f2 = lambda x2: (1 - x2 )
f3 = lambda x3: (1 - x3 + x3**2)

z = standardize(f1(x1)) + standardize(f2(x2)) #+ standardize(f3(x3))
z = standardize(z)

y += z

glm=PolyGLM(np.array([x1,x2]).T,degrees = [1,1])
# glm.orthogonalize_all()
glm.fit(y)
y_pred_r=glm.predict()


plt.figure()
plt.plot(glm.correct(y), 'k.')
plt.plot(glm.correct(z), 'b-', label='true')
plt.plot(y_pred_r, 'r-', label='GLM')
plt.legend()
plt.title('glm.AdditiveModel')


reg_params=glm.regression_parameters
indx_smthr = 0
plt.figure()
plt.subplot(2,1,1)
plt.plot(x1,standarize(y-glm.alpha-f2(x2)),'k.')
plt.plot(x1, standarize(glm.predict(glm.regressors[:,0][...,None],reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),
         'r-', label='AdditiveModel')
plt.plot(x1, standarize(f1(x1)),'b-',label='true', linewidth=2)

indx_smthr = 2+reg_params[indx_smthr+1]
plt.subplot(2,1,2)
plt.plot(x2,standarize(y-glm.alpha-f1(x1)),'k.')
plt.plot(x2, standarize(glm.predict(glm.regressors[:,1][...,None],reg_params[indx_smthr:indx_smthr+2+reg_params[indx_smthr+1]])),
         'r-', label='AdditiveModel')
plt.plot(x2, standarize(f2(x2)),'b-',label='true', linewidth=2)

plt.show()

# plt.figure()
# plt.plot(x1,y-standarize(f2(x2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
# plt.plot(x1, standardize(glm.__predict__()), 'r-', label='AdditiveModel')
# plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)
