from nonlinear2.GAM import GAM, SplinesSmoother, PolynomialSmoother
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)


standardize = lambda x: x#(x - x.mean()) / x.std()
standarize = lambda x: (x - x.mean())# / x.std()
nobs = 150
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()
y= 0.5*R.standard_normal(nobs)#np.zeros(nobs)#

f1 = lambda x1: (1 + x1  )
f2 = lambda x2: (1 + x2 - x2**2)
f3 = lambda x3: (1 - x3 - x3**2)

z = standardize(f1(x1)) + standardize(f2(x2)) + standardize(f3(x3))
z = standardize(z)

y += z
d = np.array([x1,x2,x3]).T


smoothers=[]
smoothers.append(PolynomialSmoother(x1,order=1,name='PolySmoother1'))
smoothers.append(PolynomialSmoother(x2,order=2,name='PolySmoother2'))
smoothers.append(PolynomialSmoother(x3,order=2,name='PolySmoother3'))

gam=GAM()
smoother_results=gam.fit(y,smoothers)
y_pred=gam.predict()


plt.figure()
plt.plot(y, '.')
plt.plot(z, 'b-', label='true')
plt.plot(y_pred, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

plt.figure()
plt.plot(x1,y-standarize(f2(x2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
plt.plot(x1, standardize(smoother_results), 'r-', label='AdditiveModel')
plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)

plt.figure()
plt.plot(x2,y-standarize(f1(x1))-standarize(f3(x3))-m.results.alpha, 'k.')
plt.plot(x2,   standardize(m.results.smoothers[1](x2))+m.results.offset[1], 'r-', label='AdditiveModel')
plt.plot(x2, standarize(f2(x2)), label='true', linewidth=2)