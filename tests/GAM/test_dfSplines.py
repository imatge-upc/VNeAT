import sys
sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
from Fitters.GAM import GAM, SmootherSet, SplinesSmoother, PolynomialSmoother
from Fitters.CurveFitting import AdditiveCurveFitter
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt


standarize = lambda x: x#(x - x.mean()) / x.std()
standardize = lambda x: (x - x.mean())# / x.std()



x = np.linspace(1,50,50)
# [4,4,7,7,8,9,10,10,10,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,16,16,17,17,17,18,18,18,18,19,19,19,20,20,
#      20,20,20,22,23,24,24,24,24,25]

y = np.asarray([2,10, 4,22,16,10,18,26,34,17,28,14,20,24,28,26,34,34,46,26,36,60,80,20,26,54,32,40,32,40,50,42,56,76,84,36,46,68,
     32,48,52,56,64,66,54,70,92,93, 120,85])
nobs = len(x)



spline = SplinesSmoother(x,order=5,smoothing_factor=7)
spline.fit(y)
y_pred = spline.predict(x)

plt.figure()
plt.scatter(x,y, s=50, facecolors='none', edgecolors='r')
plt.plot(x,y_pred, 'b-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel' + str(spline.df_model()))
plt.show()

a=1

