import sys

sys.path.insert(1, 'C:\\Users\\upcnet\\Repositoris\\neuroimatge\\nonlinear2')
from Fitters.GAM import GAM, SmootherSet, SplinesSmoother
from Fitters.CurveFitting import AdditiveCurveFitter
import numpy as np
import numpy.random as R
import matplotlib.pyplot as plt

standarize = lambda x: x  # (x - x.mean()) / x.std()
standardize = lambda x: (x - x.mean())  # / x.std()
nobs = 129
x1 = R.standard_normal(nobs)
x1.sort()
x2 = R.standard_normal(nobs)
x2.sort()
x3 = R.standard_normal(nobs)
x3.sort()

x = [x1, x2, x3]
y = np.zeros(nobs)  # R.standard_normal(nobs)#
f1 = lambda x1: (1 + x1 + x1 ** 2)
f2 = lambda x2: (1 + x2 - x2 ** 2)
f3 = lambda x3: (1 - x3 + x3 ** 2)
f = [f1(x1), f2(x2), f3]

z = f1(x1)  # + f2(x2) #+ standardize(f3(x3))
z = z

y += z

# corrector=np.array((x1,)).T
# regressor=np.array((x2,x3)).T

predictor_smoother = SmootherSet()
corrector_smoother = SmootherSet()
predictor_smoother.append(SplinesSmoother(x1, order=5, smoothing_factor=0.8))
# predictor_smoother.append(SplinesSmoother(x2,order=2,smoothing_factor=0.8))
# regressor_smoother.append(PolynomialSmoother(x3,order=2))

gam = GAM(corrector_smoothers=corrector_smoother, predictor_smoothers=predictor_smoother,
          intercept=AdditiveCurveFitter.CorrectionIntercept)
gam.fit(y)
y_pred_r = gam.predict()

plt.figure()
plt.plot(gam.correct(y), '.')
plt.plot(gam.correct(z), 'b-', label='true')
plt.plot(y_pred_r, 'r-', label='AdditiveModel')
plt.legend()
plt.title('gam.AdditiveModel')

plt.figure()
n_parameters_init = 0
for index, pred in enumerate(predictor_smoother):
    n_parameters_final = gam.prediction_parameters[n_parameters_init + 1] + n_parameters_init + 1
    plt.subplot(2, 1, index + 1)
    plt.plot(x[index], gam.predict(gam.predictors[:, index][..., None],
                                   gam.prediction_parameters[n_parameters_init:n_parameters_final]), 'r.',
             label='AdditiveModel')
    plt.plot(x[index], standardize(f[index]), 'b-', label='true', linewidth=2)
    plt.title(gam.df_model()[index])
    n_parameters_init = n_parameters_final
    #
    # plt.subplot(2,1,2)
    # plt.plot(x2,gam.correct(y)-f1(x1),'k.')
    # plt.plot(x2, gam.predict(gam.prediction_parameters[:,0][...,None],gam.prediction_parameters[n_parameters+1:]), 'r.', label='AdditiveModel')
    # plt.plot(x2, standardize(f2(x2)),'b-',label='true', linewidth=2)

plt.show()
a = 1
# plt.figure()
# plt.plot(x1,y-standarize(f2(x2))-standarize(f3(x3))-smoother_results['mean'], 'k.')
# plt.plot(x1, standardize(gam.__predict__()), 'r-', label='AdditiveModel')
# plt.plot(x1, standarize(f1(x1)),label='true', linewidth=2)
