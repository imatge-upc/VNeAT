

# So, here is the situation:
#
# We are given X = (C | R), the design matrix, and Y, the matrix of observations.
# dim(X) = (N, K)  ;  dim(C) = (N, c)  ;  dim(R) = (N, r)
# dim(Y) = (N, M)
#
# From here, we build a new matrix Z by orthogonalizing and normalizing the columns of X.
# Z = (ZC | ZR)
# dim(Z) = dim(X) = (N, K)  ;  dim(ZC) = dim(C) = (N, c)  ;  dim(ZR) = dim(R) = (N, r)
#
# In the process we also obtain the deorthonormalization matrix, Gamma, which fullfils
# X = Z * Gamma  ;  dim(Gamma) = (K, K)
# Gamma = (GammaC | GammaR)  ;  dim(GammaC) = (K, c)  ;  dim(GammaR) = (K, r)
#
# Now, we solve the equation by using Z instead of X, and obtain the matrix of parameters
# Beta2 that minimizes the energy of the error:
# Beta2 = argmin(B){ ||Y - Z*B||^2 }  ;  dim(Beta2) = (K, M)
#
# What we would like is to obtain the parameters BetaR from the alternative approach described
# next by using Beta2, Z, and Gamma exclusively.
#
# Alternative approach: 
# 
# First solve the problem for the C matrix alone, and obtain the residual error (the corrected data).
# BetaC = argmin(BC){ ||Y - C*BC||^2 }  ;  dim(BetaC) = (c, M)
# Yc = Y - C*BetaC  ;  dim(Yc) = (N, M)
#
# Then solve the problem for the R matrix by using the corrected data as observations.
# BetaR = argmin(BR){ ||Yc - R*BR||^2 }  ;  dim(BetaR) = (r, M)
#
#
#
# OK, so here is the solution:
#
# Notice that, since every column of Z is orthogonalized with respect to the previous ones, it would
# be the same to try to solve the equation Y = Z*B in one step or to divide it in two steps as in
# the alternative approach (Yc = Y - ZC*BC  ;  Yc = ZR*BR). Thus, Yc must be the same for both cases.
#
# Moreover, this corrected data will also be equal to that computed in the alternative approach when
# using X instead of Z (obviously, the parameters may differ, but the prediction made with C alone
# should be the same as that made with ZC alone).
#
# We can reach a similar conclusion by applying the same principle to the second step of the alternative
# approach, and therefore claim that the residual error will be the same when using either of the
# methods explained before, i.e.:
# eps = Yc - ZR*Beta2R = Yc - R*BetaR  ;  dim(eps) = (N, M)
# 
# where Beta2R is the lower submatrix of shape (r, M) of the Beta2 matrix.
#
# From here, we get that ZR * Beta2R = R * BetaR, and replacing R = Z * GammaR, we obtain:
# ZR * Beta2R = Z * GammaR * BetaR
#
# which is the same as:
# BetaR = left_inverse(Z * GammaR) * ZR * Beta2R
#
# The only algorithmic problem here may arise from computing the left inverse of Z * GammaR, but we can
# easily notice that this is the same as finding the parameters that minimize the residual error of the
# following equation:
# transpose(Z * GammaR) * B = I_r  ;  I_r = identity matrix of shape (r, r)
# B = argmin(Bt){ ||I_r  -  transpose(Z * GammaR) * Bt||^2 }  ;  dim(B) = (N, r)
#
# Then we only have to transpose B to obtain the mentioned left inverse matrix, i.e.,
# left_inverse(Z * GammaR) = transpose(B)
#
# Prove (ideal case, i.e., no error):
# transpose(Z * GammaR) * B = I_r
# transpose( transpose(Z * GammaR) * B ) = transpose(I_r)
# transpose(B) * transpose( transpose(Z * GammaR) ) = I_r
# transpose(B) * (Z * GammaR) = I_r
# by definition, transpose(B) = left_inverse(Z * GammaR)



import numpy as np

C = np.random.random((129, 2))
R = np.linspace(0, 1, 129).reshape(-1, 1)
R = np.concatenate([R**(i+1) for i in xrange(3)], axis = 1)

from nonlinear2.Fitters.GLM import GLM
glm = GLM(predictors = R, correctors = C, homogeneous = True)

C = glm.correctors
R = glm.predictors
X = np.concatenate((C, R), axis = 1)

Gamma = glm.orthonormalize_all()

ZC = glm.correctors
ZR = glm.predictors
Z = np.concatenate((ZC, ZR), axis = 1)


from matplotlib.pyplot import plot, show

params = np.random.uniform(-5, 5, 4)
params2 = np.random.normal(0, 10, 4)

y = np.array(sum(R[:, i]*params[i] for i in xrange(R.shape[1])))
y2 = np.array(sum(R[:, i]*params2[i] for i in xrange(R.shape[1])))
plot(R[:, 0], y, 'bs', R[:, 0], y2, 'rs')
show()

scale1 = y.max() - y.min()
scale2 = y2.max() - y2.min()
Y = np.array([y + np.random.normal(0, 0.15*scale1, y.shape), y2 + np.random.normal(0, 0.15*scale2, y2.shape)]).T

plot(R[:, 0], y, 'bs', R[:, 0], Y[:, 0], 'g^')
show()

plot(R[:, 0], y2, 'bs', R[:, 0], Y[:, 1], 'g^')
show()



# Y = np.random.random((20, 2))
glm.fit(Y)

Beta2C = glm.correction_parameters
Beta2R = glm.prediction_parameters


GammaR = Gamma[:, -(ZR.shape[1]):]
ZGR = Z.dot(GammaR)


glmInv = GLM(predictors = ZGR.T, homogeneous = False)
glmInv.fit(np.identity(ZGR.shape[1]))

ZGRInv = glmInv.prediction_parameters.T

# from sklearn.linear_model import LinearRegression as LR

# lr = LR(fit_intercept = False, normalize = True, copy_X = True, n_jobs = -1)
# lr.fit(ZGR.T, np.identity(ZGR.shape[1]))

# ZGRInv2 = lr.coef_

# print 'ZGRInv and ZGRInv2 coincide:', (ZGRInv2 == ZGRInv).all()

BetaR_denorm = ZGRInv.dot(ZR).dot(Beta2R)






glm2 = GLM(correctors = C, homogeneous = False)
glm2.fit(Y)
BetaC = glm2.correction_parameters

Yc = glm2.correct(Y)

glm3 = GLM(predictors = R, homogeneous = False)
glm3.fit(Yc)
BetaR = glm3.prediction_parameters


print 'Maximum difference between original (separated) parameters and denormalized ones:', np.abs(BetaR_denorm - BetaR).max()

Yc_norm = Y - ZC.dot(Beta2C)

print 'Maximum difference between corrected data with normalized features vs. non-normalized features:', np.abs(Yc - Yc_norm).max()





from matplotlib.pyplot import plot, show
plot(R[:, 0], Yc, 'x')

x = np.linspace(0, 1, 50)
xdata = np.array([x**(i+1) for i in xrange(3)]).T
prediction = glm.predict(xdata, BetaR_denorm)
plot(x, prediction)

show()





