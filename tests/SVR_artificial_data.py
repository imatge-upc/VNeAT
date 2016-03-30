import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

NUM_SAMPLES = 60
N_VAR = 1
C_PARAM = 10
GAMMA = 0.1
DEGREE = 3

###############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(NUM_SAMPLES, 1), axis=0)
y = np.sin(X).ravel()

###############################################################################
# Add noise to targets
y += N_VAR * (0.5 - np.random.rand(NUM_SAMPLES))

###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=C_PARAM, gamma=GAMMA)
svr_lin = SVR(kernel='linear', C=C_PARAM)
svr_poly = SVR(kernel='poly', C=C_PARAM, degree=DEGREE)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results
plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.plot(X, y_lin, c='r', label='Linear model')
plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()