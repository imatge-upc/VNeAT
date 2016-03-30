
from GLM import GLM
import numpy as np

correctors = np.random.random((129, 10))
predictors = np.random.random((129, 6))
features = np.array(list(correctors.T) + list(predictors.T)).T


# ------------------ Normalization ------------------

print
print 'Normalization results:'
print

glm = GLM(predictors, correctors, False)
denormalization_matrix = glm.normalize_all()

new_correctors = glm.correctors
new_predictors = glm.predictors
new_features = np.array(list(new_correctors.T) + list(new_predictors.T)).T

old_features = new_features.dot(denormalization_matrix)
print '    Maximum difference between the original and the denormalized matrices:', np.abs(old_features - features).max()
print
print '    Column-magnitudes before and after normalizing:'
print '        (Before)', (features**2).sum(axis = 0)
print '         (After)', (new_features**2).sum(axis = 0)
print

# ---------------- Orthogonalization ----------------

print
print 'Orthogonalization results:'
print

glm = GLM(predictors, correctors, False)
deorthogonalization_matrix = glm.orthogonalize_all()

new_correctors = glm.correctors
new_predictors = glm.predictors
new_features = np.array(list(new_correctors.T) + list(new_predictors.T)).T

old_features = new_features.dot(deorthogonalization_matrix)
print '    Maximum difference between the original and the deorthogonalized matrices:', np.abs(old_features - features).max()
print

old_products = []
new_products = []
for i in xrange(features.shape[1] - 1):
	for j in xrange(i+1, features.shape[1]):
		old_products.append(features[:, i].dot(features[:, j]))
		new_products.append(new_features[:, i].dot(new_features[:, j]))

old_products = np.array(old_products) 
new_products = np.array(new_products)
print '    Maximum dot product between columns before and after orthogonalizing:', old_products.max(), 'vs.', new_products.max()
print

# --------------- Orthonormalization ----------------

print
print 'Orthonormalization results:'
print

glm = GLM(predictors, correctors, False)
deorthonormalization_matrix = glm.orthonormalize_all()

new_correctors = glm.correctors
new_predictors = glm.predictors
new_features = np.array(list(new_correctors.T) + list(new_predictors.T)).T

old_features = new_features.dot(deorthonormalization_matrix)
print '    Maximum difference between the original and the deorthonormalized matrices:', np.abs(old_features - features).max()
print
print '    Column-magnitudes before and after orthonormalizing:'
print '        (Before)', (features**2).sum(axis = 0)
print '         (After)', (new_features**2).sum(axis = 0)
print

old_products = []
new_products = []
for i in xrange(features.shape[1] - 1):
	for j in xrange(i+1, features.shape[1]):
		old_products.append(features[:, i].dot(features[:, j]))
		new_products.append(new_features[:, i].dot(new_features[:, j]))

old_products = np.array(old_products) 
new_products = np.array(new_products)
print '    Maximum dot product between columns before and after orthonormalizing:', old_products.max(), 'vs.', new_products.max()
print





