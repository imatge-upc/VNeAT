import matplotlib.pyplot as plt
import numpy as np

import localRegressionSmoothing as lrs

""" TEST SUITE """

print "Test span computation..."
data = np.linspace(-2, 2, 100)
x = -0.2
span_percentage = 15
span = lrs.compute_span(data, x, span_percentage)
print "Span: ", span
print

print "Test weighted window..."
w = lrs.regression_weights(data, x, span)
plt.plot(data, w)
plt.show()
print
