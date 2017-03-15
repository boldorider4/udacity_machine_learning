#!/brew/bin/python

import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB

print("GaussiaNB object being creater (clf)")
clf = GaussianNB()
print("fitting data points (X) into labels (Y)")
clf.fit(X, Y)
GaussianNB(priors=None)

print("print clf.predict([[-0.8, -1]])...")
print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
print("partial fit (???)")
clf_pf.partial_fit(X, Y, np.unique(Y))
GaussianNB(priors=None)
print(clf_pf.predict([[-0.8, -1]]))
