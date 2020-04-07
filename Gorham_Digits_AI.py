from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import random

random.seed(10)

digits = datasets.load_digits()
x = digits.data / digits.data.max()
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
w_train, w_val, z_train, z_val = train_test_split(x_train, y_train, test_size = 0.20)
clf = GaussianNB()
clf.fit(x,y)
GaussianNB(priors=None, var_smoothing=1e-09)
print clf.score(w_val, z_val)
print clf.score(x_test, y_test) 


clf1 = Perceptron(tol = 1e-3, random_state= 0)
clf1.fit(w_val,z_val)
Perceptron(alpha=0.1, class_weight=None, early_stopping=False, eta0=1.0,
      fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
      penalty='elasticnet', random_state=0, shuffle=True, tol=0.001,
      validation_fraction=0.1, verbose=0, warm_start=False)
print clf1.score(w_val,z_val)

clf2 = Perceptron(tol = 1e-3, random_state= 0)
clf2.fit(x_train, y_train)
Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
      fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
      penalty='elasticnet', random_state=0, shuffle=True, tol=0.001,
      validation_fraction=0.1, verbose=0, warm_start=False)
print clf2.score(x_train, y_train)
test_pred = clf2.predict(x_test)

print 'Precision, recall, fscore, support(n/a)'
print precision_recall_fscore_support(y_test, test_pred, average='weighted')
print confusion_matrix(y_test, test_pred)

