import numpy as np
from sklearn import tree

from sklearn.datasets import load_iris


data = np.array(
    [[2.771244718, 1.784783929],
     [1.728571309, 1.169761413],
     [3.678319846, 2.81281357],
     [3.961043357, 2.61995032],
     [2.999208922, 2.209014212],
     [7.497545867, 3.162953546],
     [9.00220326, 3.339047188],
     [7.444542326, 0.476683375],
     [10.12493903, 3.234550982],
     [6.642287351, 3.319983761]]
)

target = np.array([0] * 5 + [1] * 5)

clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
clf = clf.fit(data, target)

tree.plot_tree(clf)

iris = load_iris()
X, y = iris.data, iris.target
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X, y)

# tree.plot_tree(clf2)

