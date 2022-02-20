from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import custom_plotter
import matplotlib.pyplot as plt

iris_data = load_iris()
X = iris_data.data
Y = iris_data.target

# Raw data #
custom_plotter.plot_projections(X, Y, iris_data.target_names, 'iris-raw', 4,
                                lambda x: iris_data.feature_names[x], lambda y: iris_data.feature_names[y],
                                display=False)

# PCA #
pca_nb_components = 2
pca = PCA(n_components=pca_nb_components, copy=True, svd_solver='auto')
X2 = pca.fit(X).transform(X)

# EMPIRICALLY:
# Setosa vs. the rest of the world
x_setosa = -1.5

# Between versicolor / virginica
x0_versicolor = 1.3
y0_versicolor = 0
a_versicolor = .89


fig = custom_plotter.plot_projections(X2, Y, iris_data.target_names, 'iris-pca-main2', pca_nb_components,
                                      additional_display=lambda local_plt:
                                      (local_plt.axvline(x=x_setosa, color="royalblue"),
                                       local_plt.axline((x0_versicolor, y0_versicolor), slope=a_versicolor,
                                                        color="darkorange")),
                                      display=True)


# Simulating a test data set...
# This should NOT be the learning data set
X_test_dataset = X[::-1]
X3 = pca.fit(X).transform(X_test_dataset)

Y_test_dataset = list()

for flower in X3:
    if flower[0] <= x_setosa:
        Y_test_dataset.append(0)







