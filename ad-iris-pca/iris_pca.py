import numpy as np
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
                                display=True)

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


def plot_separation_lines(local_plt):
    global x_setosa, x0_versicolor, y0_versicolor, a_versicolor
    setosa = local_plt.axvline(x=x_setosa, color="royalblue")
    versicolor = local_plt.axline((x0_versicolor, y0_versicolor), slope=a_versicolor, color="darkorange")
    return [setosa, versicolor], ["setosa limit", "versicolor limit"]


custom_plotter.plot_projections(X2, Y, iris_data.target_names, 'iris-pca-main2', pca_nb_components,
                                additional_display=plot_separation_lines, display=True)

# Simulating a test data set...
# This should NOT be the learning data set
X_test_dataset = np.copy(X)
X3 = pca.fit(X).transform(X_test_dataset)

Y_test_dataset = np.ndarray((len(X_test_dataset)), dtype=int)

for idx, flower in enumerate(X3):
    if flower[0] <= x_setosa:
        Y_test_dataset[idx] = 0

    # Flower is Virginica if the scalar product
    # with a normal vector is > 0
    elif (flower[0] - x0_versicolor) * a_versicolor + (flower[1] - y0_versicolor) * (-1) > 0:
        Y_test_dataset[idx] = 2

    # Otherwise, Versicolor
    else:
        Y_test_dataset[idx] = 1

# Looking for differences between learning dataset / test dataset
diff_idx = np.where((Y != Y_test_dataset))


def plot_separation_lines_and_errors(values, differences):
    x_errors = []
    y_errors = []
    for diff in differences:
        x_errors.append(values[diff][0])
        y_errors.append(values[diff][1])

    def result(local_plt):
        leg_elts, leg_labels = plot_separation_lines(local_plt)
        extra_points, = local_plt.plot(x_errors, y_errors, 'o', color='red')  # 'o' = style
        leg_elts.append(extra_points)
        leg_labels.append(f"errors ({len(differences)})")
        return leg_elts, leg_labels

    return result


custom_plotter.plot_projections(X3, Y_test_dataset, iris_data.target_names, 'iris-pca-test',
                                pca_nb_components,
                                additional_display=plot_separation_lines_and_errors(X3, diff_idx[0]), display=True)
