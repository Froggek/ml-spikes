from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import custom_plotter

iris_data = load_iris()
X = iris_data.data
Y = iris_data.target

# plot_id = 1
# plt.figure(figsize=(15, 25), dpi=100)
#
# for abscissa in range(4):
#     for ordinate in range(abscissa + 1, 4):
#         # create grid
#         plt.subplot(3, 2, plot_id)
#
#         plt.xlabel(iris_data.feature_names[abscissa])
#         plt.ylabel(iris_data.feature_names[ordinate])
#
#         for i in range(3):
#             plt.scatter(X[Y == i][:, abscissa], X[Y == i][:, ordinate], label=iris_data.target_names[i])
#
#         plt.legend()
#         plot_id += 1
# plt.savefig('iris-raw.png')

# Raw data #
custom_plotter.plot_projections(X, Y, iris_data.target_names, 'iris-raw', 4,
                                lambda x: iris_data.feature_names[x], lambda y: iris_data.feature_names[y])

# PCA #
pca_nb_components = 2
pca = PCA(n_components=2, copy=True, svd_solver='auto')
X2 = pca.fit(X).transform(X)

custom_plotter.plot_projections(X2, Y, iris_data.target_names, 'iris-pca', pca_nb_components)

