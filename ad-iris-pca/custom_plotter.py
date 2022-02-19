import math
from matplotlib import pyplot as plt  # replaces the name "pyplot" by "plt"


def plot_projections(data, labels, label_names, title, nb_components=2,
                     x_label_gen=lambda x: f'x #{x}', y_label_gen=lambda y: f'y #{y}'):
    """ Given a data set and the corresponding labels (and label names),
     will plot all the possible 2-D projections on a single figure,
     and then save it as a PNG file"""
    nb_plots = nb_components * (nb_components - 1) / 2
    plot_id = 1

    plt.figure(figsize=(15, 25), dpi=100)

    for abscissa in range(nb_components):
        for ordinate in range(abscissa + 1, nb_components):
            # determines the nb of rows, and columns
            plt.subplot(math.ceil(math.sqrt(nb_plots)), math.ceil(nb_plots / math.ceil(math.sqrt(nb_plots))), plot_id)

            plt.xlabel(x_label_gen(abscissa))
            plt.ylabel(y_label_gen(abscissa))

            for i in range(3):
                plt.scatter(data[labels == i][:, abscissa], data[labels == i][:, ordinate], label=label_names[i])

            plt.legend()
            plot_id += 1
    plt.savefig(f'output/{title}.png')