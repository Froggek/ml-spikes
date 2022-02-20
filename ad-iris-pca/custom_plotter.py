import math
from matplotlib import pyplot as plt  # replaces the name "pyplot" by "plt"
from matplotlib import lines as plt_lines


def plot_projections(data, labels, label_names, title, nb_components=2,
                     x_label_gen=lambda x: f'x #{x}', y_label_gen=lambda y: f'y #{y}',
                     additional_display=None, display=False):
    """ Given a data set and the corresponding labels (and label names),
     will plot all the possible 2-D projections on a single figure,
     and then save it as a PNG file"""

    # if not display:
    #     plt.ioff()  # turning OFF interactive mode

    nb_plots = nb_components * (nb_components - 1) / 2
    nb_unique_labels = len(set(labels))
    plot_id = 1

    fig = plt.figure(figsize=(15, 25), dpi=100)
    displayed_objects = []
    displayed_labels = []
    for abscissa in range(nb_components):
        for ordinate in range(abscissa + 1, nb_components):
            # determines the nb of rows, and columns
            plt.subplot(math.ceil(math.sqrt(nb_plots)), math.ceil(nb_plots / math.ceil(math.sqrt(nb_plots))), plot_id)

            plt.xlabel(x_label_gen(abscissa))
            plt.ylabel(y_label_gen(abscissa))

            for i in range(nb_unique_labels):
                tmp_displayed_object = plt.scatter(data[labels == i][:, abscissa], data[labels == i][:, ordinate],
                                                   label=label_names[i])

                # We added it to the legend ONLY for the first (abscissa, ordinate) iteration
                # Otherwise, it creates duplicated entries in the legend
                if i == len(displayed_objects):
                    displayed_objects.append(tmp_displayed_object)
                    displayed_labels.append(label_names[i])

            plt.legend()
            plot_id += 1

    if additional_display is not None:
        additional_elts, additional_labels = additional_display(plt)
        displayed_objects.extend(additional_elts)
        displayed_labels.extend(additional_labels)

    plt.legend(displayed_objects, displayed_labels)

    plt.savefig(f'output/{title}.png')

    if not display:
        plt.close(fig)

    return fig

