class Node:
    def __init__(self, value):
        self._value = value
        self._lhs = None
        self._rhs = None

    @property
    def value(self):
        return self._value

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs


def gini_index(data, pivot_value):
    """Computes the Gini Index of the given variable V,
    according to the 2 classes: v < pivot_value and V >= pivot_value
    It is assumed that data[:, 0] contains the values of the variable V,
    and data[:, 1] the observations"""

    def partial_gini_index(variable_criteria, observation_value=0):
        # e.g. All the rows so that V1 < threshold
        data_subset = [row for row in data if variable_criteria(row)]
        data_subset_length = len(data_subset)
        # e.g. All the rows so that V1 < threshold AND observation == 0
        data_subset_observation = [row for row in data_subset if row[1] == observation_value]

        g_subset = 1 - ((len(data_subset_observation) / data_subset_length) ** 2
                        + ((data_subset_length - len(data_subset_observation)) / data_subset_length) ** 2)

        return g_subset, data_subset_length

    try:
        inf, sup = partial_gini_index(lambda row: row[0] < pivot_value), \
                   partial_gini_index(lambda row: row[0] >= pivot_value)

        return inf[0] * inf[1] / len(data) + sup[0] * sup[1] / len(data)

    except ZeroDivisionError:
        return 1


dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [2.999208922, 2.209014212, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1],
           [6.642287351, 3.319983761, 1]]

for obs in dataset:
    print(f'{gini_index([r[::2] for r in dataset], obs[0])}')
