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

    @lhs.setter
    def lhs(self, value):
        self._lhs = Node(value)

    @property
    def rhs(self):
        return self._rhs

    @rhs.setter
    def rhs(self, value):
        self._rhs = Node(value)


def gini_index(data, obs_idx, pivot_value):
    """Computes the Gini Index of the given variable V,
    according to the 2 classes: v < pivot_value and V >= pivot_value
    It is assumed that data[:, var_idx] contains the values of the variable V,
    and data[:, obs_idx] the observations"""

    def partial_gini_index(variable_criteria, observation_value=0):
        # e.g. All the rows so that V1 < threshold
        data_subset = [row for row in data if variable_criteria(row)]
        data_subset_length = len(data_subset)
        # e.g. All the rows so that V1 < threshold AND observation == 0
        data_subset_observation = [row for row in data_subset if row[obs_idx] == observation_value]

        g_subset = 1 - ((len(data_subset_observation) / data_subset_length) ** 2
                        + ((data_subset_length - len(data_subset_observation)) / data_subset_length) ** 2)

        return g_subset, data_subset_length

    try:
        inf, sup = partial_gini_index(lambda row: row[0] < pivot_value), \
                   partial_gini_index(lambda row: row[0] >= pivot_value)

        return inf[0] * inf[1] / len(data) + sup[0] * sup[1] / len(data)

    except ZeroDivisionError:
        return 1


def get_pivot_var_and_value(data, verbose=1):
    """Given a (sub-)dataset,
    will determine the variable and value to use for the newt tree's LHS and RHS
    result (pivot) = (var index, var value)"""

    if verbose > 0:
        print('=' * 30)

    # Var ID, Var value, Gini idx
    values = []

    for obs in data:
        for i in range(2):
            gini = gini_index(dataset, 2, obs[i])
            values.append((i, obs[i], gini))
            if verbose > 1:
                print(f'Gini(V{i + 1}={obs[i]}) = {gini}')

    min_value = min(values, key=lambda row: row[2])
    if verbose > 0:
        print(f'Min value is: {min_value}')
        print('=' * 30, end='\n\n')

    return min_value[0], min_value[1]


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


root = get_pivot_var_and_value(dataset)

lhs = get_pivot_var_and_value([row for row in dataset if row[root[0]] < root[1]])
rhs = get_pivot_var_and_value([row for row in dataset if row[root[0]] >= root[1]])





