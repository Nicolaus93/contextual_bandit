import numpy as np


def sherman_morrison(M_inv, x):
    """
    Input:
        - x: (np.array) column vector
        - M_inv: (np.array) inverse of M matrix
    Output:
        (M + x*x')^-1 computed using Sherman-Morrison formula
    """
    x = x.reshape((-1, 1))
    M_inv -= M_inv.dot(x.dot(x.T.dot(M_inv))) / (1 + x.T.dot(M_inv.dot(x)))
    return M_inv


def get_random_state(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    return random_state
