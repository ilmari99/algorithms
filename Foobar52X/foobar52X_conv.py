import numpy as np
import functools

def activation_function(x):
    """ Apply an 'activation' function to the input.
    g(x) = {1 if x == 1, 0 otherwise}
    """
    return np.where(x == 1, 1, 0)

def inv_activation_function(y : np.ndarray):
    """ Apply the inverse of the 'activation' function to the input.
    h(x) = {1 if x == 1, NaN otherwise}
    """
    return np.where(y == 1, 1, np.nan)

@functools.lru_cache(maxsize=None)
def get_convolution_matrix(n,m):
    """ Return the convolution matrix, which operates on a flattened (n*m,1) vector.
    """
    # Initialize the convolution matrix
    C = np.zeros(((n-1)*(m-1), n*m))

    # Iterate over the rows of the convolution matrix
    for i in range(n-1):
        for j in range(m-1):
            # Calculate the index in the flattened matrix
            index = i*m + j

            # Set the corresponding elements in the convolution matrix
            C[i*(m-1) + j, [index, index+1, index+m, index+m+1]] = 1
    return C

def convolution_and_activation(curr_state : np.ndarray):
    """ Apply a 2x2 convolution and the 'activation' function to the current state.
    """
    C = get_convolution_matrix(curr_state.shape[0], curr_state.shape[1])
    out = C.dot(curr_state.flatten())
    out = activation_function(out)
    out = out.reshape(curr_state.shape[0]-1, curr_state.shape[1]-1)
    return out
    
def calc_next_state(curr_state : np.ndarray):
    """ Calculate the next state from the current state.
    """
    return convolution_and_activation(curr_state)