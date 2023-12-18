import itertools
import numpy as np
import functools

POSS_PLACEMENTS = None
CONV_MAT_CACHE = {}
PREV_STATE_CACHE = set()
P_J_CORRECT_CACHE = {}
REAL_INDICES_CACHE = {}
CACHE_HITS = {}

def find_starting_index_of_submatrix(p,j):
    """ Return the starting index of submatrix j in P.
    """
    return (j // (p.shape[1]-1), j % (p.shape[1]-1))

def get_real_indices_of_submatrix(P, j):
    """ Return all the true index pairs of submatrix j in P.
    """
    cache_key = cache_P_j(P,j)
    if cache_key in REAL_INDICES_CACHE:
        return REAL_INDICES_CACHE[cache_key]
    row_begin_idx, col_begin_idx = find_starting_index_of_submatrix(P, j)
    # One left, one down, and one down-right
    indices = []
    indices.append((row_begin_idx, col_begin_idx))
    # Index of one right
    indices.append((row_begin_idx, col_begin_idx+1))
    # Index of one down
    indices.append((row_begin_idx+1, col_begin_idx))
    # Index of one down-right
    indices.append((row_begin_idx+1, col_begin_idx+1))
    REAL_INDICES_CACHE[cache_key] = indices
    return indices

def activation_function(x):
    """ Apply an 'activation' function to the input.
    g(x) = {1 if x == 1, 0 otherwise}
    """
    return np.where(x == 1, 1, 0)

def inv_activation_function(y):
    """ Apply the inverse of the 'activation' function to the input.
    h(x) = {1 if x == 1, NaN otherwise}
    """
    return np.where(y == 1, 1, np.nan)

def get_convolution_matrix(n,m):
    """ Return the convolution matrix, which operates on a flattened (n*m,1) vector.
    """
    if (n,m) in CONV_MAT_CACHE:
        return CONV_MAT_CACHE[(n,m)]
    # Initialize the convolution matrix
    C = np.zeros(((n-1)*(m-1), n*m))

    # Iterate over the rows of the convolution matrix
    for i in range(n-1):
        for j in range(m-1):
            # Calculate the index in the flattened matrix
            index = i*m + j

            # Set the corresponding elements in the convolution matrix
            C[i*(m-1) + j, [index, index+1, index+m, index+m+1]] = 1
    CONV_MAT_CACHE[(n,m)] = C
    return C

def convolution_and_activation(curr_state):
    """ Apply a 2x2 convolution and the 'activation' function to the current state.
    """
    C = get_convolution_matrix(curr_state.shape[0], curr_state.shape[1])
    out = C.dot(curr_state.flatten())
    out = activation_function(out)
    out = out.reshape(curr_state.shape[0]-1, curr_state.shape[1]-1)
    return out
    
def calc_next_state(curr_state):
    """ Calculate the next state from the current state.
    """
    return convolution_and_activation(curr_state)


def check_correctness_so_far(P, state, j):
    """ Check if the calculated previous state is correct so far.
    P is the calculated previous state so far.
    state is the current state.
    j is the current submatrix index.
    Returns True if the previous state is correct so far, False otherwise.
    """
    cache_key = cache_P_j(P,j)
    if cache_key in P_J_CORRECT_CACHE:
        return P_J_CORRECT_CACHE[cache_key]
    # Check if convoluted P is equal to state for all i up to j
    state_so_far = calc_next_state(P)
    #print(f"state_so_far (j={j}) = {state_so_far}")
    if np.array_equal(state_so_far.flatten()[:j+1], state.flatten()[:j+1]):
        P_J_CORRECT_CACHE[cache_key] = True
        return True
    P_J_CORRECT_CACHE[cache_key] = False
    return False

POSS_PLACEMENTS = None
def get_all_possible_placements():
    global POSS_PLACEMENTS
    if POSS_PLACEMENTS is not None:
        return POSS_PLACEMENTS
    fp = list(itertools.chain.from_iterable(itertools.combinations(range(4), r) for r in range(1, 5)))
    fp.append((-1,))
    POSS_PLACEMENTS = fp
    return fp

def cache_P_j(P, j):
    """ Cache the current state of P.
    """
    return (P.tobytes(), j)


def solution(state, P = None, j = 0):
    """ Count the number of previous states that lead to the current state.
    We do this recursively.
    """
    if not isinstance(state, np.ndarray):
        state = np.array(state, dtype=np.int16)
    
    if P is None:
        # Clear all caches
        CONV_MAT_CACHE.clear()
        PREV_STATE_CACHE.clear()
        P_J_CORRECT_CACHE.clear()
        REAL_INDICES_CACHE.clear()
        CACHE_HITS.clear()
        P = np.zeros((state.shape[0]+1, state.shape[1]+1), dtype=np.int16)
        
    cache_key = cache_P_j(P,j)
    
    # Check if we have already calculated the number of previous states
    if cache_key in PREV_STATE_CACHE:
        return 0
    
    # If j > 0 and P is not correct so far, return 0
    if j > 0 and not check_correctness_so_far(P, state, j-1):
        #print(f"No solutions from j = {j},P = \n{P}")
        #PREV_STATE_CACHE.add(cache_key)
        return 0
    
    # If j == (P.shape[0]-1)*(P.shape[1]-1) and P is correct, return 1
    if j == (P.shape[0]-1)*(P.shape[1]-1):# and check_correctness_so_far(P, state, j-1):
        #print(f"j = {j}, P = {P}, Found a solution")
        #PREV_STATE_CACHE.add(cache_key)
        return 1
                


    full_placements = get_all_possible_placements()
    
    submat_coords = get_real_indices_of_submatrix(P, j)
    num_poss_states = 0
    
    already_filled = False
    # Now we try all placements. If any place at P is already filled, we skip the placement
    for placement in full_placements:
        already_filled = False
        row_placements = []
        col_placements = []
        
        if -1 not in placement:
            placement_in_real_indices = [submat_coords[i] for i in placement]
            row_placements = [i[0] for i in placement_in_real_indices]
            col_placements = [i[1] for i in placement_in_real_indices]
            # Check if any of the placements are already filled
            if P[row_placements, col_placements].any():
                already_filled = True
                continue

        
        # Set the values of P at the placement to 1
        P[row_placements, col_placements] = 1
        num_sols = solution(state, P, j+1)
        PREV_STATE_CACHE.add(cache_P_j(P,j+1))
        # Reset the values of P at the placement to 0
        P[row_placements, col_placements] = 0
        num_poss_states += num_sols
    #PREV_STATE_CACHE.add(cache_key)
    return num_poss_states



if __name__ == '__main__':
    #curr_state = [[0,1,0,1],[0,0,1,0],[0,0,0,1],[1,0,0,0]]
    curr_state = [[True, False, True], [False, True, False], [True, False, True]]
    #curr_state = [[0,1,0],[1,1,1],[1,1,0]]
    #curr_state = [[1,1,1],[1,1,1],[1,1,1]]
    curr_state = np.diag(np.ones(4, dtype=np.int32))
    curr_state = [[True, True, False, True, False, True, False, True, True, False], [True, True, False, False, False, False, True, True, True, False], [True, True, False, False, False, False, False, False, False, True], [False, True, False, False, False, False, True, True, False, False]]
    #curr_state = [[True, False, True, False, False, True, True, True], [True, False, True, False, False, False, True, False], [True, True, True, False, False, False, True, False], [True, False, True, False, False, False, True, False], [True, False, True, False, False, True, True, True]]
    curr_state = np.array(curr_state, dtype=np.int16)
    print("curr_state = \n",curr_state)
    previous_states = solution(curr_state)
    print("Number of previous states = ", previous_states)
    print("Number of checked previous states:", len(PREV_STATE_CACHE))
    print("Number of cache hits:", CACHE_HITS)