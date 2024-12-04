import numpy as np
from foobar52X_conv import calc_next_state

def check_correctness_so_far(P, state, j):
    """ Check if the calculated previous state is correct so far.
    P is the calculated previous state so far.
    state is the current state.
    j is the current submatrix index.
    Returns True if the previous state is correct so far, False otherwise.
    """
    # Check if convoluted P is equal to state for all i up to j
    state_so_far = calc_next_state(P)
    #print(f"state_so_far (j={j}) = {state_so_far}")
    if np.array_equal(state_so_far.flatten()[:j+1], state.flatten()[:j+1]):
        return True
    return False

def find_starting_index_of_submatrix(p,j):
    """ Return the starting index of submatrix j in P.
    """
    return (j // (p.shape[1]-1), j % (p.shape[1]-1))

def find_sharing_submatrices(P, j, get_real_indices=False):
    """ Return indices of submatrices that share a value with submatrix j.
    Also return the within-submatrix indices of the shared values.
    Return as a dictionary: {submat_index -> [0,1,2,3]}.
    
    The shared submatrices are the submatrices that are connected to submatrix j.
    At most there are 8 connected submatrices, if j is in the middle.
    At a side (or top/bot), there are 5 connected submatrices.
    At a corner, there are 3 connected submatrices.
    """
    # Get all the real indices of submatrices
    j_to_real_indices_map = {}
    for submat_index in range((P.shape[0]-1)*(P.shape[1]-1)):
        j_to_real_indices_map[submat_index] = get_real_indices_of_submatrix(P, submat_index)
        print(f"submat_index = {submat_index}, j_to_real_indices_map[submat_index] = {j_to_real_indices_map[submat_index]}")
        
    # Find which submatrices share a value with submatrix j
    # The shared submatrices are the submatrices that are connected to submatrix j.
    submat_j_indices = j_to_real_indices_map[j]
    sharing_submatrices = {}
    for submat_index, real_indices in j_to_real_indices_map.items():
        if submat_index == j:
            continue
        for real_index in real_indices:
            if real_index in submat_j_indices:
                if submat_index not in sharing_submatrices:
                    sharing_submatrices[submat_index] = []
                if get_real_indices:
                    sharing_submatrices[submat_index].append(real_index)
                else:
                    sharing_submatrices[submat_index].append(real_indices.index(real_index))
    return sharing_submatrices

def get_jth_submatrix(P, j, copy=False):
    """ Return the jth submatrix of P.
    """
    if copy:
        P = P.copy()
    row_begin_idx, col_begin_idx = find_starting_index_of_submatrix(P, j)
    return P[row_begin_idx:row_begin_idx+2, col_begin_idx:col_begin_idx+2]

def reset_placement_at_j(P, j, placement, inplace=True):
    """ Reset the placement at submatrix j in P.
    """
    if not inplace:
        P = P.copy()
    row_begin_idx, col_begin_idx = find_starting_index_of_submatrix(P, j)
    submat = P[row_begin_idx:row_begin_idx+2, col_begin_idx:col_begin_idx+2]
    for placement_element in placement:
        if placement_element == -1:
            # Keep the submatrix as-is
            pass
        # Change the submatrix at the current placement
        row_idx = placement_element // 2
        col_idx = placement_element % 2
        submat[row_idx, col_idx] = 0
    return P

def get_real_indices_of_submatrix(P, j):
    """ Return all the true index pairs of submatrix j in P.
    """
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
    return indices