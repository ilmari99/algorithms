import itertools
import numpy as np
import time
import pulp
import functools
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations, 
    implicit_multiplication,
)
import re

from foobar52X_conv import (
    activation_function,
    inv_activation_function,
    get_convolution_matrix,
    convolution_and_activation,
    calc_next_state,
)

from foobar52X_submatrix_utils import (
    check_correctness_so_far,
    find_starting_index_of_submatrix,
    find_sharing_submatrices,
    get_jth_submatrix,
    get_real_indices_of_submatrix,
    reset_placement_at_j
    
)

def get_combinations(X,hb0, XP=None, ret_XP=False):
    """ Return all possible values for the Nan values in hb0.

    To find the list of possible values for Nan values in hb0, we use the following method:
    0. Make an empty list 'combos'.
    1. Construct a matrix XP (zeros), which is an (n+1)x(m+1) matrix.
    2. Go through each 2x2 submatrix in XP, and set the submatrix to the index of the submatrix in hb0, IF the value in hb0 is not Nan.
    3. For each submatrix in XP, count the number of 0s, and the number of unique values, and sum these two numbers to get 'maxn'.
    Then, append list(range(maxn)).remove(1) to 'combos', if maxn != 1.
    """
    combos = []
    if XP is None:
        # Construct a matrix XP (zeros), which is an (n+1)x(m+1) matrix.
        XP = np.zeros((X.shape[0]+1, X.shape[1]+1))
        # Go through each 2x2 submatrix in XP, and set the submatrix to the index of the submatrix in hb0, IF the value in hb0 is not Nan.
        for submat_ind in range((X.shape[0])*(X.shape[1])):
            # Get the real indices of the submatrix
            real_indices = get_real_indices_of_submatrix(XP, submat_ind)
            row_indices = [real_index[0] for real_index in real_indices]
            col_indices = [real_index[1] for real_index in real_indices]
            # Get the value of the submatrix in hb0
            hb0_val = hb0[submat_ind]
            # If the value is not Nan, set the submatrix to the index of the submatrix in hb0
            if not np.isnan(hb0_val):
                XP[row_indices, col_indices] = submat_ind + 2
    # For each submatrix in XP, count the number of 0s, and the number of unique values, and sum these two numbers to get 'maxn'.
    for submat_ind in range((X.shape[0])*(X.shape[1])):
        # Get the real indices of the submatrix
        real_indices = get_real_indices_of_submatrix(XP, submat_ind)
        row_indices = [real_index[0] for real_index in real_indices]
        col_indices = [real_index[1] for real_index in real_indices]
        submat = XP[row_indices, col_indices]
        # Count the number of 0s, where we can put anything
        num_zeros = np.count_nonzero(submat == 0)
        
        # Count the number of values that are greater than len(hb0), where we can put anything
        num_over_max_nindx = np.count_nonzero(submat > len(hb0))
        
        # Count the number of unique values, that are below len(hb0), where we only put one
        num_unique_vals_lt_maxnindx = len(np.unique(submat[submat < len(hb0)]))
        
        # Count the number of negative values, where can not put anything (since these are included in num_unique_vals_lt_maxnindx)
        num_negatives = np.count_nonzero(XP[row_indices, col_indices] < 0)
        
        hb0_val = hb0[submat_ind]
        
        # Calculate how many values we can put in the submatrix at most:
        # Add the number of 0s and the number of values over maxnindx, the number of unique values below maxnindx and subtract the number of negative values
        maxn = num_zeros + num_over_max_nindx + num_unique_vals_lt_maxnindx - num_negatives
        # Clip maxn to between 1 and 4
        #maxn = np.clip(maxn, 1, 4)
        if hb0_val != 1:
            # Append list(range(maxn)).remove(1) to 'combos', if maxn != 1.
            vals = list(range(maxn))
            if 1 in vals:
                vals.remove(1)
            combos.append(vals)
    # Combine all the lists in 'combos' to get all possible combinations, and return this list.
    if ret_XP:
        return combos, XP
    return combos

def nested_tuple_to_array(nested_tuple):
    """ Convert a nested tuple to a numpy array.
    """
    return np.array(nested_tuple, dtype=np.int32)

def check_XP_is_valid(hb0, XP):
    """ return whether XP is valid, given hb0.
    XP is valid if
    - For each non-nanidx in hb0, the corresponding submatrix in XP contains atleast one positive value
    """
    non_nan_indices = np.where(~np.isnan(hb0))[0]
    for non_nan_index in non_nan_indices:
        real_indices = get_real_indices_of_submatrix(XP, non_nan_index)
        row_indices = [real_index[0] for real_index in real_indices]
        col_indices = [real_index[1] for real_index in real_indices]
        # If there are no positive values, return False
        nz_indices = np.where(XP[row_indices, col_indices] >= 0)[0]
        if len(nz_indices) == 0:
            return False
    return True


def get_legal_combinations(X,hb0, XP = None, observed_xp_vals = {}, current_combo = [], found_combos = [], depth=0):
    """ Return all legal combinations of values for the Nan values in hb0.
    First, we find all possible values for the Nan values in hb0.
    We then generate combinations by doing a depth-first search.
    
    current_combo is a list of values that have already been chosen.
    
    Firstly, we calculate vals, and the matrix XP, which is an (n+1)x(m+1) matrix.
    
    We find real combinations with a similar method to 'get_combinations'.
    'nindx' is an index in hb0 where the value is Nan.
    'submatrix' is a 2x2 submatrix in XP.
    'vals' is a list of possible values for the nindx'th Nan value in hb0.
    
    If there is only one nindx, submatrix, and vals, add all current_combo + val for val in vals to found_combos.
    Else
    1. Find the 'depth' nindx, submatrix, and vals
        - Go through each val in vals.
            - if val == 0
                - Set the submatrix to -nindx.
                - Call get_legal_combinations with the new XP, and current_combo + val.
                - Set the submatrix back to 0.
    2. Return found_combos.       
    """
    if XP is None:
        vals, XP = get_combinations(X, hb0, ret_XP=True)
    else:
        vals = get_combinations(X, hb0, XP)
    
    # Add xp to observed_xp
    xp_ = tuple([tuple(row) for row in XP])
    # If the xp and vals are already observed, return
    if xp_ in observed_xp_vals.keys() and observed_xp_vals[xp_] < depth:
        #print(f"Already observed xp: {xp_}")
        #time.sleep(1)
        return found_combos
    
    observed_xp_vals[xp_] = depth
    
    
    vals_clipped = vals[depth:]
    #print(f"depth = {depth}, vals = {vals_clipped}")
    # If there is only one nindx, submatrix, and vals, add all current_combo + val for val in vals to found_combos.
    if len(vals_clipped) == 1:
        for val in vals_clipped[0]:
            combo = current_combo + [val]
            #print(f"Found legal combination: {combo}")
            yield combo
            #found_combos.append(combo)
        return found_combos
    
    if any([len(val) == 0 for val in vals_clipped]):
        return found_combos
    
    # Go through each val in vals.
    for val in vals_clipped[0]:
        # Find the 'depth's nindx, submatrix, and vals
        nindxs = np.where(np.isnan(hb0))
        nindx = nindxs[0][depth]
        submatrix = get_real_indices_of_submatrix(XP, nindx)
        row_indices = [real_index[0] for real_index in submatrix]
        col_indices = [real_index[1] for real_index in submatrix]
        # Store the value of the submatrix
        submatrix_val = XP[row_indices, col_indices].copy()
        if val == 0:
            XP[row_indices, col_indices] = -nindx
        else:
            # Set the value to max(nindx) + nindx
            XP[row_indices, col_indices] = len(hb0) + nindx
        if check_XP_is_valid(hb0, XP):
            # Call get_legal_combinations with the new XP, and current_combo + val.
            gen = get_legal_combinations(X, hb0, XP, observed_xp_vals, current_combo + [val], found_combos, depth+1)
            while True:
                try:
                    combo = next(gen)
                    yield combo
                    #found_combos.append(combo)
                except StopIteration:
                    break
                
        # Set the submatrix back to its original value
        XP[row_indices, col_indices] = submatrix_val
        #print(f"depth = {depth}, val = {val}, found_combos = {found_combos}")
    return found_combos

def get_combinations_brute(xd):
    """ Return the kartesian product of [0, 2, 3, 4] for every Nan value in xd.
    """
    # Find the indices of the Nan values in xd
    nindxs = np.where(np.isnan(xd))
    # Find the number of Nan values in xd
    num_nans = len(nindxs[0])
    # Get the kartesian product of [0, 2, 3, 4] for every Nan value in xd
    combos = itertools.product([0, 2, 3, 4], repeat=num_nans)
    return combos

def substitute_nans(xd, combo, inplace = True, nindxs = None):
    """ Return a list of matrices, where the Nan values in xd are substituted with the values in combos.
    """
    if inplace:
        # Copy xd, so that we don't change xd
        xd = xd.copy()
    if not nindxs:
        # Find the indices of the Nan values in xd
        nindxs = np.where(np.isnan(xd))
    xd[nindxs] = combo
    return xd

def get_num_previous_states_brute(curr_state):
    """ Return the number of previous states of curr_state.
    The previous states are calculated by:
    """
    combos = get_legal_combinations(curr_state, curr_state.flatten())
    for c in combos:
        print(c)
    exit()
    
    b0 = curr_state.flatten().astype(np.float32)
    
    print(f"b0 = {b0}")
    
    A = get_convolution_matrix(*curr_state.shape)
    
    nindxs = np.where(b0 != 1)
    print(f"nindxs = {nindxs}")
    
    
    xd = b0.copy()
    # Set the values in xd to nan, where the values in b0 are not 1
    xd[nindxs] = np.nan
    
    combos = get_combinations_brute(xd)
    
    print(f"xd = {xd}")

    
    num_previous_states = 0
    for combo in combos:
        xd = substitute_nans(b0, combo, nindxs=nindxs)
        b0_poss = activation_function(xd)
        if np.array_equal(b0_poss, b0):
            print(f"Found solution: {xd}")
            num_previous_states += 1
    return num_previous_states

if __name__ == '__main__':
    #curr_state = [[0,1,0,1],[0,0,1,0],[0,0,0,1],[1,0,0,0]]
    curr_state = [[True, False, True], [False, True, False], [True, False, True]]
    #curr_state = [[0,1,0],[1,1,1],[1,1,0]]
    #curr_state = [[1,1,1],[1,1,1],[1,1,1]]
    #curr_state = np.diag(np.ones(4, dtype=np.int32))
    #curr_state = [[True, True, False, True, False, True, False, True, True, False], [True, True, False, False, False, False, True, True, True, False], [True, True, False, False, False, False, False, False, False, True], [False, True, False, False, False, False, True, True, False, False]]
    #curr_state = [[True, False, True, False, False, True, True, True], [True, False, True, False, False, False, True, False], [True, True, True, False, False, False, True, False], [True, False, True, False, False, False, True, False], [True, False, True, False, False, True, True, True]]
    curr_state = np.array(curr_state, dtype=np.int32)
    print(f"State at T = 0:\n{curr_state}\n")
    next_state = calc_next_state(curr_state)
    print(f"State at T = -1:\n{next_state}\n")
    previous_states = get_num_previous_states_brute(curr_state)
    print(f"Number of found solutions: {previous_states}\n")
