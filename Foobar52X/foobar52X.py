import itertools
import numpy as np
import time
import functools

def _calc_next_state(curr_state : np.ndarray):
    """ Calculate the next state from the current state.
    """
    next_state = np.zeros((curr_state.shape[0]-1, curr_state.shape[1]-1))
    for i in range(next_state.shape[0]):
        for j in range(next_state.shape[1]):
            submat = curr_state[i:i+2, j:j+2]
            #print(f"submat = {submat}")
            #print(f"submat.sum() = {submat.sum()}")
            if submat.sum() == 1:
                next_state[i, j] = 1
            else:
                next_state[i, j] = 0
    return next_state


def activation_function(x : np.ndarray):
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
    """ Calculate the convolution matrix for a n x m matrix.
    The matrix is cached for each n and m.
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
        #print(f"submat_index = {submat_index}, j_to_real_indices_map[submat_index] = {j_to_real_indices_map[submat_index]}")
        
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
                    
    

def find_starting_index_of_submatrix(p,j):
    """ Return the starting index of submatrix j in P.
    """
    return (j // (p.shape[1]-1), j % (p.shape[1]-1))

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

def nested_tuple_to_array(nested_tuple):
    """ Convert a nested tuple to a numpy array.
    """
    return np.array(nested_tuple, dtype=np.int32)


def get_previous_states(input):
    """ Calculate a deconvolution operation.
    We can not know the previous state exactly, but we can use this to find A solution.
    This works by starting with an empty n+1 x m+1 matrix, and then filling in True or False greedily.
    Algorithm:
    1. Start with an empty n+1 x m+1 matrix (P)
    
    2. Go through each 2x2 submatrix (s_j) of P and modify s_j so that conv(s_i) == input_i for all i up to j.
    Keep track of which placements we have tried for each submatrix index.
    
    3. If s_j can not be modified reset its tried placements info, and backtrack to s_{j-1} and do 2 again with a different modification.
    4. When all s_j have been modified, return P.
    """
    # Create the empty matrix
    P = np.zeros((input.shape[0]+1, input.shape[1]+1))
    # Pre compute a list of all possible placements
    full_placements = list(itertools.chain.from_iterable(itertools.combinations(range(4), r) for r in range(1, 5)))
    full_placements.append((-1,))
    full_placements = set(full_placements)
    #print(f"full_placements = {full_placements}")
    # Keep track of which placements we have tried for each submatrix index
    # int -> list(of ints), -1 means that we do not put anything in this submatrix
    index_to_tried_placements_map = {i : set() for i in range((P.shape[0]-1)*(P.shape[1]-1))}
    # Keep track of the current placement for each submatrix index
    index_to_current_placement_map = {i : (-1,) for i in range((P.shape[0]-1)*(P.shape[1]-1))}
    # Find the sharing submatrices for each submatrix
    index_to_sharing_submatrices = {j : find_sharing_submatrices(P, j) for j in range((P.shape[0]-1)*(P.shape[1]-1))}
    #print(f"index_to_sharing_submatrices = {index_to_sharing_submatrices}")
    index_to_sharing_submatrices_real_indices = {j : find_sharing_submatrices(P, j, get_real_indices=True) for j in range((P.shape[0]-1)*(P.shape[1]-1))}
    # Go through each 2x2 submatrix
    submat_index = 0
    # Calculate the indices of the top left corner of the submatrix. The submatrix is a sliding 2x2 window.
    submat_row_begin_idx = submat_index // (P.shape[1]-1)
    submat_col_begin_idx = submat_index % (P.shape[1]-1)
    found_solutions = set()
    already_found = False
    while True:
        #print(f"P =\n{P}")
        submat_row_begin_idx, submat_col_begin_idx = find_starting_index_of_submatrix(P, submat_index)
        #print(f"submat_index = {submat_index}, submat_row_begin_idx = {submat_row_begin_idx}, submat_col_begin_idx = {submat_col_begin_idx}")
        
        # If we find a solution, print it and try to find more solutions
        if submat_row_begin_idx == P.shape[0]-1:# or len(index_to_tried_placements_map[submat_index]) == len(full_placements):
            # Convert the solution to a nested tuple
            sol = tuple(tuple(row) for row in P)
            len_before = len(found_solutions)
            found_solutions.add(sol)
            already_found = len_before == len(found_solutions)
            #print(f"Found solution number {len(found_solutions)}")
            #print(f"P =\n{P}")
            # At this point, we have found a solution (we have found a P matrix that satisfies the input)
            # To find more solutions, lets go back to the previous submatrix and let the algorithm continue
            submat_index -= 1
            if submat_index < 0:
                found_solutions = [nested_tuple_to_array(sol) for sol in found_solutions]
                # We have backtracked to the first submatrix, so we can not find a solution
                return found_solutions
            continue
            
        # Reset the current placement for this submatrix
        if index_to_current_placement_map[submat_index][0] != -1:
            curr_placements = index_to_current_placement_map[submat_index]
            P = reset_placement_at_j(P, submat_index, curr_placements)
            index_to_current_placement_map[submat_index] = (-1,)
            
        # Make a copy of P, that we modify through submat_copy
        P_copy = P.copy()
        submat_copy = get_jth_submatrix(P_copy, submat_index, copy=False)
        
        # The possible placements for all submatrices are -1:4,
        # where -1 means that we do not put anything in this submatrix.
        # But we remove all placements, that we have already tried and if squares are already filled.
        possible_unit_placements = [i for i in range(4) if submat_copy.flatten()[i] == 0]
        # All the ways to pick 1 - 4 elements from possible_unit_placements
        possible_placements = list(itertools.chain.from_iterable(itertools.combinations(possible_unit_placements, r) for r in range(1, 5)))
        
        # Add the -1 placement
        #possible_placements = list(filter(lambda x : all((submat_copy.flatten()[elem] == 0 for elem in x if elem != -1)), possible_placements))
        # Remove the placements that we have already tried
        #if submat_index == 0:
        #    print(f"Tried placements for submat_index = {submat_index} are {index_to_tried_placements_map[submat_index]}")
        #    print(f"Possible placements before removing already tried placements = {possible_placements}")
        possible_placements = list(filter(lambda x : x not in index_to_tried_placements_map[submat_index], possible_placements))
        # Sort each placement in an ascending order
        #possible_placements = [tuple(sorted(placement)) for placement in possible_placements]
        
        #if submat_index == 0:
        #    print(f"Possible placements before removing shared placements = {possible_placements}")
        
        # Check each matrix that
        # - shares a value with _this submatrix_, and
        # - its index is smaller than this submatrix
        # We'll call matrices that fill this criteria _sharing matrices_.
        # If a sharing matrix has tried a placement (a set of indices in the sharing matrix which are set to 1, for example (0,2,3)),
        # which:
        # - Is not -1 (which means that we do not put anything in this submatrix), and
        # - Contains all elements of the current placement, and
        # - The other elements are shared with this submatrix
        # Then we do not need to try the placement for this submatrix.
        
        # Note: we need to remove the elements of the placement, that are not shared with this submatrix,
        # so we can remove the already tried placement

        this_matrix_coordinates = get_real_indices_of_submatrix(P, submat_index)
        
        # Possible placements to real indices
        new_possible_placements = []
        for placement in possible_placements:
            real_coords = []
            for elem in placement:
                real_coords.append(this_matrix_coordinates[elem])
            new_possible_placements.append(tuple(real_coords))
        possible_placements = new_possible_placements
        
        this_matrix_coordinates_set = set(this_matrix_coordinates)
        #print(f"this_matrix_coordinates = {this_matrix_coordinates}")
        possible_placements_set = set(possible_placements)
        
        #print(f"submat_index = {submat_index}, possible_placements_set = {possible_placements_set}")
        
        # Convert to real indices
        #possible_placements_set = set([tuple([this_matrix_coordinates[i] for i in placement]) for placement in possible_placements_set])
        
        possible_placements.append((-1,))
        #possible_placements_set.add((-1,))
        #print(f"Possible placements before removing placement = {possible_placements}")
        #print(f"Possible placements (set, coordinates) before removing placement = {possible_placements_set}")
        # Go through each sharing matrix
        # Take all the previous sharing matrices
        for sharing_matrix_index in range(submat_index):
            if not possible_placements_set:
                break
            #print(f"Checking sharing matrix {sharing_matrix_index} with real indices {sharing_matrix_real_indices}")
            # The real indices of the sharing matrix
            sharing_matrix_real_indices = get_real_indices_of_submatrix(P, sharing_matrix_index)
            # Get the current placement of the sharing matrix
            current_placement = index_to_current_placement_map[sharing_matrix_index]
            
            if current_placement[0] != -1:
                # Convert the current placement to coordinates
                current_placement = set([sharing_matrix_real_indices[i] for i in current_placement])
            else:
                current_placement = {(-1,)}
            #print(f"Current placement of sharing matrix {sharing_matrix_index} is {current_placement}")
            
            # Go through each placement that we have tried for this sharing matrix
            for placement in index_to_tried_placements_map[sharing_matrix_index]:
                #print(f"Checking placement {placement} -> ", end="")
                placement = set([sharing_matrix_real_indices[i] for i in placement]) if -1 not in placement else {(-1,)}
                #print(f"{placement}")
                if (-1,) in placement:
                    continue
                
                #print(f"Checking placement = {placement} in sharing matrix {sharing_matrix_index}")
                # Check if all the values in the current placement are in the placement
                if current_placement.issubset(placement):
                    #print(f"Current placement is a subset of placement = {placement}")
                    
                    # Check if the other values in the placement are shared with this submatrix
                    other_values = placement.difference(current_placement)
                    # Check if all the other values are found in this_matrix_coordinates_set
                    if other_values.issubset(this_matrix_coordinates_set) or len(other_values) == 0:
                        #if len(other_values) == 0:
                        #    pass
                            #print(f"Other values are in this_matrix_coordinates_set because len(other_values) == 0")
                        #else:
                        #    pass
                            #print(f"Other values {other_values} are in this_matrix_coordinates_set {this_matrix_coordinates_set}")
                        #    time.sleep(1)
                        
                        # Remove values from the placement that are not shared with this submatrix
                        placement = placement.intersection(this_matrix_coordinates_set)
                        #print(f"The already tried values shared with this submatrix are {placement}")
                        
                        # Cehck if placement is in possible_placements
                        placement = tuple(placement)
                        if placement in possible_placements_set:
                            possible_placements_set.remove(placement)
                            # Convert to coordinates and add to tried
                            placement = tuple([this_matrix_coordinates.index(i) for i in placement])
                            # Sort the placement in an ascending order
                            #placement = tuple(sorted(placement))
                            index_to_tried_placements_map[submat_index].add(placement)
                            #exit()
                    #else:
                    #    pass
                        #print(f"Other values {other_values} are NOT in this_matrix_coordinates_set {this_matrix_coordinates_set}")
                #else:
                #    pass
                    #print(f"Current placement is NOT a subset of placement = {placement}")
        # Update the possible placements to be a list of in submatrix indices
        #possible_placements = list(possible_placements_set)
        #print(f"possible_placements_set = {possible_placements_set}")
        updated_possible_placements = []
        for placement in possible_placements_set:
            placement = set(placement)
            placement = [this_matrix_coordinates.index(i) for i in placement]
            placement = tuple(placement)
            updated_possible_placements.append(placement)
        # Add -1 if it is in the original possible_placements
        if (-1,) not in index_to_tried_placements_map[submat_index]:
            updated_possible_placements.append((-1,))
        possible_placements = updated_possible_placements.copy()

        # Try all possible placements for this submatrix
        found_correct_placement = False
        for placement in possible_placements:
            #print(f"Trying placement = {placement} in submat_index = {submat_index}")
            if placement[0] == -1:
                # Keep the submatrix as-is
                pass
            else:
                for placement_element in placement:
                    # Change the submatrix at the current placement
                    row_idx = placement_element // 2
                    col_idx = placement_element % 2
                    submat_copy[row_idx, col_idx] = 1
            # Update the tried placements
            # Sort the placement in an ascending order
            placement = tuple(sorted(placement))
            index_to_tried_placements_map[submat_index].add(placement)
            #print(f"Tried placements for submat_index = {submat_index} are {index_to_tried_placements_map[submat_index]}")
            # Check if the current placement is correct so far
            is_correct_so_far = check_correctness_so_far(P_copy, input, submat_index)
            #print(f"Trying placement = {placement}, in submat_index = {submat_index}, is_correct_so_far = {is_correct_so_far}")
            #print(f"P_copy = {P_copy}")
            #time.sleep(1)
            if is_correct_so_far:
                # Update the current placement
                found_correct_placement = True
                index_to_current_placement_map[submat_index] = placement
                break
            else:
                # Reset the current placement
                if placement[0] == -1:
                    # Keep the submatrix as-is
                    pass
                else:
                    for placement_element in placement:
                        # Change the submatrix at the current placement
                        row_idx = placement_element // 2
                        col_idx = placement_element % 2
                        submat_copy[row_idx, col_idx] = 0
        #exit()
        if found_correct_placement:
            # Go to the next submatrix
            submat_index += 1
            P = P_copy
        else:
            #print(f"Backtracking from submat_index = {submat_index}")
            # Reset the tried placements for this submatrix
            index_to_tried_placements_map[submat_index] = set()
            # Backtrack to the previous submatrix
            submat_index -= 1
            
        if submat_index < 0:
            found_solutions = [nested_tuple_to_array(sol) for sol in found_solutions]
            # We have backtracked to the first submatrix, so we can not find a solution
            return found_solutions
        #print()


if __name__ == '__main__':
    #curr_state = [[0,1,0,1],[0,0,1,0],[0,0,0,1],[1,0,0,0]]
    curr_state = [[True, False, False], [False, True, False], [True, False, True]]
    #curr_state = [[0,1,0],[1,1,1],[1,1,0]]
    #curr_state = [[1,1,1],[1,1,1],[1,1,1]]
    #curr_state = np.diag(np.ones(4, dtype=np.int32))
    #curr_state = [[True, True, False, True, False, True, False, True, True, False], [True, True, False, False, False, False, True, True, True, False], [True, True, False, False, False, False, False, False, False, True], [False, True, False, False, False, False, True, True, False, False]]
    #curr_state = [[True, False, True, False, False, True, True, True], [True, False, True, False, False, False, True, False], [True, True, True, False, False, False, True, False], [True, False, True, False, False, False, True, False], [True, False, True, False, False, True, True, True]]
    curr_state = np.array(curr_state, dtype=np.int32)
    print(f"State at T = 0:\n{curr_state}\n")
    next_state = calc_next_state(curr_state)
    print(f"State at T = -1:\n{next_state}\n")
    previous_states = get_previous_states(curr_state)
    print(f"Number of found solutions: {len(previous_states)}\n")
    #exit()
    all_true = True
    unique_previous_states = set()
    #exit()
    for previous_state in previous_states:
        #print(f"State at T = 0 (inferred from T = 1):\n{previous_state}\n")
        next_from_inferred_previous_state = calc_next_state(previous_state)
        #print(f"Next state calculated from the inferred previous state:\n{next_from_inferred_previous_state}\n")
        # Check if the inferred previous state is correct
        are_equal = np.array_equal(next_from_inferred_previous_state, next_state)
        #print(f"The next state from the inferred previous state is equal to the true next state: {are_equal}")
        if not are_equal:
            print(f"Previous state \n{previous_state} is not truely the previous state of \n{next_state}")
            all_true = False
        # Check if this inferred previous state is unique
        previous_state_as_str = str(previous_state)
        if previous_state_as_str in unique_previous_states:
            print(f"Found a duplicate inferred previous state:")#\n{previous_state}\n")
        unique_previous_states.add(previous_state_as_str)
    print(f"Number of unique inferred previous states: {len(unique_previous_states)}")
    print(f"Number of found solutions: {len(previous_states)}\n")
    print(f"All inferred previous states are correct: {all_true}")
            