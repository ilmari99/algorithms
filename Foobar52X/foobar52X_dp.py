"""
The problem is to find the number of possible previous states of a given grid.
It is a reverse Cellular Automaton problem. This solution is like the fifth iteration of
my solutions. I didn't finish the problem in time, but this solution is the fastest I have,
and it is probably too slow. I think this could be improved, maybe even drastically.

We solve the problem by finding all possible pre-images of the first row of the grid.
We then find all possible pre-images of the second row of the grid, so we know
that the first row of the grid must be one of previous layers' second rows, and so on.
We continue this process until we reach the last row of the grid, and then we count the
number of possible pre-images of the last row of the grid, which is the total number of
previous states that lead to the current state.
"""
import numpy as np
import itertools
import time
from foobar52X_conv import calc_next_state as calc_next_state_conv
from foobar52X_recursive import solution as solution_recursive
from foobar52_others_sol import solution as solution_other

# No cache in Python2.7
ALL_FULL_GRIDS_FROM_UPPER_ROW_CACHE = {}
NEXT_STATES_CACHE = {}


def calc_next_state(curr_state):
    """ Return the next state of the given grid.
    Each 2x2 submatrix becomes 1, if exactly one of its cells is 1, 0 otherwise.
    We can do this by doing a binary operation on each 2x2 submatrix.
    """
    cache_key = curr_state.tobytes()
    if cache_key in NEXT_STATES_CACHE:
        return NEXT_STATES_CACHE[cache_key]
    next_state = np.zeros((curr_state.shape[0]-1, curr_state.shape[1]-1), dtype=np.int16)
    for i in range(curr_state.shape[0]-1):
        for j in range(curr_state.shape[1]-1):
            # If exactly one of the cells is 1, the result is 1, 0 otherwise
            next_state[i,j] = 1 if np.sum(curr_state[i:i+2,j:j+2]) == 1 else 0
    NEXT_STATES_CACHE[cache_key] = next_state
    return next_state

def get_all_possible_2xn_plus_1_grids(n):
    """ Return all possible binary 2 x (n+1) grids.
    """
    a = itertools.product([0,1], repeat=2*(n+1))
    # reshape to 2 x (n+1)
    a = map(lambda x: np.array(x).reshape(2,n+1), a)
    return a


def all_full_grids_from_upper_row(upper_row):
    """ Return all possible full grids given that the upper row is fixed.
    """
    #print("upper_row: ", upper_row)
    if not isinstance(upper_row, np.ndarray):
        upper_row = np.array(upper_row)
    upper_row_hash = upper_row.tobytes()
    if upper_row_hash in ALL_FULL_GRIDS_FROM_UPPER_ROW_CACHE:
        return ALL_FULL_GRIDS_FROM_UPPER_ROW_CACHE[upper_row_hash]
    n = upper_row.shape[0]
    lower_rows = itertools.product([0,1], repeat=n)
    grids = []
    # Yield all possible full grids
    for lower_row in lower_rows:
        lower_row = np.array(lower_row)
        grid = np.vstack((upper_row, lower_row), dtype=np.int16)
        grids.append(grid)
    ALL_FULL_GRIDS_FROM_UPPER_ROW_CACHE[upper_row_hash] = grids
    return grids

def get_filter_to_check_if_grid_produces_row(row):
    """ Returns a function, that takes one input (Grid) and returns True if the grid
    produces the given row, False otherwise.
    """
    def filter_func(grid):
        next_state = calc_next_state(grid).flatten()
        return np.array_equal(next_state, row)
    return filter_func

def solution(curr_state):
    """ Return the number of possible previous states of the given grid.
    """
    if not isinstance(curr_state, np.ndarray):
        curr_state = np.array(curr_state, dtype = np.int16)
    ALL_FULL_GRIDS_FROM_UPPER_ROW_CACHE.clear()
    NEXT_STATES_CACHE.clear()
       
    # Transpose if necessary
    if curr_state.shape[0] < curr_state.shape[1]:
        curr_state = curr_state.T
    # Number of columns
    n = curr_state.shape[1]
    m = curr_state.shape[0]
    

    # First layer
    first_layer_lower_rows = []
    first_layer_lower_rows_hashes = set()
    all_possible_2grids = get_all_possible_2xn_plus_1_grids(n)
    
    # Filter out grids that do not produce the first row
    filter_func = get_filter_to_check_if_grid_produces_row(curr_state[0,:])
    for grid in all_possible_2grids:
        key = grid.tobytes()
        #if key not in first_layer_lower_rows_hashes and filter_func(grid):
        if filter_func(grid):
            first_layer_lower_rows_hashes.add(key)
            first_layer_lower_rows.append(grid[1,:])

    # For the other layers, we need to go through all previous layers grids (gp), generate
    # all possible full grids that have the same upper row as gp lower row, and filter out
    # grids that do not produce ith row of curr_state
    prev_layer_lower_rows = first_layer_lower_rows
    
    for i in range(1, m):
        curr_layer_lower_rows = []
        curr_layer_lower_rows_hashes = set()
        for lr in prev_layer_lower_rows:
            
            # Generate all possible full grids that have the same upper row as gp lower row
            all_grids = all_full_grids_from_upper_row(lr)
            # Filter out grids that do not produce ith row of curr_state
            filter_func = get_filter_to_check_if_grid_produces_row(curr_state[i,:])
            for grid in all_grids:
                key = grid.tobytes()
                #if key not in curr_layer_lower_rows_hashes and filter_func(grid):
                if filter_func(grid):
                    curr_layer_lower_rows_hashes.add(key)
                    curr_layer_lower_rows.append(grid[1,:])
        
        prev_layer_lower_rows = curr_layer_lower_rows
    return len(prev_layer_lower_rows)
        



if __name__ == '__main__':
    #curr_state = [[0,1,0,1],[0,0,1,0],[0,0,0,1],[1,0,0,0]]
    curr_state = [[True, False, False], [False, True, False], [True, False, True]]
    #curr_state = [[0,1,0],[1,1,1],[1,1,0]]
    #curr_state = [[1,1,1],[1,1,1],[1,1,1]]
    curr_state = np.diag(np.ones(4, dtype=np.int32))#1340
    curr_state = [[True, True, False, True, False, True, False, True, True, False], [True, True, False, False, False, False, True, True, True, False], [True, True, False, False, False, False, False, False, False, True], [False, True, False, False, False, False, True, True, False, False]]
    #curr_state = [[True, False, True, False, False, True, True, True], [True, False, True, False, False, False, True, False], [True, True, True, False, False, False, True, False], [True, False, True, False, False, False, True, False], [True, False, True, False, False, True, True, True]]
    
    times_with_recursive = []
    times_with_grid_images = []
    
    max_nrows = 8
    max_ncols = 8
    # Create a random grid
    for _ in range(50):
        random_grid_size = np.random.randint(1, max_nrows+1), np.random.randint(1, max_ncols+1)
        curr_state = np.random.randint(0,2, size=random_grid_size)
        curr_state = np.array(curr_state, dtype=np.int16)
        
        print("Current state: \n", curr_state)
        start = time.time()
        correct_solution = solution_other(curr_state)
        time_taken = time.time() - start
        print("Correct solution: ", correct_solution)
        print("Time taken: ", time_taken)
        times_with_recursive.append(time_taken)
        
        print()
        
        start = time.time()
        sol = solution(curr_state)
        time_taken = time.time() - start
        print("Solution: ", sol)
        print("Time taken: ", time_taken)
        times_with_grid_images.append(time.time() - start)
        
        if sol != correct_solution:
            print("Wrong solution")
            break
    
    print("Average time taken with recursive: ", np.mean(times_with_recursive))
    print("Average time taken with grid images: ", np.mean(times_with_grid_images))