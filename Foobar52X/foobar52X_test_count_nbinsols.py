import itertools
import numpy as np
import time
import pulp
import functools
import sympy

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
)

from foobar52X import (
    get_previous_states,
)

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations, 
    implicit_multiplication,
)



def nested_tuple_to_array(nested_tuple):
    """ Convert a nested tuple to a numpy array.
    """
    return np.array(nested_tuple, dtype=np.int32)


def find_a_binary_solution(A,b):
    """ Find a binary solution to the system Ax = b
    We use Pulp, binary integer programming.
    """
    # Create the problem
    prob = pulp.LpProblem("Binary solution problem", pulp.LpMinimize)
    # x is a binary vector
    x = pulp.LpVariable.dicts("x", range(A.shape[1]), cat="Binary")
    # The objective is to solve the system A x = b
    for i in range(A.shape[0]):
        prob += pulp.lpSum([A[i,j]*x[j] for j in range(A.shape[1])]) == b[i]
    
    # Solve the problem, no verbosity
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status == 1:
        # Get the solution
        solution = np.array([x[i].value() for i in range(A.shape[1])])
        return solution
    return None

        
def count_n_binary_solutions_to_system_vector(A, b):
    """ Count the number of binary solutions to the system Ax = b.
    """
    m = A.shape[0]
    n = A.shape[1]
    print(f"Number of variables: {n}")
    print(f"Number of equations: {m}")
    print(f"A = \n{A}")
    print(f"b = {b}")
    b = b.astype(np.int32)
    A = A.astype(np.int32)
    assert len(b) == m, f"b must have the same number of elements as A has variables. len(b) = {len(b)}, m = {m}"
    
    # We can drop all columns of A, where any element is larger than the corresponding element in b
    delete_cols = []
    for i in range(n):
        if (A[:,i] > b).any():
            delete_cols.append(i)
    if delete_cols:
        A = np.delete(A, delete_cols, axis=1)
        print(f"Removed {len(delete_cols)} columns from A")
    n = A.shape[1]
    
    terms = []
    combinations_temp = []
    for i in range(0,n):
        # Go through all length i combinations, so if i==2 and m = 4 -> (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        for comb in itertools.combinations(range(n), i+1):
            combinations_temp.append(comb)
            # The exponents that are added, are the columns of A, that are in comb
            exponents_to_add = A[:,comb]
            # Sum the exponents by column
            term = exponents_to_add.sum(axis=1)
            assert len(term) == m, f"len(term) = {len(term)}, m = {m}"
            #print(f"term = {term}")
            terms.append(term)
    
    terms = np.array(terms, dtype=np.int32)
    #print(f"terms = \n{terms}")
    print(f"terms.shape = {terms.shape}")
    # Find all terms/vectors that are equal to b
    b_terms = np.where((terms == b).all(axis=1))[0]
    print(f"b_terms = {b_terms}")
    print(f"Valid combinations: {[combinations_temp[i] for i in b_terms]}")
    
    #print(f"b_terms = {b_terms}")
    b_terms = terms[b_terms].astype(np.int32)
    if not b_terms.size:
        # Find the closest term to b
        # Find the distance to b for each term
        distances = np.abs(terms - b).sum(axis=1)
        #print(f"distances = {distances}")
        # Find the index of the term with the smallest distance
        min_dist_index = distances.argmin()
        #print(f"min_dist_index = {min_dist_index}")
        # Get the term with the smallest distance
        closest_term = terms[min_dist_index]
        print(f"closest_term = {closest_term}")
    
    return len(b_terms)


def count_n_binary_solutions_to_system_symbolic(A, b):
    #m is the number of equations
    m = A.shape[0]
    # n is the number of variables
    n = A.shape[1]
    print(f"Number of variables: {n}")
    print(f"Number of equations: {m}")
    print(f"b = {b}")
    
    # b must have the same number of elements as A has variables
    assert len(b) == m, f"b must have the same number of elements as A has variables. len(b) = {len(b)}, m = {m}"

    # Create variables Z = [z_1,...,z_m]
    Z = [f"z{i}" for i in range(m)]
    print(f"Z = {Z}")
    
    # For each row j in A, calculate prod z_j^{A_{j,i}}
    
    parts = []
    for i in range(n):
        part = []
        for j in range(m):
            aji = A[j,i]
            if aji == 0:
                continue
            zj = Z[j]
            p = f"{zj}**{aji}" if aji != 1 else zj
            part.append(p)
        parts.append("(1 + " + "*".join(part) + ")")
    #print(f"parts = {parts}")
    g = "*".join(parts)
    
    #print(g)
    expr = parse_expr(g, transformations=(standard_transformations + (implicit_multiplication,)))
    expr = sympy.simplify(expr)
    #print(f"g in sympy = {expr}")
    
    # expand
    g = sympy.expand(expr, mul=True, deep=False, power_exp=False, power_base=False, basic=False, multinomial=False)
    #print(f"g = {g}")
    
    # Find the coefficient of term z^{b1}_1 ... z^{bm}_m
    target_term = "*".join([f"z{i}**{int(b[i])}" for i in range(m)])
    print(f"target_term = {target_term}")
    target_term = sympy.simplify(target_term)
    print(f"target_term = {target_term}")
    
    # Get the coefficient of the term z^{b1}_1 ... z^{bm}_m
    coeff = g.coeff(target_term)
    print(f"coeff = {coeff}")
    
    # Get the number of binary solutions
    n_solutions = coeff.subs([(z,1) for z in Z])
    return n_solutions



def count_n_binary_solutions_to_system_knapsack_brute(A, b):
    """ Count the number of different subsets of the columns of A
    that sum to b.
    """
    n, m = A.shape
    count = 0

    for subset_mask in range(1 << m):  # Iterate through all subsets
        subset_sum = np.zeros((n, 1))
        #print(f"subset_mask has columns:", end=" ")
        for col in range(m):
            if (subset_mask >> col) & 1:  # Check if the column is selected
                subset_sum += A[:, col:col+1]
        if np.array_equal(subset_sum.flatten(), b):
            count += 1

    return count

def array_to_nested_tuple(arr):
    """ Convert a numpy array to a nested tuple.
    """
    return tuple([tuple(row) for row in arr])

RM_COLS_CACHE = {}
def remove_cols_from_A(A, b):
    """ Remove all columns of A, where any element is larger than the corresponding element in b
    """
    if str(A.flatten()) + str(b) in RM_COLS_CACHE:
        return RM_COLS_CACHE[str(A.flatten()) + str(b)]
    
    delete_cols = []
    for i in range(A.shape[1]):
        if (A[:,i] > b).any():
            delete_cols.append(i)
    if delete_cols:
        Amod = np.delete(A, delete_cols, axis=1)
        RM_COLS_CACHE[str(A.flatten()) + str(b)] = Amod
        A = Amod
    return A
        
    

def count_n_binary_solutions_to_system_knapsack(A, b, use_cache = False, cache={}):
    """Count the number of different subsets of the columns of A
    that sum to b, without using the column twice.
    We find the number of solutions recursively.
    """
    
    A = remove_cols_from_A(A, b)
    count = 0

    # If b is all zeros, then we have found a solution
    if not b.any():
        #if use_cache:
        #    cache[(tuple(A.flatten()), tuple(b))] = 1
        return 1


    # If we have a negative element in b, or A is empty, then there are no solutions
    if (b < 0).any() or A.size == 0:
        #if use_cache:
        #    cache[(tuple(A.flatten()), tuple(b))] = 0
        return 0
    
    
    if use_cache and (tuple(A.flatten()), tuple(b)) in cache:
        #print(f"Cache hit")
        return cache[(tuple(A.flatten()), tuple(b))]

    # Count the number of solutions where we use the first column of A
    count += count_n_binary_solutions_to_system_knapsack(A[:, 1:], b - A[:, 0], use_cache=use_cache, cache=cache)

    # Count the number of solutions where we do not use the first column of A
    count += count_n_binary_solutions_to_system_knapsack(A[:, 1:], b, use_cache=use_cache, cache=cache)

    if use_cache:
        cache[(tuple(A.flatten()), tuple(b))] = count

    return count

def count_ways_to_sum(a, b):
    
    n = len(a)

    # Create a 2D array to store the number of ways to achieve each sum for each index
    dp = np.zeros((n + 1, b + 1), dtype=np.int32)

    # There is one way to achieve the sum of 0 for any index (by not selecting any element)
    for i in range(n + 1):
        dp[i][0] = 1

    # Fill the dp array using a bottom-up approach
    for i in range(1, n + 1):
        for j in range(1, b + 1):
            # If the current element is greater than the current sum, exclude it
            if a[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                # Include the current element or exclude it
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - a[i - 1]]

    # The result is stored in the bottom-right cell of the dp array
    return dp[n][b]

def count_n_binary_solutions_to_system_knapsack_dp(A, b):
    """ Count the number of different subsets of the columns of A.
    Solve the problem using dynamic programming.
    """
    # The DP table will be a multidimensional array
    # where the first dimension (row) is the index of the column
    # And the second dimension is the index of the row in A
    
    # The third dimension is the sum (from 0 to max(b))
    # So the size of the DP table is (n, m, max(b)+1)
    
    # And where the element dp[i][j][k] is the number of ways to sum
    # A[:i,j] to k
    
    # The number of ways to sum A[:i,j] to k is the number of ways to sum
    # A[:i-1,j] to k + the number of ways to sum A[:i-1,j] to k - A[i,j]
    # Hence the recurrence relation:
    # dp[i][j][k] = dp[i-1][j][k] + dp[i-1][j][k-A[i,j]]
    A = A.astype(np.int32)
    
    n, m = A.shape
    max_sum = max(b).astype(np.int32)
    
    # Create the DP table
    dp = np.zeros((n, m, max_sum+1), dtype=np.int32)
    
    # The number of ways to sum to 0 is 
    dp[:,:,0] = 1
    
    # For values of k, we only need to consider the sums up to the value in b at index j
    # We don't necessarily need to consider up to max_sum
    for row_idx in range(0, n):
        row = A[row_idx]
        # For each column, we find the number of ways to sum to k
        for col_idx in range(0, m):
            row_part = row[:col_idx+1]
            #print(f"row_part = {row_part}")
            for k in range(1, max_sum + 1):#b[row_idx].astype(np.int32) + 1):
                # If k is 0, then there is only one way to sum to 0
                if k == 0 and len(row_part) > 1:
                    # TODO: We can sum to 0 in many ways, if there are zeros in the row
                    print(f"Number of ways to sum {row_part} to {k} = {dp[row_idx, col_idx, k]}")
                    continue
                
                # If there is only one element in the row, then there is either one or zero ways to sum to k
                if len(row_part) == 1:
                    # If there is only one element in the row, then there is only one way to sum to k
                    dp[row_idx, col_idx, k] = 1 if k == row_part[0] else 0
                    print(f"Number of ways to sum {row_part} to {k} = {dp[row_idx, col_idx, k]}")
                    continue
                    
                # The number of ways to sum 'row_part' to k
                # Is the number of ways to sum 'row_part[:-1]' to k
                # + the number of ways to sum 'row_part[:-1]' to k - row_part[-1]
                
                num_ways_to_k_prev = dp[row_idx, col_idx-1, k]
                last_element = row_part[-1]
                if last_element != 0:
                    num_ways_to_k_minus_row_part = dp[row_idx, col_idx-1, k-row_part[-1]]
                else:
                    num_ways_to_k_minus_row_part = 1
                
                #if num_ways_to_k_prev == 0:
                #    num_ways_to_k_prev = count_ways_to_sum(row_part[:-1], k)
                #    dp[row_idx, col_idx-1, k] = num_ways_to_k_prev
                #if num_ways_to_k_minus_row_part == 0:
                #    num_ways_to_k_minus_row_part = count_ways_to_sum(row_part[:-1], k-row_part[-1])
                #    dp[row_idx, col_idx-1, k-row_part[-1]] = num_ways_to_k_minus_row_part
                
                dp[row_idx, col_idx, k] = num_ways_to_k_prev + num_ways_to_k_minus_row_part
                print(f"Number of ways to sum {row_part} to {k} = {dp[row_idx, col_idx, k]}")
                
    print(f"A = \n{A}")
    print(f"b = {b}\n")
    
    # Print each layer of the DP table
    for i in range(max_sum+1):
        print(f"k = {i}")
        print(dp[:,:,i])
        print()
        
    # So dp[i,j,k] is the number of ways to sum A[:i,j] to k
        
    
    

def distribution_of_ones_in_a_state(state):
    """ Return the distribution of ones in a state, i.e. apply a convolution, but no activation function.
    """
    c = get_convolution_matrix(state.shape[0], state.shape[1])
    return c @ state.flatten()


if __name__ == '__main__':
    A = np.array([[1,1,0],[0,1,1]])
    b = np.array([1,1])
    print(f"Base test A = \n{A}")
    print(f"b = {b}")
    count_sols_by_vector = count_n_binary_solutions_to_system_vector(A,b)
    print(f"Found {count_sols_by_vector} solutions by vector")
    count_sols_by_symb = count_n_binary_solutions_to_system_symbolic(A,b)
    print(f"Found {count_sols_by_symb} solutions by symbolic")
    
    count_sols_knapsack = count_n_binary_solutions_to_system_knapsack_dp(A,b)
    print(f"Found {count_sols_knapsack} solutions by knapsack")
    print()
    #exit()
    
    A = np.array([1,0,1]).reshape(1,-1)
    b = np.array([2])
    nsols = count_n_binary_solutions_to_system_knapsack_dp(A,b)
    print(f"Found {nsols} solutions")
    #exit()
    
    # Test on a case, where we know there is only one solution
    #state_t1 = np.array([[True, False, True], [False, True, False], [True, False, True]], dtype=np.int32)
    #state_t1 = np.array([[0,1,0,1],[1,0,1,0],[0,0,0,1],[1,0,0,0]], dtype=np.int32)
    #state_t1 = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.int32)
    #state_t1 = np.diag(np.ones(3, dtype=np.int32))
    
    for i in range(100):
        n = np.random.randint(2,6)
        m = np.random.randint(2,6)
        state_t1 = np.random.randint(0,2,(n,m), dtype=np.int32)
        print(f"state_t1 = \n{state_t1}")
        state_t1_distr = distribution_of_ones_in_a_state(state_t1)
        print(f"state_t1_distr = {state_t1_distr}")
        
        state_t2 = calc_next_state(state_t1)
        print(f"state_t2 = \n{state_t2}")
        state_t2_distr = distribution_of_ones_in_a_state(state_t2)
        print(f"state_t2_distr = {state_t2_distr}")
        
        # Count the number of binary solutions (so states at time T), that fill the equation
        # C * state_t1 = state_t1_distr
        
        c = get_convolution_matrix(state_t1.shape[0], state_t1.shape[1])
        nsols = count_n_binary_solutions_to_system_vector(c, state_t1_distr.flatten())
        #nsols = count_n_binary_solutions_to_system_vector(c, state_t1_distr.flatten())
        print(f"Found {nsols} solutions using knapsack")
        #nsols = count_n_binary_solutions_to_system_vector(c, state_t1_distr.flatten())
        #print(f"Found {nsols} solutions")
        #exit()
        continue
        solutions = get_previous_states(state_t2)
        print(f"Total number of solutions for state_t1 <- state_t2 = {len(solutions)}")
        
        # Find the number of each unique distribution of ones in the solutions
        unique_distr = {}
        for s in solutions:
            d = distribution_of_ones_in_a_state(s)
            d = tuple(d)
            if d in unique_distr:
                unique_distr[d] += 1
            else:
                unique_distr[d] = 1
        print(f"Unique distributions: {unique_distr}")
        # Check how many solutions there are, where the distribution is state_t1_distr
        print(f"Number of solutions with distribution state_t1_distr = {unique_distr[tuple(state_t1_distr)]}")
        if nsols != unique_distr[tuple(state_t1_distr)]:
            print(f"Size of state_t1 = {state_t1.shape}")
            print("Did not find the correct number of solutions!")
            break