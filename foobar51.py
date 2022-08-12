import itertools as it
import math
import numpy as np

def _combinations(sample_len, els):
    """ Calculate the combinations (order doesn't matter) with replacement.
    Picks sample_len elements from els.
    
    els : List of elements to pick from OR number of elements to pick from (converts to range(els))
    sample_len : Number of elements to pick from els
    
    Returns:
        a tuple of tuples of the combinations of sample_len elements from els elements with replacement.
    """
    if isinstance(els,int):
        els = range(0,els)
    return tuple(it.combinations_with_replacement(els,sample_len))

def _permutations(sample_len, els):
    """ Calculate the permutations (order matters) with replacement.
    Picks sample_len elements from els.
    
    els : List of elements to pick from OR number of elements to pick from (converts to range(els))
    sample_len : Number of elements to pick from els
    
    Returns:
        a tuple of tuples of the permutations of sample_len elements from els elements with replacement.
    """
    if isinstance(els,int):
        els = range(0,els)
    return tuple(it.permutations(els,sample_len))

def number_of_permutations(sample_len):
    return math.factorial(sample_len)

def number_of_combinations(sample_len,els):
    """Return the number of combinations of sample_len elements from els elements with replacement.
    Args:
        n_el (_type_): number of elements to pick from
        sample_len (_type_): Number of elements to pick from els

    Returns:
        integer : number of combinations of sample_len elements from els elements with replacement.
    """
    if not isinstance(els,(int,float)):
        els = len(els)
    #https://www.calculatorsoup.com/calculators/discretemathematics/combinationsreplacement.php
    return math.factorial(els + sample_len - 1) / (math.factorial(sample_len)*math.factorial(els - 1))

def _non_equiv_permutation_matrices(M,compare_row=0):
    if compare_row >= M.shape[0]-1:
        yield M
        return
    M = M.copy()
    same_row_count = sum([True if np.array_equal(M[compare_row],M[i]) and i != compare_row else False for i in range(0,M.shape[0])])
    unique_values_index = np.unique(M[compare_row],return_index=True)[1]
    unique_values_count = len(unique_values_index)
    #perm_count = 0
    permutated = {}
    matrices = []
    if same_row_count > 0 and unique_values_count > 1:
        print("********************************************************")
        print("Matrix: {}".format(M))
        #print("Row: {} can be permutated to".format(M[0]))
        for p in _permutations(len(M[compare_row]),M[compare_row]):
            M[compare_row] = p
            if tuple(M[compare_row]) in permutated:
                continue
            permutated[tuple(M[compare_row])] = True
            for m in _non_equiv_permutation_matrices(M,compare_row+1):
                yield m
            #perm_count += len(_permutations(len(M[0]),M[0])) - 1
            #print("Permutation {}\n: {}".format(perm_count,M))
        #print("********************************************************")
        #c = number_of_permutations(unique_values)*same_row_count
    #return perm_count
    
def _non_equiv_permutation_matrices(M,compare_row=0):
    if compare_row >= M.shape[0]-1:
        #yield M
        return 0
    M = M.copy()
    same_row_count = sum([True if np.array_equal(M[compare_row],M[i]) and i != compare_row else False for i in range(0,M.shape[0])])
    unique_values_index = np.unique(M[compare_row],return_index=True)[1]
    unique_values_count = len(unique_values_index)
    #permutated = {tuple(M[compare_row]):True}
    permutated = {}
    M_count = 0
    if unique_values_count > 1:
        print("********************************************************")
        print("Matrix: {}".format(M))
        #print("Row: {} can be permutated to".format(M[0]))
        for p in _permutations(len(M[compare_row]),M[compare_row]):
            M[compare_row] = p
            if tuple(M[compare_row]) in permutated:
                continue
            permutated[tuple(M[compare_row])] = True
            M_count += number_of_permutations(len(M[compare_row])) - 1
    return M_count

def solution(w,h,s):
    """_summary_

    Args:
        w : width of grid (number of columns)
        h : height of grid (number of rows)
        s : number of distinct possible values in grid
    """
    combs = _combinations(w,s) # distinct rows of length w with atmost s distinct values
    print("Row combinations",combs)
    
    n_rows = number_of_combinations(w,s)    #Number of distinct rows of lenght w with s distinct values
    print("Number of rows: {}".format(n_rows))
    
    print("Matrices")
    solutions = _combinations(h,combs) # Base matrices: distinct matrices of height h with rows of lenght w with s distinct values
    #n_sols = number_of_combinations(h,n_rows)
    n_sols = 0
    print("Len of solutions: {}".format(len(solutions)))
    # All base matrices are non-equivalent
    for sol in solutions:
        #for a in _non_equiv_permutation_matrices(np.array(sol)):
        print(sol)
        for rn in range(0,h):
            n_sols += _non_equiv_permutation_matrices(np.array(sol[rn:]),rn)
            #print(a,"\n")
    return str(int(n_sols))

def ch_order(a,row_order,col_order):
    """_summary_

    Args:
        a : matrix to reorder
        row_order : order of rows (indexes)
        col_order : order of columns (indexes)
    """
    a_cp = a.copy()
    a = a_cp[row_order,:]
    a = a[:,col_order]
    return a

def loops_in_matrix(M,ret_loops=False):
    loops = []
    in_loop = {}
    m = M.flatten()
    loop_count = 0
    for i,el in enumerate(m):
        if i in in_loop:
                continue
        in_loop[i] = True
        while i != el:
            in_loop[el] = True
            el = m[el]
        loop_count += 1
    return loop_count
            
            
def create_array(w,h):
    a = np.ones((h,w),dtype=int)
    for i in range(0,h):
        for j in range(0,w):
            a[i,j] = i*w+j
    return a

def solution(w,h,s):
    b_arr = create_array(w,h)   #Base array: array of size hxw with elements 0,1,2,...,hxw-1
    row_orders = _permutations(h,h)
    col_orders = _permutations(w,w)
    sums = 0
    transformations = 0
    for row_order in row_orders:
        for col_order in col_orders:
            a = ch_order(b_arr,row_order,col_order)     # Change the base_matrix
            transformations += 1
            loops = loops_in_matrix(a)
            sums += s**loops
    return str(int(float(sums)/float(transformations)))

def test_loops_in_matrix():
    cases = {
        1:create_array(1,1),
        4:create_array(2,2),
        3:np.array([[1,0],[5,4],[3,2]]),
            }
    for m in cases:
        assert loops_in_matrix(cases[m]) == m

def test_create_array():
    cases = [
        (np.array([[0,1],[2,3]]),(2,2)),
        (np.array([[0,1],[2,3],[4,5]]),(2,3)),
    ]
    for m in cases:
        assert np.array_equal(m[0], create_array(*m[1]))

if __name__ == "__main__":
    test_loops_in_matrix()
    test_create_array()
    case = (1,2,2)
    ans = solution(*case)
    print("Answer:",ans)