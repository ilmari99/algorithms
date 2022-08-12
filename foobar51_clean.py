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

def forgotten_matrices(w,h,s,M):
    same_row_count = sum([True if np.array_equal(M[0],M[i]) else False for i in range(1,M.shape[0])])
    unique_values_index = np.unique(M[0],return_index=True)[1]
    unique_values_count = len(unique_values_index)
    perm_count = 0
    permutated = {tuple(M[0]):True}
    if same_row_count > 0 and unique_values_count > 1:
        for p in _permutations(len(M[0]),M[0]):
            M[0] = p
            if tuple(M[0]) in permutated:
                continue
            permutated[tuple(M[0])] = True
            perm_count += len(_permutations(len(M[0]),M[0])) - 1

        #c = number_of_permutations(unique_values)*same_row_count
    return perm_count

def solution(w,h,s):
    """_summary_

    Args:
        w : width of grid (number of columns)
        h : height of grid (number of rows)
        s : number of distinct possible values in grid
    """
    trick = h
    if h>w:
        h = w
        w = trick
    combs = _combinations(w,s) # distinct rows of length w with s distinct values
    
    n_rows = number_of_combinations(w,s)    #Number of distinct rows of lenght w with s distinct values
    solutions = _combinations(h,combs) # distinct matrices of height h with rows of lenght w with s distinct values
    #print("Len of solutions: {}".format(len(solutions)))
    incr_count = 0
    for sol in solutions:
        a = np.array(sol)
        incr_count += forgotten_matrices(w,h,s,a)
    return str(int(number_of_combinations(h,n_rows) + incr_count))

if __name__ == "__main__":
    cases = {(2,2,2):"7",
             (2,3,4):"430"
             }
    for case,ans in cases.items():
        print("case: {}, ans: {}".format(case,solution(*case)))
