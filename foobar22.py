import numpy as np
import itertools as it
L = [3, 1, 4, 1, 5, 9]
def solution(L: list):
    """
    Returns the largest integer divisible by 3 that can be arranged from the input list L containing single digits.

    Args:
        L (list): list with single digit numbers

    Returns:
        int() : largest number that can be arranged from the digits in L that is divisible by 3
    """    
    if not L:
        return 0
    if sum(L) % 3 == 0:
        L = sorted(L,reverse=True)
        str_L = [str(num) for num in L]
        return int("".join(str_L))
    candidates = []
    for i,_ in enumerate(L):
        c = [L[k] for k,__ in enumerate(L) if k != i]
        candidates.append(solution(c))
    return max(candidates)

solution = solution(L)
print(solution)

