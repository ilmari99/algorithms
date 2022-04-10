"""
Returns the largest integer divisible by 3 that can be arranged from the input list L containing single digits.
USAGE: 
`python3 foobar22.py <n1> <n2> <n3> .... <nk>`

"""

import numpy as np
import itertools as it
import sys
def solution(L: list):
    """
    Returns the largest integer divisible by 3 that can be arranged from the input list L containing single digits.

    Args:
        L (list): list with single digit numbers

    Returns:
        int() : largest number that can be arranged from the digits in L that is divisible by 3
    """    
    if not L: #No input
        return 0
    if sum(L) % 3 == 0: # If the sum of the digits is divisible by three, then the largest number is the digits sorted in descending order
        L = sorted(L,reverse=True)
        str_L = [str(num) for num in L]
        return int("".join(str_L))
    candidates = []
    for i,_ in enumerate(L):
        c = [L[k] for k,__ in enumerate(L) if k != i] # Create a new list of digits with out index i
        candidates.append(solution(c)) # Get the largest number after removing digit at index i
    return max(candidates)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        L = [3, 1, 4, 1, 5, 9]
    else:
        sys.argv.pop(0)
        L = [int(_) for _ in sys.argv]
    n = solution(L)
    print(f"The largest number that is divisible by three with digits {L} is {n}")

