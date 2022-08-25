"""
You need to pass a message to the bunny anycodings_python prisoners,
but to avoid detection, the code anycodings_python you agreed to use is...
obscure, to say the least. The bunnies are given food on
 standard-issue prison plates that are anycodings_python
stamped with the numbers 0-9 for easier sorting, and you
need to combine sets of plates to create the numbers in the code.
The signal that a number is part of the code is that it is divisible by 3. 
You can do anycodings_python smaller numbers like 15 and 45 easily, but bigger numbers
like 144 and 414 are a little anycodings_python trickier. Write a program to help yourself
quickly create large numbers for use in the anycodings_python code, given a limited number of plates towork with.

You have L, a list containing some digits (0 to 9). Write a function answer(L) 
which finds the largest number that can be made anycodings_python from some or
all of these digits and is divisible by 3. If it is not possible
to anycodings_python make such a number, return 0 as the answer. L will contain
anywhere from 1 to 9 digits. The same digit may appear multiple
times in the list, but each element in the list may only be used once.


ME
==
This wasn't that hard after figuring out that a number is divisible by 3 if the digits sum is divisible by 3.

Returns the largest integer divisible by 3 that can be arranged from the input list L containing single digits.
USAGE: 
`python3 foobar22.py <n1> <n2> <n3> .... <nk>`

"""
import itertools as it
import sys
def solution(L):
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
    print("The largest number that is divisible by three with digits",L," is",n)

