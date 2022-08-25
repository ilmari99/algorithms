"""
Find the Access Codes
In order to destroy Commander Lambda's LAMBCHOP doomsday device, 
you'll need access to it. But the only door leading to the LAMBCHOP chamber 
is secured with a unique lock system whose number of passcodes changes daily. 
Commander Lambda gets a report every day that includes the locks' access codes, 
but only the Commander knows how to figure out which of several lists contains the
access codes. You need to find a way to determine which list contains the access codes
once you're ready to go in.

Fortunately, now that you're Commander Lambda's personal assistant,
Lambda has confided to you that all the access codes are "lucky triples"
in order to make it easier to find them in the lists. A "lucky triple" is
a tuple (x, y, z) where x divides y and y divides z, such as (1, 2, 4).
With that information, you can figure out which list contains the number
of access codes that matches the number of locks on the door when you're ready
to go in (for example, if there's 5 passcodes, you'd need to find a list with
5 "lucky triple" access codes).

Write a function solution(l) that takes a list of positive integers
l and counts the number of "lucky triples" of (li, lj, lk) where
the list indices meet the requirement i < j < k. The length of l
is between 2 and 2000 inclusive. The elements of l are between 1 and 999999 inclusive.
The solution fits within a signed 32-bit integer. Some of the lists are
purposely generated without any access codes to throw off spies, so if no triples are found, return 0.

For example, [1, 2, 3, 4, 5, 6] has the triples: [1, 2, 4], [1, 2, 6], [1, 3, 6], making the solution 3 total.

Input:
solution.solution([1, 2, 3, 4, 5, 6])
Output:
3

Input:
solution.solution([1, 1, 1])
Output: 1

ME
===
This was the trickiest problem until now. After enough time spent trying different solutions, and
a few days break from the problem, I figured out a short, neat dynamic counter solution to find the access codes.
"""

import operator as op
from functools import reduce
ncrs = {(3,3):1}
def ncr23(n, r):
    if r>n:
        return 0
    if (n,r) in ncrs:
        return ncrs[(n,r)]
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    res = numer / denom
    ncrs[(n,r)] = res
    return res

def create_count_list(L):
    '''Creates a list with (key,number) pairs where key is the value and number is the consecutive appeareances'''
    L_count = []
    n_prev = L[0]
    n_count = 1
    for i,n in enumerate(L[1:]):
        if n == n_prev:
            n_count += 1
        else:
            L_count.append((n_prev,n_count))
            n_count = 1
            n_prev = n
    L_count.append((n_prev,n_count))
    return L_count

def solution_1(L):
    """First brute force solution"""
    count = 0
    for ist,first in enumerate(L):
        second_list = list(filter(lambda x : x % first == 0,L[ist+1:]))
        for ind,second in enumerate(second_list):
            third_list = list(filter(lambda x : x % second == 0,second_list[ind+1:]))
            count += len(third_list)
    return count

def solution_2(L):
    '''Second solution, counting consecutive appearances of numbers in the list and then iterating the hopefully shorter list.
    Only good for special cases with long sequences of the same number. The final solution is probably better even for such.'''
    L = create_count_list(L) # Creates a list with (key,number) pairs where key is the value and number is the consecutive appeareances
    count = 0
    for ist,fpair in enumerate(L):
        count += ncr23(fpair[1],3)                  #If there are more than three same elements, then they already create lucky triplets
        fmultip = fpair[1]                          #There are fmultip same consecutive numbers, and each of them can be selected
        fsmultip = ncr23(fpair[1],2)                #There are fsmultip 2 number combinations to take from the consecutive elements
        for ind,spair in enumerate(L[ist+1:]):
            if spair[0] % fpair[0] == 0:
                ind = ind + ist + 1
                scount = ncr23(spair[1],2)*fmultip + spair[1]*fsmultip
                count += scount
                smultip = spair[1]*fmultip
                for ird,tpair in enumerate(L[ind+1:]):
                    if tpair[0] % spair[0] == 0:
                        count += tpair[1]*smultip
    return int(count)

def solution(L):
    '''
    Third (only one efficient enough) solution with dynamic programming:
    Go through list, and for each element L[i], count with how many previous
    elements L[j] the current number L[i] is divisible by and store in a dictionary.
    If L[i] is divisible by L[j] add 1 to the divisibility counter of L[i] and add L[j]s counter to the current counter.'''
    #Create a dictionary for each index in the list with a counter of previously found divisible numbers
    C = {i:0 for i,_ in enumerate(L)}
    count = 0
    for i in range(0,len(L)):
        j=0
        for j in range(0, i):
            if L[i] % L[j] == 0: # if divisible add 1 to the counter of C[i] and increment count by the counter of C[j]
                C[i] += + 1
                count = count + C[j]
    return count



if __name__ == "__main__":
    import sys
    import random
    if len(sys.argv) == 1:
        print("Usage: python3 foobar33.py <low> <high> <how many random numbers to pick>")
        print("For a custom case: python3 foobar33.py --custom <int1> <int2> <int3> .......")
    elif len(sys.argv) == 4:
        L = [random.randint(int(sys.argv[1]), int(sys.argv[2])) for i in list(range(1,int(sys.argv[3])))]
        print("There are",solution(L),"lucky triplets in the list.")
    elif "--custom" in sys.argv:
        i = sys.argv.index("-c")
        case = list(map(int,sys.argv[i+1:]))
        print("There are",solution(case),"lucky triplets in the list.")