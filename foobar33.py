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

def short_brute_solution(L):
    """Initial brute force solution"""
    count = 0
    for ist,first in enumerate(L):
        second_list = list(filter(lambda x : x % first == 0,L[ist+1:]))
        for ind,second in enumerate(second_list):
            third_list = list(filter(lambda x : x % second == 0,second_list[ind+1:]))
            count += len(third_list)
    return count

def solution(L):
    '''Second solution, counting consecutive appearances of numbers in the list and then iterating the hopefully shorter list.
    Only good for special cases with long sequences of the same number'''
    L = create_count_list(L) # Creates a list with (key,number) pairs where key is the value and number is the consecutive appeareances
    count = 0
    for ist,fpair in enumerate(L):
        count += ncr23(fpair[1],3)        #If there are more than three same elements, then they already create lucky triplets
        fmultip = fpair[1]                      #There are fmultip same consecutive numbers, and each of them can be selected
        fsmultip = ncr23(fpair[1],2)              #There are fsmultip 2 number combinations to take from the consecutive elements
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
    '''Third (final) solution with dynamic programming:
    Go through list, and for each element l[i], count with how many previous
    elements l[j] the current number l[i] is divisible by and store in a dictionary.
    If l[i] is divisible by l[j] add 1 to the divisibility counter of l[i] and add l[j]s counter to the current counter.'''
    c = {}
    for i in range(0,len(L)): #Create a dictionary for each index in the list with a counter of previously found divisible numbers
        c[i] = 0
    count = 0
    for i in range(0,len(L)):
        j=0
        for j in range(0, i):
            if L[i] % L[j] == 0: # if divisible add 1 to the counter of c[i] and increment count by the counter of c[j]
                c[i] = c[i] + 1
                count = count + c[j]
    return count


def create_case_ans(start,stop,elems,create_answers=False,make_set=False):
    '''Returns a random list, (all the lucky triplets) and the count
    given the start, stop and number of elements in the case.'''
    case = [random.randint(start, stop) for i in list(range(1,elems))]
    answers = []
    def solution_b(L):
        if not create_answers:
            return solution(L)
        count = 0
        for ist,first in enumerate(L):
            second_list = list(filter(lambda x : x % first == 0,L[ist+1:]))
            for ind,second in enumerate(second_list):
                third_list = filter(lambda x : x % second == 0,second_list[ind+1:])
                third_list = list(third_list)
                count += len(third_list)
                for third in third_list:
                    answers.append((first,second,third))
        return count
    if make_set:
        case = list(set(case))
    count = solution_b(case)
    return case, answers, count

if __name__ == "__main__":
    import sys
    import random
    if len(sys.argv) == 1:
        print("Usage: python3 foobar33.py <start> <stop> <number of elements>")
        print("For a custom case: python3 foobar33.py -c <int1> <int2> <int3> .......")
    elif len(sys.argv) == 4:
        case,answ,count = create_case_ans(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
        print("There are",solution(case),"lucky triplets in the list.")
    elif "-c" in sys.argv:
        i = sys.argv.index("-c")
        case = list(map(int,sys.argv[i+1:]))
        print("There are",solution(case),"lucky triplets in the list.")