import numpy as np
import math
ns = [3,4,5,7,8,10,11,12,13,60,200,1000]
correct = [1,1,2,4,5,9,11,14,17,0,487067745,8635565795744155161505]
counted = [3,4,5] #Holds the values for which the list of sums has been counted
results = [[1],[1],[1,2]] #Has the number of different sums up to a threshold in descending order

def solution(n,threshold=1):
    '''
    n (int) : number
    threshold (int) : discard sums that contain a number _less_ than this threshold

    Returns the number of ways to create a sum with positive integers,
    where in every sum:
    1) There must be atleast 2 positive integers
    2) All elements in the sum must have atleast a difference of 1
    3) All elements must be >= threshold
    '''
    if n<3 or threshold >= n:
        return 0
    if n in counted:
        full_list = results[counted.index(n)]#[1,3,6,9]
        a = full_list[-1]
        b = 0
        if threshold > 1:
            try:
                b = full_list[threshold-2]
            except IndexError:
                return 0
        return a-b
    sol = [1,n-1] #Base solution
    count = 0
    i = 2
    counts = []
    while sol[0] < sol[1]:
        count = count + 1
        count = count + solution(sol[1],threshold=i)
        counts.append(count) #How many different 'stairs' that start from sol[0]
        sol = [sol[0]+1,sol[1]-1]
        i = i + 1
    if not counts:
        return 0
    results.append(counts)
    counted.append(n)
    a = counts[-1]
    b = 0
    if threshold > 1:
        try:
            b = counts[threshold-2]
        except IndexError:
            return 0
    return a-b

for n,cor in zip(ns,correct):
    ans = solution(n)
    print("case:",n, "ans:",ans,"correct:",cor)
