"""With the LAMBCHOP doomsday device finished, 
Commander Lambda is preparing to debut on the galactic stage
but in order to make a grand entrance, Lambda needs a grand staircase!
As the Commander's personal assistant, you've been tasked with figuring
out how to build the best staircase EVER.

Lambda has given you an overview of the types of bricks available,
plus a budget. You can buy different amounts of the different types
of bricks (for example, 3 little pink bricks, or 5 blue lace bricks).
Commander Lambda wants to know how many different types of staircases
can be built with each amount of bricks, so they can pick the one with the most options.

Each type of staircase should consist of 2 or more steps.
No two steps are allowed to be at the same height - each step must
be lower than the previous one. All steps must contain at least one
brick. A step's height is classified as the total amount of bricks that
make up that step. For example, when N = 3, you have only 1 choice of 
how to build the staircase, with the first step having a height of 2 
and the second step having a height of 1.

But when N = 5, there are two ways you can build a staircase from
the given bricks. The two staircases can have heights (4, 1) or (3, 2).

Write a function called solution(n) that takes a positive integer n and 
returns the number of different staircases that can be built from exactly n bricks.
n will always be at least 3 (so you can have a staircase at all), but no more than 200,
because Commander Lambda's not made of money!

ME
==
Later realized this is called integer partitioning, and would could probably have a way cleaner solution utilizing it.
"""


import math
import sys
ns = [3,4,5,7,8,10,11,12,13,60,200,1000]
correct = [1,1,2,4,5,9,11,14,17,0,487067745,8635565795744155161505]
counted = [3,4,5] #Holds the values for which the list of sums has been counted
results = [[1],[1],[1,2]] #Has the number of different sums up to a threshold in descending order

def solution(n,threshold=1):
    '''Returns the number of ways to create a sum from positive integers
    n (int) : number
    threshold (int) : discard sums that contain a number _less_ than this threshold

    Returns the number of ways to create a sum with positive integers,
    where in every sum:
    1) There must be atleast 2 positive integers
    2) All elements in the sum must have atleast a difference of 1
    3) All elements must be >= threshold
    4) The sum of the integers must be equal to n 
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

if __name__ == "__main__":
    if len(sys.argv) == 1:
        for n,cor in zip(ns,correct):
            ns = [3,4,5,7,8,10,11,12,13,60,200,1000]
            correct = [1,1,2,4,5,9,11,14,17,0,487067745,8635565795744155161505]
            ans = solution(n)
            print("case:",n, "ans:",ans,"correct:",cor)
    else:
        n = int(sys.argv[1])
        s = solution(n)
        print("There are",s," ways to create a sum that equals",n,"where every part of the sum has atleast a difference of 1")
            
