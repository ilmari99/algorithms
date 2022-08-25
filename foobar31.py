"""
Bomb, Baby!
===========

You're so close to destroying the LAMBCHOP doomsday device you can taste it! 
But in order to do so, you need to deploy special self-replicating bombs designed 
for you by the brightest scientists on Bunny Planet. 
There are two types: Mach bombs (M) and Facula bombs (F). The bombs, 
once released into the LAMBCHOP's inner workings, will automatically 
deploy to all the strategic points you've identified and destroy them 
at the same time. 

But there's a few catches. First, the bombs self-replicate via one of two distinct processes: 
Every Mach bomb retrieves a sync unit from a Facula bomb; for every Mach bomb, a Facula bomb is created;
Every Facula bomb spontaneously creates a Mach bomb.

For example, if you had 3 Mach bombs and 2 Facula bombs, they could either 
produce 3 Mach bombs and 5 Facula bombs, or 5 Mach bombs and 2 Facula bombs. 
The replication process can be changed each cycle. 

Second, you need to ensure that you have exactly the right number of Mach and 
Facula bombs to destroy the LAMBCHOP device. Too few, and the device might survive. 
Too many, and you might overload the mass capacitors and create a singularity at the 
heart of the space station - not good! 

And finally, you were only able to smuggle one of each type of bomb - one Mach, 
one Facula - aboard the ship when you arrived, so that's all you have to start with. 
(Thus it may be impossible to deploy the bombs to destroy the LAMBCHOP, but that's 
not going to stop you from trying!) 

You need to know how many replication cycles (generations) it will take to generate 
the correct amount of bombs to destroy the LAMBCHOP. Write a function answer(M, F) 
where M and F are the number of Mach and Facula bombs needed. Return the fewest number 
of generations (as a string) that need to pass before you'll have the exact number of 
bombs necessary to destroy the LAMBCHOP, or the string "impossible" if this can't be done! 
M and F will be string representations of positive integers no larger than 10^50. For example, 
if M = "2" and F = "1", one generation would need to pass, so the answer would be "1". However, 
if M = "2" and F = "4", it would not be possible.

Test cases
==========

Inputs:
    (string) M = "2"
    (string) F = "1"
Output:
    (string) "1"

Inputs:
    (string) M = "4"
    (string) F = "7"
Output:
    (string) "4"
    
ME
===
This was quite fun, and I learned to use backtracking, and
utilizing math to speed up computations: Take multiple steps at once, because the numbers get too huge.
Also properly understanding the problem took a while.



Returns the smallest amount of iterations required to reach the pair (M,F)
starting from (1,1).
Iterating from (1,1)  forwards: in every step there are two possible next iterations: (a1,a2) -> (a1+a2, a2) and (a1,a1+a2).
If there is no amount of iterations that would yield the desired pair (M,N), returns the string 'impossible'.

USAGE:
`python3 foobar31.py <int> <int>`

"""

import itertools as it
import sys

def back_step(tup,i):
    a = tup[0]
    b = tup[1]
    if a < 1 or b < 1 or (max((a,b))%min((a,b)) == 0 and min((a,b)) != 1): #impossible
        return (-1,-1),i
    if a >= b:
        return (a-b*((a-1)//b),b),i + ((a-1)//b)    # Should've documented better, because I don't remember this well :DD
    else:
        return (a,b-a*((b-1)//a)), i + ((b-1)//a)

def solution(M,F):
    """Returns the smallest amount of iterations required to reach the pair (M,F)
    starting from (1,1).
    Iterating from (1,1)  forwards: in every step there are two possible next iterations: (a1,a2) -> (a1+a2, a2) and (a1,a1+a2).
    If there is no amount of iterations that would yield the desired pair (M,N), returns the string 'impossible'.
    For example:
    Given the pair (4,7):
    because 7>4 we know the previous step must've been (4, 3) -> then (1,3) -> (1,2) -> (1,1): Hence 4 steps to reach (1,1)
    Args:
        M (str): an integer up to 10**50
        F (str): an integer up to 10**50

    Returns:
        str: iterations
    """    
    M = int(M)
    F = int(F)
    i = 0
    target_sum = sum((M,F))
    step = (M,F)
    while step != (1,1) and step != (-1,-1):
        step,i = back_step(step,i)
    if step != (1,1):
        return "impossible"
    return str(i)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        cases = [(0,0),(1,1),(2,1),(2,1),(4,7),(5,4),(16,6),(1,6),(6,5),(11,5),(16,5),(21,5),
         (26,5),(31,5),(31,36),(67,36),(46,5),(51,5),(56,5),(707,11),(10**50,5001)]
        for c in cases:
            print("case:",c)
            ans = solution(c[0],c[1])
            print("answer:",ans)
    else:
        M = sys.argv[1]
        F = sys.argv[2]
        its = solution(M, F)
        print("The smallest number of iterations to reach pair",(M,F)," from (1,1) is", its)


            
    
