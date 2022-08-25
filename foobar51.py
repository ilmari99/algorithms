"""
Disorderly Escape
=================
Oh no! You've managed to free the bunny prisoners and escape Commander Lambdas
exploding space station, but her team of elite starfighters has flanked your ship.
If you dont jump to hyperspace, and fast, youll be shot out of the sky!
Problem is, to avoid detection by galactic law enforcement, Commander Lambda planted
her space station in the middle of a quasar quantum flux field. In order to make the
jump to hyperspace, you need to know the configuration of celestial bodies in the quadrant
you plan to jump through. In order to do *that*, you need to figure out how many configurations
each quadrant could possibly have, so that you can pick the optimal quadrant through which
youll make your jump. 

There's something important to note about quasar quantum flux fields' configurations: when
drawn on a star grid, configurations are considered equivalent by grouping rather than by
order. That is, for a given set of configurations, if you exchange the position of any two
columns or any two rows some number of times, youll find that all of those configurations are
equivalent in that way - in grouping, rather than order.

Write a function answer(w, h, s) that takes 3 integers and returns the number of unique,
non-equivalent configurations that can be found on a star grid w blocks wide and h blocks
tall where each celestial body has s possible states. Equivalency is defined as above: any
two star grids with each celestial body in the same state where the actual order of the rows
and columns do not matter (and can thus be freely swapped around). Star grid standardization 
means that the width and height of the grid will always be between 1 and 12, inclusive. And while
there are a variety of celestial bodies in each grid, the number of states of those bodies is 
between 2 and 20, inclusive. The answer can be over 20 digits long, so return it as a decimal 
string.  The intermediate values can also be large, so you will likely need to use at least 64-bit integers.

For example, consider w=2, h=2, s=2. We have a 2x2 grid where each celestial body is either
in state 0 (for instance, silent) or state 1 (for instance, noisy).  We can examine which 
grids are equivalent by swapping rows and columns.
00
00
In the above configuration, all celestial bodies are "silent" - that is, they have a 
state of 0 - so any swap of row or column would keep it in the same state.
00 00 01 10
01 10 00 00
1 celestial body is emitting noise - that is, has a state of 1 - so swapping rows and columns 
can put it in any of the 4 positions.  All four of the above configurations are equivalent.
00 11
11 00
2 celestial bodies are emitting noise side-by-side.  Swapping columns leaves them unchanged, 
and swapping rows simply moves them between the top and bottom.  In both, the *groupings* are 
the same: one row with two bodies in state 0, one row with two bodies in state 1, and two columns with one of each state.
01 10
01 10
2 noisy celestial bodies adjacent vertically. This is symmetric to the side-by-side case, but 
it is different because there's no way to transpose the grid.
01 10
10 01
2 noisy celestial bodies diagonally.  Both have 2 rows and 2 columns that have one of each 
state, so they are equivalent to each other.
01 10 11 11
11 11 01 10
3 noisy celestial bodies, similar to the case where only one of four is noisy.
11
11
4 noisy celestial bodies.
There are 7 distinct, non-equivalent grids in total, so answer(2, 2, 2) would return 7.

Test cases
==========
Inputs:
    (int) w = 2
    (int) h = 2
    (int) s = 2
Output:
    (string) "7"
Inputs:
    (int) w = 2
    (int) h = 3
    (int) s = 4
Output:
    (string) "430"



ME
==
This assignment truly required a lot of work to get the correct answer.
I had to look at some explanations for the problem, and I still can't say that this would've been easy. Or that
I would completely understand some of the equations.

A great introduction to abstract algebra and group theory, symmetries and some damn cycle indices!

The nudge in the right direction came after learning about (not) Burnsides's Lemma, which states that the number of equivalency classes
(the number of orbits, number of non-equivalent matrices) equals the average number of elements in X (all matrices of size w*h) that
are fixed by g in the action group G.

After implementing it I realized it wasn't nearly efficient enough with big-o of something like O(w!h!). 
Furthermore, then I didn't know it produced correct answers outside simple cases as I had no way to test.

After the implementation of Burnsides Lemma, I was desperate and quickly lost in the lingo of group theory, symmetry groups, and
abstract algebra. I realized the problem was beyond my reach and just had to look up discussions about the problem (on Stackoverflow obviously).

The solution involved a lot of math I was very unfamiliar with, and even reading solutions led me atleast 5 research tangents deep in abstract algebra.

After many painstaking hours (or days, who knows when time flies) and gradually lowering my standards as to how much I
should understand about the solution before daring to submit it, I got the implementation correct.
I didn't understand how some of the facts came to being, but I understood enough to be very very happy about reaching a solution.

This was the last challenge in the series, and I'm proud of completing all the challenges and being thrown deep into the world of algorithms.
"""


import math
from fractions import Fraction, gcd
from collections import Counter

def solution(w,h,s):
    """
    Return the number of non-equivalent (https://en.wikipedia.org/wiki/Matrix_equivalence) 
    matrices with dimensions w x h, and s different possible values.
    
    This is the same as the number of orbits (https://en.wikipedia.org/wiki/Group_action#Orbits_and_stabilizers)
    for the action group G (all different row and column switches)
    acting on set X (all different matrices filling the restrictions).
    
    This in turn is the same as the average number of fixed points (https://en.wikipedia.org/wiki/Group_action#Fixed_points_and_stabilizer_subgroups)
    for the action group G acting on set X.
    """
    ans = 0
    row_cycle_vectors = get_cycle_vectors(h)    # Get the cycle vectors and their coefficients
    col_cycle_vectors = get_cycle_vectors(w)
    # Loop through all row and column cycles
    for rc_vec,r_coef in row_cycle_vectors.items():
        for cc_vec,c_coef in col_cycle_vectors.items():
            coeff = r_coef*c_coef
            combined_vec = combine(rc_vec,cc_vec)
            value = 1
            # Combine the cycle indices and calculate cycle index of the cartesian product of the cycles
            for _, power in combined_vec:
                value *= s ** power
            ans += coeff * value
    return str(int(ans))

def combine(a_cycles, b_cycles):
    """
    Combine the row cycles and the column cycles according to this formula:
    https://math.stackexchange.com/questions/2113657/burnsides-lemma-applied-to-grids-with-interchanging-rows-and-columns/2343500#2343500.
    len_ tells the length of the cycle
    freg_ tells the frequency of len_ subcycles in _cycles
    """
    combined = []
    for len_a, freq_a in enumerate(a_cycles):
        len_a += 1      # Add one because len_ != 0
        for len_b, freq_b in enumerate(b_cycles):
            len_b += 1
            lcm = (len_a * len_b) / gcd(len_a, len_b)   # Calculate least-common multiple
            combined.append((lcm, int(len_a * freq_a * len_b * freq_b / lcm)))
    return combined
    
def partitions(n):
    """
    Recursively yield the partitions of n as a list of the components (each list sums to n).
    """
	# base case
    if n == 0:
        yield []
        return
    # get partitions of n-1
    for p in partitions(n-1):
        yield [1] + p
        if p and (len(p) < 2 or p[1] > p[0]):
            yield [p[0] + 1] + p[1:]

def get_cycle_vectors(n):
    """
    https://en.wikipedia.org/wiki/Cycle_index#Disjoint_cycle_representation_of_permutations
    Return the unique cycles of an array of length n (conjugacy classes, which are partitions of integer n),
    and the corresponding coefficients for the cycle index.
    Returned as a dictionary with unique_cycle : coefficient -pairs.
    The cycles are encoded so that, the index(+1) denotes the length of the subcycle
    and the value at the index(+1) denotes the number of such type cycles (same length).
    Kind of like one-hot encoded vectors.
    
    Each member in the dictionary then essentially holds the dummy variable of a cycle of disctinct length (the cycle indexes term) and its coefficient.
    """
    vectors = partitions(n) # Get the distinct cycles (conjugacy classes), which are the partitions: https://en.wikipedia.org/wiki/Symmetric_group#Conjugacy_classes
    J = {}
    for v in vectors:
        base = [0 for _ in range(n)]
        c = Counter(v)
        for item, count in c.items():
            base[item-1] += count   # Modfiy base vector, leaving 0 as val
        J[tuple(base)] = coefficient_of_cycle(base)
    return J

def coefficient_of_cycle(j):
    """
    Return the coefficient of a cycle j according to a formula for counting it.
    It is the inner part of the formula for the cycle index, found here: https://en.wikipedia.org/wiki/Cycle_index#Symmetric_group_Sn
    Returns it as Fraction instance for higher accuracy.
    """
    # in j, index + 1 is the length of the subcycle and the value at index is the number of times the subcycle is repeated in the cycle.
    s = 1
    for n in range(1,len(j)+1):
        s *= math.factorial(j[n-1])*n**j[n-1]
    return Fraction(1,s)


