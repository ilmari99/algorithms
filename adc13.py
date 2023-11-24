from io import TextIOWrapper
from typing import List
import ast
import itertools as it

"""
Packet data consists of lists and integers. Each list starts with [, ends with ], and contains zero or more comma-separated values (either integers or other lists). Each packet is always a list and appears on its own line.

When comparing two values, the first value is called left and the second value is called right. Then:

If both values are integers, the lower integer should come first. If the left integer
is lower than the right integer, the inputs are in the right order. If the left integer
is higher than the right integer, the inputs are not in the right order. Otherwise, the
inputs are the same integer; continue checking the next part of the input.

If both values are lists, compare the first value of each list, then the second value,
and so on. If the left list runs out of items first, the inputs are in the right order.
If the right list runs out of items first, the inputs are not in the right order. If the
lists are the same length and no comparison makes a decision about the order, continue
checking the next part of the input.

If exactly one value is an integer, convert the integer to a list which contains that
integer as its only value, then retry the comparison. For example, if comparing [0,0,0]
and 2, convert the right value to [2] (a list containing 2); the result is then found
by instead comparing [0,0,0] and [2].

What are the indices of the pairs that are already in the right order?
(The first pair has index 1, the second pair has index 2, and so on.)
In the above example, the pairs in the right order are 1, 2, 4, and 6; the sum of these indices is 13.

Determine which pairs of packets are already in the right order. What is the sum of the indices of those pairs?
"""

def _get_next_line(fh : TextIOWrapper):
    """ Get the next line from the file, and return it as a list"""
    line = ""
    while not line:
        line = fh.readline()
        # line contains line changes etc
        # If there is absolutely nothing,
        # then the file is read and we break
        if not line:
            break
        # Remove \n and check if there is a list on the line
        line = line.strip(" \n")
        # if not, continue, because not still end of file
        if not line:
            continue
        # Now we know we have a correct line which we return
        line = ast.literal_eval(line)
        #print(line)
        return line

def get_next_packs(file):
    """ Get the next two packets from the file, and return them as a list of lists"""
    with open(file,"r") as f:
        while f:
            packs = [_get_next_line(f), _get_next_line(f)]
            if None in packs:
                return
            yield packs
    return


def cmp_packs(lpack : List, rpack : List,ind=0):
    """ Compare two packets, and return whether they are in the right order
    """

    # if both ints, return whether l is smaller than r, or None if equal
    if  isinstance(lpack,int) and isinstance(rpack,int):
        return lpack < rpack if lpack != rpack else None
    
    # Convert ints to lists
    if not isinstance(lpack, list):
        lpack = [lpack]
    if not isinstance(rpack, list):
        rpack = [rpack]
    
    val = -1
    li = 0
    ri = 0
    
    # Loop through the lists
    for i in range(max(len(lpack),len(rpack))):
        try:
            l_el = lpack[i]
            r_el = rpack[i]
        # If one list is shorter than the other, return whether the left is shorter or None if they're equal
        # Throwing an exception is expensive, so this should be changed
        except IndexError:
            li = len(lpack)
            ri = len(rpack)
            return li < ri if li != ri else None
        # Recursively call this function to compare the elements
        val = cmp_packs(l_el,r_el,ind)
        if val is None:
            continue
        return val

def flatten(x):
    ''' Creates a generator object that loops through a nested list '''
    # First see if the input is not a list
    if not isinstance(x,list):
        yield x
    # If it is iterable, loop through the list recursively
    elif x:
        for i in x:
            for j in flatten(i):
                yield j
    else:
        yield -1

def _sort_key(val : List):
    """ Sort key for the sorted function """
    flat = list(flatten(val))
    if not flat:
        return -1
    key = flat[0] + 0.01 * len(flat)
    return key
    
if __name__ == "__main__":
    pack_gen = get_next_packs("actual-input.txt")
    pack_gen = list(pack_gen)
    count = 0
    indices = []
    for i,el in enumerate(pack_gen):
        res = cmp_packs(el[0],el[1],i)
        if res:
            count += 1
            indices.append(i+1)
            print(f"{el} is {res} at index {i+1}")
    print(f"Found indices: {indices}")
    print(f"Sum of indices: {sum(indices)}")
    pack_gen = list(it.chain(*pack_gen))
    pack_gen.append([[2]])
    pack_gen.append([[6]])
    sorted_pack = pack_gen
    sorted_pack = sorted(pack_gen,key=_sort_key)
    ind_of_2 = 0
    ind_of_6 = 0
    for i,r in enumerate(sorted_pack):
        print(r)
        if r == [[2]]:
            ind_of_2 = i + 1
            print("Found 2")
        elif r == [[6]]:
            ind_of_6 = i + 1
            print("Found 6")
    print(f"Product of 2 and 6 indices: {ind_of_2*ind_of_6}")