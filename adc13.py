from io import TextIOWrapper
from typing import List
import ast
import itertools as it

def _get_next_line(fh : TextIOWrapper):
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
    with open(file,"r") as f:
        while f:
            packs = [_get_next_line(f), _get_next_line(f)]
            if None in packs:
                return
            yield packs
    return


def cmp_packs(lpack : List, rpack : List,ind=0):
    # if both ints, return whether l is smaller than r, or None if equal
    if  isinstance(lpack,int) and isinstance(rpack,int):
        return lpack < rpack if lpack != rpack else None
    
    if not isinstance(lpack, list):
        lpack = [lpack]
        
    if not isinstance(rpack, list):
        rpack = [rpack]
    val = -1
    li = 0
    ri = 0
    
    for i in range(max(len(lpack),len(rpack))):
        try:
            l_el = lpack[i]
            r_el = rpack[i]
        except IndexError:
            li = len(lpack)
            ri = len(rpack)
            return li < ri if li != ri else None
        
        val = cmp_packs(l_el,r_el,ind)
        #print(f"Compared {l_el} to {r_el}: {val}")
        if val is None:
            #print("val is none")
            continue
        return val

def flatten(x):
    ''' Creates a generator object that loops through a nested list '''
    # First see if the list is iterable
    if not isinstance(x,list):
        yield x
    # If it is iterable, loop through the list recursively
    elif x:
        for i in x:
            for j in flatten(i):
                yield j
    else:
        yield -1
        
def sort_key(val):
    flat = list(flatten(val))

def _sort_key(val : List):
    flat = list(flatten(val))
    if not flat:
        return -1
    key = flat[0] + 0.01 * len(flat)
    #key = sum([v / 10**(i) for i,v in enumerate(flat)])
    #s = map(str,flat)
    #s = "".join(s)
    #key = int(s)
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
    #for ind in range(0,1):
    #    s_key = lambda x : sort_key(x,ind)
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
        #if ind_of_2 != 0 and ind_of_6 != 0:
        #    break
    print(f"Product of 2 and 6 indices: {ind_of_2*ind_of_6}")