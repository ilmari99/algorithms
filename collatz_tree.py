"""
This file can be used to create a tree with all possible previous Collatz iterations up to a desired depth.
HENCE only if starting from 1, the levels correspond to all numbers with the level number of collatz steps.

TODO: create a graphical representation of the tree with matplotlib
"""

import itertools as it
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from collections import OrderedDict
import sympy as sp
from tqdm import tqdm
from math import gcd

class ColN:
    """An object of this class represents a node in the tree
    """
    number = None
    children = [None,None]
    iters = 0
    parent = None
    def __init__(self,n):
        self.number = n
    
    def set_children(self,children:list):
        self.children = children
    
    def set_iters(self,iters):
        self.iters = iters
    
    def set_parent(self,p):
        self.parent = p

def prev_step(n):
    """Returns a list of length 1 or 2 with the possible previous collatz numbers of n.
    the previous collatz numbers are always atleast [B*n], but can also be [2*n, (n-R*n % B)/R]
    IF (n-1) % 3 == 0 AND ((n-1)/3) % 2 != 0
    """
    vals = [B*n,0]
    # Rem tells us how much we have to add to n (R*n) to get a number divisible by B
    rem = R - B
    lower_val = (n - rem) / R
    pos = True
    #print(f"n: {n}, rem: {rem}, lower_val: {lower_val}")
    if n<0:
        pos = False
    # Check if the lower_val is compatible
    if (lower_val % 2 != 0) and ((n - 1) % 3 == 0): #If the lower_val is divisible by 2, then it can't be the previous step
        vals[1] = int(lower_val)
    # The previous number can not fall below 1 for positive numbers or above -1 for negative
    if (pos and vals[1] <= 1) or (not pos and vals[1]>=-1):
        vals.pop(-1)
    return vals

def collatz_step(n):
    """Returns the next number in the collatz sequence.
    """
    if n % B == 0:
        return int(n/B)
    else:
        return int(R*n + R - B)

def create_tree(start=1,max_level=10):
    """Creates a tree with all possible previous Collatz iterations from start up to a desired depth.
    
    Args:
        start (int, optional): From which number to start. Defaults to 1.
        max_level (int, optional):  Max depth. More than 40 levels are discouraged,
                                    because the amount of numbers in a level often grows exponentially.
                                    Defaults to 10.

    Returns:
        ColN: ColN instance, head of the collatz tree
    """
    head = ColN(start) # Create the first node
    level = head.iters
    vals = [None,None]
    nodes = [head]
    level_width = -1
    total_nodes = 0
    while nodes:
        node = nodes.pop(0)
        val = node.number
        vals = prev_step(val)
        children = [ColN(v) for v in vals]  # Create new nodes, children of the current node and set their values
        # Set the parent and iters for children of node
        for c in children:
            c.set_parent(node)
            c.set_iters(node.iters + 1)
        node.set_children(children)         # Set the children of the current node
        level_width += 1
        if level != node.iters:             # Create a new level
            print(f"Level {level} with {level_width} elements.")
            level = node.iters
            total_nodes += level_width
            level_width = 0
        if level < max_level:               # Add children to the nodes if we still want to count their children 
            nodes = nodes + children
    print("Total number of nodes:",total_nodes)
    return head


def print_smallest(head):
    """Print the smallest numbers in each level (corresponding to collatz steps) 
    """
    i = 0
    nodes = [head]
    start = head.number
    while None not in nodes:
        print(f"Smallest (abs) number with {i} iterations from {start} is {min([abs(n.number) for n in nodes])}")
        nodes = [n.children for n in nodes]
        nodes = list(it.chain(*nodes))
        i = i + 1
        
def print_level(head):
    """Print every level of the tree, aka every number with n collatz steps
    """
    nodes = [head]
    while None not in nodes:
        print(nodes[0].iters,[n.number for n in nodes])
        nodes = [n.children for n in nodes]
        nodes = list(it.chain(*nodes))

def print_levels_pairs(head):
    """ Print every level of the tree, aka every number with n collatz steps
        But with formatting, that makes it clearer, which children belong to which parent
    """
    nodes = [[head]]
    while nodes:
        for n in nodes:
            if None in n:
                break
            for i,nn in enumerate(n):
                out = str(nn.number)+", "
                if i == len(n)-1:
                    out = str(nn.number)
                print(out,end=" ")
            print("||",end="")
        print("\n")
        nodes = list(it.chain(*nodes))
        try:
            nodes = [n.children for n in nodes]
        except AttributeError:
            break

def get_levels(head):
    """Return all ColN objects of levels in a nested list
    """
    nodes = [head]
    levels = []
    while None not in nodes:
        levels.append(nodes)
        nodes = [n.children for n in nodes]
        nodes = list(it.chain(*nodes))
    return levels

def print_tree(head,format_mode="smallest"):
    """Print the values in the tree with the specified format mode
    """
    if format_mode == "smallest":
        print_smallest(head)
    elif format_mode == "levels":
        print_level(head)
    elif format_mode == "pairs":
        print_levels_pairs(head)
    else:
        print_smallest(head)
        
def get_widths(head):
    """Returns width of each level in the tree.
    ;The amount of numbers that have the same amount of Collatz steps
    """
    widths = []
    nodes = [head]
    while None not in nodes:
        widths.append(len(nodes))
        nodes = list(it.chain(*[n.children for n in nodes])) # Make 1-D list
    return widths

def flatten(head):
    """Returns the entire tree flattened in to a list
    """
    nodes = [head]
    prev_nodes = [head]
    i = 0
    while 1:
        new_nodes = list(it.chain(*[n.children for n in prev_nodes]))
        if None in new_nodes:
            break
        prev_nodes = new_nodes
        nodes = nodes + new_nodes
    return nodes

def plot_as_tree(head,show=True):
    """ 
    Plot the created tree, so that the root is one and the numbers in the next level are connected to it,
    and so on.
    """
    fig, ax = plt.subplots()
    nodes = flatten(head)
    for n in nodes:
        if n.parent:
            ax.plot([n.parent.iters,n.iters],[n.parent.number,n.number])
    ax.set_xlabel("Number of iterations to reach 1")
    ax.set_ylabel("Number (log scale)")
    ax.grid(True)
    # Logarithmic scale
    ax.set_yscale('log')
    ax.set_title("Collatz tree in log10")
    if show:
        plt.show()
    return

def plot_level_widths(head, show = True):
    """Plots the amount of numbers (y) versus the corresponding iterations
    """
    widths = get_widths(head)
    plt.figure()
    plt.title("Amount of numbers with the same amount of iterations")
    xs = list(range(0,len(widths)))
    plt.plot(list(range(0,len(widths))),widths)
    plt.xlabel(f"Iterations from {head.number}")
    plt.ylabel("Numbers in a level")
    if show:
        plt.show()
    return
    
def plot_numbers_reached(head,start=0,end=-1,show=True):
    """Plots all numbers (y) reached with (x) iterations from start.
    """
    levels = get_levels(head)[start:end+1]
    max_iters = levels[-1][0].iters
    min_iters = levels[0][0].iters
    x = range(min_iters,max_iters)
    plt.figure()
    plt.title("Values of numbers with the amount of iterations.")
    for xs,ys in zip(x,levels):
        plt.plot([xs]*len(ys), [y.number for y in ys])
    plt.xlabel(f"Iterations from {head.number}")
    plt.ylabel("Numbers reached")
    if show:
        plt.show()
    return
    

def plot_function():
    """
    Plot the function, so that we plot the function of the value, wrt. m + n
    Only consider positive integer values of m and n, and only if the function is an integer too
    """
    func = lambda m,n :(2**n - 1)/(3**m) + 1
    m = range(0,100)
    n = range(0,100)
    x = []
    y = []
    for i in m:
        for j in n:
            value = func(i,j)
            if value.is_integer() and value > 1:
                x.append(i + j)
                y.append(func(i,j))
    plt.figure()
    plt.title("Function of the value, wrt. m + n")
    plt.scatter(x,y)
    plt.xlabel("m + n")
    plt.ylabel("Value")
    plt.yscale("log",base=2)
    plt.show()
    return
    



if __name__ == "__main__":
    # Create the collatz tree with a root of 1 up to 30 reverse collatz steps
    head = create_tree(start=1,max_level=40)
    # Display the smallest number in each level
    print_tree(head,format_mode="smallest")
    # Fit a function to the amount of numbers found with the same amount of iterations
    widths = get_widths(head)
    xs = list(range(0,len(widths)))
    coefs = optimize.curve_fit(lambda x,a,b: a*2**(b*x), xs,widths)[0]
    fun = lambda x : coefs[0]*2**(coefs[1]*x)
    print(f"Best fit function for approximating the amount of numbers found with the same number of iterations {coefs[0]}*2^({coefs[1]}*x)")
    cycle_ends = find_repeating_patterns(n=100, to_positives=True)
    print(f"Found possible cycle ends: {cycle_ends}")
    print(f"Validating solutions found from empirical tests")
    for sol in cycle_ends:
        # Check that the solution is valid, and then run the collatz sequence on it
        if check_cycle_solution_vector(sol):
            random_values_from_cycle = np.random.choice(sol)
            iters, cycle = run_collatz_sequence(random_values_from_cycle)
            print(f"Time to reach cycle: {iters} with cycle: {cycle} and start value: {random_values_from_cycle}")
    print(f"\nFinding solutions for the cycle equations")
    valid_sols = []
    for nups in tqdm(range(1,7)):
        for ndowns in range(1,7):
            print(f"Trying nups={nups} and ndowns={ndowns}")
            sols = find_cycle_solutions(nups, ndowns)
            if check_cycle_solution_vector(sols):
                valid_sols.append(sols)
                print(f"Found valid solution with nups={nups} and ndowns={ndowns}: {sols}")
    #print(f"Found solutions for the cycle equations: {sols}")
    #print(f"Valid solutions: {valid_sols}")
    print(f"Validating solutions found from Cycle Equations")
    for sol in valid_sols:
        # Check that the solution is valid, and then run the collatz sequence on it
        if check_cycle_solution_vector(sol):
            for val in sol:
                iters, cycle = run_collatz_sequence(val)
                if cycle:
                    print(f"Time to reach cycle: {iters} with cycle: {cycle} and start value: {val}")
            
    
    exit()
    # Plot all numbers reached with the amount of iterations
    plot_numbers_reached(head,start=1,end=40,show=False)
    # Plot the amount of numbers found with the same amount of iterations
    plot_level_widths(head,show=False)
    # Plot the tree
    plot_as_tree(head,show=True)
    plt.show()

    
