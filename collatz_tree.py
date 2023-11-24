"""
This file can be used to create a tree with all possible previous Collatz iterations up to a desired depth.
HENCE only if starting from 1, the levels correspond to all numbers with the level number of collatz steps.

TODO: create a graphical representation of the tree with matplotlib
"""

import itertools as it
import matplotlib.pyplot as plt
from scipy import optimize
R = 5
B = 2

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
    print(f"n: {n}, rem: {rem}, lower_val: {lower_val}")
    if n<0:
        pos = False
    # Check if the lower_val is compatible
    #If the lower_val is divisible by B, then it can't be the previous step
    #if ((n - 1) % 3 == 0) and (lower_val % 2 != 0):
    if ((n - rem) % R == 0) and (lower_val % B != 0):
        vals[1] = int(lower_val)
    # The previous number can not fall below 1 for positive numbers or above -1 for negative
    if (pos and vals[1] <= 1) or (not pos and vals[1]>=-1):
        vals.pop(-1)
    return vals

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
            level_width = 0
        if level < max_level:               # Add children to the nodes if we still want to count their children 
            nodes = nodes + children
    return head


def print_smallest(head):
    """Print the smallest numbers in each level (corresponding to collatz steps) 
    """
    i = 0
    nodes = [head]
    start = head.number
    while None not in nodes:
        print(f"Smallest number with {i} iterations from {start} is {min([n.number for n in nodes])}")
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
    

if __name__ == "__main__":
    # Create the collatz tree with a root of 1 up to 30 reverse collatz steps
    head = create_tree(start=1,max_level=45)
    # Display the smallest number in each level
    print_tree(head,format_mode="smallest")

    # Fit a function to the amount of numbers found with the same amount of iterations
    widths = get_widths(head)
    xs = list(range(0,len(widths)))
    coefs = optimize.curve_fit(lambda x,a,b: a*2**(b*x), xs,widths)[0]
    fun = lambda x : coefs[0]*2**(coefs[1]*x)
    print(f"Best fit function for approximating the amount of numbers found with the same number of iterations {coefs[0]}*2^({coefs[1]}*x)")

    # Plot all numbers reached with the amount of iterations
    plot_numbers_reached(head,start=1,end=40,show=False)
    # Plot the amount of numbers found with the same amount of iterations
    plot_level_widths(head,show=False)
    # Plot the tree
    plot_as_tree(head,show=True)
    plt.show()

    
