"""
Escape Pods
===========
You've blown up the LAMBCHOP doomsday device and relieved the bunnies of their work duries --
and now you need to escape from the space station as quickly and as orderly as possible! 
The bunnies have all gathered in various locations throughout the station, and need to make 
their way towards the seemingly endless amount of escape pods positioned in other parts of the station. 
You need to get the numerous bunnies through the various rooms to the escape pods. Unfortunately, 
the corridors between the rooms can only fit so many bunnies at a time. What's more, many of the 
corridors were resized to accommodate the LAMBCHOP, so they vary in how many 
bunnies can move through them at a time.

Given the starting room numbers of the groups of bunnies, 
the room numbers of the escape pods, 
and how many bunnies can fit through at a time in each direction of every corridor in between, 
figure out how many bunnies can safely make it to the escape pods at a time at peak.

Write a function solution(entrances, exits, path) that takes an array of integers 
denoting where the groups of gathered bunnies are, an array of integers 
denoting where the escape pods are located, and an array of an array of integers 
of the corridors, returning the total number of bunnies that can get through at each 
time step as an int. The entrances and exits are disjoint and thus will never overlap. 
The path element path[A][B] = C describes that the corridor going from A to B can fit C 
bunnies at each time step. There are at most 50 rooms connected by the corridors and at 
most 2000000 bunnies that will fit at a time.
For example, if you have:
entrances = [0, 1]
exits = [4, 5]
path = [
[0, 0, 4, 6, 0, 0], # Room 0: Bunnies
[0, 0, 5, 2, 0, 0], # Room 1: Bunnies
[0, 0, 0, 0, 4, 4], # Room 2: Intermediate room
[0, 0, 0, 0, 6, 6], # Room 3: Intermediate room
[0, 0, 0, 0, 0, 0], # Room 4: Escape pods
[0, 0, 0, 0, 0, 0], # Room 5: Escape pods
]
Then in each time step, the following might happen:
0 sends 4/4 bunnies to 2 and 6/6 bunnies to 3
1 sends 4/5 bunnies to 2 and 2/2 bunnies to 3
2 sends 4/4 bunnies to 4 and 4/4 bunnies to 5
3 sends 4/6 bunnies to 4 and 4/6 bunnies to 5

So, in total, 16 bunnies could make it to the escape pods at 4 and 5 at each time step.
(Note that in this example, room 3 could have sent any variation of 8 bunnies to 4 and 5, 
such as 2/6 and 6/6, but the final solution remains the same.)

Test cases
==========
Your code should pass the following test cases.
Note that it may also be run against hidden test cases not shown here.
-- Python cases --
Input:
solution.solution([0], [3], [[0, 7, 0, 0], [0, 0, 6, 0], [0, 0, 0, 8], [9, 0, 0, 0]])
Output:
6

Input:
solution.solution([0, 1], [4, 5], [[0, 0, 4, 6, 0, 0], [0, 0, 5, 2, 0, 0], [0, 0, 0, 0, 4, 4], [0, 0, 0, 0, 6, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
Output:
16

ME
==
This was a fun problem. I had a hard time understanding the problem, and my initial solution (which passed all Google test cases (5 in total))
was actually incorrect, but it passed the test cases by chance.

There are algorithms for finding the maximum flow, but I got carried away with my idea and finally got it working.
"""
import numpy as np


def create_case(vertices=5, max_edge_weight=10, nentrances = 1, nexits = 1):
    """Create a random undirected graph with nentrances entrances and nexits exits and random edge weights

    Args:
        vertices (int, optional): The number of vertices in the graph. Defaults to 5.
        max_edge_weight (int, optional): The largest possible single edge weight. Defaults to 10.
        nentrances (int, optional): Number of entrances chosen randomly from range(vertices). Defaults to 1.
        nexits (int, optional): Number of exits chosen randomly. Defaults to 1.

    Returns:
        case (tuple): A tuple of (path, exits, entrances)
    """
    weights = np.random.randint(low=0, high=max_edge_weight+1, size=(vertices, vertices))
    path = np.triu(np.reshape(weights, (vertices, vertices)))
    path = mirror_upper_triangle(path)                          # Undirected graph
    possibilities = list(range(vertices))
    entrances = [possibilities.pop(np.random.randint(len(possibilities))) for _ in range(nentrances)]
    exits = [possibilities.pop(np.random.randint(len(possibilities))) for _ in range(nexits)]
    path[np.diag_indices_from(path)] = 0    # make diagonal 0
    for vert,_ in enumerate(path):
        if vert in entrances:
            path[vert][exits] = 0
        elif vert in exits:
            path[vert][entrances] = 0
    path[exits] = 0
    return path.tolist(), exits, entrances
    


def solution(path,exits,entrances,debug=False,max_iters=None):
    global PRIORITY_ORDER
    PRIORITY_ORDER = create_priority_order(path,exits)          # Create priority order -> The order of vertices in which we prioritise putting flow to.
    # Pre-compute the sources and sinks to dictionaries for quick look-ups with large graphs
    dentrances = {ent:i for i,ent in enumerate(entrances)}
    dexits = {exit:i for i,exit in enumerate(exits)}
    capacity_matrix = np.array(path)                            # Convert to numpy array, and avoid modifying the original path
    flow_matrix = get_flow_matrix(path, entrances)              # Initialize flow matrix: 0's except for entrance rows
    max_iters = max_iters if max_iters is not None else float("inf")
    i = 0
    while i<max_iters and not in_equilibrium(flow_matrix,dexits,dentrances):
        flow_matrix = update_residual_matrix(flow_matrix,capacity_matrix, dexits, dentrances,debug=debug)   #  Update the residual matrix
        i += 1
    return get_max_flow(flow_matrix,exits,entrances)


def get_flow_matrix(path,entrances):
    """Initialize flow matrix: 0's except for entrance rows """
    path = np.array(path)
    fm = np.zeros_like(path)
    fm[entrances] = path[entrances]
    return fm

def get_max_flow(matrix,exits,entrances):
    """ Return the theoretical maximum amount of flow in the graph.
    The maximum flow is the smallest of: The possible flows to exits, the possible flows from entrances.

    Args:
        matrix (np.ndarray): The matrix
        exits (list): The list of exits
        entrances (list): The list of entrances

    Returns:
        int: The theoretical maximum flow
    """
    return min(np.sum(matrix.T[exits]),np.sum(matrix[entrances]))

def mirror_upper_triangle(matrix):
    """ Mirror the upper triangle of a square matrix along its diagonal."""
    return matrix + matrix.T - np.diag(np.diag(matrix))

def get_vertices_from(path, rooms, visited = {},depth=0):
    """ Recursively return a dictionary of rooms with the levels they are on. Exits are on level 0.
    
    score is deprecated, but just removed for now.
    """
    if not rooms:
        return visited
    next_rooms = {}
    # Take in a list of tuples, where each tuple is (room, score)
    # Rooms at 'depth' steps from nearest exit
    for room,score in rooms:
        if room not in visited:
            visited[room] = depth#score/depth if depth != 0 else float("inf")   # Add the rooms to the visited dictionary
        # Get each accessible next room (at depth+1) from the current room
        for i,c in enumerate(path[room]):
            # Check if the room isn't already visited, and if the current room is accessible from the next room
            if all([path[i][room] > 0, i not in visited, i not in rooms, i not in next_rooms]):
                sc = 1#sum([path[i][room],score])
                next_rooms[i] = sc
    visited = get_vertices_from(path,next_rooms.items(),visited,depth+1)
    return visited

def in_equilibrium(flow_matrix,dexits,dentrances):
    """Return True if the flow matrix is in equilibrium, False otherwise.
    Equilibrium is when each vertix (except exits or entrances) has the same amount of flow in and out.

    Args:
        flow_matrix (_type_): _description_
        dexits (_type_): _description_
        dentrances (_type_): _description_

    Returns:
        _type_: _description_
    """
    for nrow,out in enumerate(flow_matrix):
        if nrow in dexits or nrow in dentrances:
            continue
        if np.sum(out) != np.sum(flow_matrix.T[nrow]):
            return False
    return True 


def create_priority_order(path, exits):
    """ Create a priority order of rooms to visit. The exits have highest priority,
    the rooms connected to exits have next highest priority, and so on.
    """
    vertices = get_vertices_from(path,[(ex,float("inf")) for ex in exits],visited = {})
    # In case some vertices are unreachable from the exits, add them to the priority order
    for i in range(len(path)):
        if i not in vertices:
            vertices[i] = float("inf")
    room_dist_pairs = sorted(vertices.items(), key=lambda x:x[1])
    return tuple(room for room,_ in room_dist_pairs)


def update_residual_matrix(flow_matrix,capacity_matrix, dexits, dentrances,debug = True):
    """Do one iteration, where each vertix is updated to be in equilibrim.
    If say: A has flow 5 from entrance, 5 units of flow must be dispersed to the connected nodes of A.
    Then A is in equilibrium. The matrix might still not be in equilibrium, after one iteration, because
    for example, B could have flow 3 from entrance, and be connected to A and flow TO A, after which A is not in equilibrium.

    Args:
        flow_matrix (np.ndarray): The flow matrix to be updated
        capacity_matrix (_type_): The amount of maximum flow between each vertix
        dexits (dict): Dictionary of exits for quick look-up
        dentrances (dict): Dictionary of entrances
        debug (bool, optional): If true, prints the flow matrix everytime a row is updated. Defaults to True.

    Returns:
        np.ndarray: The updated flow matrix
    """
    for nrow,outflows in enumerate(flow_matrix):
        if debug:
            print("Flow matrix: \n{}".format(flow_matrix))
            print("Handling row {}".format(nrow))
        # Residual matrix is already initialized with the outflows from entrances. Exits do not have outflows.
        if nrow in dentrances or nrow in dexits:
            continue
        inflows = flow_matrix.T[nrow]                       # Current flows from other vertices to the current vertix
        edge_capacities = capacity_matrix[nrow]             # Maximum capacities of the current vertices outgoing edges
        available_increases = edge_capacities - outflows    # Available increases to outflows from the current vertix
        balance = np.sum(inflows) - np.sum(outflows)        # Balance of the current vertix (inflows - outflows)
        # If the inflows are equal to outflows, then the current vertix is in balance
        if balance == 0:
            continue
        # Now we know, that the current vertix is not in balance, and it is not an entrance or exit
        # Check where are available increases and increase the outflows of the current vertix
        # Here we only increase the outflows to vertices, that have an increase available (except entrances)
        for ncol in PRIORITY_ORDER:
            incr_capacity = available_increases[ncol]
            # If there is no available increase
            # If the ncol is an entrance, then we don't want to increase the flow yet, because it is wasted
            if inflows[ncol] > 0 or incr_capacity <= 0 or ncol in dentrances:
                continue
            ch = min(incr_capacity,balance)                         # Increase the outflow by as much as possible
            flow_matrix[nrow][ncol] += ch                           # Update the residual matrix
            available_increases[ncol] -= ch                         # Update the available increases
            balance -= ch                                           # Update the balance
            # If we are in balance, or we need to pushback the flow, exit from loop
            if np.sum(available_increases) == 0 or balance <= 0:
                break
        # If the vertix is in balance, we are done with the vertix
        if balance <= 0:
            continue
        # If the vertix is not in balance, we need to pushback the flow to some other vertix
        for ncol in PRIORITY_ORDER:
            max_capacity = capacity_matrix[ncol][nrow]              # Maximum capacity of the current vertix incoming edge
            outflow = outflows[ncol]                                # How much flow is currently out from ncol to nrow
            # Check is there is a connection, and sort of pushback the flow
            if max_capacity - outflow > 0 and ncol not in dentrances and available_increases[ncol] > 0:
                ch = min(max(2*(max_capacity-outflow),max_capacity),balance)    # Pushback the flow by as much as possible
                flow_matrix[nrow][ncol] += ch
                balance -= ch
            if balance <= 0:
                break
        # If the balance is still greater than 0, then there is no connection to pushback the flow except for the entrance (overflow)
        if balance > 0:
            flow_matrix[nrow][list(dentrances.keys())[0]] += balance
            balance = 0
    return flow_matrix