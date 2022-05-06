"""
Given the starting room numbers of the groups of bunnies, 
the room numbers of the escape pods, and 
how many bunnies can fit through at a time in each direction of every corridor in between, 
figure out how many bunnies can safely make it to the escape pods at a time at peak.

Write a function solution(entrances, exits, path) that takes an array of integers
denoting where the groups of gathered bunnies are, an array of integers denoting 
where the escape pods are located, and an array of an array of integers of the corridors,
returning the total number of bunnies that can get through at each time step as an int. 
The entrances and exits are disjoint and thus will never overlap. 
The path element path[A][B] = C describes that the corridor going
from A to B can fit C bunnies at each time step.  There are at most 50 rooms 
connected by the corridors and at most 2000000 bunnies that will fit at a time.

For example, if you have:
entrances = [0, 1]
exits = [4, 5]
path = [
  [0, 0, 4, 6, 0, 0],  # Room 0: Bunnies
  [0, 0, 5, 2, 0, 0],  # Room 1: Bunnies
  [0, 0, 0, 0, 4, 4],  # Room 2: Intermediate room
  [0, 0, 0, 0, 6, 6],  # Room 3: Intermediate room
  [0, 0, 0, 0, 0, 0],  # Room 4: Escape pods
  [0, 0, 0, 0, 0, 0],  # Room 5: Escape pods
]

Then in each time step, the following might happen:
0 sends 4/4 bunnies to 2 and 6/6 bunnies to 3
1 sends 4/5 bunnies to 2 and 2/2 bunnies to 3
2 sends 4/4 bunnies to 4 and 4/4 bunnies to 5
3 sends 4/6 bunnies to 4 and 4/6 bunnies to 5

So, in total, 16 bunnies could make it to the escape pods at 4 and 5 at each time step.  (Note that in this example, room 3 could have sent any variation of 8 bunnies to 4 and 5, such as 2/6 and 6/6, but the final solution remains the same.)
"""

import numpy as np
def create_case(max_rooms = 9,max_weight = 30):
    entrs = []
    exits = []
    while not entrs and not exits:
        entrs = [np.random.randint(0,max_rooms) for i in range(np.random.randint(1,max_rooms))]
        exits = [np.random.randint(0,max_rooms) for i in range(np.random.randint(1,max_rooms))]
        comb = entrs + exits
        comb = list(set(comb))
        entrs = comb[:len(comb)//2]
        exits = comb[len(comb)//2:]
    npaths = max([max(entrs),max(exits)])+1
    path = np.random.randint(low=0,high=max_weight+1,size=(npaths,npaths))
    # Make diagonal 0
    path[np.diag_indices_from(path)] = 0
    # Make entrs and exits disconnected
    for room,cor in enumerate(path):
        if room in entrs:
            path[room][exits] = 0
        elif room in exits:
            path[room][entrs] = 0
        
    path = path.tolist()
    return entrs,exits,path

def find_upper_bound(entrs,exits,path):
    bound = 0
    for i,p in enumerate(path):
        if i in entrs:
            bound = bound + sum(p)
    return bound

def find_lower_bound(entrs,exits,path):
    # Lower bound is always zero on test cases, because there is no path from entrance straight exit
    # But in theory there could be
    bound = 0
    for i,p in enumerate(path):
        if i in entrs:
            bound = bound + sum([_ for room,_ in enumerate(p) if room in exits])
    return bound

def solution(entrs,exits,path):
    """
    Returns the number of bunnies that can make it to the escape pods at each time step,
    given the entrances, exits, and the narrowness of the corridors.
     

    Args:
        entrs (list): _description_
        exits (list): _description_
        path (list): _description_

    Returns:
        int : _description_
    """
    # Paths from entrances
    entr_paths = [p for i,p in enumerate(path) if i in entrs]
    # How many bunnies are in each room (start at 0 in each)
    room_state = [0 for _ in path]
    # Send the maximum amount of bunnies from entrances
    for i,room in enumerate(room_state):
        room_state[i] = sum([v[i] for v in entr_paths])
    # Send the maximum amount of bunnies forward ('towards exits') from each room
    # entrances are already sent, and there is no need to send from exits
    for room, state in enumerate(room_state):
        if room in exits+entrs:
            continue
        # Check all paths from room
        for target, fits in enumerate(path[room]):
            # If there are no bunnies left in the room, break
            if room_state[room] == 0:
                break
            # If the bunnies can be sent to the next room or exit
            if (target in exits) or target>room:
                if fits > room_state[room]:
                    goes = room_state[room]
                else: #room_state[room]>=fits
                    goes = fits
                room_state[room] = room_state[room] - goes
                room_state[target] = room_state[target] + goes
    # room_state is now the amount of bunnies in each room, and the amount of bunnies in each exit
    return sum([buns for i,buns in enumerate(room_state) if i in exits])

import sys

if __name__=="__main__":
    if len(sys.argv)==1:
        cases = []
        cases.append(([0], [3], [[0, 7, 0, 0], [0, 0, 6, 0], [0, 0, 0, 8], [9, 0, 0, 0]]))
        cases.append(([0,1],[4,5],[[0,0,4,6,0,0],[0,0,5,2,0,0],[0,0,0,0,4,4],[0,0,0,0,6,6],[0,0,0,0,0,0],[0,0,0,0,0,0]]))
        cases.append(create_case(max_rooms = 5,max_weight = 6))
        for case in cases:
            print("entrances:",case[0])
            print("exits:",case[1])
            print("path:")
            for p in case[2]:
                print(p)
            print("upper bound:",find_upper_bound(case[0],case[1],case[2]))
            print(solution(case[0],case[1],case[2]))
    elif len(sys.argv)==3:
        max_rooms = int(sys.argv[1])
        max_weight = int(sys.argv[2])
        entrs, exits, path = create_case(max_rooms = max_rooms,max_weight = max_weight)
        print("Created random case with at most",max_rooms,"rooms with ",max_weight,"as max weights")
        print("entrances:",entrs)
        print("exits:",exits)
        print("path:")
        for p in path:
            print(p)
        print("upper bound:",find_upper_bound(entrs,exits,path))
        print("Solution: ",solution(entrs,exits,path))
