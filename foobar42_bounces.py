
import itertools
import random
import math
import time
import matplotlib.pyplot as plt

from class_foobar42 import Shot, Room, vector_length, round_if_close, to_minimal_path

def is_bounce_seq(seq):
    """Goes through the sequence and checks if it is a bounce sequence.
    Checks if there are impossible subsequences, such that a bounce occurs twice in the same wall,
    without first having been bounced from he opposite wall back.

    Args:
        seq (tuple): bounce sequence to check

    Returns:
        bool: True if the sequence is a bounce sequence, False otherwise
    """
    for i,b in enumerate(seq):
        for r in seq[i+1:]: # TODO: maybe only check the next 3 of the sequence
            if abs(b-r) == 2:
                break
            if r == b:
                return False
    return True

def generate_bounce_tuples(bounces = 3):
    """ Generate all bounce sequences of length 'bounces'

    Args:
        bounces (int, optional): bounce sequences to create. Defaults to 3.

    Yields:
        tuple: describes the bounce sequence as bounces from 0 to 3
    """
    # Generates the kartesian product of [0,1,2,3] with 'bounces' repeats
    perms = itertools.product([0,1,2,3],repeat=bounces)
    for i,bounce_pair in enumerate(perms):
        if is_bounce_seq(bounce_pair):
            yield bounce_pair

def bounce_tuple_to_direction(bounce_tuple,my_pos,enemy_pos,bounds):
    """ Converts a bounce tuple (denoting the bounce order) to a direction vector [x,y].
    The length of the vector is also the distance to the enemy.
    
    
    """
    # Create distance arrays from my_pos to the bounds, and from enemy to the bounds
    d = to_dist_array(my_pos,bounds)
    e = to_dist_array(enemy_pos,bounds)
    xdir = []       # Holds the x changes
    ydir = []       # Holds the y changes
    # Find the first x to calculate the direction in the end
    first_x = [b for b in bounce_tuple if b in [1,3]]
    if not first_x:
        xdir = [abs(enemy_pos[0]-my_pos[0])]
        xsign = (enemy_pos[0]-my_pos[0])/abs(enemy_pos[0]-my_pos[0])
    else:
        first_x = first_x[0]
        xsign = 1 if first_x == 3 else -1
    first_y = [b for b in bounce_tuple if b in [0,2]]
    if not first_y:
        ydir = [abs(enemy_pos[1]-my_pos[1])]
        ysign = (enemy_pos[1]-my_pos[1])/abs(enemy_pos[1]-my_pos[1])
    else:
        first_y = first_y[0]
        ysign = 1 if first_y == 2 else -1
    # Keep track of the latest x and y changes, to add the corresponding distances to the direction vector in the end
    latest_x = None
    latest_y = None
    for i,bounce in enumerate(bounce_tuple):
        if bounce in [0,2]:
            if ydir:
                ydir.append(bounds[1])
            else:
                ydir.append(d[bounce])
            latest_y = bounce
            continue
        if bounce in [1,3]:
            if xdir:
                xdir.append(bounds[0])
            else:
                xdir.append(d[bounce])
            latest_x = bounce
            continue
    if latest_x is not None:
        xdir.append(e[latest_x])
    if latest_y is not None:
        ydir.append(e[latest_y])
    # Calculate the direction vector
    out = (int(xsign*sum(xdir)),int(ysign*sum(ydir)))
    return out

def to_dist_array(pos,bounds):
    """Converts a position (x,y) to a distance from bounds array (d0,d1,d2,d3)
    """
    d = []
    d.append(pos[1])
    d.append(pos[0])
    d.append(bounds[1]-pos[1])
    d.append(bounds[0] - pos[0])
    return d

def gen_up_to_n_bounces(n):
    for nn in range(n):
        gen = generate_bounce_tuples(nn)
        for t in gen:
            yield t
            
def gen_up_to_n_bounces_directions(n,my_pos,enemy_pos,bounds):
    for g in gen_up_to_n_bounces(n):
        d = bounce_tuple_to_direction(g,my_pos,enemy_pos,bounds)
        yield d
        

def solution_with_bounces(shot):
    """
    """
    bounces = math.ceil(shot.room.distance/min(shot.room.bounds)) + 2
    seq_gen = gen_up_to_n_bounces(bounces)
    directions = {}
    for seq in seq_gen:
        d = bounce_tuple_to_direction(seq,shot.room.my_pos,shot.room.enemy_pos,shot.room.bounds)
        if vector_length(d) > shot.room.distance:
            continue
        gcd = math.gcd(*d)
        d = tuple([int(comp/gcd) for comp in d])
        if d in directions:
            continue
        else:
            directions[d] = 1
        if all([i in seq for i in [0,1,2,3]]) and len(seq) == 5:
            shot.shoot(d)
            shot.plot_path(show = True)
    return len(directions.keys())
        
    
        
            
 
if __name__ == "__main__":
    case2 = {"bounds":[300,275],
             "my_pos":[150,150],
             "enemy_pos":[185,100],
             "distance":500
             }
    room = Room(bounds = [4,4],my_pos = [1,1],enemy_pos=[3,2],distance=20)
    shot = Shot(room,create_path=True)
    count = solution_with_bounces(shot)
    print("Count:",count)