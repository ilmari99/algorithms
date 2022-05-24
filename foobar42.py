"""
Uh-oh -- you've been cornered by one of Commander Lambdas elite bunny trainers! Fortunately, you grabbed a beam weapon from an abandoned storeroom while you were running through the station, so you have a chance to fight your way out. But the beam weapon is potentially dangerous to you as well as to the bunny trainers: its beams reflect off walls, meaning you'll have to be very careful where you shoot to avoid bouncing a shot toward yourself!

Luckily, the beams can only travel a certain maximum distance before becoming too weak to cause damage. You also know that if a beam hits a corner, it will bounce back in exactly the same direction. And of course, if the beam hits either you or the bunny trainer, it will stop immediately (albeit painfully). 

Write a function solution(dimensions, your_position, trainer_position, distance) that gives an array of 2 integers of the width and height of the room, an array of 2 integers of your x and y coordinates in the room, an array of 2 integers of the trainer's x and y coordinates in the room, and returns an integer of the number of distinct directions that you can fire to hit the elite trainer, given the maximum distance that the beam can travel.

The room has integer dimensions [1 < x_dim <= 1250, 1 < y_dim <= 1250]. You and the elite trainer are both positioned on the integer lattice at different distinct positions (x, y) inside the room such that [0 < x < x_dim, 0 < y < y_dim]. Finally, the maximum distance that the beam can travel before becoming harmless will be given as an integer 1 < distance <= 10000.

For example, if you and the elite trainer were positioned in a room with dimensions [3, 2], your_position [1, 1], trainer_position [2, 1], and a maximum shot distance of 4, you could shoot in seven different directions to hit the elite trainer (given as vector bearings from your location): [1, 0], [1, 2], [1, -2], [3, 2], [3, -2], [-3, 2], and [-3, -2]. As specific examples, the shot at bearing [1, 0] is the straight line horizontal shot of distance 1, the shot at bearing [-3, -2] bounces off the left wall and then the bottom wall before hitting the elite trainer with a total shot distance of sqrt(13), and the shot at bearing [1, 2] bounces off just the top wall before hitting the elite trainer with a total shot distance of sqrt(5).
"""
import random
import math
from termios import VEOF
cases = [([3,2],[1,1],[2,1],4,7), ([300,275],[150,150],[185,100],500,9)]

def generate_case(**kwargs):
    """Generates a case, with either the default parameters or the ones given in kwargs

    Returns:
        tuple: (dimensions, your_position, trainer_position, distance)
    """
    dimensions = kwargs.get("dimensions", [random.randint(1,1250), random.randint(1,1250)])
    your_position = kwargs.get("your_position", [random.randint(1,dimensions[0]-1), random.randint(1,dimensions[1]-1)])
    trainer_position = kwargs.get("trainer_position", [random.randint(1,dimensions[0]-1), random.randint(1,dimensions[1]-1)])
    distance = kwargs.get("distance", random.randint(1,10000))
    while your_position == trainer_position:
        trainer_position = [random.randint(1,dimensions[0]-1), random.randint(1,dimensions[1]-1)]
    assert 1 < distance <= 10000
    assert 1 < dimensions[0] <= 1250
    assert 1 < dimensions[1] <= 1250
    assert 0 < your_position[0] < dimensions[0]
    assert 0 < your_position[1] < dimensions[1]
    assert 0 < trainer_position[0] < dimensions[0]
    assert 0 < trainer_position[1] < dimensions[1]
    return (dimensions, your_position, trainer_position, distance)


    
def vector_length(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

def max_step_lens(direction,laser_pos):
    """Returns the longest step in x and y directions, where the longest step in the distance to the closest
    integer in x and y directions wrt. to direction.
    Assumes that if laser_pos is at bounds, then the direction is not in the direction of the bound.
    """
    # Calculate the distance to the closest integer in the x and y directions
    xdist = math.floor(laser_pos[0]) - laser_pos[0] if direction[0]<0 else math.ceil(laser_pos[0]) - laser_pos[0]
    ydist = math.floor(laser_pos[1]) - laser_pos[1] if direction[1]<0 else math.ceil(laser_pos[1]) - laser_pos[1]
    
    # If the position is at an integer coordinate, then we can step 1 in the direction of the direction
    if math.isclose(xdist, 0):
        xdist = -1 if direction[0]<0 else 1
    if math.isclose(ydist, 0):
        ydist = -1 if direction[1]<0 else 1
    return [xdist,ydist]

def check_new_direction(direction,new_direction):
    """Checks if the new direction is valid (same signs and same ratio)
    """
    msg = False
    try:
        if direction[0]<0:
            assert new_direction[0]<0
        else:
            assert new_direction[0] >= 0
        if direction[1]<0:
            assert new_direction[1]<0
        else:
            assert new_direction[1] >= 0
    except AssertionError:
        msg = "Invalid direction (incorrect signs): New direction: {}, Old direction".format(new_direction,direction)
    try:
        assert math.isclose(direction[0]/direction[1],new_direction[0]/new_direction[1])
    except AssertionError:
        msg = "Invalid direction (incorrect ratio): New direction: {}, Old direction {}".format(new_direction,direction)
    if msg:
        raise AssertionError(msg)

def make_unidir_vector(direction, laser_pos, bounds):
    """ Converts a direction vector to a step vector, that is the longest 'safe' step in the given direction
    """
    ### TODO: Currently can skip an integer position
    
    # Step towards the closest integer in the x direction
    max_steps = max_step_lens(direction,laser_pos)
    xdist = max_steps[0]
    ydist = max_steps[1]
    print("xdist: {}, ydist: {}".format(xdist, ydist))
    # Choose the smaller of x and y, and calculate the larger by using the smaller using proportionality.
    # 
    if abs(ydist) < abs(xdist):
        print("ydist is smaller")
        print(direction[0], direction[1])
        vec2 = [(direction[0]/direction[1])*ydist,ydist]
    else:
        vec2 = [xdist,((direction[1]*xdist)/direction[0])*xdist]
    # TODO: Sign changes to incorrect, need to fix more elegantly
    check_new_direction(direction,vec2)
    if (direction[0] < 0 and vec2[0]>0) or (direction[0] > 0 and vec2[0]<0):
        vec2[0] = -1*vec2[0]
    if (direction[1] < 0 and vec2[1]>0) or (direction[1] > 0 and vec2[1]<0):
        vec2[1] = -1*vec2[1]
    print("vec2 {}".format(vec2))
    # Return the smaller of the two vectors
    #if vector_length(vec) < vector_length(vec2):
        #return vec
    return vec2
    
    
def hits_walls(laser_position,dimensions):
    """Returns a logical array, with True if the laser hits a wall
    index 0 is the bottom wall, index 1 is the left wall,
    index 2 is the top wall, index 3 is the right wall
    """
    xbounds = [0,dimensions[0]]
    ybounds = [0,dimensions[1]]
    bounds = [xbounds,ybounds]
    bounds = [0,0,dimensions[0],dimensions[1]]
    walls = [False]*len(bounds)       # 0 = bottom, 1 = left, 2 = top, 3 = right
    if laser_position[0] == bounds[0]:
        walls[1] = True
    if laser_position[0] == bounds[2]:
        walls[3] = True
    if laser_position[1] == bounds[1]:
        walls[0] = True
    if laser_position[1] == bounds[3]:
        walls[2] = True
    return walls

def cal_new_direction(wall_hits,direction):
    """Calculates the new direction based on the wall hits.
    If the beam hits the bounds of the corresponding coordinate, then the direction is reversed in the corresponding direction element.
    """
    new_direction = direction.copy()
    for i,w in enumerate(wall_hits):
        if w:
            if i in [0,2]:
                new_direction[1] = -new_direction[1]
            else:
                new_direction[0] = -new_direction[0]
    print("Calculated direction: {}".format(new_direction))
    return new_direction

def fire_to_direction(direction,dimensions,your_position,trainer_position,distance):
    print("INPUTS:")
    print("direction:",direction)
    print("dimensions:",dimensions)
    print("your_position:",your_position)
    print("trainer_position:",trainer_position)
    print("distance:",distance)
    print("**********")
    laser_pos = your_position
    travelled_distance = 0
    direction = make_unidir_vector(direction,laser_pos,dimensions)
    while True:
        laser_pos = [laser_pos[0] + direction[0], laser_pos[1] + direction[1]]
        travelled_distance += vector_length(direction)
        print("Laser position at main:",laser_pos)
        print("laser direction at main:",direction)
        if all([math.isclose(lp,yp) for lp, yp in zip(laser_pos,your_position)]) or travelled_distance > distance:
            return False
        if all([math.isclose(lp,tp) for lp, tp in zip(laser_pos,trainer_position)]):
            return True
        hits_wall = hits_walls(laser_pos,dimensions)
        if any(hits_wall):
            direction = cal_new_direction(hits_wall,direction)
        direction = make_unidir_vector(direction,laser_pos,dimensions)
        
    

if __name__ == "__main__":
    print("vector:",make_unidir_vector([-(1/2),-(1/6)],[3,11/6],[3,2]))
    #exit()
    print("vecotor",make_unidir_vector([3,2],[0,5/3],[3,2]))
    print(cal_new_direction([False,True,False,False],[-3,2]))
    print(fire_to_direction([3,2],*cases[0][:-1]))