"""
Uh-oh -- you've been cornered by one of Commander Lambdas elite bunny trainers! Fortunately, you grabbed a beam weapon from an abandoned storeroom while you were running through the station, so you have a chance to fight your way out. But the beam weapon is potentially dangerous to you as well as to the bunny trainers: its beams reflect off walls, meaning you'll have to be very careful where you shoot to avoid bouncing a shot toward yourself!

Luckily, the beams can only travel a certain maximum distance before becoming too weak to cause damage. You also know that if a beam hits a corner, it will bounce back in exactly the same direction. And of course, if the beam hits either you or the bunny trainer, it will stop immediately (albeit painfully). 

Write a function solution(dimensions, your_position, trainer_position, distance) that gives an array of 2 integers of the width and height of the room, an array of 2 integers of your x and y coordinates in the room, an array of 2 integers of the trainer's x and y coordinates in the room, and returns an integer of the number of distinct directions that you can fire to hit the elite trainer, given the maximum distance that the beam can travel.

The room has integer dimensions [1 < x_dim <= 1250, 1 < y_dim <= 1250]. You and the elite trainer are both positioned on the integer lattice at different distinct positions (x, y) inside the room such that [0 < x < x_dim, 0 < y < y_dim]. Finally, the maximum distance that the beam can travel before becoming harmless will be given as an integer 1 < distance <= 10000.

For example, if you and the elite trainer were positioned in a room with dimensions [3, 2], your_position [1, 1], trainer_position [2, 1], and a maximum shot distance of 4, you could shoot in seven different directions to hit the elite trainer (given as vector bearings from your location): [1, 0], [1, 2], [1, -2], [3, 2], [3, -2], [-3, 2], and [-3, -2]. As specific examples, the shot at bearing [1, 0] is the straight line horizontal shot of distance 1, the shot at bearing [-3, -2] bounces off the left wall and then the bottom wall before hitting the elite trainer with a total shot distance of sqrt(13), and the shot at bearing [1, 2] bounces off just the top wall before hitting the elite trainer with a total shot distance of sqrt(5).
"""
import random
import math
import matplotlib.pyplot as plt
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
    a = round_if_close([xdist,ydist])
    #xdist = a[0]
    #ydist = a[1]
    # If the position is at an integer coordinate, then we can step 1 in the direction of the direction
    if xdist == 0:
        xdist = -1 if direction[0]<0 else 1
    if ydist == 0:
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
        new_ratio = new_direction[1]/new_direction[0]
        ratio = direction[1]/direction[0]
        assert math.isclose(ratio,new_ratio)
    except AssertionError:
        msg = "Invalid direction (incorrect ratio): New direction: {}, Old direction {}".format(new_ratio,ratio)
    if msg:
        raise AssertionError(msg)
    
def round_if_close(L):
    """Rounds a list of floats to the nearest integer if they are 'close' to an integer
    """
    return [round(x) if math.isclose(x,round(x),abs_tol=10**(-12)) else x for x in L]

def create_step_to_direction(direction, laser_pos, bounds):
    """ Converts a direction vector to a step vector, that is the longest 'safe' step in the given direction
    """
    # Steps towards the closest integer
    max_steps = max_step_lens(direction,laser_pos)
    xdist = max_steps[0]
    ydist = max_steps[1]
    # If the slope is 0, then the line is horizontal
    if math.isclose(direction[0],0):
        return [0,ydist] #Else would raise exception on division by zero
    k = direction[1]/direction[0]
    if math.isclose(k,0):
        return [xdist,0] ######
    xchange = ydist/k
    ychange = xdist*k
    if abs(xchange) > abs(xdist):
        vec = [xdist,ychange]
    elif abs(xchange) < abs(xdist):
        vec = [xchange,ydist]
    else:#math.isclose(abs(xchange),abs(ychange))
        vec = [xdist,ydist]
    return vec
    
    
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
    if laser_position[0]==bounds[0]:
        walls[1] = True
    if laser_position[0]==bounds[2]:
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
            # If hits the bottom or top walls, the y direction is reversed
            if i in [0,2]:
                new_direction[1] = -new_direction[1]
            else:
                new_direction[0] = -new_direction[0]
    return new_direction

def fire_to_direction(direction,dimensions,your_position,trainer_position,distance,ret_path=False):
    #print("INPUTS:")
    #print("direction:",direction)
    #print("dimensions:",dimensions)
    #print("your/laser_position:",your_position)
    #print("trainer_position:",trainer_position)
    #print("distance:",distance)
    #print("**********")
    laser_pos = your_position
    travelled_distance = 0
    #direction = create_step_to_direction(direction,laser_pos,dimensions)
    path = {0:laser_pos}
    stepno = 1
    out = None
    while True:
        direction = create_step_to_direction(direction,laser_pos,dimensions)
        laser_pos = [laser_pos[0] + direction[0], laser_pos[1] + direction[1]]
        laser_pos = round_if_close(laser_pos)
        travelled_distance += vector_length(direction)
        if any([True if lp < 0 or lp > d else False for lp,d in zip(laser_pos,dimensions)]):
            create_plot(dimensions,your_position,trainer_position,path)
            raise Exception("Laser position is outside the bounds of the arena")
        #print("Direction step:",direction)
        #print("Laser position at main:",laser_pos)
        if ret_path:
            path[stepno] = laser_pos#[sum(p) for p in zip(path.get(stepno,[0,0]),laser_pos)]
        if all([lp==yp for lp, yp in zip(laser_pos,your_position)]) or travelled_distance > distance:
            out = False
            break
        if all([lp==tp for lp, tp in zip(laser_pos,trainer_position)]):
            #print("HITS TRAINER\n")
            out = True
            break
        hits_wall = hits_walls(laser_pos,dimensions)
        if any(hits_wall):
            direction = cal_new_direction(hits_wall,direction)
        stepno += 1
    if ret_path:
        return out,path
    return out
        
        
def generate_initial_directions(distance):
    """Generate distinct shooting directions that are integer pairs (vectors), whose length < distance"""
    i = 4
    yield [0,1]
    yield [0,-1]
    yield [1,0]
    yield [-1,0]
    for x_direction in range(1,distance+1):
        ymax = int(math.sqrt(distance**2 - x_direction**2))
        for y_direction in range(1,ymax+1):
            #if vector_length([x_direction,y_direction]) > distance:
            #    continue
            if math.gcd(x_direction,y_direction) == 1:
                i += 4
                yield [x_direction,y_direction]
                yield [x_direction,-y_direction]
                yield [-x_direction,-y_direction]
                yield [-x_direction,y_direction]

def create_plot(dimensions,your_position,trainer_position,path,show=True,new_fig=True):
    """Creates a plot of the path of the laser
    """
    x = [_[0] for _ in path.values()]
    y = [_[1] for _ in path.values()]
    if new_fig:
        plt.figure()
    plt.plot(x,y,label="path")
    free_space_on_edges = 0.4
    plt.xlim(-free_space_on_edges*dimensions[0],dimensions[0]+free_space_on_edges*dimensions[0])
    plt.ylim(-free_space_on_edges*dimensions[1],dimensions[1]+free_space_on_edges*dimensions[1])
    x = [0,0,dimensions[0],dimensions[0],0]
    y = [0,dimensions[1],dimensions[1],0,0]
    plt.plot(x,y,label="bounds")
    plt.plot(your_position[0],your_position[1],'ro',label="Your position")
    plt.plot(trainer_position[0],trainer_position[1],'bo',label="enemy position"),
    plt.grid()
    if new_fig:
        plt.legend(loc="upper left")
    if show:
        plt.show()

if __name__ == "__main__":
    #print("vecotor",create_step_to_direction([3,2],[0,5/3],[3,2]))
    #print("vector:",create_step_to_direction([-(1/2),-(1/6)],[3,11/6],[3,2]))
    #exit()
    #print(cal_new_direction([False,True,False,False],[-3,2]))
    hits = {}
    dims = [4,4]#cases[1][0]
    your_position = [1,1]#cases[1][1]
    trainer_position = [3,2]#cases[1][2]
    distance = 8#cases[1][3]
    plt.figure()
    for i,d in enumerate(generate_initial_directions(distance)):
        h,path = fire_to_direction(d,dims,your_position,trainer_position,distance,ret_path = True)
        if h:
            create_plot(dims,your_position,trainer_position,path,show=False,new_fig=False)
            if d[0] in hits:
                hits[d[0]].append(d[1])
            else:
                hits[d[0]] = [d[1]]
    print("count:",sum([len(hits[k]) for k in hits]))
    for k in hits.items():
        print(k)
    plt.show()
    #print(fire_to_direction([1,2],*cases[0][:-1]))