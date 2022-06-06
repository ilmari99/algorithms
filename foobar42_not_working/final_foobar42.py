import math
import itertools
import random

def isclose(a,b,rel_tol=1e-09,abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def round_if_close(L):
    """Rounds a list of floats to the nearest integer if they are 'close' to an integer
    """
    return [round(x) if isclose(x,round(x),abs_tol=10**(-12)) else x for x in L]
class Room:
    my_pos = []
    enemy_pos = []
    bounds = []
    distance= None
    ax = None
    
    def __init__(self,**kwargs):
        """Gives the conditions for the room (size, my position, enemy position)
        """
        self.bounds,self.my_pos,self.enemy_pos,self.distance = self.generate_case(**kwargs)
        

    @classmethod
    def generate_case(cls,**kwargs):
        """Generates a case, with either the default parameters or the ones given in kwargs

        Returns:
            tuple: (bounds, your_position, trainer_position, distance)
        """
        bounds = kwargs.get("bounds", [random.randint(1,1250), random.randint(1,1250)])
        your_position = kwargs.get("my_pos", [random.randint(1,bounds[0]-1), random.randint(1,bounds[1]-1)])
        trainer_position = kwargs.get("enemy_pos", [random.randint(1,bounds[0]-1), random.randint(1,bounds[1]-1)])
        distance = kwargs.get("distance", random.randint(1,10000))
        while your_position == trainer_position:
            trainer_position = [random.randint(1,bounds[0]-1), random.randint(1,bounds[1]-1)]
        assert 1 < distance <= 10000
        assert 1 < bounds[0] <= 1250
        assert 1 < bounds[1] <= 1250
        assert 0 < your_position[0] < bounds[0]
        assert 0 < your_position[1] < bounds[1]
        assert 0 < trainer_position[0] < bounds[0]
        assert 0 < trainer_position[1] < bounds[1]
        return (bounds, your_position, trainer_position, distance)
    
class Shot:
    start_pos = []          # The starting position of the laser: my_pos of room
    create_path = False     # If the path of the shot should be created
    room = None             # The room object, where the shot happens
    max_distance = None         # How long does the laser remain deadly
    bounds = None           # Bounds of the room (0 excluded)
    hits = None             # Whether the laser hits the target
    path = {}               # The path of the laser (if created)
    start_direction = []
    ax = None
    bounces = None
    travelled_distance = None
    
    wall_hits = []          # A logical array telling which walls the laser is currently hitting
    pos = []                # The current position of the laser
    direction = []          # The current direction of the laser
    
    def set_room(self,room):
        self.room = room
        self.start_pos = room.my_pos
        self.max_distance = room.distance
        self.bounds = room.bounds
    
    def __init__(self,room,create_path=False):
        self.set_room(room)
        self.create_path = create_path
    
    
    def __max_step_lens(self,pos,direction):
        """Returns the longest step in x and y directions, where the longest step in the distance to the closest
        integer in x and y directions wrt. to direction.
        Assumes that if self.pos is at bounds, then the direction is not in the direction of the bound.
        """
        # Calculate the distance to the closest integer in the x and y directions
        xdist = math.floor(pos[0]) - pos[0] if direction[0]<0 else math.ceil(pos[0]) - pos[0]
        ydist = math.floor(pos[1]) - pos[1] if direction[1]<0 else math.ceil(pos[1]) - pos[1]
        a = round_if_close([xdist,ydist])
        # If the position is at an integer coordinate, then we can step 1 in the direction of the direction
        if xdist == 0:
            xdist = -1 if direction[0]<0 else 1
        if ydist == 0:
            ydist = -1 if direction[1]<0 else 1
        return [xdist,ydist]
    
    def check_new_direction(self,direction,new_direction):
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
    
    def create_step_to_direction(self,pos,direction):
        """ Converts a direction vector to a step vector, that is the longest 'safe' step in the given direction
        """
        # Steps towards the closest integer
        max_steps = self.__max_step_lens(pos,direction)
        xdist = max_steps[0]
        ydist = max_steps[1]
        # If the slope is 0, then the line is horizontal
        if isclose(direction[0],0):
            return [0,ydist] #Else would raise exception on division by zero
        k = direction[1]/direction[0]
        if isclose(k,0):
            return [xdist,0] ######
        xchange = ydist/k
        ychange = xdist*k
        if abs(xchange) > abs(xdist):
            step = [xdist,ychange]
        elif abs(xchange) < abs(xdist):
            step = [xchange,ydist]
        else:
            step = [xdist,ydist]
        return step
    
    def hits_walls(self,pos):
        """Returns a logical array, with True if the laser hits a wall
        index 0 is the bottom wall, index 1 is the left wall,
        index 2 is the top wall, index 3 is the right wall
        """
        bounds = [0,0,self.bounds[0],self.bounds[1]]
        walls = [False]*len(bounds)       # 0 = bottom, 1 = left, 2 = top, 3 = right
        if pos[0]==bounds[0]:
            walls[1] = True
        if pos[0]==bounds[2]:
            walls[3] = True
        if pos[1] == bounds[1]:
            walls[0] = True
        if pos[1] == bounds[3]:
            walls[2] = True
        return walls
    
    def cal_new_direction(self,wall_hits,direction):
        """Calculates the new direction based on the wall hits.
        If the beam hits the bounds of the corresponding coordinate, then the direction is reversed in the corresponding direction element.
        """
        #new_direction = direction.copy()
        for i,w in enumerate(wall_hits):
            if w:
                # If hits the bottom or top walls, the y direction is reversed
                if i in [0,2]:
                    direction[1] = -direction[1]
                else:
                    direction[0] = -direction[0]
        return direction
    
    def shoot(self,direction, allowed_bounces=float("inf")):
        pos = self.start_pos
        self.start_direction = direction
        self.travelled_distance = 0
        self.bounces = 0
        self.path = {0:pos}
        stepno = 1
        while True:
            direction = self.create_step_to_direction(pos,direction)
            pos = round_if_close([pos[0]+direction[0],pos[1]+direction[1]])
            self.travelled_distance += vector_length(direction)
            #if any([True if lp < 0 or lp > d else False for lp,d in zip(pos,self.bounds)]):
            #    raise Exception("Laser position is outside the bounds of the arena")
            if self.create_path:
                self.path[stepno] = pos
            if all([lp==yp for lp, yp in zip(pos,self.start_pos)]) or self.travelled_distance > self.max_distance:
                self.hits = False
                break
            if all([lp==tp for lp, tp in zip(pos,self.room.enemy_pos)]):
                #print("HITS TRAINER\n")
                self.hits = True
                break
            
            wall_hits = self.hits_walls(pos)
            if any(wall_hits):
                self.bounces += 1
                if self.bounces > allowed_bounces:
                    self.hits = False
                    break
                direction = self.cal_new_direction(wall_hits,direction)
            stepno += 1
        return self.hits         

def relative_pos(wrt,pos):
    return [pos[0]-wrt[0],pos[1]-wrt[1]]

def vector_length(vector):
    return math.sqrt(sum([v**2 for v in vector]))

class Tile:
    my_pos = ()
    enemy_pos = ()
    corner_pos = ()
    size = ()
    def __init__(self,my_pos,enemy_pos,corner_pos,size):
        self.my_pos = tuple(my_pos)
        self.enemy_pos = tuple(enemy_pos)
        self.corner_pos = tuple(corner_pos)
        self.size = tuple(size)
        
    def __repr__(self):
        return str(self.corner_pos)
    
    def __eq__(self, o):
        return self.my_pos == o.my_pos and self.enemy_pos == o.enemy_pos and self.corner_pos == o.corner_pos and self.size == o.size
        
    def to_relative_pos(self,pos):
        """Converts absolute position to coordinates relative to the tiles corner"""
        pos = relative_pos(self.corner_pos,pos)
        return tuple(pos)
    
    def change_position_to_relative(self,new_origin):
        self.my_pos = tuple(relative_pos(new_origin,self.my_pos))
        self.enemy_pos = tuple(relative_pos(new_origin,self.enemy_pos))
        self.corner_pos = tuple(relative_pos(new_origin,self.corner_pos))
        
    def print_relative(self):
        print("relative My pos:",self.to_relative_pos(self.my_pos))
        print("relative Enemy pos:",self.to_relative_pos(self.enemy_pos))
        print("relative Corner pos:",self.to_relative_pos(self.corner_pos))
        print("Size:",self.size)
        print("")
    
    def print_absolute(self):
        print("My pos:",self.my_pos)
        print("Enemy pos:",self.enemy_pos)
        print("Corner pos:",self.corner_pos)
        print("Size:",self.size)
        print("")


def copy_tile(tile,direction):
    cpos = list(tile.corner_pos)                    # Absolute positions of the tile to be copied
    my_pos = list(tile.my_pos)
    enemy_pos = list(tile.enemy_pos)
    assert cpos != [0,0]                            # if corner is (0,0) then the cdinates are not absolute
    #direction = list(direction)
    #direction.reverse()
    if abs(direction[1]) == 1:                      # If the tile is to be copied in the x-direction
        cpos[0] += direction[1]*tile.size[0]        # The new tiles absolute corner position is moved in the given x-direction by the width of the tile
        # The new absolute positions are calculated by adding the tile width to the original tiles absolute position,
        # then the point (my_pos and enemy_pos) is moved 2*the relative distance in the given direction,
        # finally the tile_size is added to the new absolute position
        my_pos[0] += direction[1]*tile.size[0] - 2*tile.to_relative_pos(my_pos)[0] + tile.size[0]
        enemy_pos[0] += direction[1]*tile.size[0] - 2*tile.to_relative_pos(enemy_pos)[0] + tile.size[0]
        
    elif abs(direction[0]) == 1:                    # If the tile is to be copied in the y-direction
        cpos[1] += direction[0]*tile.size[1]
        my_pos[1] += direction[0]*tile.size[1] - 2*tile.to_relative_pos(my_pos)[1] + tile.size[1]
        enemy_pos[1] += direction[0]*tile.size[1] - 2*tile.to_relative_pos(enemy_pos)[1] + tile.size[1]
    return Tile(my_pos,enemy_pos,cpos,tile.size)

def create_none_array(tile_size,distance,my_pos):
    #print("Inputs:",tile_size,distance,my_pos)
    tile_size = [float(t) for t in tile_size]
    xneq = math.ceil((distance - my_pos[0])/tile_size[0])
    xpos = math.ceil((distance + my_pos[0])/tile_size[0])
    ypos = math.ceil((distance + my_pos[1])/tile_size[1])
    yneq = math.ceil((distance - my_pos[1])/tile_size[1])
    inxdir = xpos + xneq
    inydir = ypos + yneq
    #print("inxdir: {} inydir: {}".format(inxdir,inydir))
    l = [[0 for i in range(int(inxdir))] for j in range(int(inydir))]     # create a list of lists
    return l
    
    
def insert_center_tile_inplace(tile_array,tile):
    """Inserts the given (original) tile into the tile_array
    The tile should have its coordinates relative to my_pos(=(0,0))
    """
    
    mp = tile.to_relative_pos(tile.my_pos)      # Get my_pos relative to the corner of the tile
    xt = len(tile_array[0])                     # Total number of tiles in the x direction
    yt = len(tile_array)                        # Total number of tiles in the y direction
    tile_size = [float(t) for t in tile.size]
    # Calculate the index of the tile in the tile array
    row_ind = yt//2 if mp[1]>=tile_size[1]/2 else yt//2 - 1
    col_ind = xt//2 if mp[0]<=tile_size[0]/2 else xt//2 - 1
    tile_array[row_ind][col_ind] = tile
    return (row_ind,col_ind)
    
def calc_xy_dist(dire,ind,size):
    dist = [0,0]
    for i,d in enumerate(dire):
        if i == 0:                       # x-direction
            if d == 1:
                dist[0] = size[0]-1-ind[0]
            elif d == -1:
                dist[0] = ind[0]
        elif i == 1:
            if d == 1:
                dist[1] = size[1]-1-ind[1]
            elif d == -1:
                dist[1] = ind[1]
    return dist

def tile_to_left_up_corner(tar,tile,ind):
    """Tiles the tile array to the left upper corner.
    First goes to left side of the array, and then to the top of the array."""
    size = (len(tar[0]),len(tar))
    #directions = [(-1,0) for _ in range(ind[0])] + [(0,-1) for _ in range(ind[1])]
    while True:
        if ind[1] > 0:
            d = (0,-1)#directions.pop(0)
        elif ind[0] > 0:
            d = (-1,0)
        else:
            break
        ind = [ind[0]+d[0],ind[1]+d[1]]
        try:
            if not isinstance(tar[ind[0]][ind[1]],Tile):
                tar[ind[0]][ind[1]] = copy_tile(tile,d)
            tile = tar[ind[0]][ind[1]]
        except IndexError:
            raise IndexError("IndexError: tile array index out of bounds")
    return tar,tile,ind
    
def tile_the_arr(tar,tile,ind):
    # First tile to the left upper corner
    tar,tile,ind = tile_to_left_up_corner(tar, tile, ind)
    #print("After tiling to left upper corner:")
    #for _ in tar:
    #    print(_)
    size = (len(tar[0]),len(tar))
    dist = calc_xy_dist((1,0),ind,size)
    assert ind == [0,0]
    coldir = 1
    assert dist[0] == size[0]-1
    directions = [(0,1) for _ in range(dist[0])] + [(1,0)]
    while True:
        if not directions:
            coldir = -1*coldir
            directions = [(0,coldir) for _ in range(dist[0])] + [(1,0)]#[(xdir,0)]*dist[0] + [(0,-1)]
        d = directions.pop(0)
        ind = [ind[0]+d[0],ind[1]+d[1]]
        try:
            if not isinstance(tar[ind[0]][ind[1]],Tile):
                tar[ind[0]][ind[1]] = copy_tile(tile,d)
            tile = tar[ind[0]][ind[1]]
        except IndexError:
            break
    return tar

def solution(bounds,my_pos,enemy_pos,distance):
    case = {'bounds':bounds,'my_pos':my_pos,'enemy_pos':enemy_pos,'distance':distance}
    room = Room(**case)     # Create room instance
    # Create the original tile instance, with coordinates relative to my_pos
    tile = Tile(room.my_pos,room.enemy_pos,(0,0),room.bounds)
    tile.change_position_to_relative(new_origin=room.my_pos)
    #tile.print_relative()
    # Create an empty tile array (list of lists with 0 as elements)
    tile_arr = create_none_array(tile.size,
                                 room.distance,
                                 relative_pos(tile.corner_pos,tile.my_pos))
    # Insert the original tile into the tile array, to its proper position, and return the row and column index
    rind,cind = insert_center_tile_inplace(tile_arr, tile)
    #print("Center tile (row: {}, col: {}):".format(rind,cind))
    #for _ in tile_arr:
    #    print(_)
    tile_arr = tile_the_arr(tile_arr, tile, [rind,cind])
    #print("Tiled array")
    #for _ in tile_arr:
    #    print(_)
    flat_ar = itertools.chain(*tile_arr)
    shot = Shot(room,create_path=True)
    hits = {}
    for t in flat_ar:
        vec = [float(t) for t in t.enemy_pos]
        vlen = vector_length(vec)
        uvec = tuple([round(vec[i]/vlen,7) for i in range(len(vec))])
        if uvec not in hits.keys(): # If the same direction vector has already been shot
            hit = shot.shoot(vec)
            if hit:     # If the shot hits
                hits[uvec] = True
            else:
                print("Shot at direction {} did not hit".format(uvec))
    #print(hits.keys())
    #print("Hit count: {}".format(len(hits)))
    return len(hits)
    
if __name__ == "__main__":
    cases = [{"bounds":(4,4),"my_pos":(1,1),"enemy_pos":(3,2),"distance":6,"ans":7},
             {"bounds":(4,4),"my_pos":(1,1),"enemy_pos":(3,2),"distance":15,"ans":42},
             {"bounds":(4,4),"my_pos":(2,2),"enemy_pos":(1,1),"distance":6,"ans":7},
             {"bounds":(4,4),"my_pos":(2,2),"enemy_pos":(1,1),"distance":25,"ans":99},
             {"bounds":(3,2),"my_pos":(1,1),"enemy_pos":(2,1),"distance":4,"ans":7},
             {"bounds":(4,3),"my_pos":(1,2),"enemy_pos":(1,1),"distance":5,"ans":4},
             {"bounds":(2,3),"my_pos":(1,1),"enemy_pos":(1,2),"distance":4,"ans":7},
             {"bounds":(16,2),"my_pos":(3,1),"enemy_pos":(14,1),"distance":4,"ans":7},
             ]
    for n,case in enumerate(cases):
        try:
            ans = case.pop("ans")
            sol = solution(**case)
            assert sol == ans
        except AssertionError:
            print("Case #{}: FAILED".format(n+1))
            print("Details:")
            print("Excpected: {}".format(ans))
            print("Received: {}".format(sol))
            for k in case.keys():
                print("{}: {}".format(k,case[k]))
            print("\n")