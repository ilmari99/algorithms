import math
import itertools
import random

def isclose(a,b,rel_tol=1e-09,abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def round_if_close(L):
    """Rounds a list of floats to the nearest integer if they are 'close' to an integer
    """
    return [round(x) if isclose(x,round(x),abs_tol=10**(-12)) else x for x in L]   

def relative_pos(wrt,pos):
    return [pos[0]-wrt[0],pos[1]-wrt[1]]

def vector_length(vector):
    return math.sqrt(sum([v**2 for v in vector]))

def gcd(a,b):
    """Greatest common divisor"""
    while b:
        a,b = b,a%b
    return a

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
    assert ind == [0,0]
    coldir = 1
    directions = [(0,1) for _ in range(size[0]-1)] + [(1,0)]
    while True:
        if not directions:
            coldir = -1*coldir
            directions = [(0,coldir) for _ in range(size[0]-1)] + [(1,0)]#[(xdir,0)]*dist[0] + [(0,-1)]
        d = directions.pop(0)
        ind = [ind[0]+d[0],ind[1]+d[1]]
        try:
            if not isinstance(tar[ind[0]][ind[1]],Tile):
                tar[ind[0]][ind[1]] = copy_tile(tile,d)
            tile = tar[ind[0]][ind[1]]
        except IndexError:
            break
    return tar

def gen_next_tile(tar,tile,ind):
    # First tile to the left upper corner
    tar,tile,ind = tile_to_left_up_corner(tar, tile, ind)
    size = (len(tar[0]),len(tar))
    assert ind == [0,0]
    coldir = 1
    directions = [(0,1) for _ in range(size[0]-1)] + [(1,0)]
    while True:
        if not directions:
            coldir = -1*coldir
            directions = [(0,coldir) for _ in range(size[0]-1)] + [(1,0)]#[(xdir,0)]*dist[0] + [(0,-1)]
        d = directions.pop(0)
        ind = [ind[0]+d[0],ind[1]+d[1]]
        try:
            if not isinstance(tar[ind[0]][ind[1]],Tile):
                tar[ind[0]][ind[1]] = copy_tile(tile,d)
            tile = tar[ind[0]][ind[1]]
        except IndexError:
            break
        yield tile
    
def generate_directions(bounds,my_pos,enemy_pos,distance):
    ogtile = Tile(my_pos,enemy_pos,(0,0),bounds)
    ogtile.change_position_to_relative(my_pos)
    assert ogtile.my_pos == (0,0)
    tile_arr = create_none_array(bounds, distance, my_pos)
    #print("Tile array size: (",len(tile_arr),",",len(tile_arr[0]),")")
    rind,cind = insert_center_tile_inplace(tile_arr,ogtile)
    next_tile_gen = gen_next_tile(tile_arr,ogtile,(rind,cind))
    for i,t in enumerate(next_tile_gen):
        #vlen = vector_length(t.enemy_pos)
        #uvec = (float(t.enemy_pos[0]/vlen),float(t.enemy_pos[1]/vlen))
        yield t
        #yield (float(t.enemy_pos[0]),float(t.enemy_pos[1]))
        
        
    
def solution(bounds,my_pos,enemy_pos,distance):
    hits = {} # Heading - distance pairs
    for t in generate_directions(bounds,my_pos,enemy_pos,distance):
        heading_to_enemy = math.atan2(t.enemy_pos[1],t.enemy_pos[0])
        heading_to_me = math.atan2(t.my_pos[1],t.my_pos[0])
        dist_to_enemy = vector_length(t.enemy_pos)
        dist_to_me = vector_length(t.my_pos)
        if heading_to_enemy in hits.keys():
            d = hits[heading_to_enemy]
            if d < 0 and abs(d) > dist_to_enemy:
                hits[heading_to_enemy] = dist_to_enemy
        if heading_to_me == heading_to_enemy and dist_to_me < dist_to_enemy:
            hits[heading_to_me] = -dist_to_me
    #print(hits.keys())
    #print("Hit count: {}".format(len(hits)))
    return sum([1 if d>0 and d <= distance else 0 for d in hits.values()])

def solution_2(bounds,my_pos,enemy_pos,distance):
    hits = {}
    generate_directions(bounds, my_pos, enemy_pos, distance)
    for t in generate_directions(bounds, my_pos, enemy_pos, distance):
        d_to_enemy = vector_length(t.enemy_pos)
        d_to_me = vector_length(t.my_pos)
        direction_to_enemy = (float(t.enemy_pos[0])/d_to_enemy,float(t.enemy_pos[1])/d_to_enemy)
        direction_to_me = (float(t.my_pos[0])/d_to_me,float(t.my_pos[1])/d_to_me)
        
        
        
    
if __name__ == "__main__":
    cases = [{"bounds":(4,4),"my_pos":(1,1),"enemy_pos":(3,2),"distance":6,"ans":7},
             {"bounds":(4,4),"my_pos":(1,1),"enemy_pos":(3,2),"distance":15,"ans":42},
             {"bounds":(4,4),"my_pos":(2,2),"enemy_pos":(1,1),"distance":6,"ans":7},
             {"bounds":(4,4),"my_pos":(2,2),"enemy_pos":(1,1),"distance":25,"ans":99},
             {"bounds":(3,2),"my_pos":(1,1),"enemy_pos":(2,1),"distance":4,"ans":7},
             {"bounds":(4,3),"my_pos":(1,2),"enemy_pos":(1,1),"distance":5,"ans":4},
             {"bounds":(2,3),"my_pos":(1,1),"enemy_pos":(1,2),"distance":4,"ans":7},
             {"bounds":(16,2),"my_pos":(3,1),"enemy_pos":(14,1),"distance":4,"ans":0},
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