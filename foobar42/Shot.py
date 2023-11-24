import math
from utils import round_if_close, isclose, vector_length


class Shot:
    start_pos = ()          # The starting position of the laser: my_pos of room
    create_path = False     # If the path of the shot should be created
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
    
    def __init__(self,bounds, my_pos, enemy_pos, distance, create_path=True):
        self.start_pos = my_pos
        self.max_distance = distance
        self.bounds = bounds
        self.enemy_pos = enemy_pos
        self.create_path = True
    
    
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
            if all([lp==tp for lp, tp in zip(pos,self.enemy_pos)]):
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
    
    def plot_path(self,ax):
        x = [_[0] for _ in self.path.values()]
        y = [_[1] for _ in self.path.values()]
        ax.plot(x,y)
        return ax