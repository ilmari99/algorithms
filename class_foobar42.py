"""
Uh-oh -- you've been cornered by one of Commander Lambdas elite bunny trainers! Fortunately, you grabbed a beam weapon from an abandoned storeroom while you were running through the station, so you have a chance to fight your way out. But the beam weapon is potentially dangerous to you as well as to the bunny trainers: its beams reflect off walls, meaning you'll have to be very careful where you shoot to avoid bouncing a shot toward yourself!

Luckily, the beams can only travel a certain maximum distance before becoming too weak to cause damage. You also know that if a beam hits a corner, it will bounce back in exactly the same direction. And of course, if the beam hits either you or the bunny trainer, it will stop immediately (albeit painfully). 

Write a function solution(bounds, your_position, trainer_position, distance) that gives an array of 2 integers of the width and height of the room, an array of 2 integers of your x and y coordinates in the room, an array of 2 integers of the trainer's x and y coordinates in the room, and returns an integer of the number of distinct directions that you can fire to hit the elite trainer, given the maximum distance that the beam can travel.

The room has integer bounds [1 < x_dim <= 1250, 1 < y_dim <= 1250]. You and the elite trainer are both positioned on the integer lattice at different distinct positions (x, y) inside the room such that [0 < x < x_dim, 0 < y < y_dim]. Finally, the maximum distance that the beam can travel before becoming harmless will be given as an integer 1 < distance <= 10000.

For example, if you and the elite trainer were positioned in a room with bounds [3, 2], your_position [1, 1], trainer_position [2, 1], and a maximum shot distance of 4, you could shoot in seven different directions to hit the elite trainer (given as vector bearings from your location): [1, 0], [1, 2], [1, -2], [3, 2], [3, -2], [-3, 2], and [-3, -2]. As specific examples, the shot at bearing [1, 0] is the straight line horizontal shot of distance 1, the shot at bearing [-3, -2] bounces off the left wall and then the bottom wall before hitting the elite trainer with a total shot distance of sqrt(13), and the shot at bearing [1, 2] bounces off just the top wall before hitting the elite trainer with a total shot distance of sqrt(5).
"""
import random
import math
import matplotlib.pyplot as plt
cases = [([3,2],[1,1],[2,1],4,7), ([300,275],[150,150],[185,100],500,9)]

def vector_length(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

def round_if_close(L):
    """Rounds a list of floats to the nearest integer if they are 'close' to an integer
    """
    return [round(x) if math.isclose(x,round(x),abs_tol=10**(-12)) else x for x in L]

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
    
    def draw_room(self,new_fig=True):
        if not new_fig:
            return self.ax
        fig = plt.figure()
        ax = fig.add_subplot()
        free_space_on_edges = 0.4
        xlims = (-free_space_on_edges*self.bounds[0],self.bounds[0]+free_space_on_edges*self.bounds[0])
        ylims = (-free_space_on_edges*self.bounds[1],self.bounds[1]+free_space_on_edges*self.bounds[1])
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        #fig.xlim(-free_space_on_edges*self.bounds[0],self.bounds[0]+free_space_on_edges*self.bounds[0])
        #fig.ylim(-free_space_on_edges*self.bounds[1],self.bounds[1]+free_space_on_edges*self.bounds[1])
        x = [0,0,self.bounds[0],self.bounds[0],0]
        y = [0,self.bounds[1],self.bounds[1],0,0]
        ax.plot(x,y,label="bounds")
        ax.plot(self.my_pos[0],self.my_pos[1],'ro',label="Your position")
        ax.plot(self.enemy_pos[0],self.enemy_pos[1],'bo',label="enemy position"),
        ax.grid()
        self.ax = ax
        return ax
        
    

class Shot:
    start_pos = []          # The starting position of the laser: my_pos of room
    create_path = False     # If the path of the shot should be created
    room = None             # The room object, where the shot happens
    distance = None         # How long does the laser remain deadly
    bounds = None           # Bounds of the room (0 excluded)
    hits = None             # Whether the laser hits the target
    path = {}               # The path of the laser (if created)
    start_direction = []
    ax = None
    
    wall_hits = []          # A logical array telling which walls the laser is currently hitting
    pos = []                # The current position of the laser
    direction = []          # The current direction of the laser
    
    def set_room(self,room):
        self.room = room
        self.start_pos = room.my_pos
        self.distance = room.distance
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
        if math.isclose(direction[0],0):
            return [0,ydist] #Else would raise exception on division by zero
        k = direction[1]/direction[0]
        if math.isclose(k,0):
            return [xdist,0] ######
        xchange = ydist/k
        ychange = xdist*k
        if abs(xchange) > abs(xdist):
            step = [xdist,ychange]
        elif abs(xchange) < abs(xdist):
            step = [xchange,ydist]
        else:#math.isclose(abs(xchange),abs(ychange))
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
    
    def shoot(self,direction):
        pos = self.start_pos
        self.start_direction = direction.copy()
        travelled_distance = 0
        self.path = {0:pos}
        stepno = 1
        while True:
            direction = self.create_step_to_direction(pos,direction)
            pos = round_if_close([pos[0]+direction[0],pos[1]+direction[1]])
            travelled_distance += vector_length(direction)
            if any([True if lp < 0 or lp > d else False for lp,d in zip(pos,self.bounds)]):
                raise Exception("Laser position is outside the bounds of the arena")
            if self.create_path:
                self.path[stepno] = pos
            if all([lp==yp for lp, yp in zip(pos,self.start_pos)]) or travelled_distance > self.distance:
                self.hits = False
                break
            if all([lp==tp for lp, tp in zip(pos,self.room.enemy_pos)]):
                #print("HITS TRAINER\n")
                self.hits = True
                break
            wall_hits = self.hits_walls(pos)
            if any(wall_hits):
                direction = self.cal_new_direction(wall_hits,direction)
            stepno += 1
        return self.hits
    
    def plot_path(self,ax=None,show=True,new_fig=True):
        """Creates a plot of the path of the laser
        """
        if self.ax is None:
            assert new_fig or ax
        self.fig = ax
        if new_fig:
            self.ax = room.draw_room()
        x = [_[0] for _ in self.path.values()]
        y = [_[1] for _ in self.path.values()]
        self.ax.plot(x,y,label="path")
        if new_fig:
            plt.legend(loc="upper left")
        if show:
            plt.show()

room = Room(bounds = [3,2],my_pos = [1,1],enemy_pos=[2,1],distance=4)
shot = Shot(room,create_path=True)


if __name__ == "__main__":
    room = Room(bounds = [3,2],my_pos = [1,1],enemy_pos=[2,1],distance=4)
    shot = Shot(room,create_path=True)
    shot.ax = room.draw_room()
    for d in generate_initial_directions(10):
        shot.shoot(d)
        if shot.hits:
            shot.plot_path(show=False,new_fig=False)
    plt.show()