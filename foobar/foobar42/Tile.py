from utils import relative_pos
MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
        
    def plot_tile(self,ax=None):
        if not MATPLOTLIB_AVAILABLE:
            msg = "Matplotlib could not be imported"
            raise ImportError(msg)
        if ax is None:
            ax = plt.subplots()[1]
        linewidth = 1 if self.my_pos != (0,0) else 3
        ax.plot([self.corner_pos[0],
                 self.corner_pos[0],
                 self.corner_pos[0]+self.size[0],
                 self.corner_pos[0]+self.size[0],
                 self.corner_pos[0]],
                
                [self.corner_pos[1],
                 self.corner_pos[1]+self.size[1],
                 self.corner_pos[1]+self.size[1],
                 self.corner_pos[1],
                 self.corner_pos[1]],
                linewidth=linewidth
                )
        ax.scatter(self.my_pos[0],self.my_pos[1],color="green")
        ax.scatter(self.enemy_pos[0],self.enemy_pos[1],color="red")