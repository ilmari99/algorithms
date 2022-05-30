import math
import numpy as np
from class_foobar42 import Shot, Room, vector_length, round_if_close, to_minimal_path

def relative_pos(wrt,pos):
    return [pos[0]-wrt[0],pos[1]-wrt[1]]


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
        
    def to_relative_pos(self,pos):
        """Converts absolute position to coordinates relative to the tiles corner"""
        pos = relative_pos(self.corner_pos,pos)
        return tuple(pos)
    
    def change_position_to_relative(self,new_origin):
        self.my_pos = relative_pos(new_origin,self.my_pos)
        self.enemy_pos = relative_pos(new_origin,self.enemy_pos)
        self.corner_pos = relative_pos(new_origin,self.corner_pos)
        
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
    
    if abs(direction[0]) == 1:                      # If the tile is to be copied in the x-direction
        cpos[0] += direction[0]*tile.size[0]        # The new tiles absolute corner position is moved in the given x-direction by the width of the tile
        # The new absolute positions are calculated by adding the tile width to the original tiles absolute position,
        # then the point (my_pos and enemy_pos) is moved 2*the relative distance in the given direction,
        # finally the tile_size is added to the new absolute position
        my_pos[0] += direction[0]*tile.size[0] - 2*tile.to_relative_pos(my_pos)[0] + tile.size[0]
        enemy_pos[0] += direction[0]*tile.size[0] - 2*tile.to_relative_pos(enemy_pos)[0] + tile.size[0]
        
    elif abs(direction[1]) == 1:                    # If the tile is to be copied in the y-direction
        cpos[1] += direction[1]*tile.size[1]
        my_pos[1] += direction[1]*tile.size[1] - 2*tile.to_relative_pos(my_pos)[1] + tile.size[1]
        enemy_pos[1] += direction[1]*tile.size[1] - 2*tile.to_relative_pos(enemy_pos)[1] + tile.size[1]
    return Tile(my_pos,enemy_pos,cpos,tile.size)

def tile_up_to_distance(tile_size,distance):
    """ Returns an array of the tiles, that are inside the given distance """
    # How many tiles in each direction is needed to cover the distance circle entirely
    sq_size = (((2*distance)//tile_size[0])+1,((2*distance)//tile_size[1])+1)
    
    
    
    

if __name__ == "__main__":
    room = Room(bounds = [4,4],my_pos = [2,2],enemy_pos=[1,1],distance=20)
    enemy_pos = (room.enemy_pos[0] - room.my_pos[0],room.enemy_pos[1] - room.my_pos[1])
    tile = Tile((0,0),enemy_pos,(-room.my_pos[0], -room.my_pos[1]),tuple(room.bounds))
    tile.print_relative()
    tile.print_absolute()
    new_tile = copy_tile(tile,(0,-1))
    new_tile.print_relative()
    new_tile.print_absolute()
