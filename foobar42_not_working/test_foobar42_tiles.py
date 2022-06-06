import unittest
from foobar42_tiles import *

class TestTileMethod(unittest.TestCase):
    def test_conversion_to_relative1(self):
        wrt = [0,0]
        pos = [1,1]
        assert relative_pos(wrt,pos) == [1,1]
    def test_conversion_to_relative2(self):
        wrt = [1,1]
        pos = [1,2]
        ans = relative_pos(wrt,pos)
        print(ans)
        self.assertTrue(ans == [0,1])
    
    def test_conversion_to_relative3(self):
        wrt = [-1,-1]
        pos = [1,2]
        ans = relative_pos(wrt,pos)
        print(ans)
        self.assertTrue(ans == [2,3])
    
    def test_conversion_to_relative4(self):
        wrt = [1,2]
        pos = [-3,2]
        ans = relative_pos(wrt,pos)
        print(ans)
        self.assertTrue(ans == [-4,0])
        
    def test_copying_tile1(self):
        tile = Tile((0,0),(1,0),(-1,-1),(3,2))
        direction = (1,0)
        new_tile = copy_tile(tile,direction)
        try:
            assert new_tile.corner_pos == (2,-1)
            assert new_tile.enemy_pos == (3,0)
            assert new_tile.my_pos == (4,0)
        except AssertionError:
            print("INCORRECT:")
            new_tile.print_absolute()
            self.assertTrue(False)
            
    def test_copying_tile2(self):
        tile = Tile((0,0),(1,0),(-1,-1),(3,2))
        direction = (-1,0)
        new_tile = copy_tile(tile,direction)
        try:
            assert new_tile.corner_pos == (-4,-1)
            assert new_tile.enemy_pos == (-3,0)
            assert new_tile.my_pos == (-2,0)
        except AssertionError:
            print("INCORRECT:")
            new_tile.print_absolute()
            self.assertTrue(False)
            
    def test_copying_tile3(self):
        tile = Tile((0,0),(1,0),(-1,-1),(3,2))
        direction = (0,1)
        new_tile = copy_tile(tile,direction)
        try:
            assert new_tile.corner_pos == (-1,1)
            assert new_tile.my_pos == (0,2)
            assert new_tile.enemy_pos == (1,2)
        except AssertionError:
            print("INCORRECT:")
            new_tile.print_absolute()
            self.assertTrue(False)
    
    def test_copying_tile4(self):
        tile = Tile((0,0),(1,0),(-1,-1),(3,2))
        direction = (0,-1)
        new_tile = copy_tile(tile,direction)
        try:
            assert new_tile.corner_pos == (-1,-3)
            assert new_tile.my_pos == (0,-2)
            assert new_tile.enemy_pos == (1,-2)
        except AssertionError:
            print("INCORRECT:")
            new_tile.print_absolute()
            self.assertTrue(False)
            
    def test_create_none_array1(self):
        tile_size = (4,3)
        distance = 5
        my_pos = (1,2)
        a = create_none_array(tile_size,distance,my_pos)
        self.assertEqual(len(a[0]), 3)
        self.assertEqual(len(a), 4)
    
    def test_create_none_array2(self):
        tile_size = (3,2)
        distance = 4
        my_pos = (1,1)
        a = create_none_array(tile_size,distance,my_pos)
        self.assertEqual(len(a[0]), 3)
        self.assertEqual(len(a), 5)
    
    def test_create_none_array3(self):
        tile_size = (4,4)
        distance = 6
        my_pos = (2,2)
        a = create_none_array(tile_size,distance,my_pos)
        self.assertEqual(len(a[0]), 3)
        self.assertEqual(len(a), 3)
    
    def test_insert_center_tile_inplace1(self):
        tile_size = (4,3)
        distance = 5
        my_pos = (1,2)
        a = create_none_array(tile_size, distance, my_pos)
        tile = Tile((1,2),(1,1),(0,0),(tile_size))
        tile.change_position_to_relative((1,2))
        col,row = insert_center_tile_inplace(a, tile)
        self.assertTrue((col,row) == (2,3))
    
if __name__ == "__main__":
    unittest.main()