import unittest
from foobar42_tiles import *

class Test_foo42(unittest.TestCase):
    def get_tile1(self):
        return Tile((1,1),(3,2),(0,0),(4,4))
    def get_tile1_abs(self):
        return Tile((0,0),(2,1),(-1,-1),(4,4))
    def get_tile2(self):
        return Tile((2,2),(1,1),(0,0),(4,4))
    def get_tile2_abs(self):
        return Tile((0,0),(-1,-1),(-2,-2),(4,4))
    def get_tile3(self):
        return Tile((1,1),(2,1),(0,0),(3,2))
    def get_tile3_abs(self):
        return Tile((0,0),(1,0),(-1,-1),(3,2))
    
    def test_convert_relative_to_absolute1(self):
        tile = self.get_tile1()
        tile.change_position_to_relative(tile.my_pos)
        self.assertTrue(tile == self.get_tile1_abs())
        
    def test_convert_relative_to_absolute2(self):
        tile = self.get_tile2()
        tile.change_position_to_relative(tile.my_pos)
        self.assertTrue(tile == self.get_tile2_abs())
        
    def test_convert_relative_to_absolute3(self):
        tile = self.get_tile3()
        tile.change_position_to_relative(tile.my_pos)
        self.assertTrue(tile == self.get_tile3_abs())
        
    def test_copy_tile1(self):
        tile = self.get_tile1()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        tc = copy_tile(tile,(1,0))
        #tc = copy_tile(tc,(1,0))
        try:
            assert tc.my_pos == (0,6)
            assert tc.enemy_pos == (2,5)
            assert tc.corner_pos == (-1,3)
        except AssertionError:
            print("Expected :")
            print("my_pos : (0,-2), received : {}".format(tc.my_pos))
            print("enemy_pos : (2,-3), received : {}".format(tc.enemy_pos))
            print("corner_pos : (-1,-5), received : {}".format(tc.corner_pos))
            self.assertTrue(False)
    
    def test_copy_tile11(self):
        tile = self.get_tile1()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        tc = copy_tile(tile,(-1,0))
        tc = copy_tile(tc,(0,-1))
        tc = copy_tile(tc,(0,-1))
        tc = copy_tile(tc,(-1,0))
        self.assertTrue(tc.my_pos == (-8,-8))
        self.assertTrue(tc.enemy_pos == (-6,-7))
        self.assertTrue(tc.corner_pos == (-9,-9))
    
    def test_copy_tile2(self):
        tile = self.get_tile2()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        tc = copy_tile(tile,(0,-1))
        tc = copy_tile(tc,(0,-1))
        #tc = copy_tile(tc,(0,-1))
        self.assertTrue(tc.my_pos == (0,-8))
        self.assertTrue(tc.enemy_pos == (-1,-9))
        self.assertTrue(tc.corner_pos == (-2,-10))
        
    def test_create_none_array1(self):
        tile = self.get_tile1()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        distance = 6
        a = create_none_array(tile.size,distance,relative_pos(tile.corner_pos, tile.my_pos))
        xd = len(a[0])
        yd = len(a)
        try:
            assert xd == 4
            assert yd == 4
        except AssertionError:
            for _ in a:
                print(_)
            print("xd : {} not equal to 4".format(xd))
            print("yd : {} not equal to 4".format(yd))
            self.assertTrue(False)
    
    def test_create_none_array2(self):
        tile = self.get_tile2()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        distance = 6
        a = create_none_array(tile.size,distance,relative_pos(tile.corner_pos, tile.my_pos))
        xd = len(a[0])
        print(a)
        yd = len(a)
        try:
            assert xd == 3
            assert yd == 3
        except AssertionError:
            for _ in a:
                print(_)
            print("xd : {} not equal to 3".format(xd))
            print("yd : {} not equal to 3".format(yd))
            self.assertTrue(False)
            
    def test_create_none_array3(self):
        tile = self.get_tile3()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        distance = 4
        a = create_none_array(tile.size,distance,relative_pos(tile.corner_pos, tile.my_pos))
        xd = len(a[0])
        yd = len(a)
        #insert_center_tile_inplace(a, self.get_tile3())
        try:
            assert xd == 3
            assert yd == 5
        except AssertionError:
            for _ in a:
                print(_)
            print("xd : {} not equal to 3".format(xd))
            print("yd : {} not equal to 5".format(yd))
            self.assertTrue(False)
            
    def test_create_none_array4(self):
        tile = Tile((1,1),(2,2),(0,0),(3,3))
        tile.change_position_to_relative(new_origin = tile.my_pos)
        distance = 5
        a = create_none_array(tile.size,distance,relative_pos(tile.corner_pos, tile.my_pos))
        xd = len(a[0])
        yd = len(a)
        try:
            assert xd == 4
            assert yd == 4
        except AssertionError:
            for _ in a:
                print(_)
            print("xd : {} not equal to 4".format(xd))
            print("yd : {} not equal to 4".format(yd))
            self.assertTrue(False)
            
    def test_insert_center_tile_inplace1(self):
        tile = self.get_tile1()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        distance = 6
        a = create_none_array(tile.size,distance,relative_pos(tile.corner_pos, tile.my_pos))
        row,col = insert_center_tile_inplace(a, tile)
        try:
            assert col == 2
            assert row == 1
        except AssertionError:
            print("col : {} not equal to 3".format(col))
            print("row : {} not equal to 2".format(row))
            self.assertTrue(False)
            
    def test_insert_center_tile_inplace2(self):
        tile = self.get_tile2()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        distance = 6
        a = create_none_array(tile.size,distance,relative_pos(tile.corner_pos, tile.my_pos))
        row,col = insert_center_tile_inplace(a, tile)
        try:
            assert col == 1
            assert row == 1
        except AssertionError:
            print("col : {} not equal to 3".format(col))
            print("row : {} not equal to 2".format(row))
            self.assertTrue(False)
            
    def test_insert_center_tile_inplace3(self):
        tile = self.get_tile3()
        tile.change_position_to_relative(new_origin = tile.my_pos)
        distance = 4
        a = create_none_array(tile.size,distance,relative_pos(tile.corner_pos, tile.my_pos))
        row,col = insert_center_tile_inplace(a, tile)
        try:
            assert col == 1
            assert row == 2
        except AssertionError:
            print("col : {} not equal to 1".format(col))
            print("row : {} not equal to 2".format(row))
            self.assertTrue(False)
    
    def test_insert_center_tile_inplace4(self):
        tile = Tile((1,1),(2,2),(0,0),(3,3))
        tile.change_position_to_relative(new_origin = tile.my_pos)
        distance = 5
        a = create_none_array(tile.size,distance,relative_pos(tile.corner_pos, tile.my_pos))
        row,col = insert_center_tile_inplace(a, tile)
        try:
            assert len(a[0]) == 4
            assert len(a) == 4
            assert col == 2
            assert row == 1
        except AssertionError:
            print("columns : {} not equal to 4".format(len(a[0])))
            print("rows : {} not equal to 4".format(len(a)))
            print("col : {} not equal to 2".format(col))
            print("row : {} not equal to 1".format(row))
            self.assertTrue(False)
        
        
        
        
if __name__ == '__main__':
    unittest.main()
    