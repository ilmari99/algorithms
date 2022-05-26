import unittest
import math
from foobar42 import *
class test_foobar42(unittest.TestCase):
    def test_vector_length1(self):
        vec = [1,1]
        self.assertAlmostEqual(vector_length(vec), math.sqrt(2))
    def test_vector_length2(self):
        vec = [1,0]
        self.assertAlmostEqual(vector_length(vec), 1)
        
    def test_max_step_lens1(self):
        laser_pos = [1,1]
        direction = [1,1]
        self.assertEqual(max_step_lens(direction,laser_pos),[1,1])
        
    def test_max_step_lens2(self):
        laser_pos = [0,1.5]
        direction = [0.1,0.1]
        self.assertEqual(max_step_lens(direction,laser_pos),[1,0.5])
        
    def test_max_step_lens3(self):
        laser_pos = [5/3,2]
        direction = [1,-3]
        pred = max_step_lens(direction,laser_pos)
        ans = [1/3,-1]
        self.assertTrue(all([math.isclose(p,a) for p,a in zip(pred,ans)]))
        
    def test_check_new_direction_raises_on_signs(self):
        direction = [1,1]
        new_direction = [-1,1]
        self.assertRaises(AssertionError,check_new_direction,direction,new_direction)
    
    def test_check_new_direction_raises_on_ratio(self):
        direction = [1,1]
        new_direction = [1,0.9]
        self.assertRaises(AssertionError,check_new_direction,direction,new_direction)
    
    def test_check_new_direction_doesnt_raise(self):
        direction = [1,1]
        new_direction = [1,1]
        self.assertEqual(check_new_direction(direction,new_direction),None)
        
    def test_create_step_to_direction1(self):
        laser_pos = [1,1]
        direction = [1,1]
        bounds = [3,2]
        self.assertEqual(create_step_to_direction(direction,laser_pos,bounds),[1,1])
        
    def test_create_step_to_direction2(self):
        laser_pos = [0,1/3]
        direction = [1,-2/3]
        bounds = [3,2]
        ans = create_step_to_direction(direction,laser_pos,bounds)
        ans = [round(x,7) for x in ans]
        corr_ans = [round(x,7) for x in [0.5,-1/3]]
        self.assertListEqual(ans, corr_ans)
        
    def test_create_step_to_direction3(self):
        laser_pos = [3,3]
        direction = [0,1]
        bounds = [5,5]
        ans = create_step_to_direction(direction,laser_pos,bounds)
        corr_ans = [0,1]
        self.assertListEqual(ans,corr_ans)
        
    def test_create_step_to_direction4(self):
        laser_pos = [2,2]
        direction = [1,1]
        bounds = [3,3]
        ans = create_step_to_direction(direction, laser_pos, bounds)
        corr_ans = [1,1]
        self.assertListEqual(ans, corr_ans)
        
    def test_hits_walls1(self):
        laser_pos = [1,1]
        bounds = [3,2]
        ans = hits_walls(laser_pos,bounds)
        corr_ans = [False]*4
        self.assertListEqual(ans, corr_ans)
        
    def test_hits_walls2(self):
        laser_pos = [0,2]
        bounds = [3,2]
        ans = hits_walls(laser_pos,bounds)
        corr_ans = [False, True,True,False]
        self.assertListEqual(ans, corr_ans)
        
    def test_hits_walls3(self):
        laser_pos = [4,4]
        bounds = [5,4]
        ans = hits_walls(laser_pos,bounds)
        corr_ans = [False, False, True, False]
        self.assertListEqual(ans, corr_ans)
    
    def test_hits_walls4(self):
        laser_pos = [3,0]
        bounds = [3,2]
        ans = hits_walls(laser_pos,bounds)
        corr_ans = [True,False,False,True]
        self.assertListEqual(ans, corr_ans)
        
    def test_cal_new_direction1(self):
        wall_hits = [False, False, True, False]
        direction = [1,1]
        ans = cal_new_direction(wall_hits,direction)
        corr_ans = [1,-1]
        self.assertListEqual(ans, corr_ans)
        
    def test_cal_new_direction2(self):
        wall_hits = [False,True,False,False]
        direction = [-0.5,0]
        ans = cal_new_direction(wall_hits,direction)
        corr_ans = [0.5,0]
        self.assertListEqual(ans,corr_ans)
        
    def test_cal_new_direction3(self):
        wall_hits = [True,True,False,False]
        direction = [-1,-1]
        ans = cal_new_direction(wall_hits, direction)
        corr_ans = [1,1]
        self.assertListEqual(ans, corr_ans)
    
    def test_cal_new_direction4(self):
        wall_hits = [True,False,False,True]
        direction = [1/3,-1]
        ans = cal_new_direction(wall_hits, direction)
        corr_ans = [-1/3,1]
        self.assertListEqual(ans, corr_ans)
        
    def test_fire_to_direction1(self):
        """Fire to [1,0] and hit target
        """
        direction = [1,0]
        dimensions = [3,2]
        your_position = [1,1]
        trainer_position = [2,1]
        distance = 4
        ans = fire_to_direction(direction,dimensions,your_position,trainer_position,distance)
        self.assertTrue(ans)
    
    def test_fire_to_direction2(self):
        """Fire to [1,1] and hit target
        """
        direction = [1,1]
        dimensions = [3,2]
        your_position = [1,1]
        trainer_position = [2,2]
        distance = 4
        ans = fire_to_direction(direction,dimensions,your_position,trainer_position,distance)
        self.assertTrue(ans)
    def test_fire_to_direction3(self):
        """Fire to [1,1] and miss target
        """
        direction = [1,1]
        dimensions = [4,4]
        your_position = [1,1]
        trainer_position = [2,3]
        distance = 4
        ans = fire_to_direction(direction,dimensions,your_position,trainer_position,distance)
        self.assertFalse(ans)
        
    def test_fire_to_direction4(self):
        direction = [3/2,3]
        dimensions = [4,4]
        your_position = [1,1]
        trainer_position = [2,3]
        distance = 5
        ans = fire_to_direction(direction,dimensions,your_position,trainer_position,distance)
        self.assertTrue(ans)
        
    def test_fire_to_direction5(self):
        directions = [[1, 0], [1, 2], [1, -2], [3, 2], [3, -2], [-3, 2], [-3, -2]]
        dimensions = [3,2]
        your_position = [1,1]
        trainer_position = [2,1]
        distance = 4
        corrs = []
        for d in directions:
            ans = fire_to_direction(d,dimensions,your_position,trainer_position,distance)
            corrs.append(ans)
        for ans,case in zip(corrs,directions):
            if not ans:
                print("####################",case)
        self.assertTrue(all(corrs))
    
    
    
if __name__ == "__main__":
    unittest.main()
        