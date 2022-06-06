import math
from class_foobar42 import Shot, Room, vector_length, round_if_close, to_minimal_path
from foobar42_tiles import *

def generate_initial_directions(distance):
    """Generate distinct shooting directions that are integer pairs (vectors), whose length < distance"""
    i = 4
    yield (0,1)
    yield (0,-1)
    yield (1,0)
    yield (-1,0)
    for x_direction in range(1,distance+1):
        ymax = int(math.sqrt(distance**2 - x_direction**2))
        for y_direction in range(1,ymax+1):
            #if vector_length([x_direction,y_direction]) > distance:
            #    continue
            if math.gcd(x_direction,y_direction) == 1:
                i += 4
                yield (x_direction,y_direction)
                yield (x_direction,-y_direction)
                yield (-x_direction,-y_direction)
                yield (-x_direction,y_direction)
                
def solutions_with_n_bounces2(n,shot):
    direction_gen = generate_initial_directions(shot.room.distance)
    hit_count = 0
    directions = []
    for i,d in enumerate(direction_gen):
        shot.shoot(d,allowed_bounces=n)
        if shot.hits:
            hit_count += 1
            directions.append(d)
            #shot.plot_path(show = False)
    return hit_count,sorted(directions,key=lambda x : x[0])

def solution_with_directions(shot):
    dirs = generate_initial_directions(shot.room.distance)
    hit_count = 0
    for d in dirs:
        shot.shoot(d)
        if shot.hits:
            #shot.plot_path(show=True)
            hit_count += 1
    return hit_count

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
        room = Room(**case)
        shot = Shot(room,create_path=True)
        hc = solution_with_directions(shot)
        print("case",n,"Hit count:",hc)
        
        tile_ans = solution(case["bounds"], case["my_pos"], case["enemy_pos"], case["distance"])
        print("Answer with tile method:",tile_ans)
        dirgen = generate_directions(case["bounds"], case["my_pos"], case["enemy_pos"], case["distance"])
        hc = 0
        for d in dirgen:
            hit = shot.shoot(d)
            if hit:
                hc += 1
        print("Hit count with directions:",hc)
            
    #print("Directions:",L)
