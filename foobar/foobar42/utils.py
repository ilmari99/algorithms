import random
import math

def generate_case(**kwargs):
    """Generates a case, with either the default parameters or the ones given in kwargs

    Returns:
        tuple: (dimensions, your_position, trainer_position, distance)
    """
    dimensions = kwargs.get("dimensions", [random.randint(3,1250), random.randint(3,1250)])
    your_position = kwargs.get("your_position", [random.randint(1,dimensions[0]-1), random.randint(1,dimensions[1]-1)])
    trainer_position = kwargs.get("trainer_position", [random.randint(1,dimensions[0]-1), random.randint(1,dimensions[1]-1)])
    distance = kwargs.get("distance", random.randint(2,10000))
    while your_position == trainer_position:
        trainer_position = [random.randint(1,dimensions[0]-1), random.randint(1,dimensions[1]-1)]
    assert 1 < distance <= 10000
    assert 1 < dimensions[0] <= 1250
    assert 1 < dimensions[1] <= 1250
    assert 0 < your_position[0] < dimensions[0]
    assert 0 < your_position[1] < dimensions[1]
    assert 0 < trainer_position[0] < dimensions[0]
    assert 0 < trainer_position[1] < dimensions[1]
    return (dimensions, your_position, trainer_position, distance)


def isclose(a,b,rel_tol=1e-09,abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def round_if_close(L):
    """Rounds a list of floats to the nearest integer if they are close to an integer
    """
    return [round(x) if isclose(x,round(x),abs_tol=10**(-12)) else x for x in L]   

def relative_pos(wrt,pos):
    return [pos[0]-wrt[0],pos[1]-wrt[1]]

def vector_length(vector):
    return math.sqrt(sum([v**2 for v in vector]))