import numpy as np
import random
import matplotlib.pyplot as plt

def get_circle(r):
    c = np.zeros((int(np.ceil(2*r)),int(np.ceil(2*r))))
    for ir,row in enumerate(c):
        for ic,col in enumerate(row):
            if abs(ic - r) < round(np.sqrt((r)**2 - (r - ir)**2)):
                c[ir,ic] = 1
    return c

def make_holes(circle, hole_sz=(3,3), n_holes = 5):
    for i in range(n_holes):
        #rows = random.sample(range(0,circle.shape[0]),hole_sz[0])
        row = np.random.randint(0,circle.shape[0]-hole_sz[0])
        rows = [row+n for n in range(hole_sz[0])]
        col = np.random.randint(0,circle.shape[0]-hole_sz[1])
        cols = [col+n for n in range(hole_sz[1])]
        #cols = random.sample(range(0,c.shape[1]),max_hole_sz[1])
        for row in rows:
            for col in cols:
                c[row,col] = 0
    return c
    

c = get_circle(512)
print(c)
c = make_holes(c,hole_sz=(20,20))
plt.matshow(c)
plt.show()