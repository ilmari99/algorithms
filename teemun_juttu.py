import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

class Circle:
    absorption = 1
    center = ()
    circle_mat = None
    empty_grids = []
    degree = 0
    def __init__(self, radius, hole_sz = None, nholes=0,absorption=1):
        """Create a new Circle instance. A circle instance contains an absorption matrix with values in the shape of
        a circle with square pieces missing (values 0). Each value in the matrix corresponds to an absorption coefficient.
        The absortion coefficient absorbs (1-absorption) of the beam.

        Args:
            radius (int): The radius of the absorption circle
            hole_sz (tuple of 2 ints): The size of the holes in the absorption matrix. Defaults to None.
            nholes (int, optional): Maximum number of holes to make in the matrix. Defaults to 0.
            absorption (int, optional): The absorption. Defaults to 1.

        Returns:
            _type_: _description_
        """
        self.absorption = absorption
        self.circle_mat = Circle.get_circle(radius, absorption=absorption)
        self.center = (radius,radius)
        #self.circle_mat = self.trim_zeros(axis=1)
        if hole_sz != None:
            self._make_holes(hole_sz=hole_sz, n_holes=nholes)
        return None
        
    @classmethod
    def get_circle(cls,r,absorption=1):
        c = np.zeros((2*r+1,2*r+1))
        for ir,row in enumerate(c):
            for ic,col in enumerate(row):
                if abs(ic - r) < round(np.sqrt((r)**2 - (r - ir)**2)):
                    c[ir,ic] = absorption
        return c

    def _make_holes(self, hole_sz=(3,3), n_holes = 5):
        circle = self.circle_mat
        for i in range(n_holes):
            row = np.random.randint(0,circle.shape[0]-hole_sz[0])
            rows = [row+n for n in range(hole_sz[0])]
            col = np.random.randint(0,circle.shape[0]-hole_sz[1])
            cols = [col+n for n in range(hole_sz[1])]
            for row in rows:
                for col in cols:
                    self.empty_grids.append((row,col))
                    circle[row,col] = 0
        return circle
    
    def rotate(self,angle,inplace=False):
        circle = self.circle_mat if inplace else self.circle_mat.copy()
        circle = rotate(circle, angle,reshape=False,order=1)
        if inplace:
            self.circle_mat = circle
            self.degree += angle
        return circle
        
    def show(self,mat=None):
        fig,ax = plt.subplots()
        mat = mat if mat is not None else self.circle_mat
        ax.matshow(mat)
        ax.set_title("rotated by {} degrees".format(self.degree))
        
    def trim_zeros(self,axis=0,_flip=True):
        mat = self.circle_mat
        if axis==1:
            mat = mat.T
        for nr, r in enumerate(mat):
            if any((v>float(0) for v in r)):
                break
        mat = np.delete(mat,range(nr),axis=axis)
        mat = np.flipud(mat)
        self.circle_mat = mat
        if _flip:
            mat = self.trim_zeros(_flip=False)
        return mat
    
    def get_measurement(self):
        measures = []
        for ir,row in enumerate(self.circle_mat):
            prod = 1
            for c in row:
                prod *= (1 - c)
            measures.append(prod)
        measures = np.array(measures)
        return measures
        
 

c = Circle(12, hole_sz=(4,4),nholes=6,absorption=0.1)
print(c.circle_mat)
c.show()
print(c.get_measurement())
c.rotate(90,inplace=True)
c.show()
print(c.get_measurement())
c.rotate(45,inplace=True)
c.show()
c.rotate(90,inplace=True)
c.show()
plt.show()

