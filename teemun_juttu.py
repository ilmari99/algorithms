from typing import Callable, Iterable, SupportsIndex
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import math

class AbsorptionMatrix:
    size = (0,0)
    center = (0,0)
    matrix = None
    fill = 0
    rotation_degree = 0
    absorption = 0
    
    def __init__(self,size : SupportsIndex,fill=0):
        #assert isinstance(size,SupportsIndex), "size must support indexing."
        assert size[0] == size[1], "Absorption matrix must be a square."
        assert len(size) == 2, "Only 2D arrays are supported"
        self.size = size
        self.center = (size[0]/2,size[1]/2)
        self.fill = fill
        self.matrix = self.init_matrix(size=size,fill=fill)
    
    @classmethod
    def add_absorption_from_dict(cls,matrix : np.ndarray, changes : dict):
        for rc,v in changes:
            row = rc[0]
            col = rc[1]
            matrix[row,col] = v
        return matrix
    
    @classmethod
    def init_matrix(cls, size, fill=0):
        matrix = np.full(size,fill,dtype=float)
        return matrix
    
    
    @classmethod
    def add_absorption_from_iterable(cls, matrix : np.ndarray, changes : Iterable):
        for r,c,v in changes:
            matrix[r,c] = v
        return matrix
    
    @classmethod
    def generate_random_absorption(cls, matrix : np.ndarray, max_absorption : float = 1, npoints : int = None):
        size = matrix.shape
        npoints = random.randint(0,size[0]) if npoints is None else npoints
        rows = random.sample(range(size[0]),npoints)
        cols = random.sample(range(size[1]),npoints)
        changes = {}
        for r,c in zip(rows,cols):
            yield r,c,random.random()*max_absorption
        return
    
    @classmethod
    def get_random_absorption(cls, matrix : np.ndarray, max_absorption : float = 1, npoints: int = None):
        changes = {}
        for r,c,v in cls.generate_random_absorption(matrix,max_absorption,npoints):
            changes[tuple(r,c)] = v
        return changes
    
    @classmethod
    def add_random_absortion(cls,matrix,max_absorption=1,npoints=None):
        matrix = cls.add_absorption_from_iterable(matrix,cls.generate_random_absorption(matrix,max_absorption,npoints))
        return matrix
    
    @classmethod
    def gen_holes(cls, matrix, hole_sz=(3,3), n_holes = 5):
        for i in range(n_holes):
            row = np.random.randint(0,matrix.shape[0]-hole_sz[0])
            rows = [row+n for n in range(hole_sz[0])]
            col = np.random.randint(0,matrix.shape[0]-hole_sz[1])
            cols = [col+n for n in range(hole_sz[1])]
            for row in rows:
                for col in cols:
                    yield row,col,0
        return
    
    @classmethod
    def get_measurement(self,matrix):
        measures = []
        for ir,row in enumerate(matrix):
            prod = 1
            for c in row:
                prod *= (1 - c)
            measures.append(prod)
        measures = np.array(measures)
        return measures
    
    def rotate(self,angle, spline_order=0, inplace=False):
        mat = self.matrix if inplace else self.matrix.copy()
        mat = rotate(mat,angle,reshape=False,order=spline_order)
        if inplace:
            self.rotation_degree += angle
            self.matrix = mat
        return mat
    
    def plot(self, show=False):
        fig, ax = plot_mat(self.matrix,"rotated by {} degrees".format(self.rotation_degree))
        if show:
            plt.show()
        return fig,ax
    
def plot_mat(mat,title=""):
    fig,ax = plt.subplots()
    ax.matshow(mat)
    ax.set_title(title)
    return fig,ax

class Circle(AbsorptionMatrix):
    radius = 0
    absorption = 0.1
    row_circle_width = []
    def __init__(self,radius,absorption=0.1):
        self.radius = radius
        self.init_absorption(absorption)
        super().__init__(size = [2*radius + 1, 2*radius + 1], fill = 0)
        circle_points = list(self.generate_circle_absorption())
        self.matrix = self.add_absorption_from_iterable(self.matrix,circle_points)
    
    def init_absorption(self,absorption):
        if isinstance(absorption,(int,float)):
            self.absorption = lambda : absorption
        elif isinstance(absorption,Callable):
            self.absorption = absorption
        else:
            raise TypeError("Absorption must be an integer, float or a callable distribution.")
        
    def generate_circle_absorption(self):
        r = self.radius
        self.row_circle_width = np.zeros(self.size[0])
        for ir,row in enumerate(self.matrix):
            for ic,col in enumerate(row):
                if abs(ic - r) < round(np.sqrt((r)**2 - (r - ir)**2)) and abs(ir-r) < round(np.sqrt((r)**2 - (r - ic)**2)):
                    self.row_circle_width[ir] += 1
                    yield ir,ic,self.absorption()
        return
    
    def get_parsed_measurement(self):
        measurements = self.get_measurement(self.matrix)
        parsed_measurements = []
        for measurement,cwidth in zip(measurements,self.row_circle_width):
            parsed_measurements.append(round(math.log(((1-self.absorption())**cwidth / measurement),(1-self.absorption()))))
        #assert all([pm >= 0 for pm in parsed_measurements])
        return parsed_measurements
            
    
    def make_holes(self,hole_sz,n_holes):
        self.matrix = self.add_absorption_from_iterable(self.matrix,self.gen_holes(self.matrix,hole_sz=hole_sz,n_holes=n_holes))
        return self.matrix
    


circle = Circle(12, absorption=0.1)
circle.plot()
circle.make_holes(hole_sz = (3,3),n_holes=4)
print(circle.get_parsed_measurement())
circle.plot()
circle.rotate(45,inplace=True)
print(circle.get_parsed_measurement())
circle.plot(show=True)




