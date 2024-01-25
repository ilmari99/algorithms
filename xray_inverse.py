from typing import Callable, Iterable, SupportsIndex
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import math
from abc import ABC, abstractmethod

"""
This is a short script I started doing to attempt to model where are holes
in an object by rotating it and seeing how much radiation is absorbed in which part.
"""


class AbsorptionMatrix(ABC):
    size = (0,0)
    center = (0,0)
    matrix = None
    fill = 0
    rotation_degree = 0
    absorption = 0
    
    def __init__(self,size : SupportsIndex,fill=0):
        print(f"Creating a matrix of size {size}")
        #assert isinstance(size,SupportsIndex), "size must support indexing."
        assert size[0] == size[1], "Absorption matrix must be a square."
        assert len(size) == 2, "Only 2D arrays are supported"
        self.size = size
        self.center = (size[0]//2,size[1]//2)
        self.fill = fill
        self.matrix = self.init_matrix(size=size,fill=fill)
        self.matrix = self.add_absorption()
    
    @abstractmethod
    def add_absorption(self):
        """ This method adds absorption to self.matrix, by generating tuples of (row,col,absorption)
        """
        pass
    
    def init_matrix(self, size, fill=0):
        matrix = np.full(size,fill,dtype=float)
        return matrix
    
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
    def __init__(self,radius,absorption=1):
        self.radius = radius
        self.init_absorption(absorption)
        #super creates the matrix, and adds absorption to it using the add_absorption method.
        super().__init__(size = [2*radius + 1, 2*radius + 1], fill = 0)
        
    
    def add_absorption(self):
        """ Add a circle of absorption to the matrix.
        """
        # Create a circle with radius self.radius
        mat = self.matrix
        # Add a filled circle to the matrix
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                if self.is_inside_circle((row,col)):
                    mat[row,col] = self.absorption()
        return mat
    
    def is_inside_circle(self,point):
        """ Returns True if the point is inside the circle, False otherwise.
        """
        # Get the distance from the center of the circle to the point
        dist = math.sqrt((point[0]-self.center[0])**2 + (point[1]-self.center[1])**2)
        # If the distance is less than the radius, the point is inside the circle
        return dist <= self.radius
        
    
    def init_absorption(self,absorption):
        if isinstance(absorption,(int,float)):
            self.absorption = lambda : absorption
        elif isinstance(absorption,Callable):
            self.absorption = absorption
        else:
            raise TypeError("Absorption must be an integer, float or a callable distribution.")
    
    def get_parsed_measurement(self):
        raise NotImplementedError("This method is not used.")
        measurements = self.get_measurement(self.matrix)
        parsed_measurements = []
        for measurement,cwidth in zip(measurements,self.row_circle_width):
            parsed_measurements.append(round(math.log(((1-self.absorption())**cwidth / measurement),(1-self.absorption()))))
        #assert all([pm >= 0 for pm in parsed_measurements])
        return parsed_measurements
            
    
    def make_holes(self,hole_sz,n_holes):
        """ Makes n_holes of size hole_sz to the circle in random locations where absorption is not 0.
        So sets the value of the hole to 0.
        """
        for ith_hole in range(n_holes):
            hole_lu_corner = (random.randint(0,self.size[0]-hole_sz[0]),random.randint(0,self.size[1]-hole_sz[1]))
            # Set the values at indices to 0
            self.matrix[hole_lu_corner[0]:hole_lu_corner[0]+hole_sz[0],hole_lu_corner[1]:hole_lu_corner[1]+hole_sz[1]] = 0
        return
    
    def get_measurement(self,theta : float):
        """ Returns the total absorption at each height of the circle in some angle theta.
        """
        # Rotate the matrix
        rotated_mat = self.rotate(theta)
        # Get the row sums of the rotated matrix
        return rotated_mat.sum(axis=1)


circle = Circle(128)

# Get the measurements at 0 degrees
measurements = circle.get_measurement(0)

# Make holes in the circle
circle.make_holes(hole_sz = (20,20),n_holes=10)
circle.plot()

fig, ax = plt.subplots()
for theta in range(0,180,10):
    # Get the measurements at 45 degrees
    measurements = circle.get_measurement(theta)
    ax.plot(measurements,label="theta={}".format(theta))
    
ax.legend()
plt.show()






