# Defines Classes for Use in the Selenkay Diversity Project
# PB 10/03/2022
# Build on top of cloud Class used in the Buffalo Camp Paper 

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import time
import laspy
import concurrent.futures


# Point Cloud Class
# Used for voxelization
class Cloud: 
    
    def __init__(self,
                 lasf=None,
                 metrics={},
                 gridsize=1,
                 vsize=0.25,
                 heightcol='Height',
                 numretcol='number_of_returns',
                 retnumcol='return_number',
                 classcol='classification',
                 maxh=None): 
        
        self.lasf = lasf
        
        self.las = laspy.read(lasf)
        
        # initialize empty metrics dictionary
        self.metrics_dict = metrics
        
        # Set col names
        # NOTE: in future use self.las.point_format strings to regex these
        self.heightcol = heightcol
        self.numretcol = numretcol
        self.retnumcol = retnumcol
        self.classcol = classcol
        
        # Set grid and vert sizes
        self.gridsize = gridsize
        # Vertical res for foliage and cover profiles
        self.vsize = vsize

    # Make a grid with features defining each pixel
    def makegrid(self, xmin = None, xmax = None, ymin = None, ymax = None): 
        
        # If no grid boundaries given as input
        if not xmin:
            # Use the bounds of the lasfile to set bounds of the grid
            self.xmin, self.ymin, self.xmax, self.ymax = np.min(self.las.x), np.min(self.las.y), np.max(self.las.x), np.max(self.las.y)
        else:
            # else, use the boundaries of the input grid
            self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        
        # Build a grid of cells
        # by using floor to snap all x and y coords to a grid
        # divide all x and y values by cell size,
        # Subtract boundary coords (minx, miny) to snap to grid
        # floor them to get vertex coordinate of cell (lower left)
        # then multiply by size again and add mins to turn back into UTM coords
        x_vert = np.floor((self.las.x - self.xmin)/self.gridsize)*self.gridsize + self.xmin
        y_vert = np.floor((self.las.y - self.ymin)/self.gridsize)*self.gridsize + self.ymin

        # use np.unique to find unique combinations of x and y 
        # and record their indices (for later operations)
        # can take a bit, but took less than 30 sec for 15 million points with 5m grid size
        # Numpy unique docs:
        # https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        xy_unique, idx = np.unique([x_vert, y_vert], axis=1, return_inverse=True)

        # unique indices for each cell 
        idx_unique = np.unique(idx)
        
        # output a dictionary
        self.grid_dict = {'x_cells': xy_unique[0],
                          'y_cells': xy_unique[1],
                          'idx_cells': idx_unique,
                          'idx_points': idx
                         }

        # NOTE: To run a function over all cells
        # just need to use idx_cells and idx_points in a loop:
        # for idx in grid_dict['idx_cells']: 
        #    cellsubset = las.points[idx_points == idx]
        #    do function(cellsubset)
    