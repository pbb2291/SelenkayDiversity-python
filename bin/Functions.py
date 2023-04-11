# Functions for Selenkay Diversity Project
# Pb 9/29/2022

import laspy
import pdal
import json
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
import time
import glob
import os
import matplotlib.pyplot as plt

# makes matplotlib plots big
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams.update({'font.size': 14})

# Quality check for las clipping
# make sure there aren't any LAS outputs already
# If there are, point will be appended to each file 
def lasQualityCheckExistingFiles(od):
    
    lfs = [l for l in od.glob('*.las')]

    if len(lfs) > 0:

        print(f'WARNING: Output las files already found in directory: \n \t{od} \n')
        print('To avoid overwrite issues, delete all files in output directory before proceeding.\n')


# Checks a shapefile for non-polygon features
# Removes them if found.
def shpQualityCheckPolys(shpdf):
    
    # Make a copy for testing 
    shpdf_test = shpdf.copy(deep=True)
    shpdf_test.head()

    # Label all rows with multipolygons
    shpdf_test['NotPoly'] = shpdf_test.geometry.apply(lambda x: x.type != 'Polygon')

    if shpdf_test.loc[shpdf_test['NotPoly']].size > 0:

        numnonpolys = shpdf_test.loc[shpdf_test['NotPoly']].size[0]

        print(f'{numnonpolys} non-polygon features found. \n')

        q = input('Discard non-polygon features and continue? y\n')

        if q == 'y':

            # Filter it to only include Polygon features
            shpdf = shpdf.query('NotPoly == False')

        else: 

            print('Provide a new shapefile with only polygons and restart process.\n')
    
    return shpdf

# Checks if there is a unique ID for each polygon feature in a shapefile
# Otherwise, you can get multiple polygons, overwriting issues, etc.
# Inputs: Shapefile name and path, shapefile dataframe
# Outputs: Shapefile Dataframe (potentially, edited if duplciates are found)
def shpQualityCheckDupes(shpf, shpdf, featureIDcol):
    
    # Check for duplicate ids by filtering
    shpdf_nodupes = shpdf[featureIDcol].drop_duplicates()

    # If there are duplicates
    if shpdf_nodupes.shape[0] < shpdf.shape[0]:

        numberofdupes = shpdf.shape[0] - shpdf_nodupes.shape[0]

        q = input(f"{numberofdupes} duplicate IDs found.\n\tMake new attribute with unique Feature IDs and continue? y/n \n")

        if q == "y":

            # sort the file by the original index
            shpdf.sort_values(by=featureIDcol, inplace=True)

            # Make a new column with a unique index (row number) to identify each feature with
            shpdf['FeatureID'] = shpdf.index

            # Set the featureIDcol value to be this new column:
            featureIDcol = 'FeatureID'

            # save new shapefile for future reference
            # make the new shapefile name
            oshp_name = shpf.name.split('.')[0] + '_NewFeatureID.shp'

            # set the output folder to be in same as the input file
            newshpf = Path(str(shpf.parent) + oshp_name)

            # Save it
            shpdf.to_file(str(newshpf))

            print(f'New shapefile {newshpf.name} with FeatureID column saved in {newshpf.anchor}/ \n')

        if q == "n":

            print('Operation cancelled. Provide a new shapefile with unique feature IDs.')
            
    return shpdf

# # # 
# # #
# The below section of functions are for preprocessing tiles of lidar data.

# Define normalize height function
# Uses HAG nearest neighbors instead of HAG delaunay 
# Even those this seems to be more prone to errors in ground,
# ie. sensitive to spots where the ground points jump up into the vegetation
# it does not throw colinear errors
# for consistency, run it on all point clouds (not just the ones that throw colinear errors)
# https://pdal.io/en/stable/stages/filters.hag_nn.html?highlight=hag_nn
# implemented 10/21/2022
# Note: added a "buffer" option
# this would be the distance that needs to be clipped from the edge of a las file
# for instance, if las tiles overlap by 20 m
# then the buffer value would be 20/2 = 10 m
# NOTE: the buffer value here assumes an input of rectangular tiles with equal buffers on all sides
# PB 10/03/22
def calcLasHeight_HagNN(inf, opath=None, of=None, buffer=0):
    
    inf = Path(str(inf))

    # If no outfile name or dir, make one in the same dir
    if not of:

        of = inf.name
        
    if not opath:
        
        opath = inf.parent
        
    else:
        
        opath = Path(opath)
    
    # If there's no buffer to clip
    if buffer==0:
        
        of = Path(str(opath) + '/' + inf.name.split('.')[0] + "_Height.las")
        
        # define injson for pdal pipeline
        injson= """
        [
            "in.las",
            {
                "type":"filters.hag_nn",
                "count": 10,
                "allow_extrapolation": true
            },
            {
                "type": "writers.las",
                "filename": "out.las",
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
        """
        
    # else: if there is a buffer to clip
    else:
        
        of = Path(str(opath) + '/' + inf.name.split('.')[0] + "_Height.las")
        
        # open the header of the file in laspy
        # to get coordinates boundaries from the header
        l = laspy.open(str(inf))
        
        # Take away the buffer region from the boundary
        xmin = l.header.mins[0] + buffer
        xmax = l.header.maxs[0] - buffer
        ymin = l.header.mins[1] + buffer
        ymax = l.header.maxs[1] - buffer
        
        # define injson for pdal pipeline
        injson= """
        [
            "in.las",
            {
                "type":"filters.hag_nn",
                "count": 10,
                "allow_extrapolation": true
            },
            {
                "type":"filters.crop",
                "bounds":"([xmin,xmax],[ymin,ymax])"
            },
            {
                "type": "writers.las",
                "filename": "out.las",
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
        """
        
        # Replace bounds with values of xmin, xmax, ymin, ymax
        # clipped by buffer
        injson = injson.replace("xmin", str(xmin))
        injson = injson.replace("xmax", str(xmax))
        injson = injson.replace("ymin", str(ymin))
        injson = injson.replace("ymax", str(ymax))
        
    # Replace args with in and out file
    injson = injson.replace("in.las", str(inf))
    injson = injson.replace("out.las", str(of))

    pipeline = pdal.Pipeline(injson)
    pipeline.execute()
    # arrays = pipeline.arrays
    metadata = pipeline.metadata
    # log = pipeline.log
    
    return metadata

# Define normalize height function
# Note: added a "buffer" option
# this would be the distance that needs to be clipped from the edge of a las file
# for instance, if las tiles overlap by 20 m
# then the buffer value would be 20/2 = 10 m
# NOTE: the buffer value here assumes an input of rectangular tiles with equal buffers on all sides
# PB 10/03/22
def calcLasHeight(inf, opath=None, of=None, buffer=0):
    
    inf = Path(str(inf))

    # If no outfile name or dir, make one in the same dir
    if not of:

        of = inf.name
        
    if not opath:
        
        opath = inf.parent
        
    else:
        
        opath = Path(opath)
    
    # If there's no buffer to clip
    if buffer==0:
        
        of = Path(str(opath) + '/' + inf.name.split('.')[0] + "_Height.las")
        
        # define injson for pdal pipeline
        injson= """
        [
            "in.las",
            {
                "type":"filters.hag_delaunay",
                "count": 10,
                "allow_extrapolation": true
            },
            {
                "type": "writers.las",
                "filename": "out.las",
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
        """
        
    # else: if there is a buffer to clip
    else:
        
        of = Path(str(opath) + '/' + inf.name.split('.')[0] + "_Height.las")
        
        # open the header of the file in laspy
        # to get coordinates boundaries from the header
        l = laspy.open(str(inf))
        
        # Take away the buffer region from the boundary
        xmin = l.header.mins[0] + buffer
        xmax = l.header.maxs[0] - buffer
        ymin = l.header.mins[1] + buffer
        ymax = l.header.maxs[1] - buffer
        
        # define injson for pdal pipeline
        injson= """
        [
            "in.las",
            {
                "type":"filters.hag_delaunay",
                "count": 10,
                "allow_extrapolation": true
            },
            {
                "type":"filters.crop",
                "bounds":"([xmin,xmax],[ymin,ymax])"
            },
            {
                "type": "writers.las",
                "filename": "out.las",
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
        """
        
        # Replace bounds with values of xmin, xmax, ymin, ymax
        # clipped by buffer
        injson = injson.replace("xmin", str(xmin))
        injson = injson.replace("xmax", str(xmax))
        injson = injson.replace("ymin", str(ymin))
        injson = injson.replace("ymax", str(ymax))
        
    # Replace args with in and out file
    injson = injson.replace("in.las", str(inf))
    injson = injson.replace("out.las", str(of))

    pipeline = pdal.Pipeline(injson)
    pipeline.execute()
    # arrays = pipeline.arrays
    metadata = pipeline.metadata
    # log = pipeline.log
    
    return metadata

# Define normalize height function
# NOTE: Similar Height Norm Function to the above
# but with an added voxel based point filter
# This is because of a "Collinear Points error" that ws happening with certain tiles of the selenkay data
# See: https://github.com/PDAL/PDAL/issues/3844
# and for info on the point filtering:https://pdal.io/en/stable/stages/filters.sample.html#filters-sample
# Note: added a "buffer" option
# this would be the distance that needs to be clipped from the edge of a las file
# for instance, if las tiles overlap by 20 m
# then the buffer value would be 20/2 = 10 m
# NOTE: the buffer value here assumes an input of rectangular tiles with equal buffers on all sides
# PB 10/17/22
def calcLasHeight_Filter(inf, opath=None, of=None, buffer=0, radius=None):
    
    inf = Path(str(inf))

    # If no outfile name or dir, make one in the same dir
    if not of:

        of = inf.name
        
    if not opath:
        
        opath = inf.parent
        
    else:
        
        opath = Path(opath)
    
    # If there's no buffer to clip
    if buffer == 0:
        
        of = Path(str(opath) + '/' + inf.name.split('.')[0] + "_Filter_Height.las")
        
        # define injson for pdal pipeline
        injson= """
        [
            "in.las",
            {
                "type":"filters.voxelcentroidnearestneighbor",
                "cell": 0.1
            },
            {
                "type":"filters.hag_delaunay",
                "count": 10,
                "allow_extrapolation": true
            },
            {
                "type": "writers.las",
                "filename": "out.las",
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
        """
        if radius:
            injson = injson.replace(" 0.1", str(radius))
        
    # else: if there is a buffer to clip
    else:
        
        of = Path(str(opath) + '/' + inf.name.split('.')[0] + "_Filter_Height_NoBuffer.las")
        
        # open the header of the file in laspy
        # to get coordinates boundaries from the header
        l = laspy.open(str(inf))
        
        # Take away the buffer region from the boundary
        xmin = l.header.mins[0] + buffer
        xmax = l.header.maxs[0] - buffer
        ymin = l.header.mins[1] + buffer
        ymax = l.header.maxs[1] - buffer
        
        # define injson for pdal pipeline
        injson= """
        [
            "in.las",
            {
            "type":"filters.sample",
            "radius": 0.1
            },
            {
                "type":"filters.hag_delaunay",
                "count": 10,
                "allow_extrapolation": true
            },
            {
                "type":"filters.crop",
                "bounds":"([xmin,xmax],[ymin,ymax])"
            },
            {
                "type": "writers.las",
                "filename": "out.las",
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
        """
        
        if radius:
            injson = injson.replace(" 0.1", str(radius))
        
        # Replace bounds with values of xmin, xmax, ymin, ymax
        # clipped by buffer
        injson = injson.replace("xmin", str(xmin))
        injson = injson.replace("xmax", str(xmax))
        injson = injson.replace("ymin", str(ymin))
        injson = injson.replace("ymax", str(ymax))

    # Replace args with in and out file
    injson = injson.replace("in.las", str(inf))
    injson = injson.replace("out.las", str(of))

    pipeline = pdal.Pipeline(injson)
    pipeline.execute()
    # arrays = pipeline.arrays
    metadata = pipeline.metadata
    # log = pipeline.log
    
    return metadata

# Define simple filtering function
# Using a voxel based point filter
# PB 11/03/22
def filterLAS(inf, opath=None, of=None, radius=None):
    
    inf = Path(str(inf))

    # If no outfile name or dir, make one in the same dir
    if not of:

        of = inf.name
        
    if not opath:
        
        opath = inf.parent
        
    else:
        
        opath = Path(opath)

    # define injson for pdal pipeline
    injson= """
    [
        "in.las",
        {
        "type":"filters.sample",
        "radius":0.1
        },
        {
            "type":"writers.las",
            "filename":"out.las"
        }
    ]
    """

    if radius:
        injson = injson.replace("0.1", str(radius))
        radstr = str(radius).replace('.','p') + 'm'
    else:
        radstr = "0p1m"
    
    of = Path(str(opath) + '/' + inf.name.split('.')[0] + f"_Filter{radstr}.las")
    
    # Replace args with in and out file
    injson = injson.replace("in.las", str(inf))
    injson = injson.replace("out.las", str(of))

    pipeline = pdal.Pipeline(injson)
    pipeline.execute()
    # arrays = pipeline.arrays
    metadata = pipeline.metadata
    # log = pipeline.log
    
    return metadata

# Define normalize height function
# NOTE: Similar Height Norm Function to the above
# but with an added splitter- breaking the file into tiles
# See: https://pdal.io/en/stable/stages/filters.divider.html
# Note: added a "buffer" option
# this would be the distance that needs to be clipped from the edge of a las file
# for instance, if las tiles overlap by 20 m
# then the buffer value would be 20/2 = 10 m
# NOTE: the buffer value here assumes an input of rectangular tiles with equal buffers on all sides
# PB 10/17/22
def calcLasHeight_Split(inf, opath=None, of=None, buffer=0):
    
    inf = Path(str(inf))

    # If no outfile name or dir, make one in the same dir
    if not of:

        of = inf.name
        
    if not opath:
        
        opath = inf.parent
        
    else:
        
        opath = Path(opath)
    
    # If there's no buffer to clip
    if buffer == 0:
        
        # Note the additional '#' sign here
        # since the data is being tiled into 4 sections
        # and each tile will be numbered by pdal
        of = Path(str(opath) + '/' + inf.name.split('.')[0] + "_Filter_Height#.las")
        
        # define injson for pdal pipeline
        injson= """
        [
            "in.las",
            {
                "type":"filters.divider",
                "count":"4"
            },
            {
                "type":"filters.hag_delaunay",
                "count": 10,
                "allow_extrapolation": true
            },
            {
                "type": "writers.las",
                "filename": "out.las",
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
        """
        
    # else: if there is a buffer to clip
    else:
        
        # Note the additional '#' sign here
        # since the data is being tiled into 4 sections
        # and each tile will be numbered by pdal
        of = Path(str(opath) + '/' + inf.name.split('.')[0] + "_Filter_Height_NoBuffer#.las")
        
        # open the header of the file in laspy
        # to get coordinates boundaries from the header
        l = laspy.open(str(inf))
        
        # Take away the buffer region from the boundary
        xmin = l.header.mins[0] + buffer
        xmax = l.header.maxs[0] - buffer
        ymin = l.header.mins[1] + buffer
        ymax = l.header.maxs[1] - buffer
        
        # define injson for pdal pipeline
        injson= """
        [
            "in.las",
            {
                "type":"filters.divider",
                "count":"4"
            },
            {
                "type":"filters.hag_delaunay",
                "count": 10,
                "allow_extrapolation": true
            },
            {
                "type":"filters.crop",
                "bounds":"([xmin,xmax],[ymin,ymax])"
            },
            {
                "type": "writers.las",
                "filename": "out.las",
                "extra_dims": "HeightAboveGround=float32"
            }
        ]
        """
        
        # Replace bounds with values of xmin, xmax, ymin, ymax
        # clipped by buffer
        injson = injson.replace("xmin", str(xmin))
        injson = injson.replace("xmax", str(xmax))
        injson = injson.replace("ymin", str(ymin))
        injson = injson.replace("ymax", str(ymax))

    # Replace args with in and out file
    injson = injson.replace("in.las", str(inf))
    injson = injson.replace("out.las", str(of))

    pipeline = pdal.Pipeline(injson)
    pipeline.execute()
    # arrays = pipeline.arrays
    metadata = pipeline.metadata
    # log = pipeline.log
    
    return metadata


# Make Shapefile Boundary from las file header
# PB 9/30/22
def lasBBoxShp(lasf, epsg='32737', opath=None):
    
    lasf = Path(lasf)

    # open the current las file for reading
    with laspy.open(str(lasf)) as l:

        # Make las boundary points from header into a polygon (ul, ur, lr, ll, ul)
        # Ploygon is a shapely.geometry object
        lasbounds_poly = Polygon([[l.header.mins[0], l.header.maxs[1]],
                                  [l.header.maxs[0], l.header.maxs[1]],
                                  [l.header.maxs[0], l.header.mins[1]],
                                  [l.header.mins[0], l.header.mins[1]],
                                  [l.header.mins[0], l.header.maxs[1]]])

         # Make a geodataframe from the boundaries of the las file
        lasbounds_gdf = gpd.GeoDataFrame(geometry=[lasbounds_poly],
                                        crs=f'EPSG:{epsg}')
        
        # Export 
        if not opath:
            
            opath = str(lasf.parent)
            
        # Set output file names
        lasname = str(lasf.name).split('.las')[0]
        shpf = f'{opath}/{lasname}.shp'
        
        # Export
        lasbounds_gdf.to_file(shpf)
        
        print(f'Saved {Path(shpf).name}. \n')
        
        

# Define the lasClip function for Clipping points by iterating through multiple features in a shapefile
# Features can be irregularly shaped (not squares),
# But can't handle multipolygon inputs (only polygons).
# Outputs las files to a folder (outdir), labelled by the specified featureID name/number.
#
# - Inputs -
# shpdf = a geodataframe (geopandas) of the shapefile
# lasdir = directory of tiles of las data to loop through
# outdir = directory to output individual las files
# featureIDcol = column used to iterate through features in shapefile
# epsg = epsg code of the shapefile and las files (note: they should be the same)
# verb = True/False verbose output (telling you time to compute each file)
#
def lasClip_Shape(shpdf=None, lasdir=None, outdir=None, featureIDcol='treeID', epsg='32736', verb=True, buffer=0):
    # Make absolute path for consistency
    lasdir = os.path.abspath(lasdir)
    outdir = os.path.abspath(outdir)
    
    if type(epsg) is str:
        epsg = int(epsg)
    
    # for each polygon feature in the shapefile dataframe
    for row, feature in shpdf.iterrows():

        # TIME each feature: 
        start = time.time()

        # For each las tile file in the directory
        for lasf in glob.glob(f'{lasdir}/*.las'):

            # time each las file:
            lasstart = time.time()

            # open the current las file for reading
            with laspy.open(lasf) as l:

                # Make las boundary points from header into a polygon (ul, ur, lr, ll, ul)
                # Ploygon is a shapely.geometry object
                lasbounds_poly = Polygon([[l.header.mins[0], l.header.maxs[1]],
                                          [l.header.maxs[0], l.header.maxs[1]],
                                          [l.header.maxs[0], l.header.mins[1]],
                                          [l.header.mins[0], l.header.mins[1]],
                                          [l.header.mins[0], l.header.maxs[1]]])

                # Make a geodataframe from the boundaries of the las file
                lasbounds_gdf = gpd.GeoDataFrame(geometry=[lasbounds_poly],
                                                crs=f'EPSG:{epsg}')

                # Using the lasboundary, 
                # Test whether the las file intersects with the polygon, and if so: 
                if feature.geometry.intersects(lasbounds_poly):

                    # Iteration time
                    itertime_start = time.time()

                    # Load the points in Chunks of 10 million
                    # https://laspy.readthedocs.io/en/latest/basic.html#writing
                    for pointset in l.chunk_iterator(5_000_000):

                         # subset the points to only default and ground points within the square boundary of the given shapefile feature
                        idx =  ((pointset.x <= feature.geometry.bounds[2]) &
                               (pointset.x >= feature.geometry.bounds[0]) &
                               (pointset.y <= feature.geometry.bounds[3]) &
                               (pointset.y >= feature.geometry.bounds[1]) &
                               (pointset.classification != 7))

                        points = pointset[idx]

                        # if points is not empty
                        if len(points) > 0:

                            # Make a geodataframe from the points
                            points_gdf = gpd.GeoDataFrame({'time':points.gps_time},
                                                          geometry=gpd.points_from_xy(points.X * l.header.x_scale + l.header.x_offset,
                                                                                      points.Y * l.header.y_scale + l.header.y_offset),
                                                          crs=f'EPSG:{epsg}')


                            # get the indices of points that intersect with the polygon feature
                            # use align=False to preserve the order of the index
                            intersects_idx = points_gdf.intersects(feature.geometry, align=False)

                            # NOW: subset your points again, this time, based on your intersection index
                            points_subset = points[intersects_idx.values]

                            # set the outf name and path
                            # use the featureIDcol in the shapefile to name it 
                            outf = f'{outdir}/{featureIDcol}_{feature.get(featureIDcol)}.las'

                            try:

                                # if there are any points to output 
                                if len(points_subset) > 0: 
                                    # if this file does not exist yet, make a new las file
                                    if not os.path.exists(outf):
                                        # write points
                                        with laspy.open(outf, mode="w", header=l.header) as writer:
                                            writer.write_points(points_subset)
                                    else:
                                        # else, append the points to an existing las file
                                        with laspy.open(outf, mode="a", header=l.header) as writer:
                                            writer.append_points(points_subset)

                            except:

                                print(f'Issue saving {featureIDcol}_{feature.get(featureIDcol)}.las')
                                    
            lasend = time.time()
            # lastime = lasend-lasstart
            # print(f'Processed lastile in {lastime} s.\n')
    
        end = time.time()
        totaltime = end-start

        # print(f'Total time was {totaltime} s.\n Iteration time was {itertime_end - itertime_start} s.\n Write time was {writetime_end - itertime_end}.\n')
        if verb:
            print(f'Processed {featureIDcol}_{feature.get(featureIDcol)} in {totaltime} s.\n')
            

# Define the lasClip function for Clipping points for a single feature in a shapefile
# Used for parrallel processing.
# Features can be irregularly shaped.
#
# - Inputs -
# feature = a shapely geometry object (made in geopandas) of the given polygon
# lasdir = directory of tiles of las data to loop through
# outdir = directory to output individual las files
# featureIDcol = column used to iterate through features in shapefile
# epsg = epsg code of the shapefile and las files (note: they should be the same)
# verb = True/False verbose output (telling you time to compute each file)
#
def lasClip_IndivFeature(feature=None, lasdir=None, outdir=None, featureIDcol='treeID', epsg='32736', verb=False):
    
    # Make absolute path for consistency
    lasdir = os.path.abspath(lasdir)
    outdir = os.path.abspath(outdir)
    
    if type(epsg) is str:
        epsg = int(epsg)
    
    # Counter to check if any lasfiles intersected the shapefile
    # prints a warning if not
    tilecheck = []

    # TIME each feature: 
    start = time.time()

    # For each las tile file in the directory
    for lasf in glob.glob(f'{lasdir}/*.las'):

        # time each las file:
        lasstart = time.time()

        # open the current las file for reading
        with laspy.open(lasf) as l:

            # Make las boundary points from header into a polygon (ul, ur, lr, ll, ul)
            # Ploygon is a shapely.geometry object
            lasbounds_poly = Polygon([[l.header.mins[0], l.header.maxs[1]],
                                      [l.header.maxs[0], l.header.maxs[1]],
                                      [l.header.maxs[0], l.header.mins[1]],
                                      [l.header.mins[0], l.header.mins[1]],
                                      [l.header.mins[0], l.header.maxs[1]]])

             # Make a geodataframe from the boundaries of the las file
            lasbounds_gdf = gpd.GeoDataFrame(geometry=[lasbounds_poly],
                                            crs=f'EPSG:{epsg}')

            # Using the lasboundary, 
            # Test whether the las file intersects with the polygon, and if so: 
            if feature.geometry.intersects(lasbounds_poly):
                
                # append to a list, so you can check what went in it later
                tilecheck.append(lasf)
                
                # print(f'\t\tFeature {feature.get(featureIDcol)} made it this far')

                # Iteration time
                itertime_start = time.time()

                # Load the points in Chunks of 10 million
                # https://laspy.readthedocs.io/en/latest/basic.html#writing
                for pointset in l.chunk_iterator(10_000_000):

                     # subset the points to only default and ground points within the square boundary of the given shapefile feature
                    idx = ((pointset.x <= feature.geometry.bounds[2]) &
                           (pointset.x >= feature.geometry.bounds[0]) &
                           (pointset.y <= feature.geometry.bounds[3]) &
                           (pointset.y >= feature.geometry.bounds[1]) &
                           (pointset.classification != 7))

                    points = pointset[idx]

                    # if points is not empty
                    if len(points) > 0:

                        # Make a geodataframe from the points
                        points_gdf = gpd.GeoDataFrame({'time':points.gps_time},
                                                      geometry=gpd.points_from_xy(points.X * l.header.x_scale + l.header.x_offset,
                                                                                  points.Y * l.header.y_scale + l.header.y_offset),
                                                      crs=f'EPSG:{epsg}')


                        # get the indices of points that intersect with the polygon feature
                        # use align=False to preserve the order of the index
                        intersects_idx = points_gdf.intersects(feature.geometry, align=False)

                        # NOW: subset your points again, this time, based on your intersection index
                        points_subset = points[intersects_idx.values]

                        # set the outf name and path
                        # use the featureIDcol in the shapefile to name it 
                        outf = f'{outdir}/{featureIDcol}_{feature.get(featureIDcol)}.las'

                        try:

                            # if there are any points to output 
                            if len(points_subset) > 0: 
                                # if this file does not exist yet, make a new las file
                                if not os.path.exists(outf):
                                    # write points
                                    with laspy.open(outf, mode="w", header=l.header) as writer:
                                        writer.write_points(points_subset)
                                else:
                                    # else, append the points to an existing las file
                                    with laspy.open(outf, mode="a", header=l.header) as writer:
                                        writer.append_points(points_subset)

                        except:

                            print(f'Issue saving {featureIDcol}_{feature.get(featureIDcol)}.las')
                                    
            lasend = time.time()
            # lastime = lasend-lasstart
            # print(f'Processed lastile in {lastime} s.\n')
        
        end = time.time()
        totaltime = end-start

        # print(f'Total time was {totaltime} s.\n Iteration time was {itertime_end - itertime_start} s.\n Write time was {writetime_end - itertime_end}.\n')
        if verb:
            print(f'Processed {featureIDcol}_{feature.get(featureIDcol)} in {totaltime} s.\n')
            
    if len(tilecheck) == 0:
        print(f'{featureIDcol} {feature.get(featureIDcol)} did not intersect with any las files.\n')

        
# # # 
# # # 
# The following functions compute metrics from a point cloud.
# When using gridding (Cloud.makegrid()), 
# this can run on a stack of voxels with each grid cell


# Function for Calculating Cover and Foliage Height Profiles
def calccover(points=None, hmin=0, hmax=15, step=0.25, 
              heightcol='HeightAboveGround',
              numretcol='number_of_returns',
              retnumcol='return_number',
              classcol='classification',
              groundthres=0.05,
              calcintensity=False):

    # Calc Cover for height bins
    nbins = ((hmax - hmin) / step) + 1
    heightbins = np.linspace(hmin, hmax, int(nbins))
    
   # IMPORTANT: Using groundthres, you may want to account for errors in relative accuracy
    # EX: IF the rel. accuracy of ground is about 0.06 m (6 cm) between flightlines,
    # the lowest height bin could be set to 0.06 (instead of 0) to account for this.
    # so any hit below 0.06 m counts as ground.
    # NOTE: If you want to use everything, just set groundthres to 0
    if groundthres > 0:
        # insert the groundthres into the array (right above 0)
        heightbins = np.insert(heightbins, 1, groundthres)
        # heightbins[heightbins==0] = groundthres
    if groundthres < 0:
        # insert the groundthres into the array (right below 0)
        heightbins = np.insert(heightbins, 0, groundthres)
        # heightbins[heightbins==0] = groundthres

    # intiialize arrays
    coverD1 = []
    coverD2 = []
    Npulses = []
    
    # Get the heights (z value) of all 1st return points 
    # filter for max height added (10/26/22)
    zfirst = points[heightcol][((points[retnumcol] == 1) & (points[heightcol] <= hmax))]
    zfirst_veg =  points[heightcol][((points[classcol] == 1) & (points[retnumcol] == 1) & (points[heightcol] <= hmax))]
    
    # 2 Methods for Cover Used Here
    # Method D1 treats each 1st return with a weight of 1
    # Method D2, each first return is weighted based on the number of returns in the pulse (loosely based on Armston et al., 2013)
    
    # Compute weights for D2
    weights = []
    # for each first return vegetation point, append a weight equal to 1 / the number of returns in the pulse
    for nr in points[numretcol][((points[classcol] == 1) & (points[retnumcol] == 1) & (points[heightcol] <= hmax))]:
        weights.append(1/nr)

    weights = np.array(weights)

    # Loop through Height bins and run 
    # Methods D1 and D2
    for h in heightbins:
        
        # Get the total number of pulses,
        # Calculated as the total number of first returns in the cell
        N = len(zfirst)
        
        # If there are pulses in the cell
        if N > 0:
            
            # D1 - sum the number of first returns above the given height h, and below the max height (to exclude noise)
            n = sum( (zfirst_veg <= hmax) & (zfirst_veg > h) )
            # Divide by the total number of 1st returns (aka: pulses)
            coverD1.append( n / N )

            # Also, grab the number of pulses per height bin here
            Npulses.append(n)

            # D2 - sum the weights of first returns above the griven height h
            w = sum( weights[(zfirst_veg <= hmax) & (zfirst_veg > h)] )
            # Divide by the total number of 1st returns (aka: pulses)
            coverD2.append( w / N )
        
        # Else, return 0
        else:
            
            coverD1.append(0)
            coverD2.append(0)
            Npulses.append(0)

    # Make foliage profile
    # Uses np.diff to get cover in each bin (discrete derivative)
    # 1 - cover = gap fraction
    # get foliage profile (palace et al., 2015; Macarthur & Horn)
    # See: https://www.youtube.com/watch?v=r4oPfOTZDDg for notes on np.diff
    
    # new (should be correct!!!) version - 7/20/22
    # Get the cover difference in each voxel 
    # *-1 becase numpy does the difference backwards (top-to-bottom instead of bottom-top)
    coverD1byH = -np.diff(coverD1)
    coverD2byH = -np.diff(coverD2)
    
    # PAVD = -ln(1 - cover in each voxel) / height of voxel
    FHPD1 = -np.log(1-coverD1byH, where=(1-coverD1byH)>0) / np.diff(np.array(heightbins))
    FHPD2 = -np.log(1-coverD2byH, where=(1-coverD2byH)>0) / np.diff(np.array(heightbins))
    
    # NOTE: You can get warnings from numpy - RuntimeWarning: divide by zero encountered in log
    # That is likely because coverD1 or coverD2 become fully saturated (1) - so it tries to take -np.log(0). 
    # TBD a solution for this... can't just set it to NA (becuase it has plant material, it's just the cover fraction is saturated)
    # Maybe a method using all returns (not just first) could avoid this.

    cover = {'CoverD1':np.array(coverD1),
             'CoverD2':np.array(coverD2),
             'CoverD1byH':np.array(coverD1byH),
             'CoverD2byH':np.array(coverD2byH),
             'FHPD1':np.array(FHPD1),
             'FHPD2':np.array(FHPD2),
             'HeightBins':np.array(heightbins),
             'Npulses':np.array(Npulses)}

    return cover


# Complexity Function - v3
# PB 11/09/22
# NOTE: If performing over a set of pixels, need to wrap the below in a loop
def canopyLayerMetrics(h, hbins, plot=False, smoothsigma=2, method='gauss1d', rel_height=0.75, groundthreshold=0.05):
    # Where h is an array of point height values
    # hbins is a list/array of height bin edges
    # smoothsigma = positive float - gives a smoothing parameter for gaussian smoothing in layer calculation (in meters)
    # plot = True/False - whether to plot the results
    # method = ['gauss1d'] with smoothsigma = 2 (default)
    # method = ['kde'] with rule-of-thumb approximation of sigma
    # relH_peakwidth = 0.75 - rel. height of peak to find the "trough" bottoms at
        # rel_height = 0.5 is the full width half max
        # rel_height = 1 is the bottom of the gaussian
    
    # If there are any heights in the array
    # edited 10/26 - have to have at least 2 points above the ground to do complexity stats
    # This means that a plot with only 1 point will have all metrics automatically set to 0 or nan
    if np.sum(np.array(h) > 0) >= 2:

        # sort points by height
        h = np.sort(h)
        
        # Interpolate heights and derivative to 1 cm increments so that you have more precise inflection points
        hbins_interp= np.arange(np.min(hbins), np.max(hbins), 0.01)
        
        # Group each point by height bin
        hgroups = np.digitize(h, bins=hbins)

        # Count the number of points in each bin
        # Note: minlength allows it to go to the full length of hbins, rather than stop at the max height of the points
        hcounts = np.bincount(hgroups, minlength=len(hbins))

        # Normalize the counts
        hcounts_norm = hcounts/np.sum(hcounts)
        
        # If using a gaussian kernel density estimate from the height array
        if method == 'kde':

            gausskde = gaussian_kde(h, bw_method='scott')
            smooth = gausskde.pdf(hbins_interp)

        # else, if using a smoothed version of point density by height
        else:
            
            # smooth with gaussian filter
            gauss1d = gaussian_filter1d(hcounts_norm, smoothsigma)
            
            # interpolate with hbins_interp to make it more dense
            smooth = np.interp(x=hbins_interp, xp=hbins, fp=gauss1d)
            
        # Find Peaks and troughs
        idx_peaks, meta = find_peaks(smooth)

        # Get the widths of the peak and identify the troughs using a relative heiht value
        # rel_height = 0.5 is the full width half max
        # rel_height = 1 is the bottom of the gaussian
        w, width_heights, l_ips, r_ips = peak_widths(smooth, idx_peaks, rel_height=rel_height)

        # NOTE: convert peak and trough indices back to height values using the below function
        # copied exactly from: https://stackoverflow.com/questions/66464148/peak-widths-w-r-t-to-x-axis
        def index_to_xdata(xdata, indices):
            "interpolate the values from signal.peak_widths to xdata"
            ind = np.arange(len(xdata))
            f = interp1d(ind,xdata)
            return f(indices)

        # get heights of troughs
        widths = index_to_xdata(hbins_interp, w)
        left_ips = index_to_xdata(hbins_interp, l_ips)
        right_ips = index_to_xdata(hbins_interp, r_ips)
        troughs = np.append(left_ips, right_ips)
        
        # get density of troughs for plotting
        smooth_left_ips = index_to_xdata(smooth, l_ips)
        smooth_right_ips = index_to_xdata(smooth, r_ips)
        smooth_troughs = np.append(smooth_left_ips, smooth_right_ips)
        
        # Get heights of peaks
        peaks = hbins_interp[idx_peaks]

        # Number of layers as number of peaks
        nlayers = len(peaks)

        if nlayers > 1:

            # height location of peak with largest frequency of points
            # maxpeakh = peaks[np.argmax(smooth[idx_peaks])]
            maxpeakh = np.nanmax(peaks)

            # Get the gap size as the maximum distance between consecutive peaks
            gapsize = np.max(np.diff(peaks))

            # MEAN and STD peak HEIGHT
            # made up by peter
            meanpeakh = np.nanmean(peaks)
            stdpeakh = np.nanstd(peaks)
            cvpeakh = np.nanmean(peaks)/np.nanstd(peaks)

            # Vertical Distribution Ratio (Goetz 2007) 
            # - experimental, computed from peaks instead of norm point distribution
            VDRpeak = (np.max(peaks) - np.nanmedian(peaks)) / np.max(peaks)
            
            # get the top of the herbaceous layer from the troughs
            # New addition 11/9/22
            # The herbaceous height is the 2nd trough up from the bottom
            # but it has to be higher than the groundthreshold
            # otherwise, we set the herb height to be 0
            # this means you can't detect a herbaceous layer lower than the groundthreshold (usallually 5 cm)
            herbtroughs = troughs[troughs >= groundthreshold]
            if herbtroughs.size:
                herbh = np.sort(herbtroughs)[0]
            else:
                herbh = 0
            
        else:

            # Edited 10/31/22 
            # If there's only 1 layer, 
            # set the peaks to be the only peak height
            if peaks.size:
                maxpeakh = peaks[0]
                meanpeakh = peaks[0]
                
                # New addition 11/9/22
                # The herbaceous height is the 2nd trough up from the bottom
                # but it has to be higher than the groundthreshold
                # otherwise, we set the herb height to be 0
                # this means you can't detect a herbaceous layer lower than the groundthreshold (usallually 5 cm)
                herbtroughs = troughs[troughs >= groundthreshold]
                if herbtroughs.size:
                    herbh = np.sort(herbtroughs)[0]
                else:
                    herbh = 0
                    
            # Except if there is no peak to record
            else:
                maxpeakh = 0
                meanpeakh = 0 
                herbh = 0

            # and variation metrics to 0
            stdpeakh = 0
            gapsize = 0
            cvpeakh = 0
            VDRpeak = 0
        
        # computing the PtoH ratio (Davies 2020, Asner 2014)
        perc99 = np.nanpercentile(h, 99, method='median_unbiased')
        ptoh = maxpeakh/perc99

        # Complexity Score (Davies 2020)
        # proportion of bins with points in them vs without
        # could be insensitive to striping
        cscore = np.sum(hcounts>0)/len(hcounts)

        # Vertical Distribution Ratio (Goetz 2007)
        VDR = (np.max(h) - np.median(h)) / np.max(h)
        
        # Foliage Height Diversity 
        # (Bergen 2009 and many others using PAI profile - this is using normalized point counts)
        if np.sum(hcounts_norm>0) > 0:
            FHD = -1*np.sum(hcounts_norm*np.log(hcounts_norm, where=hcounts_norm>0))
        else:
            FHD = 0

        ### SAVE Outputs
        complex_dict = {'nlayers':nlayers,
                        'gapsize':gapsize,
                        'maxpeakh':maxpeakh,
                        'ptoh':ptoh,
                        'cscore':cscore,
                        'FHD':FHD,
                        'VDR':VDR,
                        'VDRpeak':VDRpeak,
                        'meanpeakh':meanpeakh,
                        'stdpeakh':stdpeakh,
                        'cvpeakh':cvpeakh,
                        'herbh':herbh
                       }

        if plot:

            fig, ax = plt.subplots()
                
            ax.plot(hcounts_norm, hbins, label='Point Density', lw=5, c='k', alpha=0.6)
                
            ax.plot(smooth, hbins_interp, label=method, lw=3, c='c', alpha=0.9, linestyle='--')
            
            ax.axhline(y=herbh, color='olive', label=f'Herb. Height {np.round(herbh, 2)}m', alpha=0.8)
                                    
            ax.scatter(x=smooth[idx_peaks], y=hbins_interp[idx_peaks],
                       c='b', label='Peaks', marker='x', linewidths=4)
            
            ax.scatter(x=smooth_troughs, y=troughs,
                      c='r', label='Troughs', marker='x', linewidths=4)
            
            # for infl in troughs:
            #     tline = ax.axhline(y=infl, color='b', label='Trough', alpha=0.8)
            # for infl in peaks:
            #     pline = ax.axhline(y=infl, color='c', label='Peak', alpha=0.8)
            # ax.legend(handles=[tline, pline], loc='best')
            
            ax.legend(loc='best')
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Height [m]')
            ax.set_xlim(-0.01, np.max(hcounts_norm) + 0.03)
            ax.set_ylim(0, np.percentile(hbins, 50) + 1)
            ax.grid('on')

            # don't return peaks, troughs
            return complex_dict, fig, ax

        else:

            return complex_dict

    # Else, if there were no heights in the array
    # just return an array of 0s
    else:

        complex_dict = {'nlayers':0,
                        'gapsize':0,
                        'maxpeakh':0,
                        'ptoh':0,
                        'cscore':0,
                        'FHD':0,
                        'VDR':0,
                        'VDRpeak':0,
                        'meanpeakh':0,
                        'stdpeakh':0,
                        'cvpeakh':0,
                        'herbh':0
                       }

        return complex_dict

    
# Function for calculating percentile heights
def calcPercentileHeights(points, groundthres=0, returnHeights=True, hmax=15, heightcol='HeightAboveGround'):
    
    # Calculate Percentile Metrics of Height
    perc_dict= {0:[],
                25:[],
                50:[],
                75:[],
                98:[],
                100:[],
                'mean':[],
                'std':[]}
    
    # check to see if there are points between 0 and maxH
    points_between0andmaxH = points[heightcol][((points[heightcol] >= 0) & (points[heightcol] <= hmax))]
    
    # if the array of heights is not empty
    if points_between0andmaxH.size > 0:
        
        # Get the height values of all points, filtering out ground
        # Added hmax filter here 10/26/22
        heights = points[heightcol][((points[heightcol] >= groundthres) & (points[heightcol] <= hmax))]

        # If there are any heights after filtering out ground
        if heights.size > 0:

            perc_dict[0].append(np.quantile(heights, [0]).flat[0])
            perc_dict[25].append(np.quantile(heights, [0.25]).flat[0])
            perc_dict[50].append(np.quantile(heights, [0.5]).flat[0])
            perc_dict[75].append(np.quantile(heights, [0.75]).flat[0])
            perc_dict[98].append(np.quantile(heights, [0.98]).flat[0])
            perc_dict[100].append(np.quantile(heights, [1.0]).flat[0])
            perc_dict['mean'].append(np.nanmean(heights).flat[0])
            perc_dict['std'].append(np.nanstd(heights).flat[0])

        # else, height stats are 0 (all points fall between 0 and ground)
        else:

            perc_dict[0].append(0)
            perc_dict[25].append(0)
            perc_dict[50].append(0)
            perc_dict[75].append(0)
            perc_dict[98].append(0)
            perc_dict[100].append(0)
            perc_dict['mean'].append(0)
            perc_dict['std'].append(0)
            
    # else, if the array of heights is completely empty,
    # after filtering for heights between 0 and hmax
    else:
        
            # set the heights array to be empty
            heights = np.empty(0)
            
            # insert nans 
            perc_dict[0].append(np.nan)
            perc_dict[25].append(np.nan)
            perc_dict[50].append(np.nan)
            perc_dict[75].append(np.nan)
            perc_dict[98].append(np.nan)
            perc_dict[100].append(np.nan)
            perc_dict['mean'].append(np.nan)
            perc_dict['std'].append(np.nan)
            
    if returnHeights:
        
        return perc_dict, heights

    else:
        return perc_dict

    
# # # 
# # # 
# The below functions analyze the height, cover, and percentile dictionaries
# They calculate aggregate statistics of the voxels in the point cloud.
    

# Function to compute aggregate pixel metrics based on perc height dictionary
def heightAggMetrics(perc_dict, groundthreshold=0.05, outstr='', vegtypestats=True):
    # 2/9/23
    # Added a way to add a string suffix (outstr) to the variable names
    # in order to distinguish them from the other agg cover/height metrics
    # and a way to turn on/off the proportional veg cover stats (vegtypestats=True/False)
    
    # Get max heights and std heights
    maxH = np.array([v[98][0] for v in perc_dict.values()])
    meanH = np.array([v['mean'][0] for v in perc_dict.values()])
    stdH = np.array([v['std'][0] for v in perc_dict.values()])
    
    # initialize output stats dict
    stats = {}
    
    # Fill Stat dict
    
    # Horizontal variation in max height
    stats[f'max_maxH{outstr}'] = np.nanmax(maxH)
    # leaving out min, since it will always be 0
    # stats['min(maxH)'] = np.min(maxH)
    
    # Note: Leaving out the range, since all will have a min height of 0. This will just be the maxH.
    # stats['range(maxH)'] = np.max(maxH) - np.min(maxH)
    
    stats[f'mean_maxH{outstr}'] = np.nanmean(maxH)
    stats[f'sd_maxH{outstr}'] = np.nanstd(maxH)
    stats[f'cv_maxH{outstr}'] = np.nanstd(maxH) / np.nanmean(maxH)
    stats[f'median_maxH{outstr}'] = np.nanmedian(maxH)
    stats[f'iqr_maxH{outstr}'] = np.nanpercentile(maxH, q=75, method='median_unbiased') - np.nanpercentile(maxH, q=25, method='median_unbiased')
    
    # Stats of vertical deviation in point heights
    stats[f'mean_sdH{outstr}'] = np.nanmean(stdH)
    stats[f'sd_sdH{outstr}'] = np.nanstd(stdH)
    stats[f'mean_cvH{outstr}'] = np.nanmean(stdH/meanH)
    stats[f'sd_cvH{outstr}'] = np.nanstd(stdH/meanH)
    
    # Vegetation proportional cover and other stats
    # using veg. classes based on thresholds of max heights of pixels
    if vegtypestats:
    
        ground_idx = (maxH <= groundthreshold)
        grass_idx = ((maxH > groundthreshold) & (maxH <= 0.5))
        shrub_idx = ((maxH > 0.5) & (maxH <= 3))
        tree_idx = (maxH > 3)
        woody_idx = (maxH > 1)

        stats[f'horzcover_ground{outstr}'] = np.sum(ground_idx) / np.size(maxH)
        stats[f'horzcover_grass{outstr}'] = np.sum(grass_idx) / np.size(maxH)
        stats[f'horzcover_shrub{outstr}'] = np.sum(shrub_idx) / np.size(maxH)
        stats[f'horzcover_tree{outstr}'] = np.sum(tree_idx) / np.size(maxH)
        stats[f'horzcover_woody{outstr}'] = np.sum( woody_idx ) / np.size(maxH)

        # Point cloud height stats (Rappaport et al., 2018)
        # Grass
        stats[f'mean_sdH_vegtype_grass{outstr}'] = np.nanmean(stdH[grass_idx])
        stats[f'sd_sdH_vegtype_grass{outstr}'] = np.nanstd(stdH[grass_idx])
        stats[f'mean_cvH_vegtype_grass{outstr}'] = np.nanmean(stdH[grass_idx]/meanH[grass_idx])
        stats[f'sd_cvH_vegtype_grass{outstr}'] = np.nanstd(stdH[grass_idx]/meanH[grass_idx])

        # Shrub
        stats[f'mean_sdH_vegtype_shrub{outstr}'] = np.nanmean(stdH[shrub_idx])
        stats[f'sd_sdH_vegtype_shrub{outstr}'] = np.nanstd(stdH[shrub_idx])
        stats[f'mean_cvH_vegtype_shrub{outstr}'] = np.nanmean(stdH[shrub_idx]/meanH[shrub_idx])
        stats[f'sd_cvH_vegtype_shrub{outstr}'] = np.nanstd(stdH[shrub_idx]/meanH[shrub_idx])

        # Tree
        stats[f'mean_sdH_vegtype_tree{outstr}'] = np.nanmean(stdH[tree_idx])
        stats[f'sd_sdH_vegtype_tree{outstr}'] = np.nanstd(stdH[tree_idx])
        stats[f'mean_cvH_tree{outstr}'] = np.nanmean(stdH[tree_idx]/meanH[tree_idx])
        stats[f'sd_cvH_vegtype_tree{outstr}'] = np.nanstd(stdH[tree_idx]/meanH[tree_idx])

        # Woody
        stats[f'mean_sdH_vegtype_woody{outstr}'] = np.nanmean(stdH[woody_idx])
        stats[f'sd_sdH_vegtype_woody{outstr}'] = np.nanstd(stdH[woody_idx])
        stats[f'mean_cvH_vegtype_woody{outstr}'] = np.nanmean(stdH[woody_idx]/meanH[woody_idx])
        stats[f'sd_cvH_vegtype_woody{outstr}'] = np.nanstd(stdH[woody_idx]/meanH[woody_idx])

    # # Round everything to 3 decimal points
    # for k in stats.keys():
    #     stats[k] = np.round(stats[k], 3)
    
    return stats

# Function to compute aggregate pixel metrics based on cover
# Horizontal variation in cover
def coverAggMetrics(cover_dict, outstr=''):
    # note - renamed this from cover (canopy cover) to canopy density (CD)
    # So that it doesn't get confused with horizontal cover
    
    stats = {}
    
    # Canopy cover at ground level
    covG = [v['CoverD2'][1] for v in cover_dict.values()]
    
    stats[f'mean_CD_G{outstr}'] = np.nanmean(covG)
    stats[f'sd_CD_G{outstr}'] = np.nanstd(covG)
    stats[f'cv_CD_G{outstr}'] = np.nanstd(covG) / np.nanmean(covG)
    stats[f'median_CD_G{outstr}'] = np.nanmedian(covG)
    stats[f'iqr_CD_G{outstr}'] = np.nanpercentile(covG, q=75, method='median_unbiased') - np.nanpercentile(covG, q=25, method='median_unbiased')
    stats[f'max_CD_G{outstr}'] = np.nanmax(covG)
    
    # Added a couple PAI stats here
    # using spherical leaf angle dist (k=0.5)
    covG = np.array(covG) # turn into numpy array for below calc
    stats[f'mean_PAI_G'] = np.nanmean(-np.log(1-covG, where=(1-covG)>0) / 0.5)
    stats[f'sd_PAI_G'] = np.nanstd(-np.log(1-covG, where=(1-covG)>0) / 0.5)
    
    # Cover 1 step above ground
    covAG = [v['CoverD2'][2] for v in cover_dict.values()]
    
    stats[f'mean_CD_AboveG{outstr}'] = np.nanmean(covAG)
    stats[f'sd_CD_AboveG{outstr}'] = np.nanstd(covAG)
    stats[f'cv_CD_AboveG{outstr}'] = np.nanstd(covAG) / np.nanmean(covAG)
    stats[f'median_CD_AboveG{outstr}'] = np.nanmedian(covAG)
    stats[f'iqr_CD_AboveG{outstr}'] = np.nanpercentile(covAG, q=75, method='median_unbiased') - np.nanpercentile(covAG, q=25, method='median_unbiased')
    stats[f'max_CD_AboveG{outstr}'] = np.nanmax(covAG)
    
    # Added a couple PAI stats here
    # using spherical leaf angle dist (k=0.5)
    covAG = np.array(covAG) # turn into numpy array for below calc
    stats[f'mean_PAI_AboveG'] = np.nanmean(-np.log(1-covAG, where=(1-covAG)>0) / 0.5)
    stats[f'sd_PAI_AboveG'] = np.nanstd(-np.log(1-covAG, where=(1-covAG)>0) / 0.5)
    
    # # Round everything to 3 decimal points
    # for k in stats.keys():
    #     stats[k] = np.round(stats[k], 3)
    
    return stats


def complexityAggMetrics(complexity_dict, outstr=''):
    
    stats = {}
    
    # Get all the data vals from the nested complexity dictionary
    vc = list(complexity_dict.values())

    for k in vc[0].keys():

        values = [v[k] for v in vc]

        stats[f'mean_{k}{outstr}'] = np.nanmean(values)
        stats[f'sd_{k}{outstr}'] = np.nanstd(values)
        stats[f'cv_{k}{outstr}'] = np.nanstd(values)/np.nanmean(values)
        stats[f'median_{k}{outstr}'] = np.nanmedian(values)
        stats[f'iqr_{k}{outstr}'] = np.nanpercentile(values, q=75, method='median_unbiased') - np.nanpercentile(values, q=25, method='median_unbiased')
        stats[f'max_{k}{outstr}'] = np.nanmax(values)
        # Note: Leaving out the range, since all will have a min of 0.
        # stats[f'range({k})'] = np.round(np.nanmax(values) - np.nanmin(values, where=values>0))
        
        if k == 'nlayers':
            
            # Also, return some stats about the number of layers
            # proportion of pixels with more than 1 layer in them
            stats[f'propMultiLayer{outstr}'] = np.sum(np.array(values) > 1) / np.array(values).size

    
    return stats



# Function for classifying veg type by max height of pixel
def classifyVeg_GGST(maxh, groundthreshold):
    
    if maxh <= groundthreshold:
    
        # 'ground'
        vegtype = int(1)
    
    elif ((maxh > groundthreshold) & (maxh <= 0.5)):
        
        # 'grass'
        vegtype = int(2)
        
    elif ((maxh > 0.5) & (maxh <=3)):
        
         # 'shrub'
        vegtype = int(3)
    
    elif maxh > 3:
        
        # 'tree'
        vegtype = int(4)
        
    else:
        
        # noise/nan class
        vegtype = int(-9999)
        
    return vegtype


# Function for classifying veg type by max height of pixel
def classifyVeg_GGW(maxh, groundthreshold):
    
    # tree_grass_shrub_bareground = []
    if maxh <= groundthreshold:
    
        # 'ground'
        vegtype = int(1)
    
    elif ((maxh > groundthreshold) & (maxh <= 1)):
        
        # 'grass'
        vegtype = int(2)
        
    elif maxh > 1:
        
        # 'woody'
        vegtype = int(5)
        
    else:
        
        # noise/nan class
        vegtype = int(-9999)
        
    return vegtype

# Combining metric dicts. from the whole plot
def wholePlotMetrics(perc_dict, cover_dict, complex_dict):
    
    wholeplot_dict = {}
    
    # Get all perc_dict values, except for 0
    for k in perc_dict.keys():
        
        if k != 0:
            
            if ((k != 'mean') & (k != 'std')):
                wholeplot_dict[f'{k}thPerc_plot'] = perc_dict[k][0]
            else:
                wholeplot_dict[f'{k}H_plot'] = perc_dict[k][0]
    
    # get D2 cover values 
    for i in [0, 1, 2, 3, 7]:
        
        h = str(cover_dict['HeightBins'][i]).replace('.', 'p')
        
        wholeplot_dict[f'Cover{h}m_plot'] = cover_dict['CoverD2'][i]

    # get all the complexity metrics
    for c in complex_dict:
        
        wholeplot_dict[f'{c}_plot'] = complex_dict[c]
        
    return wholeplot_dict

# # #
# Below functions are for saving metrics and outputting 2d/3d arrays

# define a function for filling arrays
def fill2Darray(data, shape, xindices, yindices, filteridx=None, plot=False):
    
    # make an empty output array, filled with nans
    output_array = np.full(shape, np.nan)
    
    # if that data needs to be filtered
    if filteridx:
        
        data = np.array(data)[filteridx]
        xindices = np.array(xindices)[filteridx]
        yindices = np.array(yindices)[filteridx]
    
    # Convert to int
    xindices = np.array(xindices).astype(int)
    yindices = np.array(yindices).astype(int)
    
    # xidx = [int(x) for x in xidx]
    # yidx = [int(y) for y in yidx]
        
    # fill the output array with data values, using the above indices
    output_array[yindices, xindices] = data
    
    if plot:
        fig, ax = plt.subplots()
        a = ax.imshow(output_array, cmap='magma')
        fig.colorbar(a)
        
    return output_array