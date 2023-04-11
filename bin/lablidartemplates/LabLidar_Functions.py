# Lab Lidar Functions Accessible to the Davies Lab
# PBB 8/19/22

import geopandas as gpd
import pandas as pd
import numpy as np
import laspy
from shapely.geometry import Polygon
import time
import glob
import os


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
def lasClip_Shape(shpdf=None, lasdir=None, outdir=None, featureIDcol='treeID', epsg='32736', verb=True):
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
    
    # for each polygon feature in the shapefile dataframe
    # for idx, feature in shpdf.iterrows():

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
    zfirst = points[heightcol][(points[retnumcol] == 1)]
    zfirst_veg =  points[heightcol][(points[classcol] == 1) & (points[retnumcol] == 1)]
    
    # 2 Methods for Cover Used Here
    # Method D1 treats each 1st return with a weight of 1
    # Method D2, each first return is weighted based on the number of returns in the pulse (loosely based on Armston et al., 2013)
    
    # Compute weights for D2
    weights = []
    # for each first return, vegetation point, append a weight equal to 1 / the number of returns in the pulse
    for nr in points[numretcol][(points[classcol] == 1) & (points[retnumcol] == 1)]:
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
    FHPD1 = -np.log(1-coverD1byH) / np.diff(np.array(heightbins))
    FHPD2 = -np.log(1-coverD2byH) / np.diff(np.array(heightbins))
    
    # NOTE: You will likely get warnings from numpy - RuntimeWarning: divide by zero encountered in log
    # That is likely because coverD1 or coverD2 become fully saturated (1) - so it tries to take -np.log(0). 
    # TBD a solution for this... can't just set it to NA (becuase it has plant material, it's just the cover fraction is saturated)
    # Maybe a method using all returns (not just first) could avoid this

    cover = {'CoverD1':np.array(coverD1),
             'CoverD2':np.array(coverD2),
             'CoverD1byH':np.array(coverD1byH),
             'CoverD2byH':np.array(coverD2byH),
             'FHPD1':np.array(FHPD1),
             'FHPD2':np.array(FHPD2),
             'HeightBins':np.array(heightbins),
             'Npulses':np.array(Npulses)}

    return cover


# Function for calculating percentile heights
def calcPercentileHeights(points, groundthres=0, returnHeights=True, heightcol='HeightAboveGround'):
    
    # Calculate Percentile Metrics of Height
    perc_dict= {0:[],
                25:[],
                50:[],
                75:[],
                98:[],
                100:[],
                'mean':[],
                'std':[]}

    # Get Heights for given cell
    heights = points[heightcol]

    # If there are any heights left
    if heights.size > 0:

        perc_dict[0].append(np.quantile(heights, [0]).flat[0])
        perc_dict[25].append(np.quantile(heights, [0.25]).flat[0])
        perc_dict[50].append(np.quantile(heights, [0.5]).flat[0])
        perc_dict[75].append(np.quantile(heights, [0.75]).flat[0])
        perc_dict[98].append(np.quantile(heights, [0.98]).flat[0])
        perc_dict[100].append(np.quantile(heights, [1.0]).flat[0])
        perc_dict['mean'].append(np.mean(heights).flat[0])
        perc_dict['std'].append(np.std(heights).flat[0])

    # else, height stats are 0
    else:
        perc_dict[0].append(0)
        perc_dict[25].append(0)
        perc_dict[50].append(0)
        perc_dict[75].append(0)
        perc_dict[98].append(0)
        perc_dict[100].append(0)
        perc_dict['mean'].append(0)
        perc_dict['std'].append(0)

    if returnHeights:
        return perc_dict, heights

    else:
        return perc_dict