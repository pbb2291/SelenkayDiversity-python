# CalcLasHeight Function for Point Clouds
# Using pdal: https://pdal.io/en/stable/
import laspy
import pdal
from pathlib import Path

# A function to compute height above ground in las files
# Note: It assumes that points in the input las files
# have already been classified-
# with class 2 points being ground. 
# Inputs: 
#  inf = las file path as a string
#  opath = file path of output directory as a string
#  of = output file name as a string
#  buffer = a numeric value indicating a distance to clip 
#           from the edges of the point cloud (generally, leave this as 0)
# Outputs:
#  metadata = a json-formatted string describing 
#             the arguments and outputs in the pdal pipeline
#  Writes an output .las file with a "HeightAboveGround" attribute for each point
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