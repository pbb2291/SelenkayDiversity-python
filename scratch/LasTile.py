import laspy
import os
import pdal
import glob
import json
import geopandas as gpd
from shapely.geometry import Polygon
import fiona

# Using Pdal instead of laspy
# The normalizeheights function works well
# can be run on all tiles
# Pdal is probably the way to go for merging cloud info as well.
# Only thing- it's confusing how you can get at metadata (it seems to be within the pipeline object after it's executed...)
# useful links:
# https://pdal.io/about.html#id7
# https://pdal.io/pipeline.html#option-files
# https://pdal.io/stages/filters.hexbin.html#filters-hexbin - hexbin would be great for pdenisties and data extents
# https://www.spatialised.net/lidar-qa-with-pdal-part-1/
# https://docs.tiledb.com/geospatial/pdal

#Laspy:
# https://pythonhosted.org/laspy/tut_part_1.html
# https://github.com/laspy/laspy

class LasTile:

    def __init__(self, lasfile=None,
                 bbox = [], tilenum=None, fields=[],
                 pden=None, pcount=None):

        self.filename = os.path.abspath(lasfile)
        # make projdir (can probably get this from project class)
        self.projdir = '\\'.join(self.filename.split('\\')[0:-3])
        self.tilenum = self.filename.split('\\')[-1].split('_')[-1].split('.')[0]

        # Get stats of tiles
        # self.calcstats()

        # Get data from it
        # self.GetData()

        # Get Fields...
        # self.fields = []

    def convertfromtsolidbin(self, of=None):

        # If no outfile name, make one in the same dir
        if of==None:

            of = self.filename.split('\\')[-1]
            opath = self.filename.split(of.split('.')[0])[0]
            of = of.split('.')[0] + "_converted.las"
            of = opath + of

        injson = """
                    [
                        {
                            "type": "readers.terrasolid",
                            "filename": "in.bin"
                        },
                        {
                            "type": "writers.las",
                            "filename": "out.las"
                        }
                    ]
                    """

        injson = injson.replace("in.bin", self.filename.replace('C:', '').replace('\\', '/'))
        injson = injson.replace("out.las", of.replace('C:', '').replace('\\', '/'))

        pipeline = pdal.Pipeline(injson)
        pipeline.execute()

    def normalizeheights(self, of=None, overwriteZ=False):

        # If no outfile name, make one in the same dir
        if of==None:

            of = self.filename.split('\\')[-1]
            opath = self.filename.split(of.split('.')[0])[0]
            of = of.split('.')[0] + "_NormHeight.las"
            of = opath + of

        if overwriteZ == False:

            # define injson for pdal pipeline
            injson= """
            [
                "in.las",
                {
                    "type":"filters.hag_delaunay"
                },
                {
                    "type": "writers.las",
                    "filename": "out.las",
                    "extra_dims": "HeightAboveGround=float32"
                }
            ]
            """

        else:
            # Note: This overwrite z values with height above ground
            # to undo, remove the "filter ferry" section
            injson= """
            [
                "in.las",
                {
                    "type":"filters.hag_delaunay"
                },
                {
                    "type": "filters.ferry",
                    "dimensions": "HeightAboveGround=>Z"
                },
                {
                    "type": "writers.las",
                    "pdal_metadata":"true",
                    "filename": "out.las",
                    "extra_dims": "HeightAboveGround=float32"
                }
            ]
            """

        # Replace args with in and out file
        # need to redo file path names (removing "C:" and stuff) for json
        # NOTE: this will be interesting on the server, maybe have an if statement here to check for C: or drive file paths
        injson = injson.replace("in.las", self.filename.replace('C:', '').replace('\\', '/'))
        injson = injson.replace("out.las", of.replace('C:', '').replace('\\', '/'))

        pipeline = pdal.Pipeline(injson)
        pipeline.execute()
        # arrays = pipeline.arrays
        # metadata = pipeline.metadata
        # log = pipeline.log

    def calcstats(self, outputkml=True, plotboundary=False, verbose=False):

        injson= """
        [
            "in.las",
            {
                "type":"filters.hexbin"
            }
        ]
        """

        #Edge length of 10 means hexagons for sampling are 10 m in size
        injson= injson.replace("in.las", self.filename.replace('C:', '').replace('\\', '/'))

        pipeline = pdal.Pipeline(injson)
        self.pcount = pipeline.execute()

        # Load the metadata stored as a json formatted string in the pipeline
        stats = json.loads(pipeline.metadata)

        self.coverage = stats['metadata']['filters.hexbin']['area']
        self.pdensity = stats['metadata']['filters.hexbin']['avg_pt_per_sq_unit']
        self.pspacing = stats['metadata']['filters.hexbin']['avg_pt_spacing']

        # Output KML and SHP files of boundary into docs folder
        # https://geopandas.org/gallery/create_geopandas_from_pandas.html
        # fiona.supported_drivers

        # set settings for opening KML files in fiona w/ geopandas
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

        # Note: May want to adjust these stats... they aren't really correct
        # See: https://www.spatialised.net/lidar-qa-with-pdal-part-1/
        try:
            geodict = {'f': [self.filename],
                        'tilenum': [self.tilenum],
                        'area': [self.coverage],
                        'avg_pt_per_sq_unit': [self.pdensity],
                        'pspacing': [self.pspacing],
                        'density': [stats['metadata']['filters.hexbin']['density']],
                        'boundary': Polygon(stats['metadata']['filters.hexbin']['boundary_json']['coordinates'][0])}

            self.geodf = gpd.GeoDataFrame(geodict, geometry='boundary', crs="EPSG:32736")

            # Make output dirs in docs folder
            # shpdir = 'C:\\Users\\peter\\DaviesLab\\data\\UAV\\2020\\Kruger_Dec\\20191119-195840_OTSExclosureS\\docs\\lidarboundaryshp\\'
            # kmldir = 'C:\\Users\\peter\\DaviesLab\\data\\UAV\\2020\\Kruger_Dec\\20191119-195840_OTSExclosureS\\docs\\lidarboundarykml\\'
            shpdir = self.projdir + '\\docs\\lidarboundaryshp\\'
            kmldir = self.projdir + '\\docs\\lidarboundarykml\\'
            if not os.path.exists(kmldir):
                os.mkdir(kmldir)
            if not os.path.exists(shpdir):
                os.mkdir(shpdir)

            self.geodf.to_file(shpdir + 'Tile' + self.tilenum + '.shp')
            self.geodf.to_file(kmldir + 'Tile' + self.tilenum + '.kml', driver='KML')

            if plotboundary == True:
                self.geodf.plot()
                # HA it twerks!
            if verbose == True:
                print(f'Boundary files in {shpdir} and {kmldir}.')

        except:
            print(f'Tile {self.tilenum} errored out.')

    def GetData(self):

        # Load in data as Np arrays with Laspy
        self.data = laspy.file.File(self.filename, mode="r")

        # Pipe out important stuff
        # can use the "label" argument to create subsets, like wood, grass, trees, etc.
        # perhaps, not the easiest way to do it! 11/30/2020 - think of better ways

        # self.data[self.label] = {
        #     "X":temp.get_x_scaled(),
        #     "Y":temp.get_y_scaled(),
        #     "Z":temp.get_z_scaled(),
        #     "Red":temp.red,
        #     "Green":temp.green,
        #     "Blue":temp.blue,
        #     "GCC": np.array(temp.green / (temp.red + temp.blue + temp.green)),
        #     "Class":temp.Classification,
        #     "Amplitude":temp.amplitude,
        #     "Deviation":temp.deviation,
        #     "Distance":temp.distance,
        #     "ScanAngle":temp.scan_angle
        # }

        # def clipshp(self, shapefile=None):
        #  # Honestly, just use lasclip, much easier...
        #     # https: // pdal.io / stages / filters.crop.html  # filters-crop
        #
        #     try:
        #
        #         injson= """
        #         [
        #             "in.las",
        #             {
        #                 "type":"filters.crop",
        #                 "point":"POINT(0 0 0)",
        #                 "distance": 500
        #             },
        #             {
        #                 "type":"writers.las",
        #                 "filename":"file-cropped.las"
        #             }
        #         ]
        #         """

from pathlib import Path

# Use to make heights at Uhuru Plots
laspaths = Path('C:\\Users\\peter\\DaviesLab\\server\\download\\Nkulu_PointCloud')

for path in laspaths.glob('*.las'):
    templas = LasTile(path)
    templas.normalizeheights(overwriteZ=False)