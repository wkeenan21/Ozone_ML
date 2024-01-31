import numpy as np
from matplotlib import pyplot as plt
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging

# our bbox in 4326
bbox_4326 = [39.259770,-105.632996,40.172953,-104.237732]
# transform to 3857
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
point1 = transformer.transform(bbox_4326[0],bbox_4326[1])
point2 = transformer.transform(bbox_4326[2], bbox_4326[3])
bbox = [point1[0], point1[1], point2[0], point2[1]]

# get the spacing for a grid
x_steps = (bbox[0] - bbox[2]) / 1000
y_steps = (bbox[1] - bbox[3]) / 1000
# Generate random data following a uniform spatial distribution
# of nodes and a uniform distribution of values in the interval
# [2.0, 5.5]:
N = 7
lon = 360.0 * np.random.random(N)
lat = 180.0 / np.pi * np.arcsin(2 * np.random.random(N) - 1)
z = 3.5 * np.random.rand(N) + 2.0

# Generate a regular grid with 60° longitude and 30° latitude steps:
grid_lon = np.linspace(bbox[0], bbox[2], abs(int(x_steps)))
grid_lat = np.linspace(bbox[1], bbox[3], abs(int(y_steps)))

# Create ordinary kriging object:
OK = OrdinaryKriging(
    lon,
    lat,
    z,
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    coordinates_type="geographic",
)

# Execute on grid:
z1, ss1 = OK.execute("grid", grid_lon, grid_lat)

# Create ordinary kriging object ignoring curvature:
OK = OrdinaryKriging(
    lon, lat, z, variogram_model="linear", verbose=False, enable_plotting=False
)

# Execute on grid:
z2, ss2 = OK.execute("grid", grid_lon, grid_lat)