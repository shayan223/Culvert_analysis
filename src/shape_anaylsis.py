import os

from PIL import Image
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from src.config import IMAGE_SPLIT_DIM


def disparity_normalization(disp):
    _min = np.amin(disp)
    _max = np.amax(disp)

    disp_norm = (disp - _min) * 255.0 / (_max - _min)
    disp_norm = np.uint8(disp_norm)

    return disp_norm


im = Image.open("./small_geo_data/mosaic_dem800.tif")
norm = disparity_normalization(im)


if not os.path.exists("./small_geo_data/normalized_images"):
    os.makedirs("./small_geo_data/normalized_images")

converted_image = Image.fromarray(norm)
converted_image.save("./small_geo_data/normalized_images/sample1.jpg")

if not os.path.exists("./small_geo_data/split_images"):
    os.makedirs("./small_geo_data/split_images")

if not os.path.exists("./small_geo_data/split_images/sample1"):
    os.makedirs("./small_geo_data/split_images/sample1")

img = cv2.imread("./small_geo_data/normalized_images/sample1.jpg")
for r in range(0, img.shape[0], IMAGE_SPLIT_DIM):
    for c in range(0, img.shape[1], IMAGE_SPLIT_DIM):
        cv2.imwrite(
            f"./small_geo_data/split_images/sample1/img{r}_{c}.png",
            img[r : r + IMAGE_SPLIT_DIM, c : c + IMAGE_SPLIT_DIM, :],
        )

points = pd.read_excel("./small_geo_data/point.xlsx")

print(points.iloc[2])

top_bound = points.iloc[0]["Unnamed: 1"]
bottom_bound = points.iloc[1]["Unnamed: 1"]
left_bound = points.iloc[2]["Unnamed: 1"]
right_bound = points.iloc[3]["Unnamed: 1"]
height = top_bound - bottom_bound

width = right_bound - left_bound

print("HEIGHT: ", height)
print("WIDTH: ", width)

true_points = []
false_points = []

for i in range(points["X"].shape[0]):
    if not np.isnan(points.iloc[i]["X"]):
        coord_true = Point(points.iloc[i]["X"], points.iloc[i]["Y"])
        true_points.append(coord_true)

    if not np.isnan(points.iloc[i]["X.1"]):
        coord_false = Point(points.iloc[i]["X.1"], points.iloc[i]["Y.1"])
        false_points.append(coord_false)

print("Number of TRUE points: ", len(true_points))
print("Number of FALSE points:  ", len(false_points))

geodata_true = gpd.GeoDataFrame(true_points, geometry=true_points)
geodata_false = gpd.GeoDataFrame(false_points, geometry=false_points)

img = plt.imread("./small_geo_data/normalized_images/sample1.jpg")
ax = plt.axes()

xlim = [left_bound, right_bound]
ylim = [bottom_bound, top_bound]

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.imshow(img, extent=[left_bound, right_bound, bottom_bound, top_bound])

geodata_true.plot(ax=ax, color="yellow")
geodata_false.plot(ax=ax, color="red")

if not os.path.exists("./small_geo_data/labeled_images"):
    os.makedirs("./small_geo_data/labeled_images")

plt.savefig("./small_geo_data/labeled_images/sample1.jpg")