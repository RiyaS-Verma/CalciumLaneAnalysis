import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '60' #needed for slurm numpy executions
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import mstats
import pickle
import statistics
from pylab import *
from skimage import io
from sklearn import linear_model
from pathlib import Path
from scipy.signal import savgol_filter
from skimage.transform import rescale
from skimage.exposure import rescale_intensity
from skimage.exposure import rescale_intensity
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import probabilistic_hough_line
from skimage import feature
from skimage import data
from skimage import filters
import matplotlib.pyplot as plt
from skimage.util import invert
from skimage.filters import threshold_multiotsu
from skimage.filters import butterworth
import cv2 as cv
from skimage import morphology
from skimage import  (measure, morphology, segmentation)
from scipy import ndimage as ndi
from skimage.segmentation import flood, flood_fill
from collections import deque
import numpy as np

#adjust this main function to be compatible with snakemake
def main():
    if not os.path.exists('csv'):
        os.makedirs('csv')
    csvdir = directory+"csv/"
    tifdir = directory+"tifs/"
    tifs = Path(tifdir).glob('*.tiff')
    fps = 100 #hardcoded for riya's samples make params snakemake
    for tif in tifs:
        filename = str(os.path.basename(tif).split(".ome.tiff")[0])
        image_path = directory+"tifs/"+filename+".ome.tiff"
        lanes = findLanes(image_path, filename)
        islands = extract_cell_islands(lanes)
        meanDF = extractROIMeanIntensity(islands, image)
        df_csv = addTimeToCSV(meanDF,fps)
        df_csv.to_csv(filename+'.csv', index=False)


# This method finds the brightest frames, uses the frame to detect ROIs using scipy image filtering
def findLanes(image,filename):
    #index is the first element of brightest frame
    print("Reading image "+filename)
    image = io.imread(image)
    print("Image loaded")
    brightest_frame = max([(idx, np.sum(arr)) for idx, arr in enumerate(image)], key= lambda x: x[1])
    p21, p98 = np.percentile(image[brightest_frame[0]], (2, 98))
    print("Rescaling image")
    rescaled_image = rescale_intensity(image[brightest_frame[0]], in_range=(p21, p98))
    plt.imsave(filename+'_brightestframe_contrasted.pdf',lrescaled_image, cmap='gray')
    print("Buttering image")
    buttering = butterworth(rescaled_image)
    print("Smoothing image")
    smooth = filters.gaussian(buttering, sigma=1.5)
    print("Otsu thresholding")
    thresh_value = filters.threshold_otsu(smooth)
    thresh = smooth > thresh_value
    print("Cleaning lanes")
    lanes_cleaned = morphology.remove_small_objects(thresh,5000)
    plt.imsave(filename+'_lanes_detected.pdf',lanes_cleaned, cmap='gray')
    return lanes_cleaned, image

# This method has contributions from mohit kumar and borislav milkov
# This method isolates the ROI (where there is calcium signal) and uses breath first search to isolate pixels corresponding to the coordinates of the ROI
def extract_cell_islands(mat) -> list:
    cell_sections = []
    # A function to check if a given cell
    # (u, v) can be included in DFS
    print("Performing breath first search for detecting lanes")
    def isSafe(mat, i, j, vis, x_max, y_max):

        return ((i >= 0) and (i < x_max) and
                (j >= 0) and (j < y_max) and
            (mat[i][j] and (not vis[i][j])))

    def BFS(mat, vis, si, sj, x_max, y_max):

        # These arrays are used to get row and
        # column numbers of 8 neighbours of
        # a given cell
        row = [-1, -1, -1, 0, 0, 1, 1, 1]
        col = [-1, 0, 1, -1, 1, -1, 0, 1]

        # Simple BFS first step, we enqueue
        # source and mark it as visited
        q = deque()
        q.append([si, sj])
        vis[si][sj] = True

        # Next step of BFS. We take out
        # items one by one from queue and
        # enqueue their unvisited adjacent
        this_cell_section_island = []
        this_cell_section_island.append((si,sj))
        while (len(q) > 0):
            temp = q.popleft()

            i = temp[0]
            j = temp[1]

            # Go through all 8 adjacent
            for k in range(8):
                if (isSafe(mat, i + row[k], j + col[k], vis, x_max, y_max)):
                    vis[i + row[k]][j + col[k]] = True
                    q.append([i + row[k], j + col[k]])
                    this_cell_section_island.append((i + row[k], j + col[k]))
        cell_sections.append(this_cell_section_island)


    # This function returns number islands (connected
    # components) in a graph. It simply works as
    # BFS for disconnected graph and returns count
    # of BFS calls.
    def countIslands(mat):

        x_max, y_max = mat.shape
        # Mark all cells as not visited
        vis = [[False for i in range(y_max)]
                    for i in range(x_max)]

        # 5all BFS for every unvisited vertex
        # Whenever we see an univisted vertex,
        # we increment res (number of islands)
        # also.
        res = 0

        for i in range(x_max):
            for j in range(y_max):
                if (mat[i][j] and not vis[i][j]):
                    BFS(mat, vis, i, j, x_max, y_max)
                    res += 1

        return res
    countIslands(mat)
    return cell_sections


# This method has contributions from borislav milkov
## This method isolates the mean frame intensity for each
def extractROIMeanIntensity(ROIs, image):
    all_roi_means = [[] for _ in range(len(ROIs))]
    res_dict = {}
    for frame in image:
        for idx, roi in enumerate(ROIs):
            pixel_vals = []
            for t_coords in roi:
                pixel_vals.append(frame[t_coords[0]][t_coords[1]])
            all_roi_means[idx].append(mean(pixel_vals))
    df = pd.DataFrame(all_roi_means)
    df = df.transpose()
    df.columns = ["ROI_{}".format(x+1) for x in range(len(all_roi_means))]
    print("Found Mean Intensity of "+str(len(all_roi_means))+" lanes")
    return df

# This method adds the frame time into (ms) of the dataframe based on the fps
## Need to write a method that isolates FPS from .ome.tiff metadata

def addTimeToCSV(ROI_df,fps):
    #hardcoded for 100fps videos will need to be modified to convert fps to sec idx
    ROI_df.insert(0, column='time',value=ROI_df.index * (1000/fps))
    return ROI_df
