#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
#from scipy import ndimage as ndi
import pandas as pd
import pathlib
import glob
import os
import re
import cv2 as cv2
import math
import argparse

# python /Users/MiguelT/Box/My_scripts/count_colonies.py --folder /Users/MiguelT/Box/Jordi/Image/Fotos_last/Images_delivery_3/ --thres 75

parser = argparse.ArgumentParser(description='Count Colonies')
parser.add_argument('--folder', metavar='folder', type=str, help='Input folder')
parser.add_argument('--thres', metavar='Qorts_folder', type=str, help='size threshold')

args = parser.parse_args()

folder = str(args.folder)
thres = int(args.thres)

# Choose main folder
#computer = '/Users/torres3'
#computer = '/Users/MiguelT'
#computer = '/home/jovyan/work/'
#folder = f'/{computer}Jordi/Image/Fotos_last/Images_delivery_2/'


# Create and define folders
folder_out_colored = f'{folder}/colored/'
folder_out_BW = f'{folder}/BW/'
folder_out_filtered = f'{folder}/filtered/'

pathlib.Path(f'{folder_out_colored}').mkdir(exist_ok=True)
pathlib.Path(f'{folder_out_BW}').mkdir(exist_ok=True)
pathlib.Path(f'{folder_out_filtered}').mkdir(exist_ok=True)

# Threshold for contour

threshold_area = thres


# Get files from main folder
files = glob.glob(f'{folder}/*.jpg')
name_list = []
for file in files:
    name = re.findall(fr'(?<={folder}).*.(?=.jpg)', file)[0]
    name_list.append(name)

# Full pipeline

results = {}

for file in name_list:
    # Load the  image and convert to HSV colourspace
    image = cv2.imread(f'{folder}{file}.jpg')
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    # Define lower and uppper limits of our background
    brown_lo=np.array([0,0,0])
    brown_hi=np.array([50,60,255])
    
    # Mask image to only select limits
    mask=cv2.inRange(hsv,brown_lo,brown_hi)
    
    # Change image to red where we found limits
    image[mask>0]=(255,255,255)
    
    #image = cv2.fastNlMeansDenoising(image);
    
    # Crop with a circle
    hh, ww = image.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2
    radius1 = hh2 - 20
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (hh2, ww2), radius1 , 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    
    # create zeros mask 2 pixels larger in each dimension
    mask = np.ones([hh + 2, ww + 2], np.uint8)
    cv2.floodFill(masked, mask, (0,0), 255)
    
    # Save image
    cv2.imwrite(f'{folder_out_filtered}{file}_backcorr.jpg',masked)
    
    # Second part 
    img = cv2.imread(f'{folder_out_filtered}{file}_backcorr.jpg', 0)
    img_2 = cv2.imread(f'{folder}{file}.jpg')
    
    # Denoising
    #denoisedImg = cv2.fastNlMeansDenoising(img);
    denoisedImg = img
    
    # Threshold (binary image)
    th, threshedImg = cv2.threshold(denoisedImg, 200, 255,cv2.THRESH_BINARY_INV|cv2.ADAPTIVE_THRESH_GAUSSIAN_C) # src, thresh, maxval, type
    
    # Perform morphological transformations 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)
    
    # Find and draw contours
    radius2 = radius1 + 5
    morphImg = cv2.circle(morphImg, (hh2, ww2), radius2 , 1, 10)
    
    # Print % of covered plate
    number_of_white_pix = np.sum(img == 255)
    perc_cells = round((100 - ((number_of_white_pix / (round(math.pi * radius2**2))) * 100)), 2)
    perc = str(perc_cells) + '% covered'
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (30, 70)  
    fontScale = 2
    color = (0, 0, 255) 
    thickness = 3
    cv2.putText(morphImg, perc, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    perc = None
    cv2.imwrite(f'{folder_out_BW}{file}_morphImg.jpg', morphImg)
    img_3 = cv2.imread(f'{folder_out_BW}{file}_morphImg.jpg')
    img_3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY) #gray scale image
    thresh, im_bw = cv2.threshold(img_3, 100, 255, cv2.THRESH_BINARY) #im_bw: binary image
    contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for index in reversed(range(len(contours))):
        area = cv2.contourArea(contours[index]) 
        if area < threshold_area:  
            contours.pop(index)
    
    threshold_area_2 = 1000000
    for index in reversed(range(len(contours))):
        area = cv2.contourArea(contours[index]) 
        if area > threshold_area_2:  
            contours.pop(index)
    
    cv2.drawContours(img_2, contours, -1, (0,0,255), 3)
    # Print number colonies
    number = len(contours) 
    org = (30, 140)  
    color = (0, 230, 100) 
    colonies = str(number) + ' colonies'
    cv2.putText(img_2, colonies, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    
    cv2.imwrite(f'{folder_out_colored}{file}_original_colored.jpg', img_2)
    to_table = number,perc_cells
    results[file] = to_table
    results_df = pd.DataFrame.from_dict(results,orient='Index')
    
cv2.destroyAllWindows()

results_df.reset_index(inplace = True)
results_df.columns = ['Photo','Number_colonies','Covered_colonies']
min_number = min(name_list)
max_number = max(name_list)
table_name = f"{folder}Table_{min_number}_to_{max_number}.csv"
results_df.to_csv(table_name, sep=",", header=True, index=False, mode='w')
