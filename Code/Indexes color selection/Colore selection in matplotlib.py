# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 22:59:34 2022

@author: manoj
"""
from PIL import Image
import os
import cv2
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import pandas as pd


path = r"//172.20.125.61/mtech/Major Jarmal Singh/Demmin_2015_10_01/Hyspex/Smoothed_images/Resempled_Big_Images/SOC Image/zero_to_50_soc_img.npy"
ndvi_img = Image.open(path)

cmaps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
len(cmaps)
cmaps[11]

x = 4
skip = 16*10
f, axarr = plt.subplots(4, 4)
for i in range (4):
    axarr[i, 0].imshow(ndvi_img, cmap=cmaps[i*x+skip])
    axarr[i, 0].set_title(i*x+skip, fontsize=10)
    
    axarr[i, 1].imshow(ndvi_img, cmap=cmaps[i*x+1+skip])
    axarr[i, 1].set_title(i*x+1+skip, fontsize=10)
    
    axarr[i, 2].imshow(ndvi_img, cmap=cmaps[i*x+2+skip])
    axarr[i, 2].set_title(i*x+2+skip, fontsize=10)
    
    axarr[i, 3].imshow(ndvi_img, cmap=cmaps[i*x+3+skip])
    axarr[i, 3].set_title(i*x+3+skip, fontsize=10)
plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)

# test the best color index
img = np.load(path, allow_pickle=True)
fig = plt.figure(num=None, figsize=(16, 16), dpi=200, facecolor='w', edgecolor='red')
plt.imshow(img, cmap=cmaps[11])
plt.colorbar()
# plt.show()

# Saving figure by changing parameter values
save_path = r"//172.20.125.61/mtech/Major Jarmal Singh/Demmin_2015_10_01/Hyspex/Smoothed_images/Resempled_Big_Images/SOC Image/zero_to_50_soc_index11.png"
plt.savefig(save_path,
            bbox_inches="tight",
            dpi=200,
            pad_inches=0.3, 
            transparent=True)




# plotting x and y
filename = r"\\172.20.125.61\mtech\Major Jarmal Singh\Chilika_aviris_NG_India\ang20151226t043231_rfl_v2m2\Chilika_spectral_spatial_subsets_smooth_resampled\Soil_Segmented_image\part-1_loc_soc.xlsx"
data = pd.read_excel(filename, sheet_name='part-1_loc', engine='openpyxl')

y = data['Actual'].values
y_cv = data['Predicted'].values

plt.figure(figsize=(6, 6))
with plt.style.context('ggplot'):
    plt.scatter(y, y_cv, color='red')
    plt.plot(y, y, '-g', label='Actual regression line')
    z = np.polyfit(y, y_cv, 1)
    plt.plot(np.polyval(z, y), y, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
