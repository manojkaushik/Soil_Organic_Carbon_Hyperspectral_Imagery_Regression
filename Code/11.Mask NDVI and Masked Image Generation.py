# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:08:30 2022

@author: IISTDBT
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
from osgeo import gdal
from osgeo import osr

base_path = r"..\Data_Chilika_AVIRIS_NG\ang20151226t044452_rfl_v2m2"
Source_image_path = base_path + "\\2. Spectral Subsetting Full Image\\ang20151226t044452_corr_v2m2_img_smoothened_SpectralSubset"
Image_data = rasterio.open(Source_image_path)

print ("Image_data:", Image_data)
Crs = Image_data.crs
trans = Image_data.transform
cols = Image_data.width
rows = Image_data.height
bands= Image_data.count
print("rows:", rows, ", cols:", cols, ', bands:', bands)
Image_data = Image_data.read()

Image_data = np.moveaxis(Image_data, 0, -1) # moving number of bands at last

# NDVI calculation function
# In AVIRIS_NG HyperSpectral Image Red is 41st Index (651.9200 nm) band and NIR is 79th Index (842.2500 nm) band
# In PRISMA HyperSpectral Image BGR_Nir bands are is 10th, 18th, 35th, and 122th (approx corrosponding wavelengths are: 475, 535, 684, and 1554 nm respectively)

def ndvi(img):
    r=img[:, :, 41]
    nir=img[:, :, 79]
    dinom = (nir+r)
    numer = (nir-r)
    ndvi = np.where(dinom==0.0, 0.0, ((numer/dinom)*1.0))
    plt.imshow(ndvi,cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar()
    return ndvi

Image_ndvi = ndvi(Image_data)

plt.imshow(Image_ndvi, cmap="jet", vmin=-1, vmax=1)
plt.colorbar()

# Save NDVI Image
path = base_path + "\\3. Soil Segmentation Full Image\\NDVI_n_Mask_Image"
# np.save(path+"\\ndvi", Image_ndvi) # saving as .npy file

# Saving NDVI image after Normalization
# Normalized_ndvi = (((Image_ndvi - Image_ndvi.min()) / (Image_ndvi.max() - Image_ndvi.min())) * 255.9).astype(np.uint8) # Normalization
# im = Image.fromarray(Normalized_ndvi)
# im.save(path+"\\NDVI.png")


# Creating the mask for soil
def create_soil_mask(ndvi_image):
    soil_mask = np.empty([Image_ndvi.shape[0]*Image_ndvi.shape[1]])
    for i in range(0, len(ndvi_image)):
        if ndvi_image[i] > 0.2 and ndvi_image[i] <= 0.3: 
            soil_mask[i] = 1
        else:
            soil_mask[i] = 0
    return soil_mask.reshape(Image_ndvi.shape[0], Image_ndvi.shape[1])
            
soil_mask = create_soil_mask(Image_ndvi.reshape(Image_ndvi.shape[0]*Image_ndvi.shape[1]))

plt.imshow(soil_mask, cmap="gray", vmin=-1, vmax=1)
plt.colorbar()

np.unique(soil_mask, return_counts=True)

# Save Binary Mask Image
# np.save(path+"\\Soil_mask", soil_mask)
plt.imsave(path+"\\Soil_mask.png", soil_mask, cmap=cm.gray)


# Overlapping mask one by one on each channel
masked_val = []
for i in range(bands):
    band_array = Image_data[:,:,i]
    masked_val.append(np.multiply(band_array, soil_mask))

masked_value_array = np.asanyarray(masked_val) # Here [bands, row, col] 338, 2874, 1098


# rotate channels for just view
masked_value_array = np.moveaxis(masked_value_array,0,-1) # # Here [row, col, bands] 2874, 1098, 338
plt.imshow(masked_value_array[:,:,75]) # RGB Indecis 46, 22, 6


# rerotate the bands for further processing
masked_value_array = np.moveaxis(masked_value_array,0,-1) # rotating the bands
masked_value_array = np.moveaxis(masked_value_array,0,-1) # Here [bands, row, col] 338, 2874, 1098


# writing the output file
Output_filename = base_path + "\\3. Soil Segmentation Full Image\Soil_segmented_full_image.tiff"

dataset = gdal.Open(Source_image_path, gdal.GA_ReadOnly)
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32
driver = gdal.GetDriverByName( "GTiff" )
originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform()
output_file = driver.Create(Output_filename, cols, rows, bands, gdal_datatype)
output_file.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

for i, image in enumerate(masked_value_array, 1):
        output_file.GetRasterBand(i).WriteArray( image )

prj = dataset.GetProjection()
outRasterSRS = osr.SpatialReference(wkt=prj)
output_file.SetProjection(outRasterSRS.ExportToWkt())
output_file.FlushCache()
output_file = None



















