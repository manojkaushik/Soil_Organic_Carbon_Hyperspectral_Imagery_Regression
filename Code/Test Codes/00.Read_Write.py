# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:51:44 2020
@IIST: MAnohar
"""
#numpy, scipy, matplotlib, rasterio, gdal, sklearn

# import gdal
# from gdal import *
# from gdal import osr
from osgeo import gdal
import rasterio
import numpy as np
import matplotlib.pyplot as plt

#input the file
filename = r"E:\Hymap_correction_product_ISPRS_symposium\080807_Doeberitz_01_rad.bsqFACT_corrected_refl.tif"

#Reading File Information
dataset = rasterio.open(filename)
print (dataset)
Crs = dataset.crs
trans = dataset.transform
cols = dataset.width
rows = dataset.height
bands= dataset.count
print(cols,rows,bands)


with rasterio.open(filename) as r:
    raster_matrix_image= r.read()


nir = raster_matrix_image[75,:,:]
red = raster_matrix_image[55,:,:]

dinom = nir+red
numer = nir-red

ndvi = np.where(dinom==0.0, 0.0, numer/dinom)

ndvi_1= numer/dinom

plt.figure()
plt.imshow(ndvi)

#writing the output file
Output_filename = r"C:\Users\USER\Desktop\otpt.tif"
dataset_1 = gdal.Open(filename, gdal.GA_ReadOnly)
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32
driver = gdal.GetDriverByName( "GTiff" )
originX, pixelWidth, b, originY, d, pixelHeight = dataset_1.GetGeoTransform() 
output_file = driver.Create(Output_filename, cols, rows, 1, gdal_datatype )
output_file.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
# for i, image in enumerate(ndvi, 1):
output_file.GetRasterBand(i).WriteArray( ndvi)
prj=dataset_1.GetProjection()
outRasterSRS = gdal.osr.SpatialReference(wkt=prj)
output_file.SetProjection(outRasterSRS.ExportToWkt())
output_file.FlushCache()
output_file = None




















