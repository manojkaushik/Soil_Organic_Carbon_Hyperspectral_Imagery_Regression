

import pickle
import rasterio
import numpy as np
from osgeo import gdal
from osgeo import osr
import matplotlib.pyplot as plt


wd_path = r"..\Data_Chilika_AVIRIS_NG\ang20151226t044452_rfl_v2m2 worked"

# load the Masked Image(Segmented Image)
path_source_image = wd_path + "\Full Image\\3.Soil Segmented\Soil_segmented.tiff"
Xs = rasterio.open(path_source_image)
Xs = Xs.read()

binary_mask = np.where(Xs[0] != 0, 1, 0)
plt.imshow(binary_mask, cmap="jet", vmin=-1, vmax=1)


# Reading File Information
dataset = rasterio.open(path_source_image)
print (dataset)
Crs = dataset.crs
trans = dataset.transform
cols = dataset.width
rows = dataset.height
bands= dataset.count
print(cols,rows,bands)


# normalize the image according to ASD reflactance values between 0 to 1
Xs = np.moveaxis(Xs, 0, -1) # moving number of bands at last
Xs = Xs.reshape(Xs.shape[0]*Xs.shape[1], Xs.shape[2])
Xs = Xs / 10000 # 10000 bcz of ENVI standards


# loading model for testing again
model_path = r"..\SOC Project\Saved_model\Sheet15_338_bands_Chilka_AVIRIS_NG_HSI\ENR_model.sav"
loaded_model = pickle.load(open(model_path, 'rb'))
y_pred = loaded_model.predict(Xs[0].reshape(1, -1))
print(y_pred[0])


# SOC prediction and masking
soc_pred_img = loaded_model.predict(Xs) # SOC prediction
soc_pred_img = soc_pred_img.reshape(rows, cols) # Reshaping
soc_pred_img = np.multiply(soc_pred_img, binary_mask) # AND opration


# Values narrowing 
non_neg_soc_pred_img = np.where(soc_pred_img<0.0, 0, soc_pred_img)
zero_to_50_soc_pred_img = np.where(non_neg_soc_pred_img>50.0, 0, non_neg_soc_pred_img)
zero_to_20_soc_pred_img = np.where(non_neg_soc_pred_img>20.0, 0, non_neg_soc_pred_img)


# Histograms
plt.hist(soc_pred_img)
plt.hist(non_neg_soc_pred_img)
plt.hist(zero_to_50_soc_pred_img)
plt.hist(zero_to_20_soc_pred_img)


# Set image name and variable
image_variable = zero_to_50_soc_pred_img
image_name = "zero_to_2zero_to_50_soc_pred_img0_soc_pred_img"


# path to save SOC prediction images
path = wd_path + "\Full Image\\4.Final Prediction"
fig = plt.figure(num=None, figsize=(16, 16), dpi=200, facecolor='w', edgecolor='red')
plt.imshow(image_variable, cmap="CMRmap_r")
plt.colorbar()
# plt.axis('off')
plt.savefig(path + "\\" + image_name + ".png",
            bbox_inches="tight", dpi=200,
            # pad_inches=0.3,
            # transparent=False
            )


# saving SOC predicted image as tiff with geo cordinates
Output_filename = path + "\\" + image_name + ".tiff"
dataset = gdal.Open(path_source_image, gdal.GA_ReadOnly)
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32
driver = gdal.GetDriverByName( "GTiff" )
originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform()
output_file = driver.Create(Output_filename, cols, rows, 1, gdal_datatype)
output_file.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

for i, image in enumerate([image_variable], 1):
    # print(type(i), " ", type(image))
    # print(i, " ", image.shape)
    output_file.GetRasterBand(i).WriteArray(image)

prj = dataset.GetProjection()
outRasterSRS = osr.SpatialReference(wkt=prj)
output_file.SetProjection(outRasterSRS.ExportToWkt())
output_file.FlushCache()
output_file = None




