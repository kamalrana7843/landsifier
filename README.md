# Landsifier
A python based library to estimate likely triggers of mapped landslides

# Landsifier Library Description

The library consitute three machine learning based method for finding the trigger of Landslide inventories.

# 1. Random Forest Based Method

This method is based on using 2D landslide polygon shape properties for classification. This method calculates various geometric properties of landslide polygon and based on these geometric properties it classify trigger of landslide inventories.


# 2. CNN (Convolutional Neural Networks) Based Method

This method convert landslide polygon data to landslide polygon Images. These converted landslide images are used as a input to CNN for landslide classification



# 3. TDA (Topological Data Analysis) Based Method

This method uses 3D shape of landslide by incorporating elevation data of landslides via SRTM 30 meters DEM.


# Sample output of each one of the method
![sample_output_landsifier](https://user-images.githubusercontent.com/63171258/154713717-884bcc0e-0817-48ef-a3b4-973c335a4c26.png)

# References


