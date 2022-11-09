
Topological Feature-Based Method Example 
=========================================
This method uses 3D shape of landslide by incorporating elevation data of landslides via SRTM 30 meters DEM.




Import topological features based module functions from Landsifier library 
------------------------------------------------------------------------
.. code:: ipython3

   import topo_based_model 
   from topo_based_model import read_shapefiles
   from topo_based_model import make_3d_polygons
   from topo_based_model import get_ml_features
   from topo_based_model import classify_inventory_tda
   from topo_based_model import plot_topological_results
   import numpy as np
    
    
In this example, we are using six landslide inventories spread over Japan archipaelogo. Out of six-inventories, three inventories are earthquake-triggered
and rest three inventories are rainfall-induced inventories.

- Earthquake-triggered inventories (Hokkaido, Iwata and Niigata region)

- Rainfall-induced inventories (Kumamoto, Fukuoka and Saka region)

Import Shapefiles of Earthquake and Rainfall triggered landslide Inventories
----------------------------------------------------------------------------

- Earthquake triggered inventories

.. code:: ipython3

    earth_hokkaido_shapefile = read_shapefiles("E:/Germany Visit/Landslide Datasets/Japan Inventory/Earthquake_hokkaido_polygons.shp")
    earth_iwata_shapefile = read_shapefiles("E:/Germany Visit/Landslide Datasets/Japan Inventory/Earthquake_iwata_polygons.shp")
    earth_niigata_shapefile =read_shapefiles("E:/Germany Visit/Landslide Datasets/Japan Inventory/Earthquake_niigata_polygons.shp")
    
    
- Rainfall triggered inventories

.. code:: ipython3

    rain_kumamoto_shapefile = read_shapefiles("E:/Germany Visit/Landslide Datasets/Japan Inventory/Rainfall_kumamoto_polygons.shp")
    rain_fukuoka_shapefile = read_shapefiles("E:/Germany Visit/Landslide Datasets/Japan Inventory/Rainfall_fukuoka_polygons.shp")
    rain_saka_shapefile = read_shapefiles("E:/Germany Visit/Landslide Datasets/Japan Inventory/Rainfall_saka_polygons.shp")
    
DEM files 
----------------------------------------------------------------------------  
.. code:: ipython3
    
   dem_location="E:/dem_japan/"
   inventory_name_list=['hokkaido.tif','iwata.tif','niigata.tif','kumamoto.tif','fukuoka.tif','saka.tif']   
    

Convert 2D landslide polygons to 3D landslide point cloud 
----------------------------------------------------------
- Earthquake triggered inventories


.. code:: ipython3

    pointcloud_earth_hokkaido=make_3d_polygons(earth_hokkaido_shapefile,dem_location,inventory_name_list[0],1)
    pointcloud_earth_iwata=make_3d_polygons(earth_iwata_shapefile,dem_location,inventory_name_list[1],1)
    pointcloud_earth_niigata=make_3d_polygons(earth_niigata_shapefile,dem_location,inventory_name_list[2],1)

- Rainfall triggered inventories

.. code:: ipython3

  pointcloud_rain_kumamoto=make_3d_polygons(rain_kumamoto_shapefile,dem_location,inventory_name_list[3],1)
  pointcloud_rain_fukuoka=make_3d_polygons(rain_fukuoka_shapefile,dem_location,inventory_name_list[4],1)
  pointcloud_rain_saka=make_3d_polygons(rain_saka_shapefile,dem_location,inventory_name_list[5],1)
  
  
Get ML features from 3d point cloud data
------------------------------------------

- Earthquake triggered inventories


.. code:: ipython3

   features_earth_hokkaido=get_tda_features(pointcloud_earth_hokkaido)
   features_earth_iwata=get_tda_features(pointcloud_earth_iwata)
   features_earth_niigata=get_tda_features(pointcloud_earth_niigata)

- Rainfall triggered inventories

.. code:: ipython3

  features_rain_kumamoto=get_tda_features(pointcloud_rain_kumamoto)
  features_rain_fukuoka=get_tda_features(pointcloud_rain_fukuoka)
  features_rain_saka=get_tda_features(pointcloud_rain_saka)


Taking one of the landslide inventory as the Testing inventory
---------------------------------------------------------------

- Case 1: Hokkaido (Earthquake -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Hokkaido inventory.

.. code:: ipython3

   earthquake_inventory_features=np.vstack((features_earth_iwata,features_earth_niigata))
   rainfall_inventory_features=np.vstack((features_rain_kumamoto,features_rain_fukuoka,features_rain_saka))
   test_inventory_features=features_earth_hokkaido
   predict_probability=classify_inventory_tda(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)


Visualization of the classification results



.. code:: ipython3

    plot_topological_results(predict_probability_hokkaido)
    
.. image:: Images/hokkaido_top.png
   :width: 1200    
        
- Case 2: Iwata (Earthquake -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Iwata inventory.

.. code:: ipython3

   earthquake_inventory_features=np.vstack((features_earth_hokkaido,features_earth_niigata))
   rainfall_inventory_features=np.vstack((features_rain_kumamoto,features_rain_fukuoka,features_rain_saka))
   test_inventory_features=  features_earth_iwata
   predict_probability_iwata=classify_inventory_tda(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)

Visualization of the classification results



.. code:: ipython3

    plot_topological_results(predict_probability_iwata)
    
.. image:: Images/iwata_topo.png
   :width: 1200       
    
- Case 3: Niigata (Earthquake -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Iwata inventory.

.. code:: ipython3

   earthquake_inventory_features=np.vstack((features_earth_hokkaido,features_earth_iwata))
   rainfall_inventory_features=np.vstack((features_rain_kumamoto,features_rain_fukuoka,features_rain_saka))
   test_inventory_features=features_earth_niigata  
   predict_probability_niigata=classify_inventory_tda(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)

Visualization of the classification results



.. code:: ipython3

     plot_topological_results(predict_probability_niigata)

.. image:: Images/niigata_topo.png
   :width: 1200   
     
- Case 4: Kumamoto (Rainfall -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Kumamoto inventory.

.. code:: ipython3

   earthquake_inventory_features=np.vstack((features_earth_hokkaido,features_earth_iwata,features_earth_niigata))
   rainfall_inventory_features=np.vstack((features_rain_fukuoka,features_rain_saka))
   test_inventory_features=features_rain_kumamoto  
   predict_probability_kumamoto=classify_inventory_tda(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)


Visualization of the classification results

.. code:: ipython3

     plot_topological_results(predict_probability_kumamoto)
  
.. image:: Images/kumamoto_topo.png
   :width: 1200     
     
- Case 5: Fukuoka (Rainfall -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Kumamoto inventory.

.. code:: ipython3

   earthquake_inventory_features=np.vstack((features_earth_hokkaido,features_earth_iwata,features_earth_niigata))
   rainfall_inventory_features=np.vstack((features_rain_kumamoto,features_rain_saka))
   test_inventory_features=features_rain_fukuoka  
   predict_probability_fukuoka=classify_inventory_tda(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)


Visualization of the classification results

.. code:: ipython3

    plot_geometric_results(predict_probability_fukuoka)
 
.. image:: Images/fukuoka_topo.png
   :width: 1200    
    
- Case 6: Saka (Rainfall -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Saka inventory.

.. code:: ipython3

   earthquake_inventory_features=np.vstack((features_earth_hokkaido,features_earth_iwata,features_earth_niigata))
   rainfall_inventory_features=np.vstack((features_rain_kumamoto,features_rain_fukuoka))
   test_inventory_features=features_rain_saka
   predict_probability_saka=classify_inventory_tda(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)


Visualization of the classification results

.. code:: ipython3

    plot_topological_results(predict_probability_saka)    
    
.. image:: Images/iwata_topo.png
   :width: 1200     
    




    
    



