
Image-Based Method Example 
=========================================
This method convert landslide polygon data to landslide polygon Images.
These converted landslide images are used as a input to CNN for landslide classification


Import Image based module functions from Landsifier library 
------------------------------------------------------------------------
.. code:: ipython3

    from image_based_model import make_ls_images
    from image_based_model import classify_inventory_cnn
    from image_based_model import plot_geometric_results
    from image_based_model import read_shapefiles
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

Convert landslide polygons to grayscale binary images  
----------------------------------------------------------------------------
- Earthquake triggered inventories


.. code:: ipython3

    feature_Earth_hokkaido=make_ls_images(earth_hokkaido_shapefile)
    feature_Earth_iwata=make_ls_images(earth_iwata_shapefile)
    feature_Earth_niigata = make_ls_images(earth_niigata_shapefile)

- Rainfall triggered inventories

.. code:: ipython3

 feature_Rain_kumamoto = make_ls_images(rain_kumamoto_shapefile)
 feature_Rain_fukuoka = make_ls_images(rain_fukuoka_shapefile)
 feature_Rain_saka = make_ls_images(rain_saka_shapefile)


Taking one of the landslide inventory as the Testing inventory
---------------------------------------------------------------

- Case 1: Hokkaido (Earthquake -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Hokkaido inventory.

.. code:: ipython3

  earthquake_inventory_features=np.vstack((feature_Earth_iwata,feature_Earth_niigata))
  rainfall_inventory_features=np.vstack((feature_Rain_fukuoka,feature_Rain_saka,feature_Rain_kumamoto))
  test_inventory_features=feature_Earth_hokkaido
  predict_probability_hokkaido=classify_inventory_cnn(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)


Visualization of the classification results



.. code:: ipython3

    plot_geometric_results(predict_probability_hokkaido)

    
.. image:: /docs/Images/hokkaido_image.png
   :width: 1200    
        
- Case 2: Iwata (Earthquake -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Iwata inventory.

.. code:: ipython3

  earthquake_inventory_features=np.vstack((feature_Earth_hokkaido,feature_Earth_niigata))
  rainfall_inventory_features=np.vstack((feature_Rain_fukuoka,feature_Rain_saka,feature_Rain_kumamoto))
  test_inventory_features=feature_Earth_iwata
  predict_probability_iwata=classify_inventory_cnn(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)

Visualization of the classification results



.. code:: ipython3

    plot_geometric_results(predict_probability_iwata)
    
.. image:: /docs/Images/iwata_image.png
   :width: 1200       
    
- Case 3: Niigata (Earthquake -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Iwata inventory.

.. code:: ipython3

    earthquake_inventory_features=np.vstack((feature_Earth_hokkaido,feature_Earth_iwata))
    rainfall_inventory_features=np.vstack((feature_Rain_fukuoka,feature_Rain_saka,feature_Rain_kumamoto))
    test_inventory_features=feature_Earth_niigata
    predict_probability_niigata=classify_inventory_cnn(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)

Visualization of the classification results



.. code:: ipython3

     plot_geometric_results(predict_probability_niigata)

.. image:: /docs/Images/niigata_image.png
   :width: 1200   
     
- Case 4: Kumamoto (Rainfall -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Kumamoto inventory.

.. code:: ipython3

   earthquake_inventory_features=np.vstack((feature_Earth_hokkaido,feature_Earth_iwata,feature_Earth_niigata))
   rainfall_inventory_features=np.vstack((feature_Rain_fukuoka,feature_Rain_saka))
   test_inventory_features=feature_Rain_kumamoto
   predict_probability_kumamoto=classify_inventory_cnn(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)


Visualization of the classification results

.. code:: ipython3

     plot_geometric_results(predict_probability_kumamoto)
  
.. image:: /docs/Images/kumamoto_image.png
   :width: 1200     
     
- Case 5: Fukuoka (Rainfall -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Kumamoto inventory.

.. code:: ipython3

   earthquake_inventory_features=np.vstack((feature_Earth_hokkaido,feature_Earth_iwata,feature_Earth_niigata))
   rainfall_inventory_features=np.vstack((feature_Rain_kumamoto,feature_Rain_saka))
   test_inventory_features=feature_Rain_fukuoka
   predict_probability_fukuoka=classify_inventory_cnn(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)


Visualization of the classification results

.. code:: ipython3

    plot_geometric_results(predict_probability_fukuoka)
 
.. image:: /docs/Images/fukuoka_image.png
   :width: 1200    
    
- Case 6: Saka (Rainfall -triggerd) is a testing inventory 

Training the algorithm on rest five inventories and test it on Saka inventory.

.. code:: ipython3

  earthquake_inventory_features=np.vstack((feature_Earth_hokkaido,feature_Earth_iwata,feature_Earth_niigata))
  rainfall_inventory_features=np.vstack((feature_Rain_kumamoto,feature_Rain_fukuoka))
  test_inventory_features=feature_Rain_saka
  predict_probability_saka=classify_inventory_cnn(earthquake_inventory_features,rainfall_inventory_features,test_inventory_features)


Visualization of the classification results

.. code:: ipython3

    plot_geometric_results(predict_probability_saka)    
    
.. image:: /docs/Images/saka_image.png
   :width: 1200     
    




    
    




