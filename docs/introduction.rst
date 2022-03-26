
############
Introduction
############

``Landsifier`` is a Python based library to estimate likely triggers of mapped landslides.
The Beta version of library consitute three machine learning based method for finding the trigger of Landslide inventories.



1. Geometric feature based method
==================================
This method is based on using 2D landslide polygon geometric properties for classification. This method calculates various geometric properties of landslide polygon and these geometric properties are used as a feature space for machine learning based algorithm. 

Sample landslide polygons
--------------------------
The below plot shows the sample landslide polygons of earthquake and rainfall-induced inventories.

.. image:: https://user-images.githubusercontent.com/63171258/160248721-85ff4e9c-56a3-4c56-9a24-eabbde9300e5.png
   :width: 1200 

Geometric properties of landslide polygon
-----------------------------------------

The geometric properties of landslide polygons used are:-

- Area (A) of landslide Polygon
- Perimetre (P) of Landslide Polygon
- Ratio of Area (A) to Perimetre(P)
- Convex hull based measures (Ratio of area of polygon to area of convex hull fitted to polygon)
- Width of minimum area bounding box fitted to polygon
- Eccentricity of ellipse fitted to polygon having area A and perimetre P
- minor-axis of ellipse fitted to polygon having area A and perimetre P

The below plot shows the various geometric properties of landslide polygon

.. image:: https://user-images.githubusercontent.com/63171258/160248555-f38d8d88-0901-4ec9-9f81-ef57b3f8d12f.png
   :width: 1200 






2. Topological feature based method
====================================

This method convert landslide polygon data to landslide polygon Images. These converted landslide images are used as a input to CNN for landslide classification



3.Image based method
=====================

This method convert landslide polygon data to landslide polygon Images. These converted landslide images are used as a input to CNN for landslide classification








