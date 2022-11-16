##  Python libraries needed to run the RF based method package ##




import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import MultiPoint
import random
import math
import shapely.affinity
from scipy.spatial import distance
from scipy.spatial import ConvexHull
import geopandas as gpd
import matplotlib.pyplot as plt
import utm
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import time


def read_shapefiles (path_filename):
    
    """
    function to read the shapefile from the local file path of landslide inventory
    
  
    Parameters:
         :path_filename (str): path to local inventory shapefiles
    
    
    Returns:
         read shapefile from file path

    """
    
    return gpd.read_file(path_filename)



def latlon_to_eastnorth (lonlat_polydata):
    
    """ 
    function to convert the (longitude latitude) coordinates of polygons to (easting, northing) coordinates
    
    
    Parameters:
          :lonllat_polydata (array_like): longitude and latitude coordinates data
                      
    Returns:
           (array_like) easting and northing coordinates of landslide polygon data when polygon data has longitude latitude coordinates
     
    """
     
    east_north_polydata=[]
    for i in range(np.shape(lonlat_polydata)[0]):
        u = utm.from_latlon(lonlat_polydata[i][1], lonlat_polydata[i][0])   ### (lat,lon) to (east,north)
        east_north_polydata.append([u[0],u[1]])
    east_north_polydata=np.asarray(east_north_polydata) 

    return  east_north_polydata  

def get_geometric_properties_landslide (poly_data):
    
    """
    function to calculate the geometric properties of landslide polygon
    
    Parameters:
          :poly_data: readed landslide inventory shapefile (output of read_shapefile function)
             
                
    Returns:
           (array_like) Geometric features of landslide polygon.
           
    """
    
    store_geometric_features_all_landslides=[]    
    for l in range((np.shape(poly_data)[0])):
        if poly_data['geometry'][l].geom_type=='Polygon':
            z=np.asarray(poly_data['geometry'][l].exterior.coords)
            
            if np.nanmin(z) < 100:
                z=latlon_to_eastnorth(z)
             
            ze=z[~np.isnan(z).any(axis=1)]           ### remove any Nan Values in polygon
            centre=np.array(Polygon(ze).centroid)       ### Centroid of a polygon
            if np.shape(centre)[0]>0:

                polygon = Polygon(ze)
                area_polygon,perimeter_polygon=Polygon(ze).area,Polygon(ze).length  ## Area,Perimetre

                ##### Fit minimum area bounding box to polygons to calculate minimum width and theta 
                bounding_box=MultiPoint(ze).minimum_rotated_rectangle
                coordinates_BB = np.asarray(bounding_box.exterior.coords)
                tan_y=(coordinates_BB[1,1]-coordinates_BB[0,1])/(coordinates_BB[1,0]-coordinates_BB[0,0])
                dist_firstside=((coordinates_BB[2,1]-coordinates_BB[1,1])**2+(coordinates_BB[2,0]-coordinates_BB[1,0])**2)**0.5
                dist_secondside=((coordinates_BB[1,1]-coordinates_BB[0,1])**2+(coordinates_BB[1,0]-coordinates_BB[0,0])**2)**0.5
                width_minimum_boundingbox=min(dist_firstside,dist_secondside)   ### width of minimum bounding box
                theta=math.degrees(math.atan(tan_y))   ### rotation angle of landslide polygon

                ##  Fit an ellipse to polygon having same area and perimetre as polygon

                dw=np.sqrt((perimeter_polygon**4)-16*((np.pi)**2)*(area_polygon**2))
                delta=(4*np.pi*area_polygon)/((perimeter_polygon**2)-dw)
                major_axis_ellipse=2*(np.sqrt((area_polygon*delta)/np.pi))        ## major axis of fitted ellipse
                minor_axis_ellipse=2*(np.sqrt((area_polygon)/(np.pi*delta)))      ## minor axis of fitted ellipse

                ###############################################################################
                ellipse = ((centre[0], centre[1]),(major_axis_ellipse, minor_axis_ellipse),theta)
                circ = shapely.geometry.Point(ellipse[0]).buffer(1)
                ell  = shapely.affinity.scale(circ, int(ellipse[1][0]), int(ellipse[1][1]))
                ellr = shapely.affinity.rotate(ell,ellipse[2])
                elrv = shapely.affinity.rotate(ell,90-ellipse[2])
                coordinates_ellip = np.asarray(elrv.exterior.coords)
                #################################################################################

                p=Polygon(ze)
                q=Polygon(coordinates_ellip)        ### coordinates of a ellipse in an array
                
                #print(q.is_valid)

                #if p.is_valid==True and q.is_valid==True:
                if p.is_valid==True:    
                    ##########################################################
                    hull=MultiPoint(ze).convex_hull
                    convex_hull_area=hull.area
                    convex_hull_measure=area_polygon/convex_hull_area   ## convex hull measure of the polygon         
                    eccentricity_ellipse=np.sqrt(1-(minor_axis_ellipse)**2/(major_axis_ellipse**2)) ## eccentricity of the fitted ellipse
                    #######################################
                    geometric_features_one_landslide=np.hstack((area_polygon,perimeter_polygon,
                                                   area_polygon/perimeter_polygon,convex_hull_measure,minor_axis_ellipse,
                                                                eccentricity_ellipse,width_minimum_boundingbox))
                    store_geometric_features_all_landslides.append(geometric_features_one_landslide)      ## getting data inside poylgons
    store_geometric_features_all_landslides=np.asarray(store_geometric_features_all_landslides)  
    print(np.shape(store_geometric_features_all_landslides))
    return store_geometric_features_all_landslides  


def classify_inventory_rf (earthquake_inventory_features,rainfall_inventory_features,test_inventory_features):
    
    """
    function to predict the trigger of landslides in testing inventory
    
    Parameters:
   	   :earthquake_inventory_features (array_like): geometric features of known earthquake inventories landslides
                                   
    	   :rainfall_inventory_features (array_like): geometric features of known rainfall inventories landslides
                                 
           :test_inventory_features (array_like): geometric features of known testing inventory landslides  
                             
    Returns:
            (array_like) probability of testing inventory landslides belonging to earthquake and rainfall class
                                                   
    """
    
    earthquake_label=np.zeros((np.shape(earthquake_inventory_features)[0],1))
    rainfall_label=np.ones((np.shape(rainfall_inventory_features)[0],1))
    
    n1=np.shape(earthquake_inventory_features)[0]
    n2=np.shape(rainfall_inventory_features)[0]
    if n1>n2:  ### n1 is number of earth samples and n2 is number of rainfall samples #####
        indi_earth=random.sample(range(n1),n2)
        train_earth=earthquake_inventory_features[indi_earth,:]
        train_label_earth=earthquake_label[indi_earth]  ## checked
        train_rain=rainfall_inventory_features
        train_label_rain=rainfall_label
        #######################################################################################
        train_data=np.vstack((train_earth,train_rain))
        train_label=np.vstack((train_label_earth,train_label_rain))
        #print(np.shape(train_data)[0],np.shape(train_label)[0],np.shape(test_inventory_features)[0])

    else:
        indi_rain=random.sample(range(n2),n1)
        train_rain=rainfall_inventory_features[indi_rain,:]
        train_label_rain=rainfall_label[indi_rain]   
        train_earth=earthquake_inventory_features
        train_label_earth=earthquake_label        
        train_data=np.vstack((train_earth,train_rain))
        train_label=np.vstack((train_label_earth,train_label_rain))
        #print(np.shape(train_data)[0],np.shape(train_label)[0],np.shape(test_inventory_features)[0])

        #########################################################################################
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    
    test_data=test_inventory_features
    test_data = scaler.transform(test_data)    
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(train_data,np.ravel(train_label))
    y_pred=clf.predict(test_data)
    predictions = clf.predict_proba(test_data)

    
    number_rainfall_predicted_landslides=np.sum(y_pred)
    number_earthquake_predicted_landslides=np.shape(y_pred)[0]-number_rainfall_predicted_landslides
    
    probability_earthquake_triggered_inventory=(number_earthquake_predicted_landslides/np.shape(y_pred)[0])*100
    probability_rainfall_triggered_inventory=(number_rainfall_predicted_landslides/np.shape(y_pred)[0])*100
    
    probability_earthquake_triggered_inventory=np.round(probability_earthquake_triggered_inventory,2)
    probability_rainfall_triggered_inventory=np.round(probability_rainfall_triggered_inventory,2)
   
    
    print("Probability of inventory triggered by Earthquake: ", str(float(probability_earthquake_triggered_inventory))+'%')
    print("Probability of inventory triggered by Rainfall: ",str(float(probability_rainfall_triggered_inventory))+'%')
    
 
    
    return predictions
    
    
def plot_geometric_results (predict_proba):
    
    
    """
    function to visualize the trigger prediction of landslides in testing inventory
    
     
    Parameters:
         :predict_proba (array_like): probability of each landslide in inventory class belonging to earthquake and rainfall class.
                   
                   
    Returns:
         Visualization of landslide probabilities belong to earthquake and rainfall class and trigger prediction of entire landslide 
         inventory 
                
    """
    
    plt.rc('text', usetex=True)
    # chage xtick and ytick fontsize 
    # chage xtick and ytick fontsize 
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    n1=np.shape(np.argwhere(predict_proba[:,0]>0.5))[0]
    n2=np.shape(np.argwhere(predict_proba[:,1]>0.5))[0]
    
    def RF_image(predict_proba):
        predict_proba=np.int32(np.round(predict_proba*100))
        data=np.zeros((np.shape(predict_proba)[0],100))
        for i in range(np.shape(predict_proba)[0]):
            a,b=predict_proba[i,0],predict_proba[i,1]
            #################
            c=np.zeros(int(a),)
            d=np.ones(int(b),)
            if int(a)==100:
                mat=c
            elif int(b)==100:
                mat=d
            else:   
                mat=np.hstack((c,d))
            data[i,:]=mat
        data=np.transpose(data)
        return data 

    import matplotlib as mpl
    fig,ax=plt.subplots(1, 1,figsize=(14,6), constrained_layout=True)
    cm = mpl.colors.ListedColormap([[230/255,204/255,179/255],[30/255,100/255,185/255]])
    
    if n1>n2: 
        earthquake_accuracy=np.round((n1/(n1+n2))*100,2) 
        ax.text(np.shape(predict_proba)[0]//4,110,' Probability of Earthquake: %s  '%earthquake_accuracy,fontsize=26)
        
        ind=np.argsort(predict_proba[:,0])
        predict_proba=predict_proba[ind,:]        
        matrix_probability=RF_image(predict_proba)
        image=np.dstack((matrix_probability,matrix_probability,matrix_probability))
        image[matrix_probability[:,:]==0]=[230,204,179]
        image[matrix_probability[:,:]==1]=[30,100,185]
        image=np.int32(image)
        
        
    else:
        rainfall_accuracy=np.round((n2/(n1+n2))*100,2) 
        ind=np.argsort(predict_proba[:,1])
        predict_proba=predict_proba[ind,:]        
        matrix_probability=RF_image(predict_proba)
        image=np.dstack((matrix_probability,matrix_probability,matrix_probability))
        image[matrix_probability[:,:]==0]=[230,204,179]
        image[matrix_probability[:,:]==1]=[30,100,185]
        image=np.int32(image)
        ax.text(np.shape(predict_proba)[0]//4,110,' Probability of Rainfall: %s '%rainfall_accuracy+'%',fontsize=26)
        #ax.text(np.shape(predict_proba)[0]//4,110,' Probability of Rainfall: %s '%rainfall_accuracy,fontsize=26)

        image=np.flipud(image)

    
    
    
    pcm = ax.imshow(image,aspect='auto',cmap=cm,origin='lower')
    ax.set_xlabel('Test Sample Index',fontsize=26)
    ax.set_ylabel('Class Probability',fontsize=26)
    

    #ax.text(1380,110,'85.31 $\pm$ 0.19 \%',fontsize=26)


    cb=plt.colorbar(pcm, location='top',pad=0.03,ax=ax)
    cb.ax.set_xticklabels([],length=0)                 # vertically oriented colorbar
    cb.ax.set_yticklabels([],length=0)                 # vertically oriented colorbar
    cb.ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.text(np.shape(predict_proba)[0]//6,135,'Earthquake',fontsize=26)
    ax.text(np.shape(predict_proba)[0]//1.4,135,'Rainfall',fontsize=26)

    #cb.set_label('Earthquake                            Rainfall ',fontsize=26)
    plt.show()
    ################################################################################## 

