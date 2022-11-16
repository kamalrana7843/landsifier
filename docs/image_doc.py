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


def increase_resolution_polygon (data):

    """
    function to increase the data points between two neighbouring vertex of landslide polygon to get smooth images
    
    Parameters:
           :data (array_like): easting and northing coordinates data of landslide polygon

     
    Returns:
         (array_like) linear interpolated data of landslide polygon

    """

    N=100
    n=np.shape(data)[0]-1
    dat=[]
    for j in range(n):
        x1,y1=data[j]
        x2,y2=data[j+1]
        x_dis,y_dis=np.abs(x1-x2),np.abs(y1-y2)   
        x=min(x1,x2)
        for i in range(1,N):
            xx=x+(x_dis/N)*i
            yy=y1+((y2-y1)/(x2-x1))*(xx-x1)
            dat.append([xx,yy])
    dat=np.asarray(dat)

    return dat

def make_ls_images (poly_data):
    
    """
    function to convert landslide polygon to images
    
    Parameters:
          :poly_data: readed landslide inventory shapefile 

                
    Returns:
        (array_like) binary pixels values of landslide polygon Image 

    """   
    
    DATA=[]
    
    for l in range(np.shape(poly_data)[0]):
        if poly_data['geometry'][l].geom_type=='Polygon':
            polygon_data=np.asarray(poly_data['geometry'][l].exterior.coords)
            
            if np.nanmin(z) < 100:
               z=latlon_to_eastnorth(z)
            
            polygon_data=polygon_data[~np.isnan(polygon_data).any(axis=1)]      ### remove any Nan Values in polygon
            polygon_new_data=increase_resolution_polygon(polygon_data) 
            x_new=polygon_new_data[:,0]-np.nanmin(polygon_new_data[:,0])
            y_new=polygon_new_data[:,1]-np.nanmin(polygon_new_data[:,1])

            if (np.max(x_new)<180) & (np.max(y_new)<180):
                div=3
                #x1,y1=np.int32(x_new/div),np.int32(y_new/div)
                x1,y1=np.around(x_new/div),np.around(y_new/div)
                x1,y1=np.int32(x1),np.int32(y1)
                
                k1,k2=32-int(np.max(x1)/2),32-int(np.max(y1)/2)
                x1,y1=x1+k1,y1+k2
                
                
                image=np.zeros((64,64))
                for i,j in zip(x1,y1):
                    image[j,i]=255
                image=np.flip(image,axis=0)    
                DATA.append(image)
    DATA=np.asarray(DATA)
    print(np.shape(DATA))
    
    return DATA   

def train_augment (train_data,train_label):

    """ 
    This function is used to augment the training data by rotating image by 90, 180, 270 degree and flipping image horizontally and vertically
    
    Parameters:
           :train_data (array_like): training data

           :train_label (array_like): training label

    
    Returns:
        augmented training data and labels 
    
    """
    
    
    new_train=[]
    new_train_label=[]
    for i in range(np.shape(train_data)[0]):
        aa=train_data[i,:,:]
        bb=train_label[i]
        new_train_label.append(bb)
        new_train.append(aa)
        aa_1=np.fliplr(aa)
        new_train.append(aa_1)
        new_train_label.append(bb)
        aa_2=np.flipud(aa)
        new_train.append(aa_2)
        new_train_label.append(bb)
        bb_1=np.rot90(aa)
        new_train.append(bb_1)
        new_train_label.append(bb)
        cc_1=np.rot90(bb_1)
        new_train.append(cc_1)
        new_train_label.append(bb)
        dd_1=np.rot90(cc_1)
        new_train.append(dd_1)
        new_train_label.append(bb)     
        
    new_train=np.asarray(new_train)[:,:,:]
    new_train_label=np.asarray(new_train_label)[:,:]

    return new_train,new_train_label


def classify_inventory_cnn (earthquake_inventory_features,rainfall_inventory_features,test_inventory_features):
    
    """
    function to give probability of testing inventory belonging to earthquake and rainfall class
    
    Parameters:
        :earthquake_inventory_features (array_like): is landslide images of known earthquake inventories landslides

        :rainfall_inventory_features (array_like): is landslide images of known rainfall inventories landslides

        :test_inventory_features (array_like): is landslide images of known testing inventory landslides  

                
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

    
    x_train,y_train=train_augment(train_data,train_label)
    x_test=test_inventory_features
    #y_test=test_inventory_labels
    
    img_rows, img_cols,dim = 64, 64,1

    batch_size=64
    epochs=30
    num_classes=2
    input_shape=(64,64,1)
    
    num_classes=2
    img_rows, img_cols,dim = 64, 64,1
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    #y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #x_train = np.expand_dims(x_train, axis=-1)
    #x_test = np.expand_dims(x_test, -1)
    
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64,1)))
    model.add(MaxPooling2D((2, 2)))
    layers.Dropout(0.25),
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    layers.Dropout(0.25),
    model.add(Dense(40, activation='relu'))
    layers.Dropout(0.25),
    model.add(Dense(2, activation='softmax'))
    #model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    hist=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0) 
    predictions=model.predict(x_test)
    
    
    
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