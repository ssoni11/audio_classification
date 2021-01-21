# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 03:00:36 2021

@author: Sagar Soni
"""

###############################################################################
###############################################################################
###########################     IMPORT LIBRARIES      #########################
###############################################################################
###############################################################################
import pandas as pd
import csv
#from tqdm import tqdm
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


###############################################################################
###############################################################################
###########################     DATASET & PREPROCESS      #####################
###############################################################################
###############################################################################



#@staticmethod
'''
    ##########################################################################
    prepared_dataset() will generate a CSV file features and labels both
    short explaination for each of the feature has been added.
    ##########################################################################
'''

def prepared_dataset(audiofeature_path, validated_training_data_path):
    
    # read CSV files required for the project
    dataset = pd.read_csv(audiofeature_path)
    
    #to extract the labels
    other = pd.read_csv(validated_training_data_path)
    
    # change column name to map labels of the dataset
    other = other.rename(columns={'fname':'filename'})
    
    # merging two files dataset.csv and train.csv on filename
    dataset = dataset.merge(other, on='filename', how='left')
    
    # drop unnecessary feature from the dataframe
    dataset = dataset.drop(columns=['filename','label_x','manually_verified'])
    
    y = dataset['label_y']
    
    # OneHotEncoding of the labels
    y = (OneHotEncoder().fit_transform(dataset.label_y.values.reshape(-1,1))).toarray()
    
    dataset = dataset.drop(columns='label_y')
    
    # standard scalling of the features 
    x = StandardScaler().fit_transform(np.array(dataset.iloc[:,:], dtype=float))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    return x_train, x_test, y_train, y_test     #training and test set arrays




#@staticmethod
# create csv file header
'''
    ###################################################################
    create_dataset_file() will generate a CSV file to store the fetures and 
    label you use for the model.
    ###################################################################
'''

def create_dataset_file(audiofeature_path):
    
    
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    
    file = open(audiofeature_path, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    file = pd.read_csv(audiofeature_path)
    
    return file # empty dataframe return

#@staticmethod
'''
    ###################################################################
    extract_features() will generate a CSV file to store the fetures and 
    label you use for the model.
    short explaination for each of the feature has been added.
    ###################################################################
'''

def extract_features(audiofeature_path, audiofile_path):
    
    '''
        create dataset csv file
    '''
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    
    file = open(audiofeature_path, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    '''
        extract feature to CSV file
    '''
    for filename in (os.listdir(audiofile_path)):
        
        '''
        audiofile : load audio file
        '''
        audiofile = f'dataset/audiofiles/{filename}'
        
        '''
        Load an audio file as a floating point time series
        mono signal loading upto 30seconds
        '''
        y, sr = librosa.load(audiofile, mono=True, duration=30)
        
        # RMS energy of each frame from audio 'y'
        '''
            Compute root-mean-square (RMS) energy for each frame, either from the
            audio samples `y` or from a spectrogram `S`.
    
            Computing the energy from audio samples is faster as it doesn't require a
            STFT calculation. However, using a spectrogram will give a more accurate
            representation of energy over time because its frames can be windowed,
            thus prefer using `S` if it's already available.
        '''
        rmse = librosa.feature.rms(y=y)
        
        # chromagram from a waveform
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # spectral centroid
        '''
            Each frame of a magnitude spectrogram is normalized and treated as a
            distribution over frequency bins, from which the mean (centroid) is
            extracted per frame.
        '''
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # compute p'th-order spectral bandwidth:
        '''
            (sum_k S[k] * (freq[k] - centroid)**p)**(1/p)
        '''
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # compute rolloff frequency
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Compute the zero-crossing rate of an audio time series
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        
        # appending feature value of the audio waveform to CSV file
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for mfcc in mfccs:
            to_append += f' {np.mean(mfcc)}'
        #to_append += f' {g}'
        file = open(audiofeature_path, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
    
    '''
    pull the data in DataFrame format
    '''
    file = pd.read_csv(audiofeature_path)
        
    return file # return the dataframe


###############################################################################
###############################################################################
#############     NEURAL NETWORK MODEL CREATION & TUNING      #################
###############################################################################
###############################################################################


'''
##########################################################################
The "create_model" function defines the topography of the deep neural net, specifying the following:
    * The number of layers in the deep neural net. 
    * The number of nodes in each layer.
The create_model function also defines the activation function of each layer.
##########################################################################
'''

#@staticmethod

def create_model(n_features:int, init:str):
            
    '''
    The sequential model is used as no layer sharing is required for this model.
    
    The create_model function defines the topography of the deep neural net, specifying the following:
    
    The number of layers in the deep neural net.
    The number of nodes in each layer.
    The create_model function also defines the activation function of each layer.
    
    
    Parameters
    ----------
    x_train : numpy.ndarray
        feature array from Dataset.dataset()
    init_mode : str, optional
        The default initialization of is 'uniform'.
        Initializers define the way to set the initial random weights of Keras layers. 
        The keyword arguments used for passing initializers to layers depends on the layer.

    Returns
    -------
    ann : tensorflow.python.keras.engine.sequential.Sequential
        Keras Sequential model with input shape (None, 26) -'None' is the batch_size.
        Model: "sequential_1"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        dense_4 (Dense)              (None, 20)                540       
        _________________________________________________________________
        dense_5 (Dense)              (None, 20)                420       
        _________________________________________________________________
        dense_6 (Dense)              (None, 20)                420       
        _________________________________________________________________
        dense_7 (Dense)              (None, 41)                861       
        =================================================================
        Total params: 2,241
        Trainable params: 2,241
        Non-trainable params: 0
        
    '''
    
    # Build the ANN
    
    '''
    Initialization the ANN
    '''
    ann = tf.keras.models.Sequential()
    '''
    Describe the topography of the model by calling the tf.keras.layers.Dense
    method once for each layer. We've specified the following arguments:
        * units specifies the number of nodes in this layer.
        * activation specifies the activation function (Rectified Linear Unit).
        * name is just a string that can be useful when debugging.
    '''
    '''
    Adding the input layer and define the first hidden layer with 20 nodes.
    '''
    ann.add(tf.keras.layers.Dense(units=20, kernel_initializer=init ,activation='relu', input_shape=(n_features,), name='Hidden1'))
    '''
    Adding the second hidden layer with 20 nodes.
    '''
    ann.add(tf.keras.layers.Dense(units=20, kernel_initializer=init,activation='relu', name='Hidden2'))
    '''
    Adding the third hidden layer with 20 nodes.
    '''
    ann.add(tf.keras.layers.Dense(units=20, kernel_initializer=init,activation='relu', name='Hidden3'))
    '''
    Adding the output layer.
    '''
    ann.add(tf.keras.layers.Dense(units=41, kernel_initializer=init,activation='softmax', name='Output'))
    '''
    compiling the ANN
    '''
    ann.compile(loss='categorical_crossentropy', 
                 optimizer='adam', 
                 metrics=['acc']
                 )
    '''
    returns sequencial model
    '''
    return ann


'''
##########################################################################
The "train_model" function trains the model from the input features and labels. 
The tf.keras.Model.fit method performs the actual training. 
The x : parameter of the fit method is very flexible, enabling you to pass feature data in a variety of ways.
The following implementation passes numpy array.
##########################################################################
'''

#@staticmethod

def train_model(x, 
                y, 
                model, 
                x_test, 
                y_test, 
                epochs=60, 
                batch_size=50, 
                shuffle=True):
    '''
    
    Parameters
    ----------
    x : numpy.ndarray
        x_train feature array from Dataset.dataset().
    y : numpy.ndarray
        y_train label array from Dataset.dataset().
    model : tensorflow.python.keras.engine.sequential.Sequential
        The ANN model created in create_model function.
    epochs : int, optional
        Number of passes for entire training dataset. The default is 60.
    batch_size : int, optional
        the number of training examples in one forward/backward pass. 
        The higher the batch size, the more memory space you'll need.. The default is 50.

    Returns
    -------
    hist : DataFrame
        Dataframe of training loss values and metrices.
    epochs : int
        Recording training loss values and metrics values at successive epochs.
        
    '''

    #split dataset into features and label.
    #features = {name:np.array(value) for name, value in dataset.items()}

    #label = np.array(features.pop(label_name))
    
    '''
    fit compiled model to training samples
    '''
    history = model.fit(x, 
                        y,  
                        epochs,
                        batch_size,
                        shuffle=True, 
                        validation_data=(x_test, y_test))

    '''
    Get details that will be useful for plotting the loss curve.
    '''
    History = history.history # dictionary
    epochs = history.epoch
    hist = pd.DataFrame(History)
    '''
    return history epochs and training loss and matrices values
    '''
    return hist # DataFrame



'''
##############################################################################
The "model_tuning" function implement classifier wrapper for Scikit-Learn API. 
The tf.keras.Model.fit method performs the actual training. 
The x : parameter of the fit method is very flexible, enabling you to pass feature data in a variety of ways.
The following implementation passes numpy array.

        FOLLOWING PARAMETERS ARE TUNED FOR THE SEQUENTIAL MODEL
        1. init_mode
        2. batch_size
        3. epochs
##############################################################################
'''
initializers = ['glorot_uniform','uniform', 'normal']   # different weight initializers
batch_size_list = [32, 128, 512]    # different batch sizes
epochs_list = [8, 15]    # different epochs sizes

#param_grid = dict(epochs=epochs_list, batch_size=batch_size_list, init=initializers)


#@staticmethod
def model_tuning(x, y, model, x_test, y_test):
    '''
    
    Parameters
    ----------
    
    x : numpy.ndarray
        x_train feature array from Dataset.dataset().
    y : numpy.ndarray
        y_train label array from Dataset.dataset().
    model : tensorflow.python.keras.engine.sequential.Sequential
        The ANN model created in create_model function.
    init_mode : str, optional
        The default initialization of is 'uniform'.
        Initializers define the way to set the initial random weights of Keras layers. 
        The keyword arguments used for passing initializers to layers depends on the layer.
    batch_size : int, optional
        the number of training examples in one forward/backward pass. 
        The higher the batch size, the more memory space you'll need.. The default is 50.
    epochs : int, optional
        Number of passes for entire training dataset. The default is 60

    Returns
    -------
    best_mode : DataFrame
        Dataframe of training loss values and metrices.
        
    '''
    
    '''
    Wrapper for using the Scikit-Learn API with Keras models.
    Implementation of the scikit-learn classifier API for Keras.
    **KerasClassifer expects a build function, not the model instance itself, 
    **which upon its call returns a compiled instance of a Keras model
    '''
    #keraswrapper = KerasClassifier(build_fn=model, verbose=1) # use it if model is already created in built in variable
    keraswrapper = KerasClassifier(build_fn=create_model, 
                                   n_features= x.shape[1], 
                                   init='glorot_uniform', verbose=1) # use it if model is not created in built in variable
    
    '''
    parameter grid : dictionary, parameter to be tuned for
    '''
    param_grid = dict(epochs=epochs_list, batch_size=batch_size_list, init=initializers)
    
    '''
    loop through predefined parameters and fit the estimator (model) on training dataset
    '''
    grid = GridSearchCV(estimator=keraswrapper, param_grid=param_grid, n_jobs=-1, cv=3)
    
    #split dataset into features and label.
    #features = {name:np.array(value) for name, value in dataset.items()
    #label = np.array(features.pop(label_name))
    
    output_gridsearch = grid.fit(x, y)
    
    '''
    get mean_test_score for each params
    '''
    parameter_accuracy = pd.DataFrame(pd.concat([pd.DataFrame(output_gridsearch.cv_results_["params"]),
                            pd.DataFrame(output_gridsearch.cv_results_["mean_test_score"],
                            columns=["Accuracy"])],axis=1))
    
    
    print("Best: %f using %s" % (output_gridsearch.best_score_, output_gridsearch.best_params_))
    print("Train model on the best parameter")
    
    bestParameters = output_gridsearch.best_params_
    
    '''
    call train_model function on the best parameters achieved through the GridSearch
    '''
    best_model = train_model(x, 
                             y, 
                             model, 
                             x_test, 
                             y_test, 
                             epochs=bestParameters.get('epochs'), 
                             batch_size=bestParameters.get('batch_size'), 
                             shuffle=True)
    
    '''
    history = model.fit(x_train, y_train,
                        batch_size=bestParameters.get('batch_size'), epochs = bestParameters.get('epochs'),
                        shuffle=True)
    '''

    '''
    Get details that will be useful for plotting the loss curve.
    '''
    
    '''
    epochs=history.epoch
    hist = pd.DataFrame(history.history)
    #mse = hist["mean_squared_error"]
    '''

    return best_model

###############################################################################
###############################################################################
##########################     TRANSFER LEARNIGN      #########################
###############################################################################
###############################################################################

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")

# Presumably you would want to first load pre-trained weights.
model.load_weights(...)

# Freeze all layers except the last one.
for layer in model.layers[:-1]:
  layer.trainable = False

# Recompile and train (this will only update the weights of the last layer).
model.compile(...)
model.fit(...)




if __name__ == '__main__':
    
    audiofeature_path = input("Enter the path to \"dataset.csv\" file i.e. folder_name/file_name.csv :: ")
    validated_training_data_path = input("Enter the path to the validated data file \"training.csv\" i.e. folder_name/file_name.csv :: ")
    audiofile_path = input("Enter the path to audiofiles \"dataset/audiofiles\" i.e. folder_name/folder_name :: ")
    
    #create and extract features to dataset file
    dataset = extract_features(audiofeature_path, audiofile_path)
    
    #call dataset
    x_train, x_test, y_train, y_test = prepared_dataset(audiofeature_path, validated_training_data_path)
    
    #create a model
    model = create_model(x_train.shape[1], 'glorot_uniform')
    
    #train model on training set
    #model_training = modelselection.train_model(x_train, y_train, model, x_test, y_test, epochs=60, batch_size=50)
    
    #tuning the model
    tuningOfModel = model_tuning(x_train, y_train, model, x_test, y_test)