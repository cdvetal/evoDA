""" Data Augmentation """

import keras
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import imgaug.parameters as iap
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from numpy import mean, std
import pickle
np.random.seed(42)
ia.random.seed(42)

def getState():  
    return np.random.get_state(), ia.random.get_global_rng().state

def setState(state_np, state_ia):
    np.random.set_state(state_np)
    ia.random.get_global_rng().set_state_(state_ia)

class DataGenerator(keras.utils.Sequence):
    ''' Generates batches of data '''
    def __init__(self, images, labels, functions, pr, batch_size = 32, shuffle = True, augment = True):
        self.images = images           # array of images
        self.labels = labels           # array of labels
        self.functions = functions     # array of transformation functions
        self.pr = pr                   # array of functions parameters
        self.batch_size = batch_size   # batch size
        self.shuffle = shuffle         # shuffle bool
        self.augment = augment         # augment bool
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
       
        # select data and load images
        batch_images = [self.images[i] for i in indexes] 
        batch_images = np.array(batch_images)
        batch_labels = self.labels[indexes]
        
		# preprocess and augment data
        if self.augment == True:
            #imgaug works better with the uint8 input type
            aug_images = self.augmentor((batch_images*255).astype(np.uint8))
            batch_images = np.array(aug_images)/255.0

        return batch_images, batch_labels
	
    def augmentor(self, batch_images):
        'Apply data augmentation'
        mode_op = ['constant','edge','symmetric','reflect','wrap']
        lista_augmentors = []
        for i in range(len(self.functions)):
            p1,p2,p3 = self.pr[0,i],self.pr[1,i],self.pr[2,i]
            lista_augmentors.append(iaa.Sometimes(self.functions[i][0], eval('iaa.'+self.functions[i][1])))
        seq = iaa.Sequential(lista_augmentors)
        return seq.augment_images(batch_images)
    


class NetModel():
    ''' Defines classifier model ''' 
    def __init__(self, epochs=10, batch_size=32,  image_dimensions=(50,50 ,3)):
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_dim = image_dimensions
        self.model = self.create_model()

    def summary(self):
        self.model.summary()
        
    def evaluate(self, data):
        return self.model.evaluate(data[0],data[1],verbose=0)
        
    def predict(self, y):
        return self.model.predict(y)

    def create_model(self):            
        model = Sequential()
        model.add(Conv2D(32, (3, 3), strides=2, activation='relu', input_shape=self.input_dim))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model


    def train(self, train_data, val_data, test_data):
        'Train model'
        self.model.fit(x=train_data, validation_data = val_data,
                                 epochs = self.epochs, steps_per_epoch = len(train_data),
                                 validation_steps = len(val_data[0])//self.batch_size,
                                 class_weight = [0.998, 1.002],
                                 verbose = 0)

def train_with_population(individual,input_functions):
    ''' Train model using the function population '''    
    
    
    ''' Set training parameters '''
    dataset_path = 'dataset_train_reduced4.pickle'
    epochs = 20
    batch_size = 32
    n_folds = 5
    tf = individual[0]
    pr = individual[1]
    functions = [input_functions[i] for i in tf]
    
    
    
    ''' Load dataset '''
    # open dataset file
    file = open(dataset_path, 'rb')
    [X_train,Y_train] = pickle.load(file)
    file.close()    
    
    
    ''' Train classifier '''
    ypredVL = []; ytrueVL = []
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for ii, (tr, val) in enumerate(cv.split(X=X_train, y=Y_train)):   
        print("Fold",ii+1)
        # create model
        model = NetModel(epochs=epochs, batch_size=batch_size)
        
        # split into train and validation sets
        idxtr = np.zeros(len(X_train), dtype=bool); idxtr[tr] = True
        idxval = np.zeros(len(X_train), dtype=bool); idxval[val] = True
        val_data = (X_train[idxval], Y_train[idxval])
        
        # train fold
        model.train(DataGenerator(X_train[idxtr], Y_train[idxtr], functions, pr, batch_size=batch_size, augment = True), val_data, [])
         
        # store folds's results   
        ypredVL.append(model.predict(val_data[0]))
        ytrueVL.append(val_data[1])    
    
    ''' Evaluation ''' 
    def get_metrics(yT, yP):
        'Calculates several metrics'    
        metrics = {'auc': []}

        for ii in range(len(yP)):
            ypred = np.argmax(yP[ii],axis=1) 
            ytrue = np.argmax(yT[ii],axis=1)
                    
            metrics['auc'].append(roc_auc_score(ytrue,ypred))
        return metrics
           
    metricsVL = get_metrics(ytrueVL,ypredVL)
    print("ROC AUC Score:",mean(metricsVL['auc']))

    return mean(metricsVL['auc']), std(metricsVL['auc'])

def get_AUC_values(population,input_functions):
    auc = np.empty((len(population)))
    std_auc = np.empty((len(population)))
    for pop in range(len(population)):
        print("\nTraining individual",pop+1,"...")
        [auc[pop], std_auc[pop]] = train_with_population(population[pop], input_functions)
    return auc, std_auc
        
