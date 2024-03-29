# Import necessary modules
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model, load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
import pickle

#define height and width of the image
height=300
width=300

#create a ResNet50 model instance without the top layer as we will add our own top layer
base_model=ResNet50(weights='imagenet',include_top=False,input_shape=(height,width,3))


# In[36]:


#define directory containing training and validation data
train_dir="Data\Train"
validation_dir="Data\Valid"

#number of batches the data has to be divided into
batch_size=16

#create datagen and generator to load the data from training directory
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=90,horizontal_flip=True,vertical_flip=True)
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(height,width),batch_size=batch_size)

#create datagen and generator to load the data from validation directory
validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=90,horizontal_flip=True,vertical_flip=True)
validation_generator=validation_datagen.flow_from_directory(validation_dir,target_size=(height,width),batch_size=batch_size)


# In[37]:


#our own model which will be added onto the ResNet50 model
def build_finetune_model(base_model,dropout,fc_layers,num_classes):
    for layer in base_model.layers:
        layer.trainable=False

    x=base_model.output
    x=Flatten()(x)
    for fc in fc_layers:
        x=Dense(fc,activation='relu')(x)
        x=Dropout(dropout)(x)
    
    predictions=Dense(num_classes,activation='softmax')(x)

    finetune_model=Model(inputs=base_model.input,outputs=predictions) 
    
    return finetune_model

class_list=['Real','Fake'] #the labels of our data
FC_Layers=[1024,1024]
dropout=0.5

finetune_model=build_finetune_model(base_model,dropout=dropout,fc_layers=FC_Layers,num_classes=len(class_list))





#define number of epochs(the number of times the model will be trained) and number of training images
num_epochs=50
num_train_images=110
entropy=30


checkpoint = ModelCheckpoint("Final_model.keras", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode="max")

finetune_model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=0.000001, momentum=0.9), metrics=['accuracy'])

finetune_model.fit(x=train_generator, epochs=num_epochs, validation_data=validation_generator, callbacks=[checkpoint, early])

# Evaluate the model on the test set
evaluation = finetune_model.evaluate(validation_generator)

# Save the model weights
# finetune_model.save_weights("Final_model.weights.h5")
finetune_model.save_weights("Final_model.weights.h5")
# finetune_model.load_weights("Final_model.weights.h5")
finetune_model.load_weights("Final_model.weights.h5")
pickle.dump(finetune_model,open('mil.pkl','wb'))



