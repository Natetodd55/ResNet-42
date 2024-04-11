import tensorflow as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.models import Sequential



"""
ResNet-42
"""
channels = 3
num_classes = 15
img_size = (256, 256)
batch_size = 15

#create my data generator
train_datagen = ImageDataGenerator( rescale = 1.0/255 ) #normalize color data
test_datagen = ImageDataGenerator( rescale = 1.0/255 )

#create tensor datasets via the image data generator object
train_set = train_datagen.flow_from_directory('celebdataset/train',
                                              target_size=img_size,
                                              batch_size = batch_size,
                                              class_mode='categorical')

test_set = test_datagen.flow_from_directory('celebdataset/test',
                                              target_size=img_size,
                                              batch_size = batch_size,
                                              class_mode='categorical')

############### START OF CONV RESNET BLOCK ##################
############# Build a major subblock 0
##### 2 Conv, 2 Identity
#input shape is 56x56x64
#64/2 1x1
#64 3x3
#256 1x1
block0_input = keras.Input(shape=(64, 64, 64), name='block0_input')
x = keras.layers.Conv2D(64, kernel_size=1, activation='relu', padding='same', strides=2, name='b0_conv1')(block0_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', strides=1, name='b0_conv2')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(256, kernel_size=1, activation='relu', padding='same', strides=1, name='b0_conv3')(x)
x = keras.layers.BatchNormalization()(x)

#Fast forward ResNet layer now
# ? what is the input to the ResNet layer
#   -same dimensions as last output (56, 56, 256)
#block transform will take input and reshape it to desires dimensions
block0_transform = keras.layers.Conv2D(256, kernel_size=1, strides=2, activation='relu', padding='same')(block0_input)
x1 = keras.layers.BatchNormalization()(block0_transform)

#combine the resNet fast forward sequence to conv sequence
x = keras.layers.Add()([x,x1]) #add up values from both the resnet path and regular path

#after adding up, just to a relu of the entire thing
block0_output = keras.layers.ReLU()(x)

#create a variable for the model of this entire complex thing
block0_conv = keras.Model(inputs = block0_input, outputs = block0_output, name = 'block0_conv')


############### END OF CONV RESNET BLOCK ##################


############### START OF CONV1 RESNET BLOCK ##################
############# Build a major subblock 0
##### 2 Conv, 2 Identity
#input shape is 56x56x64
#64/2 1x1
#64 3x3
#256 1x1
block0_conv1_input = keras.Input(shape=(32, 32, 256), name='block0_input1')
x = keras.layers.Conv2D(64, kernel_size=1, activation='relu', padding='same', strides=2, name='b0_conv1.1')(block0_conv1_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', strides=1, name='b0_conv2.1')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(256, kernel_size=1, activation='relu', padding='same', strides=1, name='b0_conv3.1')(x)
x = keras.layers.BatchNormalization()(x)

#Fast forward ResNet layer now
# ? what is the input to the ResNet layer
#   -same dimensions as last output (56, 56, 256)
#block transform will take input and reshape it to desires dimensions
block0_conv1_transform = keras.layers.Conv2D(256, kernel_size=1, strides=2, activation='relu', padding='same')(block0_conv1_input)
x1 = keras.layers.BatchNormalization()(block0_conv1_transform)

#combine the resNet fast forward sequence to conv sequence
x = keras.layers.Add()([x,x1]) #add up values from both the resnet path and regular path

#after adding up, just to a relu of the entire thing
block0_conv1_output = keras.layers.ReLU()(x)

#create a variable for the model of this entire complex thing
block0_conv1 = keras.Model(inputs = block0_conv1_input, outputs = block0_conv1_output, name = 'block0_conv1')


############### END OF CONV1 RESNET BLOCK ##################

############### START OF IDENTITY RESNET BLOCK ##################

block0_identity_input = keras.Input(shape=(16, 16, 256), name='block0_identity_input')
x = keras.layers.Conv2D(64, kernel_size=1, activation='relu', padding='same', strides=1)(block0_identity_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(256, kernel_size=1, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization()(x)

#combine the resNet fast forward sequence to conv sequence
x = keras.layers.Add()([x,block0_identity_input]) #add up values from both the resnet path and regular path

#after adding up, just to a relu of the entire thing
block0_identity_output = keras.layers.ReLU()(x)

#create a variable for the model of this entire complex thing
block0_identity = keras.Model(inputs = block0_identity_input, outputs = block0_identity_output, name = 'block0_identity')

############### END OF IDENTITY RESNET BLOCK ##################

######### FINALLY, lets build a model that holds the block0 conv resnet and two block0 id resnets

major_block_in = keras.Input(shape=(64,64,64), name='major_block_in')
#Per diagram, 1 RESNET CONV layer, followed by two id layers
x = block0_conv(major_block_in)
x = block0_conv1(x)
x = block0_identity(x)
x = block0_identity(x)


#Build another model to capture this sequence of models
block0 = keras.Model(inputs=major_block_in, outputs = x, name='block0')


"""
############# End of block 0 ###############
"""



"""
################# BEGINNING OF MAJOR BLOCK 1 #######################
"""
#128/2
#128
#512
#one pool to reduce dimensions

block1_input = keras.Input(shape=(16, 16, 256), name='block1_input')
x = keras.layers.Conv2D(128, kernel_size=1, activation='relu', padding='same', strides=2, name='b1_conv1')(block1_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', strides=1, name='b1_conv2')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(512, kernel_size=1, activation='relu', padding='same', strides=1, name='b1_conv3')(x)
x = keras.layers.BatchNormalization()(x)

#Fast forward ResNet layer now
# ? what is the input to the ResNet layer
#   -same dimensions as last output (56, 56, 256)
#block transform will take input and reshape it to desires dimensions
#also need strides = 2
#This conv should output a shape of 28x28x512
block1_transform = keras.layers.Conv2D(512, kernel_size=1, strides=2, activation='relu', padding='same')(block1_input)
x1 = keras.layers.BatchNormalization()(block1_transform)

#combine the resNet fast forward sequence to conv sequence
x = keras.layers.Add()([x,x1]) #add up values from both the resnet path and regular path

#after adding up, just to a relu of the entire thing
block1_output = keras.layers.ReLU()(x)

#create a variable for the model of this entire complex thing
block1_conv = keras.Model(inputs = block1_input, outputs = block1_output, name = 'block1_conv')

"""
###### END OF BLOCK 1 CONV #########
"""


"""
##### START OF BLOCK 1 IDENTITY ########
"""

block1_identity_input = keras.Input(shape=(8, 8, 512), name='block1_identity_input')
x = keras.layers.Conv2D(128, kernel_size=1, activation='relu', padding='same', strides=1)(block1_identity_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(512, kernel_size=1, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization()(x)

#combine the resNet fast forward sequence to conv sequence
x = keras.layers.Add()([x,block1_identity_input]) #add up values from both the resnet path and regular path

#after adding up, just to a relu of the entire thing
block1_identity_output = keras.layers.ReLU()(x)

#create a variable for the model of this entire complex thing
block1_identity = keras.Model(inputs = block1_identity_input, outputs = block1_identity_output, name = 'block1_identity')

"""
############### END OF BLOCK1 IDENTITY ##################
"""

######### FINALLY, lets build a model that holds the block0 conv resnet and two block0 id resnets

major_block1_in = keras.Input(shape=(16,16,256), name='major_block1_in')
#Per diagram, 1 RESNET CONV layer, followed by two id layers
x = block1_conv(major_block1_in)
x = block1_identity(x)
x = block1_identity(x)




#Build another model to capture this sequence of models
block1 = keras.Model(inputs=major_block1_in, outputs = x, name='block1')

"""

############## END OF BLOCK 1 #################

"""





"""
################# BEGINNING OF MAJOR BLOCK 2 #######################
"""
#256 1x1
#256 3x3
#1024 1x1
#one pool to reduce dimensions

block2_input = keras.Input(shape=(8, 8, 512), name='block2_input')
x = keras.layers.Conv2D(256, kernel_size=1, activation='relu', padding='same', strides=1, name='b2_conv1')(block2_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', strides=1, name='b2_conv2')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(1024, kernel_size=1, activation='relu', padding='same', strides=1, name='b2_conv3')(x)
x = keras.layers.BatchNormalization()(x)

#Fast forward ResNet layer now
# ? what is the input to the ResNet layer
#   -same dimensions as last output (56, 56, 256)
#block transform will take input and reshape it to desires dimensions
#also need strides = 2
#This conv should output a shape of 28x28x512
block2_transform = keras.layers.Conv2D(1024, kernel_size=1, strides=1, activation='relu', padding='same')(block2_input)
x1 = keras.layers.BatchNormalization()(block2_transform)

#combine the resNet fast forward sequence to conv sequence
x = keras.layers.Add()([x,x1]) #add up values from both the resnet path and regular path

#after adding up, just to a relu of the entire thing
block2_output = keras.layers.ReLU()(x)

#create a variable for the model of this entire complex thing
block2_conv = keras.Model(inputs = block2_input, outputs = block2_output, name = 'block2_conv')

"""
###### END OF BLOCK 2 CONV #########
"""


"""
##### START OF BLOCK 2 IDENTITY ########
"""

block2_identity_input = keras.Input(shape=(8, 8, 1024), name='block2_identity_input')
x = keras.layers.Conv2D(256, kernel_size=1, activation='relu', padding='same', strides=1)(block2_identity_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(1024, kernel_size=1, activation='relu', padding='same', strides=1)(x)
x = keras.layers.BatchNormalization()(x)

#combine the resNet fast forward sequence to conv sequence
x = keras.layers.Add()([x,block2_identity_input]) #add up values from both the resnet path and regular path

#after adding up, just to a relu of the entire thing
block2_identity_output = keras.layers.ReLU()(x)

#create a variable for the model of this entire complex thing
block2_identity = keras.Model(inputs = block2_identity_input, outputs = block2_identity_output, name = 'block2_identity')

"""
############### END OF BLOCK2 IDENTITY ##################
"""

######### FINALLY, lets build a model that holds the block0 conv resnet and two block0 id resnets

major_block2_in = keras.Input(shape=(8,8,512), name='major_block2_in')
#Per diagram, 1 RESNET CONV layer, followed by two id layers
x = block2_conv(major_block2_in)
x = block2_identity(x)
x = block2_identity(x)


#Build another model to capture this sequence of models
block2 = keras.Model(inputs=major_block2_in, outputs = x, name='block2')

"""

############## END OF BLOCK 2 #################

"""

def build_resnet_model():
    #define input shape to the mode
    #use structural connectivity to build the model, NOT SEQUENTIAL
    resnet_input = keras.Input(shape=(img_size[0], img_size[1], channels), name='input')

    x = keras.layers.Conv2D(64, kernel_size=7, activation='relu', padding='same', strides=2, name='conv7x7')(resnet_input)

    x = keras.layers.MaxPooling2D(padding = 'same', strides = 2, name='firstPool')(x)

    #Build that major subblock0
    x = block0(x)
    x = block1(x)
    x = block2(x)

    #Add the pooling layer
    x = keras.layers.MaxPooling2D(padding='same', strides=2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100, activation='relu')(x)

    #Output layer
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    #Lets build the model
    resnet_model = keras.Model(inputs=resnet_input, outputs = x, name='resnet_model')

    print(resnet_model.summary())
    return resnet_model


#call the build resnet func
model = build_resnet_model()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_set, epochs=20)
model.save('resnet42.h5', include_optimizer = True)


loss, accuracy = model.evaluate(test_set)
print("Loss of the ResNet42 on the celebrity dataset is: ", loss)
print("Accuracy of the ResNet42 on the celebrity dataset is: ", accuracy)