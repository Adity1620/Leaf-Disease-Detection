#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[2]:


IMAGE_SIZE = 256
BATCH_SIZE = 32  #Due to how memory is allocated on GPUs, batch sizes that are powers of 2 (e.g., 32, 64, 128) are often more efficient.
CHANNELS = 3     #The batch size needs to fit the memory requirement of GPU and the architecture of cpu


# In[3]:


data = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True, #for random shuffling
    image_size = (IMAGE_SIZE,IMAGE_SIZE),#from properties of dataset
    batch_size = BATCH_SIZE
)


# In[4]:


class_name = data.class_names  #The Folder name
class_name


# In[5]:


len(data)  #because every element is batch of 32 elements (68*32=2176) 


# In[6]:


for image_batch, label_batch in data.take(1):
    print(image_batch.shape)  #prints the no.of. images in one batch (32)
    print(label_batch.numpy())  #3 channels rgb color
    #potato_early_bright - 0; late_blight - 1; helathy - 2)
    


# In[7]:


for image_batch, label_batch in data.take(1):
    print(image_batch[0])  #basically a tensor value


# In[8]:


for image_batch, label_batch in data.take(1):
    print(image_batch[0].numpy())  #converting tensor to numpy


# In[9]:


for image_batch, label_batch in data.take(1):
    print(image_batch[0].shape)   #dimension of first image


# In[10]:


for image_batch, label_batch in data.take(1):
    print(image_batch.shape) 


# In[11]:


for image_batch, label_batch in data.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))#float value converted to int
    plt.title(class_name[label_batch[0]])


# In[12]:


plt.figure(figsize=(10,10))
for image_batch, label_batch in data.take(1):
    for i in range(12):
        x = plt.subplot(3,4,i+1) #dimensions
        plt.imshow(image_batch[i].numpy().astype("uint8"))#float value converted to int
        plt.title(class_name[label_batch[i]])


# In[13]:


# 80 percent as training and 20 percent as 10 percent as validation and 10 percent as test


# In[14]:


train_size = 0.8 #because 80 percent training
len(data)*train_size


# In[15]:


train_ds = data.take(54) #we take 54 sets and each set contain 32 elements
len(train_ds)  #first 54


# In[16]:


test_ds = data.skip(54)
len(test_ds) #taking the rest 54


# In[17]:


val_size = 0.1 #validation size 10 percent so taking 0.1
len(data)*val_size


# In[18]:


val_ds = test_ds.take(6) #we need actually 6 from test dataset
len(val_ds)


# In[19]:


test_ds = test_ds.skip(6)
len(test_ds)   #the rest is returned


# In[20]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)  #seed is to prevent same program running same time
        
    train_size = int(train_split*ds_size)   #result is float so converting to int
    val_size = int(val_split*ds_size)
    
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[21]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(data)  #function call


# In[22]:


len(train_ds)


# In[23]:


len(val_ds)


# In[24]:


len(test_ds)


# In[25]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)   #read from image from disk and for next iteration when we need same image it will keep the image in memory so improve performance of pipeline
#if we are using cpu and gpu, if gpu is busy training pre batch will load the next set of batch from disk and improve performance
#while gpu train batch 1, cpu will load that batch (batch 2)
#cache -> save time reading other images
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[26]:


# 0 to 255 -> rgb scale divide by 255 so we get a number between 0 and 1


# In[27]:


resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])  #we supply this layer when we actually build model

#our image has been already resized correctly as 256x256, but this layer will eventually go to model and when we have trained model, incase in prediction if we give different dimension image it take cares of it
# we use data augmentation inorder to apply transformation like make a image into more parts and add filters change staright to horizontal,etc.. and train


# In[28]:


#data augmentation

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),

])


# In[29]:


# Actually created a layer that makes preprocessing before entering into actual model


# In[30]:


input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes = 3

model= models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu',input_shape = input_shape), 
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'), 
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'), 
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'), 
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'), 
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'), 
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')   #last class with 3 neuron
    
])

model.build(input_shape = input_shape)
#conv is the convulation layer and first resize will happen followed by augmentation and convalution
#filter  is used to find the image splitted part (lets say i have a table, it finds its edges, legs,etc..)
#proper activation layer is relu ,activation later turns the input to meaningful representation of data
#max pool -> splits the image into type of matrix and and each parts and takes the max value of each part then flatten it


# In[31]:


model.summary()


# In[32]:


#configuring the learning process

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']  #matrix to calculate the training process
)


# In[37]:


history = model.fit(
    train_ds,
    epochs=50,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)


# In[39]:


scores = model.evaluate(test_ds)


# In[40]:


scores


# In[41]:


history


# In[42]:


history.params


# In[43]:


history.history.keys()


# In[44]:


history.history['accuracy']  #Python list with 50 values where we ran


# In[45]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[50]:


#Plot accuracy and validation accuracy using matplotlib

EPOCHS = 50
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),acc, label='Training Accuracy')
plt.plot(range(EPOCHS),val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(EPOCHS),loss, label='Training Loss')
plt.plot(range(EPOCHS),val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')  #Loss Basically an error when proceed forward it improves
plt.show()


# In[58]:


np.argmax([9.9985552e-01, 1.4440327e-04, 4.5913260e-18])  #at oth location it is maximum


# In[62]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8') #basically returns tensor converting to numpy
    first_label = labels_batch[0]
    
    print("first image to predict")
    plt.imshow(first_image)
    print("first image's actual label: ",class_name[first_label])
    
    batch_prediction = model.predict(images_batch)  #prediction for 32 images
    print("predicted label: ",class_name[np.argmax(batch_prediction[0])])  #prediction of first image
    #We get 3 dimension like 9.99 5.43 1.26 because we have 3 neurons n_classes and we used softmax basically softmax is probabaility
    #highest probability is the class [9.999... is actual class]
    


# In[72]:


#Simple function that takes model and image as input and says predicted class and confidence level

def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())  #converted image to image array and we created a batch out of it
    img_array = tf.expand_dims(img_array,0)   #Create a batch
    
    predictions = model.predict(img_array)
    
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class,confidence


# In[74]:


plt.figure(figsize=(15,15))
for images, labels in test_ds.take(1):  #Checks for only 9 images instead of taking for all
    for i in range(9):
        
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_name[labels[i]]
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")
        


# In[75]:


model_version = 1
model.save("C:/Users/avina/Downloads/Project/potato disease/models/{model_version}")


# In[76]:


model_version_1 = 2
model.save("C:/Users/avina/Downloads/Project/potato disease/models/{model_version_1}")


# In[80]:


import os
os.listdir("C:/Users/avina/Downloads/Project/potato disease/models")


# In[5]:


#import tensorflow as tf
#model_path = r"C:\Users\avina\Downloads\Project\potato disease\saved_models\{model_version}"
#model = tf.keras.models.load_model(model_path)


# In[8]:


#def predict(model,img):
    #img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())  #converted image to image array and we created a batch out of it
    #img_array = tf.expand_dims(img_array,0)   #Create a batch
    
    #predictions = model.predict(img_array)
    
    #predicted_class = class_name[np.argmax(predictions[0])]
    #confidence = round(100*(np.max(predictions[0])),2)
    #return predicted_class,confidence


# In[ ]:




