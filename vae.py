
# coding: utf-8

# In[2]:


import random
import copy
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


# In[ ]:


x_train = []
x_val = []
x_test = []
num_fil = 0
num_train = 0
num_val = 0
dir_path = "img_align_celeba/"
i = 0
for filename in os.listdir(dir_path):
    if filename.endswith(".jpg"):
        i +=1
        if i == 10000:
            break
        num_fil += 1
num_train = int(round(num_fil*0.7))
print("num_train: " + str(num_train))
num_val = int(round((num_fil-num_train)*0.5))
print("num_val: " + str(num_val))
num_test = int(round(num_fil-num_train-num_val))
print("num_test: " + str(num_test))
i = 0
for filename in os.listdir(dir_path):
    if i == 10000:
        break
    if filename.endswith(".jpg"):
#         print("Yes")
        filepath = dir_path+"/"+filename
        if i<num_train:
            img = cv2.imread(filepath, 1)
            img = cv2.resize(img, (64,64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train.append(img)
        elif i<(num_train+num_val):
            img = cv2.imread(filepath, 1)
            img = cv2.resize(img, (64,64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_val.append(img)
        elif i<(num_train+num_val+num_train):
            img = cv2.imread(filepath, 1)
            img = cv2.resize(img, (64,64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_test.append(img)
    i += 1
x_train = np.array(x_train)
print("x_train size: " + str(len(x_train)))
x_val = np.array(x_val)
print("x_val size: " + str(len(x_val)))
x_test = np.array(x_test)
print("x_test size: " + str(len(x_test)))


# In[ ]:


plt.subplot(1, 2, 1)
plt.imshow(x_train[0])
plt.subplot(1, 2, 2)
plt.imshow(x_val[10])
plt.show()


# In[ ]:


#Normalizing the data 
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

x_train = np.reshape(x_train, (len(x_train), 64, 64, 3))
x_val = np.reshape(x_val, (len(x_val), 64, 64, 3))
x_test = np.reshape(x_test, (len(x_test), 64, 64, 3))


# In[ ]:


#Model

input_img = Input(shape=(64, 64, 3))  # shape of the images that wil be on input

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[ ]:


#Obfuscating the images

stripes_no = 24
x_train_inpaint = copy.deepcopy(x_train)
x_val_inpaint = copy.deepcopy(x_val)
x_test_inpaint = copy.deepcopy(x_test)
for i in range(len(x_train_inpaint)):
    stripes = random.sample(range(64), stripes_no)
    for j in range(len(x_train_inpaint[i])):
        if j in stripes:
            x_train_inpaint[i][j] = np.zeros((64,1))
            
for i in range(len(x_val_inpaint)):
    stripes = random.sample(range(64), stripes_no)
    for j in range(len(x_val_inpaint[i])):
        if j in stripes:
            x_val_inpaint[i][j] = np.zeros((64,1))

for i in range(len(x_test_inpaint)):
    stripes = random.sample(range(64), stripes_no)
    for j in range(len(x_test_inpaint[i])):
        if j in stripes:
            x_test_inpaint[i][j] = np.zeros((64,1))


# In[ ]:


n = 7
plt.figure(figsize=(20, 4))
for i in range(1,n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train_inpaint[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:


log = autoencoder.fit(x_train_inpaint, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_val_inpaint, x_val),
                verbose=2)


# In[ ]:


plt.figure()
[plt.plot(v,label=str(k)) for k,v in log.history.items()]
plt.legend()
plt.show()


# In[ ]:


decoded_imgs_inpaint = autoencoder.predict(x_test_inpaint)
n = 7
plt.figure(figsize=(20, 6))
for i in range(1,n):
    r = random.randint(0,1501)
    # display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test[r].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_inpaint[r].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i +n+ n)
    plt.imshow(decoded_imgs_inpaint[r].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

