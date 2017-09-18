import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import csv
from sklearn import cross_validation
from math import exp, pi

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.advanced_activations import ELU
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D

def perspective_transformation(image, delta = 15):
    rows, cols, _ = image.shape
    
    d = np.random.randint(delta, size = 8)
    
    pts1 = np.float32([[d[0],d[1]],[rows-d[2],d[3]],[d[4],cols-d[5]],[rows-d[6],cols-d[7]]])
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(image,M,(cols, rows))
    
    return dst

def image_change_brightness(image, max_change = 60):
    delta = np.random.randint(0, max_change, dtype='uint8')
    dst = np.where((255-image) < delta, 255, (image+delta))
    return dst

def random_image_change(image):
    new_image = image_change_brightness(image)
    new_image = perspective_transformation(new_image)
    return new_image

def random_steering_change(f, max_change = 0.04):
    random = 2 * np.random.random_sample() - 1
    dst = f + random * max_change
    return dst

def normalize_image(image):
    image = image.astype(float)
    image = (imgage / 255) - 0.5
    return imagge

# read data:
# steering angles form csv file
# image files paths from cvs file
# images from jpg files

csv_file = './udacity_data/driving_log.csv'
image_files = './udacity_data/'

steering = []
image_paths = []
images = []

with open(csv_file, newline='') as f:
    driving_data = list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE))

print('Number of data rows:', len(driving_data[1:]))

for row in range(1, len(driving_data)):
    if float(driving_data[row][6]) > 0.1:
        correction = 0.15
        steering_center = float(driving_data[row][3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        # center
        steering.append(steering_center)
        image_paths.append(image_files + driving_data[row][0].strip())
        # left
        steering.append(steering_left)
        image_paths.append(image_files + driving_data[row][1].strip())
        # right
        steering.append(steering_right)
        image_paths.append(image_files + driving_data[row][2].strip())
		
print('steerings and image paths loaded')

# crop and resize image
# the same must be done at drive.py

for i in range (len(image_paths)):
    image = mpimg.imread(image_paths[i])
    image = image[60:140, :]
    image = random_image_change(image)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    
    images.append(image)
	
print('Images added.')

# Flipping Images And Steering Measurements
# multuply number of data by 2
for i in range(len(images)):
    image = images[i]
    image_flipped = np.fliplr(image)
    steering_flipped = -steering[i]
    
    images.append(image_flipped)
    steering.append(steering_flipped)
	
print('Images flipped, number of data:', len(steering))

#
# remove overloaded data
# add data:
# - get Gaussian distribution number of data in rate of angles
# - get minimum number of data equal around 200
#
prob_to_save  = []
remove = []
added = 0
num_of_bins = 23
min_data_in_bin = 200.0
avg = []
sig_sq = 0.1
k = 2500

n, bins = np.histogram(steering, num_of_bins)

for i in range(len(n)):
    avg.append(k * exp(-(bins[i]**2) / (2*sig_sq)) / (2 * pi * sig_sq)**(0.5))
    
print('Average', avg)
print('n', n)

for i in range(len(n)):
    if avg[i] > min_data_in_bin:
        prob_to_save.append(avg[i]/n[i])
    else:
        prob_to_save.append(min_data_in_bin/n[i])

for i in range(len(steering)):
    random = np.random.rand()
    images[i] = image_change_brightness(images[i])
    
    for j in range(len(n)):
        if steering[i] >= bins[j] and steering[i] < bins[j+1]:
            random = np.random.rand()
            prob_to_add = 1.0/prob_to_save[j]
            if random > prob_to_save[j]:
                remove.append(i)
            
            if random > (prob_to_add):
                images_to_add = 1
                if n[j] < min_data_in_bin:
                    images_to_add = min_data_in_bin // n[j] + 1
                for k in range(int(images_to_add)):    
                    steering.append(random_steering_change(steering[i]))
                    images.append(random_image_change(images[i]))
                    added = added + 1

print('Images added', added)

for i in range(len(steering)):
    images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV)
	
print('rgb2hsv')

steering = np.asarray(steering)
images = np.asarray(images)

steering = np.delete(steering, remove)
images = np.delete(images, remove, axis=0)

print('Data removed', len(remove))
print('Number of data after removing:', len(steering))

n, bins = np.histogram(steering, num_of_bins)

print('n after gausian funcion used to add and remove images', n)

print('Images shape:', images.shape)
print('Steering shape:', steering.shape)

#
# create model 
#

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
model.add(Convolution2D(12,5,5, border_mode='valid', subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(18,5,5, border_mode='valid', subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(24,5,5, border_mode='valid', subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(32,3,3,  border_mode='valid'))
model.add(ELU())
model.add(Convolution2D(32,3,3,  border_mode='valid'))
model.add(ELU())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))

#model.compile(optimizer='adam', loss='mse')
model.compile(optimizer=Adam(lr=0.0001), loss='mse')
# train data
model.fit(images, steering, validation_split=0.2, shuffle=True, nb_epoch=35)

model.save('model.h5')
