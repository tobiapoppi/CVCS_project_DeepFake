import os
import glob
import argparse
from pathlib import Path
import random
import numpy as np
from numpy import sort
import shutil
import cv2
import glob
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import random


train_path = '/home/elena/repos/CVCS_project_DeepFake/output/train'
test_path = '/home/elena/repos/CVCS_project_DeepFake/output/test'
val_path = '/home/elena/repos/CVCS_project_DeepFake/output/validation'

train_files = glob.glob(train_path+'/*.png')+glob.glob(train_path+'/*.jpg')
train_lab_files = glob.glob(train_path+'/*.txt')

test_files = glob.glob(test_path+'/*.png')+glob.glob(test_path+'/*.jpg')
test_lab_files = glob.glob(test_path+'/*.txt')

val_files = glob.glob(val_path+'/*.png')+glob.glob(val_path+'/*.jpg')
val_lab_files = glob.glob(val_path+'/*.txt')

# x = np.zeros((len(train_files), 55*128))
x = []
sift = cv2.SIFT_create(200)

for i, im in enumerate(train_files):
    image = cv2.imread(im)
    kp, des = sift.detectAndCompute(image, None)
    [x.append(des[i]) for i in range(len(des))]
    # surf = cv2.xfeatures2d.SURF_CUDA(400)
    # print(i)
    # print(des.shape[0])
    # x[i, :(des.shape[0]*128)] = des.flatten()

# img2 = cv2.drawKeypoints(image, kp, None, (255, 0, 0), 4)
# plt.imshow(img2), plt.show()


C = 256
kmeans = KMeans(n_clusters=C)
kmeans.fit_transform(np.array(x))
BoW = kmeans.cluster_centers_

sift_unlimited = cv2.SIFT_create(500)
new_img = []
labels = []
for i, im in enumerate(train_files):
    image = cv2.imread(im)
    kp, des = sift_unlimited.detectAndCompute(image, None)
    clusters = kmeans.predict(des)
    epsilon = random.random()
    histogram = [list(clusters).count(i) for i in range(C)]
    new_img.append(histogram)
    with open(train_lab_files[i]) as y:
        labels.append(y.read())
    # if epsilon>0.9:
    #     print('Image example-----------------------------------------')
    #     print('Label', labels[-1])
    #     print('Number of key-points', len(des))
    #     print('Histogram vector', histogram)
assert len(train_files)==len(labels)
yTrain = np.array(labels)
xTrain = np.array(new_img)
np.save('yTrain', yTrain)
np.save('xTrain', xTrain)

new_img = []
labels = []
for i, im in enumerate(test_files):
    image = cv2.imread(im)
    kp, des = sift.detectAndCompute(image, None)
    clusters = kmeans.predict(des)
    histogram = [list(clusters).count(i) for i in range(C)]
    new_img.append(histogram)
    with open(test_lab_files[i]) as y:
        labels.append(y.read())
assert len(test_files)==len(labels)
yTest = np.array(labels)
xTest = np.array(new_img)
np.save('yTest', yTest)
np.save('xTest', xTest)

new_img = []
labels = []
for i, im in enumerate(val_files):
    image = cv2.imread(im, 0)
    kp, des = sift.detectAndCompute(image, None)
    clusters = kmeans.predict(des)
    histogram = [list(clusters).count(i) for i in range(C)]
    new_img.append(histogram)
    with open(val_lab_files[i]) as y:
        labels.append(y.read())
assert len(val_files)==len(labels)
yVal = np.array(labels)
xVal = np.array(new_img)
np.save('yVal', yVal)
np.save('xVal', xVal)


