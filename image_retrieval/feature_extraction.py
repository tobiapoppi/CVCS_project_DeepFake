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
from matplotlib import pyplot as plt
from torchvision import models
from pytorchcv.model_provider import get_model
from torch.nn import Module
#from PIL import image
import torch
from torchvision import transforms
from PIL import Image

train_path = '/home/elena/repos/CVCS_project_DeepFake/output/train'
test_path = '/home/elena/repos/CVCS_project_DeepFake/output/test'
val_path = '/home/elena/repos/CVCS_project_DeepFake/output/validation'

train_files = glob.glob(train_path+'/*.png')+glob.glob(train_path+'/*.jpg')
train_lab_files = glob.glob(train_path+'/*.txt')

test_files = glob.glob(test_path+'/*.png')+glob.glob(test_path+'/*.jpg')
test_lab_files = glob.glob(test_path+'/*.txt')

val_files = glob.glob(val_path+'/*.png')+glob.glob(val_path+'/*.jpg')
val_lab_files = glob.glob(val_path+'/*.txt')

torchvision_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
transform = transforms.Compose(
        [transforms.RandomCrop(224, padding=10, pad_if_needed=True),
         #transforms.ToTensor(),
         torchvision_normalization])

feature_extraction = None

'Feature extraction from each image'
if feature_extraction == 'resnet':
    depth = 18
    dataset = 'imagenet'
    resnet = get_model(f'resnet{depth}', pretrained=True)
    features_extractor = resnet.features
    new_img = []
    labels = []
    new_img_name = []
    tensor_ds = torch.tensor([])
    for i, im in enumerate(train_files):
        image = cv2.imread(im)

        'Normalize take as input a tensor of shape (B, C, H, W)'
        image = torch.swapaxes(torch.from_numpy(image).to(torch.float), 1, 2)
        image = torch.swapaxes(image, 0, 1)
        image = torch.unsqueeze(image, 0)
        image = transform(image)
        features = torch.squeeze(torch.squeeze(features_extractor(image).detach(), 2), 2)
        tensor_ds = torch.concat((tensor_ds, features), 0)
        #new_img.append(image)
        with open(train_lab_files[i]) as y:
            labels.append(y.read())

    #features = torch.squeeze(torch.squeeze(features_extractor(tensor_ds), 2), 2).detach()
    print('features size {}'.format(features.size()))


else:
    labels = []
    new_img_name = []
    tensor_ds = torch.tensor([])
    for i, im in enumerate(train_files):
        image = cv2.imread(im)

        'Normalize take as input a tensor of shape (B, C, H, W)'
        image = torch.swapaxes(torch.from_numpy(image).to(torch.float), 1, 2)
        image = torch.swapaxes(image, 0, 1)
        image = torch.unsqueeze(image, 0)
        image = transform(image)
        features = torch.flatten(image)
        tensor_ds = torch.concat((tensor_ds, features), -1)
        with open(train_lab_files[i]) as y:
            labels.append(y.read())

    # features = torch.squeeze(torch.squeeze(features_extractor(tensor_ds), 2), 2).detach()
    print('features size {}'.format(features.size()))



query = tensor_ds[0].numpy()

def euclidean_distance(query, data):
    query = np.expand_dims(query,0)
    dist = np.sum((query - data)**2, axis=1)
    return dist

def cosine_similarity(query, data):
    similarities = [0]
    for img in data:
        dot_prod = np.dot(query, img)
        norm_dot_prod = np.dot(np.norm(query), np.norm(img))
        cosine_similarity = dot_prod/norm_dot_prod
        similarities.append(cosine_similarity)
    return similarities


distances = euclidean_distance(query, tensor_ds.numpy())
matching_imgs = np.argsort(distances)[1:11]
similar_images = [train_files[idx] for idx in matching_imgs]
dismatching_imgs = np.argsort(distances)[-10:]
unsimilar_images = [train_files[idx] for idx in dismatching_imgs]
print('query:', train_files[5])
print('similar images:', similar_images)
print('unsimilar images:', unsimilar_images)




i = cv2.imread(train_files[0])
plt.imshow(i)
plt.show()
plt.title('Query')
# create figure
sim_fig = plt.figure(figsize=(10, 7))
unsim_fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 5
for k in range(10):
    sim = cv2.imread(similar_images[k])
    sim_fig.add_subplot(rows, columns, k+1)
    plt.imshow(sim)
    plt.show()
    plt.title('{}-th Similar'.format(k))
    unsim = cv2.imread(unsimilar_images[k])
    unsim_fig.add_subplot(rows, columns, k+1)
    plt.imshow(unsim)
    plt.show()
    plt.title('{}-th Unsimilar '.format(k))



