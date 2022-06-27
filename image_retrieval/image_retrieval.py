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
import torch
from torchvision import transforms
import argparse
from sklearn import metrics as m
from torch import nn

def euclidean_distance(query, data):
    query = np.expand_dims(query,0)
    dist = np.sum((query - data)**2, axis=1)
    return dist

def cosine_similarity(query, data):
    dot_prod = np.dot(query, np.transpose(data))
    norm_dot_prod = np.dot(np.linalg.norm(query), np.linalg.norm(data))
    cosine_similarity = dot_prod/norm_dot_prod
    return cosine_similarity

def get_args():
    parser = argparse.ArgumentParser("Image Retrieval for DeepFake Detection Task")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path for txt results",
    )
    parser.add_argument(
        "-m",
        "--metric",
        default='EuclideanDistance',
        type=str,
        help="Cosine Similarity or Euclidean Distance",
    )
    parser.add_argument(
        "-fe",
        "--features",
        default='resnet',
        type=str,
        help="Type of features extraction applied before the metric.",
    )
    parser.add_argument(
        "-q",
        "--query",
        default=5,
        type=int,
        help="Index of the chosen query.",
    )
    parser.add_argument(
        "-nm",
        "--normalization",
        default=True,
        type=bool,
        help="Apply normalization.",
    )
    parser.add_argument(
        "-d",
        "--resnetdepth",
        default=18,
        type=int,
        help="Choose the resnet depth",
    )
    parser.add_argument(
        "-n",
        "--matching_number",
        default=5,
        type=int,
        help='Choose the number of similar/unsimilar images you are looking for.'
    )
    args = parser.parse_args()
    return args


def main(opt):
    metric_options = ['CosineSimilarity', 'EuclideanDistance']
    metric = opt.metric
    assert metric in metric_options
    features_options = ['resnet', None, 'HarrisDetection']
    feat_extractor = opt.features
    assert feat_extractor in features_options
    query_id = opt.query
    norm = opt.normalization
    n = opt.matching_number
    train_path = '/home/elena/repos/CVCS_project_DeepFake/cvcs_dataset/train'
    test_path = '/home/elena/repos/CVCS_project_DeepFake/cvcs_dataset/test'
    val_path = '/home/elena/repos/CVCS_project_DeepFake/cvcs_dataset/validation'

    train_files = glob.glob(train_path + '/*.png') + glob.glob(train_path + '/*.jpg')
    train_lab_files = glob.glob(train_path + '/*.txt')

    test_files = glob.glob(test_path + '/*.png') + glob.glob(test_path + '/*.jpg')
    test_lab_files = glob.glob(test_path + '/*.txt')

    val_files = glob.glob(val_path + '/*.png') + glob.glob(val_path + '/*.jpg')
    val_lab_files = glob.glob(val_path + '/*.txt')

    tot_files = train_files+test_files+val_files
    path = opt.path

    torchvision_normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    maxpool = nn.MaxPool2d(3)

    if norm:
        transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         # transforms.ToTensor(),
         torchvision_normalization])
    else:
        transform = transforms.Resize((224, 224))

    if feat_extractor=='resnet':
        depth = opt.resnetdepth
        resnet = get_model(f'resnet{depth}', pretrained=True)
        features_extractor = resnet.features
        query = cv2.imread(train_files[query_id])
        query = torch.swapaxes(torch.from_numpy(query).to(torch.float), 1, 2)
        query = torch.swapaxes(query, 0, 1)
        query = torch.unsqueeze(query, 0)
        query = transform(query)
        query_features = torch.squeeze(torch.squeeze(features_extractor(query).detach(), 2), 2)
        if metric=='CosineSimilarity':
            similarities = []
            for i, im in enumerate(tot_files):
                image = cv2.imread(im)
                # if image.shape[0]==1024:
                #     image = maxpool(image)
                image = torch.swapaxes(torch.from_numpy(image).to(torch.float), 1, 2)
                image = torch.swapaxes(image, 0, 1)
                image = torch.unsqueeze(image, 0)
                image = transform(image)
                features = torch.squeeze(torch.squeeze(features_extractor(image).detach(), 2), 2)
                assert len(query_features)==len(features)
                #cos_sim = cosine_similarity(query_features, features)[0][0]
                cos_sim = m.pairwise.cosine_similarity(query_features, features)[0][0]
                similarities.append(cos_sim)
            matching_imgs = np.argsort(similarities)[-n:]
            similar_images = [tot_files[idx] for idx in matching_imgs]
            dismatching_imgs = np.argsort(similarities)[:n]
            unsimilar_images = [tot_files[idx] for idx in dismatching_imgs]
        if metric=='EuclideanDistance':
            distances = []
            for i, im in enumerate(tot_files):
                image = cv2.imread(im)
                # if image.shape[0] == 1024:
                #     image = maxpool(image)
                image = torch.swapaxes(torch.from_numpy(image).to(torch.float), 1, 2)
                image = torch.swapaxes(image, 0, 1)
                image = torch.unsqueeze(image, 0)
                image = transform(image)
                features = torch.squeeze(torch.squeeze(features_extractor(image).detach(), 2), 2)
                assert len(query_features) == len(features)
                dist =m.pairwise.euclidean_distances(query_features, features)[0][0]
                distances.append(dist)
            matching_imgs = np.argsort(distances)[:n]
            similar_images = [tot_files[idx] for idx in matching_imgs]
            dismatching_imgs = np.argsort(distances)[-n:]
            unsimilar_images = [tot_files[idx] for idx in dismatching_imgs]



        with open(path, 'a') as f:
            f.write('##\n')
            f.write(feat_extractor+'-'+str(depth))
            f.write(metric)
            f.write('\n')
            f.write('# \n')
            f.write(tot_files[query_id]+'\n')
            f.write('# \n')
            f.write(str(similar_images)+'\n')
            f.write('# \n')
            f.write(str(unsimilar_images))

    if feat_extractor == 'HarrisDetection':
        transform = transforms.Resize((299, 299))
        query = cv2.imread(train_files[query_id])
        print(type(query))
        query = torch.swapaxes(torch.from_numpy(query).to(torch.float), 1, 2)
        query = torch.swapaxes(query, 0, 1)
        query = torch.unsqueeze(query, 0)
        query = np.uint8(torch.swapaxes(torch.squeeze(transform(query), 0), 0, 2))
        qgray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        qgray = np.float32(qgray)
        query_features = cv2.cornerHarris(qgray, 2, 3, 0.04)
        query_features = np.reshape(query_features, -1)
        query_features = np.expand_dims(query_features, 0)
        if metric == 'CosineSimilarity':
            similarities = []
            for i, im in enumerate(tot_files):
                image = cv2.imread(im)
                # if image.shape[0]==1024:

                image = torch.swapaxes(torch.from_numpy(image).to(torch.float), 1, 2)
                image = torch.swapaxes(image, 0, 1)
                image = torch.unsqueeze(image, 0)
                image = np.uint8(torch.swapaxes(torch.squeeze(transform(image), 0), 0, 2))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                features = cv2.cornerHarris(gray, 2, 3, 0.04)
                features = np.reshape(features, -1)
                features = np.expand_dims(features, 0)
                assert len(query_features) == len(features)
                # cos_sim = cosine_similarity(query_features, features)[0][0]
                cos_sim = m.pairwise.cosine_similarity(query_features, features)[0][0]
                similarities.append(cos_sim)
            matching_imgs = np.argsort(similarities)[-n:]
            similar_images = [tot_files[idx] for idx in matching_imgs]
            dismatching_imgs = np.argsort(similarities)[:n]
            unsimilar_images = [tot_files[idx] for idx in dismatching_imgs]

        if metric == 'EuclideanDistance':
            distances = []
            #query_features = np.expand_dims(query_features, 0)
            for i, im in enumerate(tot_files):
                image = cv2.imread(im)
                image = torch.swapaxes(torch.from_numpy(image).to(torch.float), 1, 2)
                image = torch.swapaxes(image, 0, 1)
                image = torch.unsqueeze(image, 0)
                image = np.uint8(torch.swapaxes(torch.squeeze(transform(image), 0), 0, 2))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = np.float32(gray)
                features = cv2.cornerHarris(gray, 2, 3, 0.04)
                features = np.reshape(features, -1)
                #assert len(query_features) == len(features)
                features = np.expand_dims(features, 0)
                dist = m.pairwise.euclidean_distances(query_features, features)[0][0]
                distances.append(dist)
            matching_imgs = np.argsort(distances)[:n]
            similar_images = [tot_files[idx] for idx in matching_imgs]
            dismatching_imgs = np.argsort(distances)[-n:]
            unsimilar_images = [tot_files[idx] for idx in dismatching_imgs]

        with open(path, 'a') as f:
            f.write('##\n')
            f.write(feat_extractor + '-' )
            f.write(metric)
            f.write('\n')
            f.write('# \n')
            f.write(tot_files[query_id] + '\n')
            f.write('# \n')
            f.write(str(similar_images) + '\n')
            f.write('# \n')
            f.write(str(unsimilar_images))


if __name__ == "__main__":
    options = get_args()
    main(options)