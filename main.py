from cProfile import label
from functools import reduce
from json import load
from unittest import loader
import torch
from torch import optim, nn
from torchvision import models, transforms
import utils
import numpy as np

if __name__ == "__main__":
    # Initialize the model
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    vgg16_fe = utils.FeatureExtractor(vgg16)
    #print(vgg16_fe)
    images = utils.get_images("images")
    print("No. of images ", len(images))

    loaded_features = np.load("features.npy")
    print("Shape of features ", np.shape(loaded_features))
    transformed_features = utils.pca(loaded_features)

    labels = utils.cluster_data(transformed_features, 3)
    utils.save("labels_3_clusters.txt", images, labels)

    labels = utils.cluster_data(transformed_features, 2) 
    utils.save("labels_2_clusters.txt", images, labels)

    exit(4)

    #print(images)
    features = []
    for i in range(0,len(images), 1):
        print("Processing ", images[i])
        features.append(utils.FeatureExtractor.extract_features(vgg16_fe, images[i]))
    #print(features)
    #print("=====================================0")
    features = np.asarray(features)
    #np.save('features.npy', features)
    

    #labels = utils.cluster_data(features, 3)
    #utils.save("labels_3_clusters.txt", images, labels)

    #labels = utils.cluster_data(features, 2) 
    #utils.save("labels_2_clusters.txt", images, labels)