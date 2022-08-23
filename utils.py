import torch
from torch import nn
from torchvision import transforms
import numpy as np
import cv2
import os 
from sklearn.cluster import KMeans

def save(path, images, labels):
    with open(path, 'w') as f:
        for (image, label) in zip(images, labels):
            f.write(image + ' ' + str(label) + '\n')

def get_images(root_path):
    paths = []
    for (root,dirs,files) in os.walk(root_path):
        for file in files:
            if ".JPG" in file:
                paths.append(os.path.join(root, file))
    return paths

def pca(features):
    # https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598
    # https://github.com/AdityaDutt/PCATutorial/blob/main/PCA_tutorial.ipynb 
    
    # Normalization
    mean = np.mean(features, axis= 0)
    mean_data = features - mean
    
    # Covariance
    cov = np.cov(mean_data.T)
    cov = np.round(cov, 2)
    # print("Covariance matrix ", cov.shape, "\n")

    # Eigen values
    eig_val, eig_vec = np.linalg.eig(cov)
    #print("Eigen vectors ", eig_vec.shape)
    #print("Eigen values ", eig_val.shape, "\n")

    # Sort eigen values and corresponding eigen vectors in descending order
    indices = np.arange(0,len(eig_val), 1)
    indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
    eig_val = eig_val[indices]
    eig_vec = eig_vec[:,indices]
    #print("Sorted Eigen vectors ", eig_vec)
    #print("Sorted Eigen values ", eig_val, "\n")

    # Get explained variance
    sum_eig_val = np.sum(eig_val)
    explained_variance = eig_val/ sum_eig_val
    #print(explained_variance)
    cumulative_variance = np.cumsum(explained_variance)
    n_components = np.searchsorted(cumulative_variance, 0.95, side="right")
    ## We will 2 components
    eig_vec = eig_vec[:,:n_components]
    print(eig_vec.shape)

    transformed_features = features.dot(eig_vec)
    print("Transformed data ", transformed_features.shape)
    return transformed_features

def cluster_data(features, no_of_clusters):
# Initialize the model
    model = KMeans(n_clusters=no_of_clusters, random_state=54)
    # Fit the data into the model
    model.fit(features)
    # Extract the labels
    labels = model.labels_
    return labels


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc = model.classifier[0]

    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        out = self.flatten(out)
        out = self.fc(out) 
        return out

    
    def extract_features(self, path):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(4000),
            transforms.Resize(224),
            transforms.ToTensor()                              
            ])

        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        self.to(device)
        #print("so far so good")

        img = cv2.imread(path)
        img = transform(img)
        # Reshape the image. PyTorch model reads 4-dimensional tensor
        # [batch_size, channels, width, height]
        img = img.reshape(1, 3, 224, 224)
        img = img.to(device)
        # We only extract features, so we don't need gradient
        with torch.no_grad():
            # Extract the feature from the image
            feature = self(img)
        feature = feature.cpu().detach().numpy().reshape(-1)
        #print("executed")
        #print(feature)
        return feature






