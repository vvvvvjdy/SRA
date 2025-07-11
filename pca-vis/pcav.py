import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import torch
def pca_v(nfeatures, n_components,thr):
    rgbs=[]
    for features in nfeatures:
        features = features.squeeze(0).cpu().numpy()
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features)
        features_forground = pca_features

        # Transform and visualize the first 3 PCA components
        for i in range(3):
            features_forground[:, i] = (features_forground[:, i] - features_forground[:, i].min()) / (
                        features_forground[:, i].max() - features_forground[:, i].min())
        rgb = pca_features.copy()
        rgb = rgb.reshape(16,16,3)
        rgbs.append(rgb)


    return rgbs




