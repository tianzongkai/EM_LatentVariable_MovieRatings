import numpy as np
from numpy import linalg as la
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics.cluster import homogeneity_score, completeness_score
from sklearn.manifold import TSNE

header = ["user_id","movie_id","rating"]
train_data = pd.read_csv("data/ratings.csv", names = header) # (95000, 3)
test_data = pd.read_csv("data/ratings_test.csv", names = header) # (5000, 3)
print train_data.shape
print test_data.shape