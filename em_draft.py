import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.linalg import inv

from numpy import linalg as la
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics.cluster import homogeneity_score, completeness_score
from sklearn.manifold import TSNE

"""
1. try avoid double looping user i and movie j, for example, 
in the per calculation of $E \Phi_{ij}$, or per $u_i^T v_j$. 
You could calculate the whole matrix like $\Phi$ or $U^T V$ in advance, 
sine matrix multiplication is paralleld to be faster operation 
(especially for sparsr matrix), than doing a double loop.

2. try to use Numpy operations to sum over an axis, 
or perform operation in selected entries in a matrix 
(you could look at numpy mask array to get inspiration)

3. Basically,  I make the rating matrix to be like: 
1 for positive rating, -1 for negative rating, and 0 for no rating. 
You could actually create \Phi as a 2d masked array, 
where the filter is where rating == 0.
"""

d = 5
c = 1
sigma = 1

header = ["user_id","movie_id","rating"]
train_data = pd.read_csv("data/ratings.csv", names = header) # (95000, 3)
test_data = pd.read_csv("data/ratings_test.csv", names = header) # (5000, 3)
num_users = np.amax(train_data.user_id) # user_id range 1-943
num_movies = np.amax(train_data.movie_id) # movie_id range 1-1682
I = np.identity(d)
U = np.transpose(np.asarray([np.random.normal(scale=0.1, size=d)
                             for idx in range(num_users)])) # (5, 943)
V = np.transpose(np.asarray([np.random.normal(scale=0.1, size=d)
                             for idx in range(num_movies)])) # (5, 1682)

rating_matrix = np.full((num_users, num_movies), np.nan) # (943 users, 1682 movies)

for index, row in train_data.iterrows():
    rating_matrix[row["user_id"] - 1, row["movie_id"] - 1] = row["rating"]

# phi_df = pd.DataFrame(np.full((num_users, num_movies),np.nan),
#                       columns=range(1,num_movies+1),
#                       index=range(1,num_users+1))
e_phi_matrix = np.full((num_users, num_movies), 0) # (934, 1682)

def e_step():
    UT_dot_V = np.matmul(np.transpose(U), V) # (# of u_i, # of v_j) - (934, 1682)
    UT_dot_V_over_sigma = UT_dot_V / sigma
    pdf = norm.pdf(-UT_dot_V_over_sigma)
    cdf = norm.cdf(-UT_dot_V_over_sigma)
    positive = UT_dot_V + pdf / (1 - cdf)
    negative = UT_dot_V - pdf / cdf
    e_phi_matrix[rating_matrix == 1] = positive[rating_matrix == 1]
    e_phi_matrix[rating_matrix == -1] = negative[rating_matrix == -1]
# e_step()

def m_step():
    # update u_i first
    V_dot_VT = np.apply_along_axis(lambda x: np.outer(x, x), 1, V.transpose()) # (1682, 5, 5)
    V_dot_VT_plus_I = V_dot_VT + np.asarray([I for _ in range(num_movies)]) # (1682, 5, 5)
    V_dot_e_phi = np.apply_along_axis(
        lambda e_phi_i: V * e_phi_i, 1, e_phi_matrix) # (943, 5, 1682)

    for idx in range(num_users):
        first_sum_V = np.sum(V_dot_VT_plus_I[np.logical_not(np.isnan(rating_matrix[idx]))],
                             axis=0) # (5, 5)
        second_sum_v = np.sum(V_dot_e_phi[idx], axis=1) # (5,)
        U[:,idx] = np.matmul(inv(first_sum_V), second_sum_v) # (5,)

    idx = 123
    first_sum_V = np.sum(V_dot_VT_plus_I[np.logical_not(np.isnan(rating_matrix[idx]))],
                         axis=0)  # (5, 5)
    second_sum_v = np.sum(V_dot_e_phi[idx], axis=1)  # (5,)
    U[:, idx] = np.matmul(inv(first_sum_V), second_sum_v)
    print U[:, idx].shape


    # then update v_j
    U_dot_UT = np.apply_along_axis(lambda x: np.outer(x, x), 1, U.transpose()) #  (943, 5, 5)
    U_dot_UT_plus_I = U_dot_UT + np.asarray([I for _ in range(num_users)])
    U_dot_e_phi = np.apply_along_axis(lambda e_phi_j: U * e_phi_j, 0, e_phi_matrix)





m_step()
