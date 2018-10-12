import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.linalg import inv
import math
import time
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

# of +1: 52,641
# of -1: 42,359
total: 95,000
943 x 1682 = 1,586,126

5 movies have no ratings:
[1325, 1414, 1577, 1604, 1637, 1681]
"""
start = time.clock()
d = 5
c = 1
sigma = 1

header = ["user_id","movie_id","rating"]
train_data = pd.read_csv("data/ratings.csv", names = header) # (95000, 3)
sample_size = train_data.shape[0] # 95,000
test_data = pd.read_csv("data/ratings_test.csv", names = header) # (5000, 3)
num_users = np.amax(train_data.user_id) # user_id range 1-943
num_movies = np.amax(train_data.movie_id) # movie_id range 1-1682
I = np.identity(d)
U = np.transpose(np.asarray([np.random.normal(scale=math.sqrt(0.1), size=d)
                             for idx in range(num_users)])) # (5, 943)
V = np.transpose(np.asarray([np.random.normal(scale=math.sqrt(0.1), size=d)
                             for idx in range(num_movies)])) # (5, 1682)

rating_matrix = np.full((num_users, num_movies), np.nan) # (943 users, 1682 movies)
for index, row in train_data.iterrows():
    rating_matrix[row["user_id"] - 1, row["movie_id"] - 1] = row["rating"]
boolean_rating_all = np.logical_not(np.isnan(rating_matrix))
boolean_rating_positive = rating_matrix == 1
boolean_rating_negative = rating_matrix == -1

# phi_df = pd.DataFrame(np.full((num_users, num_movies),np.nan),
#                       columns=range(1,num_movies+1),
#                       index=range(1,num_users+1))
e_phi_matrix = np.full((num_users, num_movies), 0.0) # (934, 1682)

UT_dot_V = np.matmul(U.T, V) # (# of u_i, # of v_j) - (934, 1682)
UT_dot_V_over_sigma = UT_dot_V / sigma
i= 444
j= 234
print np.dot(U[:,i], V[:,j]) , UT_dot_V[i][j]

def e_step():
    # UT_dot_V = np.matmul(np.transpose(U), V) # (# of u_i, # of v_j) - (934, 1682)
    # UT_dot_V_over_sigma = UT_dot_V / sigma
    pdf = norm.pdf(-UT_dot_V_over_sigma)
    cdf = norm.cdf(-UT_dot_V_over_sigma)
    positive = UT_dot_V + pdf / (1 - cdf)
    negative = UT_dot_V - pdf / cdf
    e_phi_matrix[boolean_rating_positive] = positive[boolean_rating_positive]
    e_phi_matrix[boolean_rating_negative] = negative[boolean_rating_negative]

def m_step():
    # first update u_i
    V_dot_VT = np.apply_along_axis(lambda x: np.outer(x, x), 1, V.transpose()) # (1682, 5, 5)
    V_dot_VT_plus_I = V_dot_VT + np.asarray([I for _ in range(num_movies)]) # (1682, 5, 5)
    V_dot_e_phi = np.apply_along_axis(
        lambda e_phi_i: V * e_phi_i, 1, e_phi_matrix) # (943, 5, 1682)
    for idx in range(num_users):
        first_sum_V = np.sum(V_dot_VT_plus_I[boolean_rating_all[idx]],
                             axis=0) # (5, 5)
        second_sum_v = np.sum(V_dot_e_phi[idx], axis=1) # (5,)
        U[:,idx] = np.matmul(inv(first_sum_V), second_sum_v) # (5,)


    # then update v_j
    U_dot_UT = np.apply_along_axis(lambda x: np.outer(x, x), 1, U.transpose()) # (943, 5, 5)
    U_dot_UT_plus_I = U_dot_UT + np.asarray([I for _ in range(num_users)]) # (943, 5, 5)
    U_dot_e_phi = np.apply_along_axis(
        lambda e_phi_j: U * e_phi_j, 1, e_phi_matrix.T) # (1682, 5, 943)
    for idx in range(num_movies):
        sum = np.sum(boolean_rating_all[:,idx])
        if sum != 0:
            first_sum_U = np.sum(U_dot_UT_plus_I[boolean_rating_all[:,idx]],
                                 axis=0) # (5, 5)
            second_sum_U = np.sum(U_dot_e_phi[idx], axis=1) # (5,)
            V[:,idx] = np.matmul(inv(first_sum_U), second_sum_U) # (5,)

def calculate_ln():
    UT_dot_V = np.matmul(np.transpose(U), V)  # (# of u_i, # of v_j) - (934, 1682)
    UT_dot_V_over_sigma = UT_dot_V / sigma
    logcdf = norm.logcdf(UT_dot_V_over_sigma) # (943, 1682)
    log_one_min_cdf = np.log(1 - norm.cdf(UT_dot_V_over_sigma))
    UT_dot_U = np.sum(U*U, axis=0) # (943,)
    VT_dot_V = np.sum(V*V, axis=0) # (1682,)
    UV_sum = np.asarray([[UT_dot_U[i] + VT_dot_V[j] for j in range(num_movies)]
                         for i in range(num_users)]) # (943, 1682)


    first_sum = sample_size * (-d) * math.log(2 * math.pi * c)
    second_sum = (-0.5 * c) * np.sum(UV_sum[boolean_rating_all])
    third_sum = np.sum(logcdf[boolean_rating_positive])
    fourth_sum = np.sum(log_one_min_cdf[boolean_rating_negative])
    sum = first_sum + second_sum + third_sum + fourth_sum
    print sum

e_step()
m_step()
calculate_ln()


iteration = 100
def run():
    for i in range(iteration):
        e_step()
        m_step()
        calculate_ln()

# run()
end = time.clock()
print "running time %.2f minutes" % ((end-start)/60.0)