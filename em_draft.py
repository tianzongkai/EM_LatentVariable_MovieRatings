import numpy as np
import pandas as pd
from scipy.stats import norm
from numpy.linalg import inv, pinv
import math
import time
import random
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

for a 2D matrix axis=0 means column-wise and axis=1 means row-wise
"""
start = time.clock()
d = 5
c = 1
sigma = 1

header = ["user_id","movie_id","rating"]
train_data = pd.read_csv("data/ratings.csv", names = header) # (95000, 3)
# train_data = train_data.loc[train_data['user_id'] < 10]
sample_size = train_data.shape[0] # 95,000
# print sample_size
test_data = pd.read_csv("data/ratings_test.csv", names = header) # (5000, 3)
num_users = np.amax(train_data.user_id) # user_id range 1-943
num_movies = np.amax(train_data.movie_id) # movie_id range 1-1682
I = np.identity(d)
# U = 0.1 * np.random.randn(d, num_users)
# V = 0.1 * np.random.randn(d, num_movies)

rating_matrix = np.full((num_users, num_movies), 0) # (943 users, 1682 movies)
for index, row in train_data.iterrows():
    rating_matrix[row["user_id"] - 1, row["movie_id"] - 1] = row["rating"]
# boolean_rating_all = np.logical_not(np.isnan(rating_matrix))
boolean_rating_all = rating_matrix != 0
boolean_rating_positive = rating_matrix == 1
boolean_rating_negative = rating_matrix == -1

# phi_df = pd.DataFrame(np.full((num_users, num_movies),np.nan),
#                       columns=range(1,num_movies+1),
#                       index=range(1,num_users+1))
e_phi_matrix = np.full((num_users, num_movies), 0.0) # (934, 1682)

# UT_dot_V = np.matmul(U.T, V) # (# of u_i, # of v_j) - (934, 1682)
# UT_dot_V = np.dot(U.T,V)
# UT_dot_V_over_sigma = UT_dot_V / sigma
ratings_i, ratings_j = rating_matrix.nonzero()

U = np.full((d, num_users), 0.0)
V = np.full((d, num_movies), 0.0)

def initialize_UV(seed):
    global U, V
    random.seed(seed)
    U = np.transpose(np.asarray([np.random.normal(scale=math.sqrt(0.1), size=d)
                             for idx in range(num_users)])) # (5, 943)
    V = np.transpose(np.asarray([np.random.normal(scale=math.sqrt(0.1), size=d)
                             for idx in range(num_movies)])) # (5, 1682)


def update_phi():
    global e_phi_matrix
    UT_dot_V = np.matmul(U.T, V) # (# of u_i, # of v_j) - (934, 1682)
    UT_dot_V_over_sigma = UT_dot_V / sigma
    pdf = norm.pdf(-UT_dot_V_over_sigma)
    cdf = norm.cdf(-UT_dot_V_over_sigma)
    one_minus_cdf = 1 - cdf
    positive = UT_dot_V + pdf / (np.clip(one_minus_cdf,1E-75, None))
    negative = UT_dot_V - pdf / cdf
    e_phi_matrix[boolean_rating_positive] = positive[boolean_rating_positive]
    e_phi_matrix[boolean_rating_negative] = negative[boolean_rating_negative]

def e_step():
    global e_phi_matrix
    # Calculate all the expectations first
    UT_dot_V = np.matmul(U.T, V)
    pdfs = norm.pdf(-UT_dot_V / sigma)
    cdfs = norm.cdf(-UT_dot_V / sigma)
    for idx in range(len(ratings_i)):
        i = ratings_i[idx]
        j = ratings_j[idx]
        ut_dot_v = UT_dot_V[i, j]
        if np.sign(rating_matrix[i, j]) == 1:
            e_phi_matrix[i, j] = ut_dot_v + sigma * pdfs[i, j] / (1 - cdfs[i, j])
        elif np.sign(rating_matrix[i, j]) == -1:
            e_phi_matrix[i, j] = ut_dot_v + sigma * - pdfs[i, j] / cdfs[i, j]

def update_u():
    # update u_i
    global U#, UT_dot_V, UT_dot_V_over_sigma
    V_dot_VT = np.apply_along_axis(lambda v: np.outer(v, v), 1, V.T) # (1682, 5, 5)
    V_dot_VT_plus_I = V_dot_VT + np.asarray([I for _ in range(num_movies)]) # (1682, 5, 5)
    V_dot_e_phi = np.apply_along_axis(
        lambda e_phi_i: V * e_phi_i, 1, e_phi_matrix) # (943, 5, 1682)
    # V_dot_e_phi = np.matmul(e_phi_matrix, V.T) # (943,5)
    for i in range(num_users):
        # first_sum_V = np.sum(V_dot_VT_plus_I[boolean_rating_all[i]],axis=0) # (5, 5)
        first_sum_V = I + np.sum(V_dot_VT[boolean_rating_all[i]], axis=0) # (5, 5)
        # first_sum_V = np.sum(V_dot_VT_plus_I,axis=0) # (5, 5)
        second_sum_v = np.sum(V_dot_e_phi[i], axis=1) # (5,)
        # second_sum_v = np.sum(V_dot_e_phi.T, axis=1)  # (5,)
        U[:,i] = np.matmul(pinv(first_sum_V), second_sum_v) # (5,)
        # U[:,i] = np.clip(U[:,i], -0.95, 0.95)
        # U[:, i] = np.matmul(inv(first_sum_V), V_dot_e_phi[i].T)

    # update U^T*V
    # UT_dot_V = np.matmul(U.T, V)  # (# of u_i, # of v_j) - (934, 1682)
    UT_dot_V = np.dot(U.T, V)

    UT_dot_V_over_sigma = UT_dot_V / sigma

def update_v():
    # update v_j
    global V#, UT_dot_V, UT_dot_V_over_sigma
    U_dot_UT = np.apply_along_axis(lambda u: np.outer(u, u), 1, U.T) # (943, 5, 5)
    U_dot_UT_plus_I = U_dot_UT + np.asarray([I for _ in range(num_users)]) # (943, 5, 5)
    U_dot_e_phi = np.apply_along_axis(
        lambda e_phi_j: U * e_phi_j, 1, e_phi_matrix.T) # (1682, 5, 943)
    # U_dot_e_phi = np.matmul(e_phi_matrix.T, U.T) # (1682,5)
    for j in range(num_movies):
        sum = np.sum(boolean_rating_all[:,j])
        if sum != 0:
            # first_sum_U = np.sum(U_dot_UT_plus_I[boolean_rating_all[:,j]], axis=0) # (5, 5)
            first_sum_U = I + np.sum(U_dot_UT[boolean_rating_all[:,j]], axis=0) # (5, 5)
            # first_sum_U = np.sum(U_dot_UT_plus_I, axis=0) # (5, 5)
            second_sum_U = np.sum(U_dot_e_phi[j], axis=1) # (5,)
            # second_sum_U = np.sum(U_dot_e_phi.T, axis=1) # (5,)
            V[:,j] = np.matmul(pinv(first_sum_U), second_sum_U) # (5,)
            # V[:,j] = np.clip(V[:,j], -0.95, 0.95)
            # V[:,j] = np.matmul(inv(first_sum_U), U_dot_e_phi[j]) # (5,)
    # update U^T*V
    # UT_dot_V = np.matmul(U.T, V)  # (# of u_i, # of v_j) - (934, 1682)
    # UT_dot_V = np.dot(U.T, V)
    # UT_dot_V_over_sigma = UT_dot_V / sigma

def calculate_ln():
    first_sum = (num_users + num_movies) * (-d/2) * math.log(2 * math.pi * c)

    UT_dot_U = np.sum(U*U) # (u_1^T*u_1 + u_2^T*u_2 + ...)
    VT_dot_V = np.sum(V*V) # (v_1^T*v_1 + v_2^T*v_2 + ...)
    second_sum = (-0.5/c) * (UT_dot_U + VT_dot_V)

    UT_dot_V = np.matmul(U.T, V) # (943,1682)
    UT_dot_V_over_sigma = UT_dot_V / sigma
    logcdf = norm.logcdf(UT_dot_V_over_sigma) # (943, 1682)
    third_sum = np.sum(logcdf[boolean_rating_positive])
    # print "non-zeros:", np.count_nonzero(1 - norm.cdf(UT_dot_V_over_sigma))
    one_minus_cdf = 1 - norm.cdf(UT_dot_V_over_sigma)
    log_one_min_cdf = np.log(np.clip(one_minus_cdf, 1E-75, None))
    # log_one_min_cdf = np.log(1 - norm.cdf(UT_dot_V_over_sigma))
    fourth_sum = np.sum(log_one_min_cdf[boolean_rating_negative])

    ln_p = first_sum + second_sum + third_sum + fourth_sum
    return ln_p

def one_iteration():
    update_phi()
    # e_step()
    update_u()
    update_phi()
    # e_step()
    update_v()


def test2():
    test_rating_matrix = np.full((num_users, num_movies), np.nan)
    for index, row in test_data.iterrows():
        test_rating_matrix[row["user_id"] - 1, row["movie_id"] - 1] = row["rating"]
    boolean_test_rating_all = np.logical_not(np.isnan(test_rating_matrix))
    # print np.sum(boolean_test_rating_all)
    boolean_test_rating_positive = test_rating_matrix == 1
    # print np.sum(boolean_test_rating_positive)
    boolean_test_rating_negative = test_rating_matrix == -1
    # print np.sum(boolean_test_rating_negative)
    predict = np.sign(UT_dot_V_over_sigma)
    result = predict == boolean_test_rating_all
    print np.sum(result)

def test():
    UT_dot_V_over_sigma = np.dot(U.T, V)
    predict = np.sign(UT_dot_V_over_sigma)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for index, row in test_data.iterrows():
        i = row["user_id"] - 1
        j = row["movie_id"] - 1
        rating_truth = row["rating"]
        if rating_truth == 1:
            if predict[i,j] == 1:
                tp += 1
            else:
                fn += 1
        else: # truth = -1
            if predict[i,j] == -1:
                tn += 1
            else:
                fp += 1
    correct = tp + tn
    confusion_matrix = {"tp":tp, "tn":tn, "fn":fn, "fp":fp}
    print 'correct:', correct
    print 'confusion_matrix', confusion_matrix

def plot_result(results, iteration, seed):
    pass



def run_a():
    seed = 756
    initialize_UV(seed)
    iteration = 100
    log_likelihood_array = []
    plt.figure(figsize=(9, 6))
    for i in range(iteration):
        if i%10 == 0: print "iteration:", i
        one_iteration()
        if i>0: log_likelihood_array.append(calculate_ln())
    plt.plot(range(2, iteration + 1), log_likelihood_array, label=('seed=%d' % seed))
    plt.xlim((2, iteration))
    plt.xticks(np.linspace(0, iteration, 11))
    plt.grid(True)
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("ln p(R,U,V)", fontsize=14)
    plt.legend(loc='best')
    plt.title(("Problem 2.a), random seed = %d" % seed), fontsize=14)

    plot_result(log_likelihood_array,iteration, seed)
    print "\nafter 100 iterations:"
    test()
# run_a()

def run_b():
    seeds = [201, 333, 456, 852, 965]
    iteration = 100
    plt.figure(figsize=(9, 6))
    for s in range(5):
        seed = seeds[s]
        initialize_UV(seed)
        log_likelihood_array = []
        for i in range(iteration):
            if i % 10 == 0: print "iteration:", i
            one_iteration()
            if i > 18: log_likelihood_array.append(calculate_ln())
        print "\nafter 100 iterations:"
        test()
        plt.plot(range(20, iteration + 1), log_likelihood_array, label=('seed=%d' % seed))
    plt.xlim((20, iteration))
    plt.xticks(np.linspace(20, iteration, 9))
    plt.grid(True)
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("ln p(R,U,V)", fontsize=14)
    plt.legend(loc='best')
    plt.title("Problem 2.b)", fontsize=14)
    plt.show()
run_b()

end = time.clock()
print "running time %.2f minutes" % ((end-start)/60.0)
plt.show()






