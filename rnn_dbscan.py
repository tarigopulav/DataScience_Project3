# -*- coding: utf-8 -*-
"""RNN_DBSCAN.ipynb

# IMPLEMENTATION

For performance reasons, a k-nn, k-rnn map will be constructed offline as an indexing table. 
Nearest neighbors are found using scikit-learn's highly optimized KDTree implementation as a pure numpy implementation turned out to be very slow.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import itertools
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from sklearn import metrics
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

def dist(x,y):
  return np.linalg.norm(x-y)


def n_k(x,k):
  #use the pretrained indexed nn map
  global kdt
  global X
  return (X[kdt.query(x.reshape(1, -1), k=k, return_distance=False)]).\
  squeeze()
   

def r_k(x, k, c_indexes=None):
  '''
  returns the k-RNN of x
  
  x: single element (2d numpy array)
  X: total data matrix (n, d)
  k: int  
  c_indexes: index list to sub set
  '''
  global X
  
  # condition 2: all the points in N that have x as a NN
  ## first, find the nearest neaghbors for each vector in N
  NN = [n_k(X[i,:], k) for i in range(X.shape[0])]  
  
  ## then, find the indexes of the vectors in X that have x in their NN set
  idx = np.isclose(NN, x).all(axis=(2)).any(axis=1)
  RNN = X[idx, :]

  return RNN

  
def is_directly_density_reachable(x, y, k):
  '''
  x, y: two vectors (each is a 2d numpy array)
  k: int
  '''
  global X
  global k_nn
  global k_rnn
  # condition 1: x is in the k-NN set of y
  NN_y = n_k(y, k) 
  cond_1 = np.isclose(NN_y, x).all(axis=(1)).any()
  
  # condition 2: k-RNN(y) >= k
  RNN_y = k_rnn[match_indexes(y, X)][0]#r_k(y, k)
  cond_2 = RNN_y.shape[0] >= k
  
  return cond_1 & cond_2

def find_density_reachable_points(C, k):
  '''
  returns the pairs of vectors in :vectors: that are density reachable
  
  C: Cluster we're interested in (2D numpy array)
  k: int  
  '''

  if C.shape[0] ==1:
    return None
  
  idx_range = range(C.shape[0])
  indexes = []
  
  for subset in itertools.combinations(idx_range, 2):
    indexes.append(subset)
  
  indexes = np.array(indexes, dtype=int)
  
  # vectors: numpy array of arrays (containing candidate vectors)
  vectors = C

  # map indexes to their vrector
  index_vector_map = {i:vectors[i,:] for i in np.ravel(indexes)}
  
   
  xy_df = pd.DataFrame(indexes)
  xy_df['x'] = xy_df[0].map(index_vector_map)#list(f[:, 0, :])
  xy_df['y'] = xy_df[1].map(index_vector_map)#list(f[:, 1, :])
  xy_df['is_den_con'] = 0
  xy_df['is_den_con'] = xy_df.apply(\
                                    lambda x: \
                                    is_directly_density_reachable(x.x,\
                                                                  x.y,\
                                                                  k
                                                                 ), axis=1)
  xy_df = xy_df.loc[xy_df['is_den_con']==True, ['x','y']]  
  
  return xy_df.values 
  

def density(C, k):
  '''
  returns the density of a cluster
  returns np.inf if there are no x,y points in C that satisfy the conditions \
  stated in Definition 7
  
  C: 2D numpy array [clustetr?]
  X: Total dataset
  '''

  # find the indexes of C in X
  global X
  global k_rnn
  c_indexes = match_indexes(C, X)
  
  # add a new column to check the eligibility to be x and y (|R_k(x)| > k)
  C_ = pd.DataFrame(C)
  C_['RNN'] = [k_rnn[match_indexes(C[i,:], X)][0] for i in range(C.shape[0])]
  C_['shape'] = C_['RNN'].apply(lambda x: x.shape[0])
  C_.drop(['RNN'], axis=1, inplace=True)
  
  
  C_ = C_.loc[C_['shape']>=k, :]
  C_ = C_.values[:, :-1] # only the elgible vectors
  
  # no eligible points in the cluster
  if C_.shape[0]==0:
    return np.inf

  else:
    # narrowing down to only the densely connected pairs
    C_d = find_density_reachable_points(C_, k)
    # no eligible points in the cluster
    if (C_d is None) or (C_d.shape[0]==0):
      return np.inf
    else:
      # calculate distances
      distances = [dist(C_d[i][1], C_d[i][0]) for i in range(C_d.shape[0])]
      density = np.max(distances)
      return density  
  
def neighborhood_helper(x, k):
  '''
  calculates the set described in {y \in R_k(x) : |R_k(y)|>=k}
  (elements(y) in k-RNN(x) that has more than (or equal) k elements in each of\
  their respective k-RNN(y)s)
  
  x: single element (2d numpy array)
  k: int
  '''
  global k_rnn
  
  # first find the k-RNN of x
  RNN_x = k_rnn[match_indexes(x, X)][0]#r_k(x, k)
  
  # for each vector y in k-RNN(x): find k-RNN(y)...
  RNN_y = np.array([k_rnn[match_indexes(y, X)][0] for y in RNN_x])
  # .. such that |k-RNN(y)| > k
  RNN_y_size = np.array([y.shape[0] for y in RNN_y])>k
  
  result = RNN_x[RNN_y_size]
  
  return result  

def match_indexes(n, X):
  '''
  matches n with X and returns the indexes of X that have n's elements
  
  n: 2D numpy array
  X: 2D numpy array
  '''
#   print(n)
  
  if n.ndim > 1:
    index_list = [np.isclose(X, n[i]).all(axis=1).nonzero()[0][0] for i in \
                range(n.shape[0])]
  else:
    index_list = [np.isclose(X, n).all(axis=1).nonzero()[0][0]]
    
  
  return index_list
  
def neighborhood(x, k):
  '''
  Algorithm 3
  -----------
  
  In addition to what's in the text, this returns the list of indexes(axis=0) \
  of the neighborhood vectors in X matrix
  '''
  global X

  NN = n_k(x, k)
  
  other_vectors = neighborhood_helper(x, k)
  
  if other_vectors.shape[0]!=0:
    out = np.vstack((NN, other_vectors))
    out = np.unique(out, axis=0)
    return out, match_indexes(out, X)
    
  else:
    return NN, match_indexes(NN, X)
  
def expand_cluster(x, cluster, assign, k, i):
  '''
  Algorithm 2
  -----------
  
  returns True
  
  returns False: if x is noise
  
  x: data point of interest (2D numpy array)
  cluster: already assigned cluster (int)
  assign: numpy array
  k: int
  i: index of x in X (passed from rnn_dbscan())
  '''

  global k_rnn
  global X

  r = k_rnn[match_indexes(x, X)][0]
  
  if r.shape[0] < k:
    assign[i] = -1
    return False
  
  else:
    neighborhood_x, idx_x = neighborhood(x, k)
    seeds = deque(neighborhood_x)
    assign[i] = cluster
    assign[idx_x] = cluster
    
    while(len(seeds)) > 0:
      y = seeds.pop()
      r_y = k_rnn[match_indexes(y, X)][0]#r_k(y, k)
      
      if r_y.shape[0] >= k:
        neighborhood_y, idx_y = neighborhood(y, k)
        for z,j in zip(neighborhood_y, idx_y):
          if assign[j] == 0:
            seeds.extend([z])
            assign[j] = cluster
          elif assign[j] == -1:
            assign[j] = cluster
            
    return True
  
  
def expand_clusters(k, assign):

  global X
  global k_rnn
  
  for i,x in enumerate(X):
    
    if assign[i] == -1:
      
           
      neighbors_x = n_k(x, k)
      min_cluster = -1
      min_dist = np.inf
      
      for j,n in enumerate(neighbors_x):

   
        idx = match_indexes(n, X)
        cluster = assign[idx][0]
        cluster_elements = X[idx,:]#X[idx,:]
        d = dist(x, n)
        
        r_k_n = k_rnn[match_indexes(n, X)][0] #r_k(n, k)
        if (r_k_n.shape[0] >= k) & (d <= density(cluster_elements, k)) &\
        (d < min_dist):
          min_cluster = cluster
          min_dist = d
      
      assign[i] = min_cluster  

def rnn_dbscan(X, k):
  '''
  returns a numpy array of shape X.shape[0] that contains the cluster \
  assignment of each point.
  
  0: unclassified
  -1: noise
  
  X: Data matrix (2D numpy array)
  k: int (< X.shape[0])
  '''
    
  assign = np.zeros((X.shape[0]), dtype=object)

  if (k<=1 or k>X.shape[0]):
    print("K must satisfy k>1 or k<X.shape[0]\n")
    return assign
  
  cluster = 1
  
  for i in range(X.shape[0]):
    x = X[i, :]
    if assign[i] == 0:
      if expand_cluster(x, cluster, assign, k, i):
        cluster += cluster
        
  start = datetime.now()
  print("Expanding clusters ... ")      
  expand_clusters(k, assign)
  
  return assign
  

def initialize_nn(X, k):
  
  nn = np.array([n_k(X[i, :], k) for i in range(X.shape[0])])
  
  return nn

def initialize_rnn(X, k):
  
  rnn = np.array([r_k(X[i, :], k) for i in range(X.shape[0])])
  
  return rnn

def pre_process(df):
  
  X = df.values[:, :-1]
  target = df.values[:, -1]
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  return X, target

def run(df,k_range=range(2,10), plots=False):
  
  global X, kdt, k_rnn
  
  X, target = pre_process(df)  
  
  kdt = KDTree(X, leaf_size=30, metric='euclidean')
    
  ARI = []

  for k in k_range:
        
    if (k<=1 or k>X.shape[0]):
        print("K must satisfy k>1 or k<X.shape[0]\n")
        ARI.append(0)
        pass
    else:
        print("\nInitializing for k = {0}".format(k))    
        k_rnn = initialize_rnn(X, k)
        
        asg = rnn_dbscan(X, k)
        ari = metrics.adjusted_rand_score(target, asg)
        ARI.append(ari)
        print("ARI_{0} = {1}".format(k, ari))
        print("Number of clusters found: {0}".format(len(np.unique(asg))))
        
        if (plots==True):
            
            df_X = pd.DataFrame(X)
            df_X['cluster'] = asg
            sns.lmplot(x="0", y="1", data=df_X.rename(columns=lambda x: str(x)),
                       hue='cluster', fit_reg=False, legend=False, markers=".")                      
            plt.scatter(X[asg==-1, 1], X[asg==-1,0], marker='+');
            plt.title("k={0} | Number of clusters={1} | ARI: {2}".format(k, 
                                                  len(np.unique(asg)),
                                                   ari))
            plt.show()
            
  res = pd.DataFrame({'k':k_range, 'ARI':ARI})
  res.sort_values(by='ARI', ascending=False,\
                  inplace=True)
  res.reset_index(inplace=True, drop=True)
  print('\n######\nk values:\n')
  print(res.head())


### FLAME

"""
df = pd.read_csv('data/flame.txt', sep='\t', header=None)
df.head()
k_range=[9]
run(df, k_range=k_range, plots=True)

### GRID

"""
df = pd.read_csv('data/grid.csv', header=None)
df.head()
k_range=[4]
run(df, k_range=k_range, plots=True)


### IRIS

"""
df = pd.read_csv('data/iris.txt', header=None)
df.head()
k_range = [20]
run(df, k_range=k_range, plots=True)

### Bank

"""
df = pd.read_csv('data/data_banknote_authentication.txt', header=None)
df.head()
k_range = [20]
run(df, k_range=k_range, plots=True)

### D31

"""
df = pd.read_csv('data/D31.txt', sep='\t', header=None)
df.head()
k_range=[15]
run(df, k_range=k_range, plots=True)

### CTG

"""
df = pd.read_csv('data/ctg.csv',header=None)
df.head()
k_range=[10]
run(df, k_range=np.arange(10)
*/
