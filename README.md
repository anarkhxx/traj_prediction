# Cell-ID Trajectory Prediction Using Multi-Graph Embedding and Encoder-Decoder Network
   Trajectory prediction for mobile phone users is a cornerstone component to support many higher-level applications in LBSs (Location-Based Services). Most existing methods are designed based on the assumption that the explicit location information of the trajectories is available (e.g., GPS trajectories). However, collecting such kind of trajectories lays a heavy burden on the mobile phones and incurs privacy concerns. In this paper, we study the problem of trajectory prediction based on cell-id trajectories without explicit location information and propose a deep learning framework (called DeepCTP) to solve this problem. Specifically, we use a multi-graph embedding method to learn the latent spatial correlations between cell towers by exploiting handoff patterns. Then, we design a novel spatial-aware loss function for the encoder-decoder network to generate cell-id trajectory predictions.
   
The paper can be visited at https://ieeexplore.ieee.org/abstract/document/9318015

# Requirement

       1.python=3.6
       2.tensorflow-gpu=1.8.0
       3.keras=2.2.0

# Data Preparation
   
1.1 src/data/dl-data/couplet/train/in.txt  ==>Base station label trajectory in the first half of training set

1.2 src/data/dl-data/couplet/train/out.txt  ==>Base station label track in the second half of training set

1.3 src/data/dl-data/couplet/test/intest.txt  ==>Label trace of base station in the first half of test set1.3

1.4 src/data/dl-data/couplet/test/outtest.txt  ==>Label trace of base station in the second half of test set

1.5 src/data/dl-data/couplet/vocabs  ==> Label of all cellular base stations

1.6 src/distance_cells/out1.txt  ==> Label trace of original cellular base station

1.7 src/distance_cells/out1_train.txt ==>Part of training set of original cellular base station label trajectory


### Data format description
`t is the trajectory, c is the base station`

(1)The data formats of 1.1-1.4 are described below：

      ... c1 c2 c3 c4 c5 ...
      ... c6 c7 c8 c9 c10 ...
      ...   ...   ....   ...  

 (2) The data formats of 1.5 are described below：
 
       <s>
       </s>
       c1
       c2
       ...
       cn
 
 (3)The data format of 1.6-1.7 is as follows. The data here is original, and the training and testing are segmented.
 
       t1c1 t1c2 t1c3 t1c4 ...
 
       t2c1 t2c2 t2c3 t2c4 ...
        
       t3c1 t3c2 t3c3 t3c4 ...

       ... ...  ...   ... ...    

# Run the Project

### Run the command below to train the model:

      python couplet.py       
   
   The training results are in the src/data/dl-data/models
   
### Run the command below to test the model:

      python forqatest.py     
  
  The results are output in src/ result.txt  ==> This part forqatest.py The code can be modified.

