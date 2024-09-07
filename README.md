## This repository contains the codes and results of the article "MetaML: A Multi-Label Meta-Learning Approach for Pipeline Recommendation".

There are two main files: the first is Search_Space.ipynb, which corresponds to the algorithm for projecting the search space, and the second is Meta_dataset_n5_k_5.ipynb, which corresponds to the use of the meta-base (meta_dataset.csv). The meta-features_OpenML.ipynb and meta_features_PYMFE.py files correspond to the meta-features extraction process.

This repository is also divided into folders: 

##### csv_meta_features
The first folder contains information for each type of meta-feature, along with its extraction time.

##### results_AutoML

This second folder corresponds to the results of each AutoML method (AutoGluon, Auto-Sklearn, Auto-Weka, H2O AutoML, Naive AutoML, FlaML-Zero Shot and TPOT) for each dataset.

##### results_PCC_n_5_k5

This third folder contains the MetaML results using PCC considering the results to check the best, considering a ranking of 3 for each dataset.

##### results_LP_CC_n_5_k5

This third folder contains the MetaML results using LP and CC.
