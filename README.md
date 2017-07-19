# test-ml
Testing some ML.

# Installation
- pew new TESTML
- pip install jupyter sklearn keras tensorflow ...
- `jupyter notebook`

# Files
1. `t1a_pca_ae_simple.ipynb`
  - trying to verify that PCA ahead of AE does not affect anything about non-linearity
  - asked question on [SE/stats](https://stats.stackexchange.com/questions/292181/does-pca-ahead-of-an-autoencoder-deter-it-from-detecting-non-linearity)
  - demonstrates that AE without PCA is worse than AE with PCA
  - also shows that PCA is better than the combination PCA-AE for this simple data

2. `t1b_pca_ae_advanced.ipynb`
  - same as `t1a...`, but with an added "0" signal
  - if modified to use the PCA without dimensionality reduction, will show that the "0" signal affects the AE result
  - not very useful though, so just skip

3. `t1c_pca_ae_nonlinear.ipynb`
  - same as `t1a...`, but with an added signal being a superposition of 2 sine waves
    - both of which are already in the input
  - demonstrates that AE without PCA is worse than AE with PCA
  - also demonstrates that the PCA-AE combination is better than PCA alone
