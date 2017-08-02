# test-ml
Testing some ML.

Can be viewed at [nbviewer/shadiakiki1986/test-ml](https://nbviewer.jupyter.org/github/shadiakiki1986/test-ml/)

# Installation
```
pew new TESTML
pip install jupyter sklearn keras tensorflow ...
jupyter notebook
```

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

3. `t1c_pca_ae_linear.ipynb`
  - same as `t1a...`, but with an added signal being a superposition of 2 sine waves
    - both of which are already in the input
  - demonstrates that AE without PCA is worse than AE with PCA
  - also demonstrates that the PCA-AE combination is better than PCA alone

4. `t1d_pca_ae_nonlinear.ipynb`
  - as `t1c` but with added signal being `sin1*sin2` instead of `sin1+sin2`
  - demonstrates that AE works the same with and without PCA
  - pending feedback from community / gabriel at my [SE/stats](https://stats.stackexchange.com/questions/292181/does-pca-ahead-of-an-autoencoder-deter-it-from-detecting-non-linearity) post

5. `t1e1_pca_ae_nonlinear-2-unchained.ipynb`
  - similar to `t1d`, but using the data from [this SE/stats question](https://stats.stackexchange.com/questions/190148/autoencoder-pca-tensorflow?rq=1)
  - adds to `t1d` using the MSE to quantify the replicated matrix's closeness to the original
  - removes from `t1d` the chained PCA+AE
  - chaining PCA+AE is done in `t1e2_pca_ae_nonlinear-2-chained.ipynb`
  - Models in ascending MSE (for both chained and unchained)
    - Notes
      - AE3 = autoencoder with dimensionality reduction to 3
      - PCA3 = same but with PCA
      - Explained variance ratio (cumsum): `[ 0.54268743  0.75709937  0.9540557   0.98194078  0.99972961  1.        ]`

| MSE   | PCA6 et al |PCA5 et al |PCA4 et al |PCA3 et al |PCA2 et al |PCA1 et al |AE |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| 9e-15 | PCA6       |           |           |           |           |           |   |
| .0094 |            |PCA5       |           |           |           |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .0502 |            |           |           |           |           |           |AE5|
| .0507 |            |           |           |           |           |           |AE6|
| .0514 |            |           |           |           |           |           |AE4|
| .0532 |            |           |           |           |           |           |AE3|
| .0659 | PCA6 + AE6 |           |           |           |           |           |   |
| .0736 |            |PCA5 + AE6 |           |           |           |           |   |
| .0755 |            |PCA5 + AE5 |           |           |           |           |   |
| .0772 |            |           |PCA4       |           |           |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .1231 |            |           |           |PCA3       |           |           |   |
| .1404 | PCA6 + AE5 |           |           |           |           |           |   |
| .1500 |            |PCA5 + AE4 |           |           |           |           |   |
| .1936 |            |           |PCA4 + AE5 |           |           |           |   |
| .1937 |            |           |PCA4 + AE6 |           |           |           |   |
| .1943 |            |           |PCA4 + AE4 |           |           |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .2447 |            |           |PCA4 + AE3 |           |           |           |   |
| .2588 |            |           |           |           |           |           |AE2|
| .2817 |            |           |           |PCA3 + AE6 |           |           |   |
| .2821 |            |           |           |PCA3 + AE4 |           |           |   |
| .2823 |            |           |           |PCA3 + AE5 |           |           |   |
| .2825 |            |           |           |PCA3 + AE3 |           |           |   |
| .2831 |            |           |           |           |PCA2       |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .3347 | PCA6 + AE4 |           |           |           |           |           |   |
| .3773 |            |           |           |           |           |           |AE1|
| .3885 |            |           |           |           |           |PCA1       |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .4281 |            |PCA5 + AE3 |           |           |           |           |   |
| .4852 |            |           |           |           |PCA2 + AE5 |           |   |
| .4853 |            |           |           |           |PCA2 + AE6 |           |   |
| .4948 |            |           |           |           |PCA2 + AE2 |           |   |
| .4949 |            |           |           |           |PCA2 + AE3 |           |   |
| .4950 |            |           |           |           |PCA2 + AE4 |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .5312 | PCA6 + AE3 |           |           |           |           |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .6700 |            |           |PCA4 + AE2 |           |           |           |   |
| .6835 |            |           |           |           |           |PCA1 + AE5 |   |
| .6836 |            |           |           |           |           |PCA1 + AE4 |   |
| .68365|            |           |           |           |           |PCA1 + AE2 |   |
| .69012|            |           |           |           |           |PCA1 + AE3 |   |
| .69013|            |           |           |           |           |PCA1 + AE1 |   |
| .6903 |            |           |           |           |           |PCA1 + AE6 |   |
| .6911 |            |           |           |PCA3 + AE2 |           |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .7446 |            |           |           |           |PCA2 + AE1 |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .8002 | PCA6 + AE2 |           |           |           |           |           |   |
| .8296 |            |PCA5 + AE2 |           |           |           |           |   |
| .8972 | PCA6 + AE1 |           |           |           |           |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|
| .9120 |            |           |PCA4 + AE1 |           |           |           |   |
| .9128 |            |           |           |PCA3 + AE1 |           |           |   |
| .9141 |            |PCA5 + AE1 |           |           |           |           |   |
|-------|------------|-----------|-----------|-----------|-----------|-----------|---|


6. `t1e2_pca_ae_nonlinear-2-chained.ipynb`
  - check above `t1e1...ipynb`

7. `t1e3_pca_ae_nonlinear-2-deep.ipynb`
  - same data as before, just deep AE, and an experiment of PCA + deep AE
