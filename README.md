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
    - 4e-15 : PCA6
    - .015  : PCA5
    - .0278 : PCA2 + AE6
    - .0282 : PCA2 + AE5
    - .0286 : PCA1 + AE4
    - .0287 : PCA1 + AE6
    - .029  : PCA1 + AE5
    - .039  : PCA3 + AE6
    - .047  : PCA6 + AE6
    - .048  : PCA3 + AE3
    - .050  : PCA4 + AE5
    - .0502 :        AE5
    - .0508 :        AE6
    - .0516 : PCA5 + AE5
    - .0519 : PCA5 + AE6
    - .0521 : PCA3 + AE3
    - .0522 : PCA4 + AE6
    - .0523 : PCA3 + AE5
    - .0524 : PCA4 + AE4
    - .051  :        AE4
    - .053  :        AE3
    - .0540 : PCA2 + AE3
    - .0541 : PCA2 + AE2
    - .0549 : PCA2 + AE4
    - .0914 : PCA1 + AE1
    - .09168: PCA1 + AE3
    - .09169: PCA1 + AE2
    - .137  : PCA6 + AE5
    - .151  : PCA5 + AE4
    - .157  : PCA4
    - .164  : PCA4 + AE3
    - .213  : PCA6 + AE4
    - .229  : PCA3 + AE2
    - .242  : PCA5 + AE3
    - .258  :        AE2
    - .259  : PCA3
    - .263  : PCA4 + AE2
    - .280  : PCA6 + AE3
    - .315  : PCA2 + AE1
    - .324  : PCA5 + AE2
    - .329  : PCA6 + AE2
    - .337  : PCA3 + AE1
    - .351  : PCA4 + AE1
    - .377  :        AE1
    - .405  : PCA6 + AE1
    - .407  : PCA5 + AE1
    - .483  : PCA2
    - .682  : PCA1

6. `t1e2_pca_ae_nonlinear-2-chained.ipynb`
  - check above `t1e1...ipynb`
