# CMD
Central Moment Discrepancy for Domain-Invariant Representation Learning - ICLR 2017

This repository contains code for reproducing the experiments reported in the paper [Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning](http://openreview.net/forum?id=SkB-_mcel) published at the International Conference on Learning Representations (ICLR2017) by Werner Zellinger, Edwin Lughofer and Susanne Saminger-Platz from the Department of Knowledge Based Mathematical Systems at the JKU Linz, and, Thomas Grubinger and Thomas Natschläger from the Data Analysis Systems Group at the Software Competence Hagenberg.

In the source code, the CMD domain-regularizer is denoted by 'mmatch'. Please note that our work has been refined and its source code is available on [github](https://github.com/wzell/mann/). A refined version of our paper with extended theory and experiments is also [available](https://arxiv.org/abs/1711.06114).

# Requirements
The implementation is based on Theano and the neural networks library Keras. For installing Theano and Keras please follow the installation instruction on the respective github pages. You will also need: numpy, pandas, seaborn, matplotlib, sklearn and scipy

# Datasets
We report results for two different benchmark datasets in our paper: AmazonReview and Office. In addition, the model weights of the VGG_16 model pre-trained on Imagenet are used. The AmazonReview data set can be downloaded from http://www.cse.wustl.edu/~mchen/code/mSDA.tar. The file mSDA/examples/amazon.mat should be copied to utils/amazon_dataset/. The Office data set can be downloaded from https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view. Copy the folders amazon, dslr and webcam to utils/office_dataset/. Download the VGG16 weights file from http://files.heuritech.com/weights/vgg16_weights.h5 and copy it to  utils/office_dataset/.

# Experiments
Use the files exp_office.py, exp_amazon_review.py and parameter_sensitivity.py to run all the experiments and create all the images from the paper. Please note that the code runs the full grid searches and random restarts and can therefore run some days.

# Precomputed weights
If you don't get the exact same results but similar ones, the random number generator of Theano could be the reason. This can be solved by using our precomputed weights in the folder utils/amazon_dataset/precomputed_weights.
