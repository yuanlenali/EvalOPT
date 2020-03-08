#!/usr/bin/env bash

# Default options
DATA_FOLDER=data_evalOPT

# Create all folders
echo "*** Creating Folders..."
mkdir -p $DATA_FOLDER
mkdir -p $DATA_FOLDER/cifar-10
mkdir -p $DATA_FOLDER/imagenet
mkdir -p $DATA_FOLDER/mnist
mkdir -p $DATA_FOLDER/svhn

# Cifar-10
echo "*** Preparing Cifar-10..."
echo "Downloading Cifar-10 Binary..."
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -O $DATA_FOLDER/cifar-10/cifar-10-binary.tar.gz
echo "Extracting..."
tar -xvf $DATA_FOLDER/cifar-10/cifar-10-binary.tar.gz -C $DATA_FOLDER/cifar-10/ --strip 1
rm -f $DATA_FOLDER/cifar-10/cifar-10-binary.tar.gz

# ImageNet
echo "*** Preparing ImageNet..."
echo "CURRENTLY NOT IMPLEMENTED!!!!"

# MNIST
echo "*** Preparing MNIST..."
echo "Downloading Train Images..."
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O $DATA_FOLDER/mnist/train-images-idx3-ubyte.gz
echo "Downloading Train Labels..."
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O $DATA_FOLDER/mnist/train-labels-idx1-ubyte.gz
echo "Downloading Test Images..."
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O $DATA_FOLDER/mnist/t10k-images-idx3-ubyte.gz
echo "Downloading Test Labels..."
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O $DATA_FOLDER/mnist/t10k-labels-idx1-ubyte.gz

# SVHN
echo "*** Preparing SVHN..."
echo "Downloading Train..."
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat -O $DATA_FOLDER/svhn/train_32x32.mat
echo "Downloading Test..."
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat -O $DATA_FOLDER/svhn/test_32x32.mat
echo "Preprocessing SVHN..."
python -c "import deepobs.scripts._svhn_preprocess as _svhn_preprocess; _svhn_preprocess.preprocess(file_path='$DATA_FOLDER/svhn/', save_path='$DATA_FOLDER/svhn/')"
rm -f $DATA_FOLDER/svhn/train_32x32.mat
rm -f $DATA_FOLDER/svhn/test_32x32.mat

echo "*** Done."
echo "*** More datasets will be supported."