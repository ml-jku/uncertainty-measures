#!/bin/sh

##########################
# Deep Ensemble Training #
##########################

CUDA="cuda:0"

#### CIFAR10 ####
python train_mcdropout.py --dataset cifar10 --model resnet18 --cuda $CUDA
python train_mcdropout.py --dataset cifar10 --model densenet169 --cuda $CUDA
python train_mcdropout.py --dataset cifar10 --model regnet --cuda $CUDA

#### CIFAR100 ####
python train_mcdropout.py --dataset cifar100 --model resnet18 --cuda $CUDA
python train_mcdropout.py --dataset cifar100 --model densenet169 --cuda $CUDA
python train_mcdropout.py --dataset cifar100 --model regnet --cuda $CUDA

#### SVHN ####
python train_mcdropout.py --dataset svhn --model resnet18 --cuda $CUDA
python train_mcdropout.py --dataset svhn --model densenet169 --cuda $CUDA
python train_mcdropout.py --dataset svhn --model regnet --cuda $CUDA

#### TIN ####
python train_mcdropout.py --dataset tin --model resnet18 --cuda $CUDA
python train_mcdropout.py --dataset tin --model densenet169 --cuda $CUDA
python train_mcdropout.py --dataset tin --model regnet --cuda $CUDA
