#!/bin/sh

######################################
# RUN ADVERSARIAL EXAMPLE GENERATION #
######################################

CUDA="cuda:0"

python generate_adversarial_examples.py --dataset=cifar10 --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --dataset=cifar10 --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --dataset=cifar100 --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --dataset=cifar100 --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --dataset=svhn --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --dataset=svhn --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --dataset=tin --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --dataset=tin --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --method=la --dataset=cifar10 --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --method=la  --dataset=cifar10 --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --method=la  --dataset=cifar100 --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --method=la  --dataset=cifar100 --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --method=la  --dataset=svhn --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --method=la  --dataset=svhn --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --method=la  --dataset=tin --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --method=la  --dataset=tin --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --method=mcd --dataset=cifar10 --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --method=mcd  --dataset=cifar10 --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --method=mcd  --dataset=cifar100 --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --method=mcd  --dataset=cifar100 --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --method=mcd  --dataset=svhn --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --method=mcd  --dataset=svhn --aa=linfpgd --device=$CUDA

python generate_adversarial_examples.py --method=mcd  --dataset=tin --aa=fgsm --device=$CUDA
python generate_adversarial_examples.py --method=mcd  --dataset=tin --aa=linfpgd --device=$CUDA