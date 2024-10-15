#!/bin/sh

#########################
#  MCD Active Learning  #
#########################

CUDA="cuda:0"

###### MNIST ######

# EU C3
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --train_size=1000

# EU C2
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --train_size=1000

# Random
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --train_size=1000

# EU B3
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --train_size=1000

# AU C
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --train_size=1000

# TU BC3
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --train_size=1000

## TU BC2 AU B
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --train_size=1000
python run_active_learning.py --dataset=mnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --train_size=1000


###### FMNIST ######


# EU C3
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15

# EU C2
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_c2 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15

# Random
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=random --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15

# EU B3
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=eu_b3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15

# AU C
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=au_c --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15

# TU BC3
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc3 --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15

## TU BC2 AU B
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=42 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=142 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=242 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=342 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15
python run_active_learning.py --dataset=fmnist --method=mc_dropout --acquisition_function=tu_bc2_au_b --n_samples=50 --p_drop=0.2 --device=$CUDA --seed=442 --start_samples_per_class=40 --n_iterations=41 --n_samples_per_iteration=15

