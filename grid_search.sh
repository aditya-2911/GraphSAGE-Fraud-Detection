#!/bin/bash

echo "🚀 Starting Automated Grid Search..."

# We define the "Grid" of parameters we want to test
LEARNING_RATES=(0.01 0.005)
HIDDEN_DIMS=(64 128)
DROPOUTS=(0.5 0.7)  # Higher dropout to prevent the overfitting we just saw

for lr in "${LEARNING_RATES[@]}"; do
  for hidden in "${HIDDEN_DIMS[@]}"; do
    for dropout in "${DROPOUTS[@]}"; do
      
      echo "---------------------------------------------------"
      echo "Training with LR=$lr | Hidden=$hidden | Dropout=$dropout"
      echo "---------------------------------------------------"
      
      # Run the training script with these specific parameters
      python train.py --model graphsage --aggr mean --lr $lr --hidden_dim $hidden --dropout $dropout --epochs 100
      
    done
  done
done

echo "Grid Search Complete! Check WandB for the best run."