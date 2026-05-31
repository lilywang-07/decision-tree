# decision-tree

## What's Implemented
- ID3 algorithm with information gain and gain ratio
- Reduced-error pruning
- Stopping criteria: max_depth, min_samples_split, min_samples_leaf
- Multi-class support
- Train/validation/test split
- Accuracy and confusion matrix

## How to Build
cmake -S . -B build
cmake --build build

## How to Run
./build/tree_exe data/mushroom_fixed.csv

## Results
| Dataset | Rows | Test Accuracy | Training Time |
|---|---|---|---|
| Tennis | 14 | 100% | <1ms |
| Zoo | 101 | 76.7% | <1ms |
| Car Evaluation | 1727 | 66.8% | 1ms |
| Mushroom | 8123 | 97.0% | 3ms |

## Benchmark vs scikit-learn

| Dataset | Metric | This C++ ID3 | scikit-learn Decision Tree | scikit-learn Random Forest |
|---|---|---|---|
| Mushroom | Test accuracy | 97.0% | ~79.575% | 79.6% |
| Mushroom | Training time | ~3.4322 ms | ~2.402 ms | ~74.428 ms |
| Car Evaluation | Test accuracy | 66.8% | 63.9% | 63.9% |
| Car Evaluation | Training time | ~1.0086 ms | ~1.2742 ms | ~42.015 ms |

Same 60/20/20 train/validation/test split, information gain criterion. Averaged across 4 trials.