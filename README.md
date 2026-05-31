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
./build/tree_exe path/to/dataset.csv

## Results
| Dataset | Rows | Test Accuracy | Training Time |
|---|---|---|---|
| Zoo | 101 | 81.0% | <0.5ms |
| Car Evaluation | 1728 | 90.5% | ~1.2919 ms |
| Mushroom | 8124 | 99.9% | ~2.6919 ms |

## Benchmark vs scikit-learn
| Dataset | Metric | This C++ ID3 | scikit-learn Decision Tree | scikit-learn Random Forest |
|---|---|---|---|---|
| Mushroom | Test accuracy | 99.9% | ~79.575% | 79.6% |
| Mushroom | Training time | ~2.6919 ms | ~2.402 ms | ~74.428 ms |
| Car Evaluation | Test accuracy | 90.5% | 63.9% | 63.9% |
| Car Evaluation | Training time | ~1.2919 ms | ~1.2742 ms | ~42.015 ms |

Same 60/20/20 train/validation/test split, information gain criterion. Averaged on 4 trials each.