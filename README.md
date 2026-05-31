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
| Dataset | Metric | This C++ ID3 | scikit-learn CART | scikit-learn Random Forest |
|---|---|---|---|---|
| Mushroom | Test accuracy | 99.9% | 100.0% | 100.0% |
| Mushroom | Training time | ~2.6919 ms | ~3.3041 ms | ~87.050 ms |
| Car Evaluation | Test accuracy | 90.5% | 97.7% | 97.3% |
| Car Evaluation | Training time | ~1.2919 ms | ~0.9291 ms | ~47.008 ms |

Same 60/20/20 train/validation/test split, information gain criterion. Averaged on 4 trials each.