#pragma once
#include <vector>
#include <memory>

// Shannon entropy (in bits) of a label vector.
// Returns 0 for empty or pure (single-class) input.
double entropy(const std::vector<int>& labels);

// Information gain of splitting on a feature, given the rows, labels, and
// feature index to split on.
double information_gain(const std::vector<std::vector<int>>& rows,
                        const std::vector<int>&              labels,
                        int                                  feature_index);

class Tree {
 public:
  // Train the tree on rows + labels.
  void fit(const std::vector<std::vector<int>>& rows,
           const std::vector<int>& labels);

  // Predict the label for a single row.
  int predict(const std::vector<int>& row) const;

  // Predict labels for many rows.
  std::vector<int> predict(const std::vector<std::vector<int>>& rows) const;

 private:
  struct Node;                       // forward declaration
  std::unique_ptr<Node> root_;
};