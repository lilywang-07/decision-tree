#pragma once
#include <vector>
#include <memory>

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