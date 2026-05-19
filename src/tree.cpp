#include "tree.h"
#include <cmath>
#include <unordered_map>

// entropy calculates the Shannon entropy (in bits) of a label vector.
double entropy(const std::vector<int>& labels) {
  if (labels.empty()) return 0.0;

  std::unordered_map<int, int> counts;
  for (int label : labels) counts[label]++;

  double h = 0.0;
  const double n = static_cast<double>(labels.size());
  for (const auto& [label, count] : counts) {
    const double p = count / n;
    h -= p * std::log2(p);
  }
  return h;
}

// information_gain calculates the information gain of splitting on a feature, 
// given the rows, labels, and feature index to split on.
double information_gain(const std::vector<std::vector<int>>& rows,
                        const std::vector<int>&              labels,
                        int                                  feature_index) {
    // Partition labels by the value of the chosen feature
    std::unordered_map<int, std::vector<int>> partitions;
    for (int i = 0; i < static_cast<int>(rows.size()); ++i)
        partitions[rows[i][feature_index]].push_back(labels[i]);

    // Weighted entropy of the children.
    double weighted_h = 0.0;
    const double n    = static_cast<double>(labels.size());
    for (const auto& [val, part] : partitions)
        weighted_h += (static_cast<double>(part.size()) / n) * entropy(part);

    return entropy(labels) - weighted_h;
}

void Tree::fit(const std::vector<std::vector<int>>& rows,
               const std::vector<int>& labels) {
  // TODO: implement ID3
}

int Tree::predict(const std::vector<int>& row) const {
  // TODO
  return 0;
}