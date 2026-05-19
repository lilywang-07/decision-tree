#include "tree.h"
#include <cmath>
#include <unordered_map>
#include <algorithm>

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

// Tree ------------------------------------------------------------------------

// returns the majority label among a vector of labels, with tie-breaking
// favoring the smaller class
int Tree::majority(const std::vector<int>& labels) {
    std::unordered_map<int, int> counts;
    for (int l : labels) counts[l]++;
    return std::max_element(counts.begin(), counts.end(),
        [](const auto& a, const auto& b){ return a.second < b.second; })->first;
}

// Recursive ID3 builder
Node* Tree::build(const std::vector<std::vector<int>>& rows,
                  const std::vector<int>&              labels,
                  std::vector<int>                     features_left) {
  Node* node = new Node();

  // Base case 1: all labels are the same -> pure leaf.
  if (std::all_of(labels.begin(), labels.end(),
                  [&](int l) { return l == labels[0]; })) {
      node->is_leaf = true;
      node->label   = labels[0];
      return node;
  }

  // Base case 2: no features left -> majority-vote leaf.
  if (features_left.empty()) {
      node->is_leaf = true;
      node->label   = majority(labels);
      return node;
  }

  // Pick the feature with the highest information gain
  int best_feat = features_left[0];
  double best_gain = -1.0;
  for (int f : features_left) {
      double g = information_gain(rows, labels, f);
      if (g > best_gain) { best_gain = g; best_feat = f; }
  }
  node->feature_idx = best_feat;

  // Remove best_feat from the remaining feature list
  std::vector<int> remaining;
  remaining.reserve(features_left.size() - 1);
  for (int f : features_left)
      if (f != best_feat) remaining.push_back(f);

  // Partition rows by their value for best_feat
  std::unordered_map<int, std::vector<int>> idx_by_val;
  for (int i = 0; i < static_cast<int>(rows.size()); ++i)
      idx_by_val[rows[i][best_feat]].push_back(i);

  // Recurse on each partition.
  for (const auto& [val, indices] : idx_by_val) {
      std::vector<std::vector<int>> sub_rows;
      std::vector<int>              sub_labels;
      sub_rows.reserve(indices.size());
      sub_labels.reserve(indices.size());
      for (int i : indices) {
          sub_rows.push_back(rows[i]);
          sub_labels.push_back(labels[i]);
      }
      node->children[val] = build(sub_rows, sub_labels, remaining);
  }

  return node;
}

void Tree::fit(const std::vector<std::vector<int>>& rows,
               const std::vector<int>& labels) {
  // TODO: implement ID3
}

int Tree::predict(const std::vector<int>& row) const {
  // TODO
  return 0;
}