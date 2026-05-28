#include "tree.h"
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <numeric>

static double split_time = 0.0;

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

void Tree::prune(const std::vector<std::vector<int>>& val_rows,
                 const std::vector<int>& val_labels) {
    prune(root_, val_rows, val_labels);
}

double split_info(const std::vector<std::vector<int>>& rows, int feature_index) {
  std::unordered_map<int, int> counts;
  for (const auto& row : rows) counts[row[feature_index]]++;

  double si = 0.0;
  const double n = static_cast<double>(rows.size());
  for (const auto& [val, count] : counts) {
      double p = count / n;
      if (p > 0) si -= p * std::log2(p);
  }
  return si;
}

double gain_ratio(const std::vector<std::vector<int>>& rows,
                  const std::vector<int>& labels, int feature_index) {
  double si = split_info(rows, feature_index);
  if (si == 0.0) return 0.0;  // avoid divide by zero
  return information_gain(rows, labels, feature_index) / si;
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
                  const std::vector<int>& labels,
                  const std::vector<int>& indices,
                  std::vector<int> features_left, int depth,int max_depth, 
                  int min_samples_split, int min_samples_leaf, int split_criteria) {
  Node* node = new Node();
  node->label = majority(labels); // default label
  
  // work on indices instead of copying rows
  std::vector<int> sub_labels;
  sub_labels.reserve(indices.size());
  for (int i : indices) sub_labels.push_back(labels[i]);

  // Base case 1: all labels are the same -> pure leaf.
  if (std::all_of(sub_labels.begin(), sub_labels.end(),
                  [&](int l) { return l == sub_labels[0]; })) {
      node->is_leaf = true;
      node->label   = sub_labels[0];
      return node;
  }

  // Base case 2: no features left -> majority-vote leaf.
  if (features_left.empty()) {
      node->is_leaf = true;
      node->label   = majority(sub_labels);
      return node;
  }

  // Base case 3: max depth reached
  if (depth >= max_depth) {
      node->is_leaf = true;
      return node;
  }

  // Base case 4: not enough samples to split
  if ((int)sub_labels.size() < min_samples_split) {
      node->is_leaf = true;
      return node;
  }

  // timing the split decision
  auto split_t0 = std::chrono::high_resolution_clock::now();

  // Pick the feature with the highest information gain
  int best_feat = features_left[0];
  double best_gain = -1.0;
  for (int f : features_left) {
    // split criteria: 0 for information gain, 1 for gain ratio
    double g = (split_criteria == 0)
      ? information_gain(rows, sub_labels, f)
      : gain_ratio(rows, sub_labels, f);
    if (g > best_gain) { best_gain = g; best_feat = f; }
  }
  node->feature_idx = best_feat;

  auto split_t1 = std::chrono::high_resolution_clock::now();
  split_time += std::chrono::duration<double, std::milli>(split_t1 - split_t0).count();

  // Remove best_feat from the remaining feature list
  std::vector<int> remaining;
  remaining.reserve(features_left.size() - 1);
  for (int f : features_left)
      if (f != best_feat) remaining.push_back(f);

  // Partition rows by their value for best_feat
  std::unordered_map<int, std::vector<int>> idx_by_val;
  for (int i : indices)
    idx_by_val[rows[i][best_feat]].push_back(i);

  // before recursing, check partitions aren't too small
  bool any_too_small = false;
  for (const auto& [val, indices] : idx_by_val)
      if ((int)indices.size() < min_samples_leaf) { any_too_small = true; break; }
  if (any_too_small) {
      node->is_leaf = true;
      return node;
  }

  // Recurse on each partition.
  for (const auto& [val, child_indices] : idx_by_val) {
    node->children[val] = build(rows, labels, child_indices,
                                remaining, depth + 1,
                                max_depth, min_samples_split, min_samples_leaf, split_criteria);
  }

  return node;
}

// split criteria: 0 for information gain, 1 for gain ratio
void Tree::fit(const std::vector<std::vector<int>>& rows,
               const std::vector<int>& labels, int split_criteria) {
  delete root_;
  root_ = nullptr;
  if (rows.empty()) return;

  // All feature indices: 0 … num_features-1
  std::vector<int> all_indices(rows.size());
  std::iota(all_indices.begin(), all_indices.end(), 0);

  // initialize the list of features left to split on
  std::vector<int> all_features(rows[0].size());
  for (int i = 0; i < (int)all_features.size(); ++i)
    all_features[i] = i;

  root_ = build(rows, labels, all_indices, all_features, 0, 10, 5, 2, 1);

  std::cout << "Split-finding time: " << split_time << " ms\n";
}

int Tree::predict(const std::vector<int>& row) const {
  if (!root_) throw std::runtime_error("Tree::predict called before fit()");

  const Node* cur = root_;
  while (!cur->is_leaf) {
      int val = row[cur->feature_idx];
      auto it = cur->children.find(val);
      if (it == cur->children.end())
          // Unseen value at inference time — fall back to majority among known children
          return cur->label;
      cur = it->second;
  }
  return cur->label;
}

double Tree::accuracy(const std::vector<std::vector<int>>& val_rows,
                const std::vector<int>&              val_labels) {
  int correct = 0;
  for (int i = 0; i < static_cast<int>(val_rows.size()); ++i)
    correct += (predict(val_rows[i]) == val_labels[i]);
  return static_cast<double>(correct) / val_labels.size();
}

// walks the tree bottom-up, tries replacing each internal node with a leaf, and 
// keeps the cut if validation accuracy doesn't drop.
void Tree::prune(Node* node, const std::vector<std::vector<int>>& val_rows,
           const std::vector<int>& val_labels){
  if (node->is_leaf) return;

  // recurse into children first
  for (auto& [val, child] : node->children)
    prune(child, val_rows, val_labels);

  // current accuracy before pruning
  double old_acc = accuracy(val_rows, val_labels);
  
  // try replacing this node with a leaf
  int old_label = node->label;
  bool old_is_leaf = node->is_leaf;
  std::unordered_map<int, Node*> old_children = std::move(node->children);
  node->is_leaf = true;

  // if accuracy dropped, revert the change
  double new_acc = accuracy(val_rows, val_labels);
  if (new_acc < old_acc) {
    node->is_leaf = old_is_leaf;
    node->label = old_label;
    node->children = std::move(old_children);
  }
}