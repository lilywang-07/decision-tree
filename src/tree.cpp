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

void Tree::fit(const std::vector<std::vector<int>>& rows,
               const std::vector<int>& labels) {
  // TODO: implement ID3
}

int Tree::predict(const std::vector<int>& row) const {
  // TODO
  return 0;
}