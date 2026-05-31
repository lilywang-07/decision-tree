#include "csv.h"
#include "tree.h"

#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <set>

using Data = std::pair<std::vector<std::vector<int>>, std::vector<int>>;

// loads a given csv file and returns a pair of (rows, targets)
Data load_csv_data(const std::string& filename) {
  const auto path = std::filesystem::path{filename};
  
  std::ifstream file(path.string());
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path.string());
  }

  std::vector<std::vector<int>> row_data;
  std::vector<int> target_data;

  std::string line;
  
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string val;
    std::vector<int> row;

    while (std::getline(ss, val, ',')) {
      row.push_back(std::stoi(val));
    }

    row_data.push_back(std::vector<int>(row.begin(), row.end() - 1));
    target_data.push_back(row.back());
  }

  return {std::move(row_data), std::move(target_data)};
}

// Metrics ----------------------------------------------------------------

double accuracy(const std::vector<int>& truth, const std::vector<int>& pred) {
    int correct = 0;
    for (int i = 0; i < static_cast<int>(truth.size()); ++i)
        correct += (truth[i] == pred[i]);
    return 100.0 * correct / static_cast<double>(truth.size());
}

void print_confusion_matrix(const std::vector<int>& truth,
                            const std::vector<int>& pred) {
    // Find all unique classes
    std::set<int> class_set(truth.begin(), truth.end());
    std::vector<int> classes(class_set.begin(), class_set.end());
    int n = classes.size();

    // Map class value to index
    std::map<int, int> idx;
    for (int i = 0; i < n; ++i) idx[classes[i]] = i;

    // Build matrix
    std::vector<std::vector<int>> mat(n, std::vector<int>(n, 0));
    for (int i = 0; i < (int)truth.size(); ++i)
        mat[idx[truth[i]]][idx[pred[i]]]++;

    // Print
    std::cout << "\nConfusion Matrix (actual rows x predicted cols):\n";
    std::cout << "       ";
    for (int c : classes) std::cout << std::setw(4) << c;
    std::cout << "\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "  [" << classes[i] << "]  ";
        for (int j = 0; j < n; ++j)
            std::cout << std::setw(4) << mat[i][j];
        std::cout << "\n";
    }
}

// Main ------------------------------------------------------------------

int main(int argc, char* argv[]) {
  // loads data
  const std::string csv_path = (argc > 1) ? argv[1] : "datasets/mushroom_fixed.csv";
  const auto [rows, targets] = load_csv_data(csv_path);
  const int n_samples  = static_cast<int>(rows.size());
  std::cout << "\nLoaded rows/samples: " << n_samples
            << ", labels: " << targets.size() << '\n';

  // train / test split — 60% train, 20% test, 20% validation
  const int train_end = static_cast<int>(n_samples * 0.60 + 0.5);  // round
  int val_end = static_cast<int>(n_samples * 0.80);

  std::vector<std::vector<int>> train_rows (rows.begin(), rows.begin() + train_end);
  std::vector<int> train_labels(targets.begin(), targets.begin() + train_end);
  std::vector<std::vector<int>> val_rows   (rows.begin() + train_end, rows.begin() + val_end);
  std::vector<int> val_labels (targets.begin() + train_end, targets.begin() + val_end);
  std::vector<std::vector<int>> test_rows (rows.begin() + val_end,  rows.end());
  std::vector<int> test_labels (targets.begin() + val_end, targets.end());

  std::cout << "Split: " << train_rows.size() << " train, "
            << val_rows.size()   << " validation, "
            << test_rows.size()  << " test.\n";

  // fit, times how long it takes to build the tree
  Tree tree;
  auto t0 = std::chrono::high_resolution_clock::now();

  // CHANGE THIS LINE to switch between information gain and gain ratio split criteria
  // split criteria: 0 for information gain, 1 for gain ratio
  tree.fit(train_rows, train_labels, 0);
  auto t1 = std::chrono::high_resolution_clock::now();
  double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // evaluate on training set
  std::vector<int> train_preds;
  for (const auto& row : train_rows)
      train_preds.push_back(tree.predict(row));
  double train_acc = accuracy(train_labels, train_preds);

  // prune the tree using the validation set
  tree.prune(val_rows, val_labels);

  // evaluate on test set
  std::vector<int> test_preds;
  for (const auto& row : test_rows)
      test_preds.push_back(tree.predict(row));
  double test_acc = accuracy(test_labels, test_preds);

  // report metrics
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "\n=== Results ===\n";
  std::cout << "Training accuracy : " << train_acc << "%\n";
  std::cout << "Test accuracy     : " << test_acc  << "%\n";
  std::cout << "Training time     : " << std::setprecision(4)
            << train_ms << " ms\n";

  std::cout << "\n--- Training set ---";
  print_confusion_matrix(train_labels, train_preds);

  std::cout << "\n--- Test set ---";
  print_confusion_matrix(test_labels, test_preds);
  int correct = 0;
  for (int i = 0; i < (int)test_labels.size(); ++i)
      correct += (test_labels[i] == test_preds[i]);

  std::cout << "\nCorrect:   " << correct << " / " << test_labels.size() << "\n";
  std::cout << "Incorrect: " << test_labels.size() - correct << " / " << test_labels.size() << "\n";
  std::cout << "\n";

  return 0;
}
