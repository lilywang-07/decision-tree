#include "csv.h"
#include "tree.h"

#include <filesystem>
#include <iostream>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>

using Data = std::pair<std::vector<std::vector<int>>, std::vector<int>>;

// loads a given csv file and returns a pair of (rows, targets)
Data load_csv_data(const std::string& filename) {
  const auto path = std::filesystem::path{"datasets"} / filename;
  
  std::ifstream file(path.string());
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path.string());
  }

  // Read header row to determine column count
  std::string header_line;
  std::getline(file, header_line);
  
  std::stringstream header_ss(header_line);
  std::string col;
  int num_cols = 0;
  while (std::getline(header_ss, col, ',')) {
    num_cols++;
  }

  // Last column is the target, rest are features
  int num_features = num_cols - 1;

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

    if ((int)row.size() != num_cols) {
      throw std::runtime_error("Row has unexpected number of columns");
    }

    row_data.push_back(std::vector<int>(row.begin(), row.begin() + num_features));
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

// Binary confusion matrix (classes 0 and 1).
//          Predicted 0   Predicted 1
// Actual 0    TN            FP
// Actual 1    FN            TP
void print_confusion_matrix(const std::vector<int>& truth,
                            const std::vector<int>& pred) {
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < static_cast<int>(truth.size()); ++i) {
        if      (truth[i] == 1 && pred[i] == 1) ++tp;
        else if (truth[i] == 0 && pred[i] == 0) ++tn;
        else if (truth[i] == 0 && pred[i] == 1) ++fp;
        else                                     ++fn;
    }

    std::cout << "\nConfusion Matrix (actual rows × predicted cols):\n";
    std::cout << "              Pred 0   Pred 1\n";
    std::cout << "  Actual 0:     " << std::setw(3) << tn
              << "      " << std::setw(3) << fp << "\n";
    std::cout << "  Actual 1:     " << std::setw(3) << fn
              << "      " << std::setw(3) << tp << "\n";
    std::cout << "\n  TP=" << tp << "  TN=" << tn
              << "  FP=" << fp << "  FN=" << fn << "\n";
}

// Main ------------------------------------------------------------------

int main() {
  // loads data
  const auto [rows, targets] = load_csv_data("zoo.csv");
  const int n_samples  = static_cast<int>(rows.size());
  std::cout << "\nLoaded rows/samples: " << n_samples
            << ", labels: " << targets.size() << '\n';

  // train / test split — 70% train, 30% test
  const int split = static_cast<int>(n_samples * 0.70 + 0.5);  // round

  std::vector<std::vector<int>> train_rows (rows.begin(), rows.begin() + split);
  std::vector<int> train_labels(targets.begin(), targets.begin() + split);
  std::vector<std::vector<int>> test_rows (rows.begin() + split, rows.end());
  std::vector<int> test_labels(targets.begin() + split, targets.end());

  std::cout << "Split: " << train_rows.size() << " train, "
            << test_rows.size()  << " test.\n";

  // fit, times how long it takes to build the tree
  Tree tree;
  auto t0 = std::chrono::high_resolution_clock::now();
  tree.fit(train_rows, train_labels);
  auto t1 = std::chrono::high_resolution_clock::now();
  double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // evaluate on training set
  std::vector<int> train_preds;
  for (const auto& row : train_rows)
      train_preds.push_back(tree.predict(row));
  double train_acc = accuracy(train_labels, train_preds);

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

  return 0;
}
