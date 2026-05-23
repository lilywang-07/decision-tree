#pragma once
#include <vector>
#include <memory>
#include <unordered_map>

// Shannon entropy (in bits) of a label vector.
// Returns 0 for empty or pure (single-class) input.
double entropy(const std::vector<int>& labels);

// Information gain of splitting on a feature, given the rows, labels, and
// feature index to split on.
double information_gain(const std::vector<std::vector<int>>& rows,
                        const std::vector<int>&              labels,
                        int                                  feature_index);

struct Node {
    bool is_leaf    = false;
    int  label      = -1;  // meaningful only when is_leaf == true
    int  feature_idx = -1;  // meaningful only when is_leaf == false

    // For categorical splits: children[feature_value] → child node
    std::unordered_map<int, Node*> children;

    ~Node() {
        for (auto& [val, child] : children) delete child;
    }
};

// Tree ------------------------------------------------------------------------

class Tree {
  public:
    Tree()  = default;
    ~Tree() { delete root_; }

    // Disable copy (owns raw pointers)
    Tree(const Tree&)            = delete;
    Tree& operator=(const Tree&) = delete;

    // Train the tree on rows + labels.
    void fit(const std::vector<std::vector<int>>& rows,
            const std::vector<int>& labels);

    // Predict the label for a single row.
    int predict(const std::vector<int>& row) const;

    // Predict labels for many rows.
    std::vector<int> predict(const std::vector<std::vector<int>>& rows) const;
    
    void prune(const std::vector<std::vector<int>>& val_rows,
           const std::vector<int>& val_labels);

    double accuracy(const std::vector<std::vector<int>>& val_rows,
                    const std::vector<int>&              val_labels);

  private:
    Node* root_ = nullptr;
    std::vector<int> train_labels_; // stored for pruning

    // Recursive ID3 builder
    Node* build(const std::vector<std::vector<int>>& rows,
                const std::vector<int>&              labels,
                std::vector<int>                     features_left);
    
    void prune(Node* node,
               const std::vector<std::vector<int>>& val_rows,
               const std::vector<int>& val_labels);

    // Majority-vote among labels (tie-break: smaller class wins).
    static int majority(const std::vector<int>& labels);
  
};