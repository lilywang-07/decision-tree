#include "tree.h"
#include <gtest/gtest.h>

const std::vector<int> tennis_labels = {
      0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0
  };

const std::vector<std::vector<int>> tennis_rows = {
    {0,0,0,0}, {0,0,0,1}, {1,0,0,0}, {2,1,0,0},
    {2,2,1,0}, {2,2,1,1}, {1,2,1,1}, {0,1,0,0},
    {0,2,1,0}, {2,1,1,0}, {0,1,1,1}, {1,1,0,1},
    {1,0,1,0}, {2,1,0,1}
};

TEST(Dataset, LoadsTennis) { EXPECT_EQ(2 + 2, 4); }

TEST(Entropy, EmptyIsZero) { // handles empty input without crashing, and returns 0 entropy.
  EXPECT_DOUBLE_EQ(entropy({}), 0.0);
}

// handles pure (single-class) input without crashing, and returns 0 entropy
TEST(Entropy, PureSetIsZero) {
  EXPECT_DOUBLE_EQ(entropy({1, 1, 1, 1}), 0.0);
  EXPECT_DOUBLE_EQ(entropy({0, 0, 0}), 0.0);
}

// handles a fair binary distribution, and returns 1 bit of entropy.
TEST(Entropy, FairCoinIsOneBit) {
  EXPECT_DOUBLE_EQ(entropy({0, 1}), 1.0);
  EXPECT_DOUBLE_EQ(entropy({0, 0, 1, 1}), 1.0);
}

// handles the tennis dataset labels, and returns ~0.940 bits of entropy.
TEST(Entropy, TennisFullDataset) {
  // 9 "play" (1) + 5 "don't play" (0) out of 14
  EXPECT_NEAR(entropy(tennis_labels), 0.940, 0.001);
}

 TEST(InformationGain, OutlookGain) {
    // Textbook value for outlook (feature 0) on the full tennis set ≈ 0.247
    EXPECT_NEAR(information_gain(tennis_rows, tennis_labels, 0), 0.247, 0.001);
}

TEST(InformationGain, ZeroGain) {
    // If every row has the same feature value, the split changes nothing so gain = 0.
    const std::vector<std::vector<int>> rows   = {{7},{7},{7},{7}};
    const std::vector<int>              labels = {  0,  0,  1,  1};
    EXPECT_NEAR(information_gain(rows, labels, 0), 0.0, 1e-9);
}

// Tree ------------------------------------------------------------------------

TEST(Tree, Builds) {
    Tree tree;
    EXPECT_TRUE(true);
}

TEST(Tree, FitAndPredict) {
    Tree tree;
    tree.fit(tennis_rows, tennis_labels, 0);
    int result = tree.predict(tennis_rows[0]);
    EXPECT_TRUE(result == 0 || result == 1);
}

TEST(Tree, PerfectTrainingAccuracy) {
    // ID3 with no pruning must memorize every training row.
    Tree tree;
    tree.fit(tennis_rows, tennis_labels, 0);

    int correct = 0;
    for (int i = 0; i < (int)tennis_rows.size(); ++i)
        correct += (tree.predict(tennis_rows[i]) == tennis_labels[i]);

    EXPECT_EQ(correct, (int)tennis_rows.size());
}

TEST(Tree, PredictKnownRows) {
    Tree tree;
    tree.fit(tennis_rows, tennis_labels, 0);

    // Row 0: outlook=0,temp=0,humidity=0,wind=0 → label 0 (don't play)
    EXPECT_EQ(tree.predict({0, 0, 0, 0}), 0);
    // Row 2: outlook=1,temp=0,humidity=0,wind=0 → label 1 (play)
    EXPECT_EQ(tree.predict({1, 0, 0, 0}), 1);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}