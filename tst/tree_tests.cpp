#include "tree.h"
#include <gtest/gtest.h>

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
  // 9 "play" (1) + 5 "don't play" (0) out of 14 — the labels from tennis.csv.
  const std::vector<int> tennis_labels = {
      0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0
  };
  EXPECT_NEAR(entropy(tennis_labels), 0.940, 0.001);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}