/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <limits.h>
#include <math.h>
#include <algorithm>
#include <vector>

#include "aom_dsp/noise_model.h"
#include "aom_dsp/noise_util.h"
#include "config/aom_dsp_rtcd.h"
#include "test/acm_random.h"
#include "third_party/googletest/src/googletest/include/gtest/gtest.h"

namespace {

// Return normally distrbuted values with standard deviation of sigma.
double randn(libaom_test::ACMRandom *random, double sigma) {
  while (1) {
    const double u = 2.0 * ((double)random->Rand31() /
                            testing::internal::Random::kMaxRange) -
                     1.0;
    const double v = 2.0 * ((double)random->Rand31() /
                            testing::internal::Random::kMaxRange) -
                     1.0;
    const double s = u * u + v * v;
    if (s > 0 && s < 1) {
      return sigma * (u * sqrt(-2.0 * log(s) / s));
    }
  }
  return 0;
}

// Synthesizes noise using the auto-regressive filter of the given lag,
// with the provided n coefficients sampled at the given coords.
void noise_synth(libaom_test::ACMRandom *random, int lag, int n,
                 const int (*coords)[2], const double *coeffs, double *data,
                 int w, int h) {
  const int pad_size = 3 * lag;
  const int padded_w = w + pad_size;
  const int padded_h = h + pad_size;
  int x = 0, y = 0;
  std::vector<double> padded(padded_w * padded_h);

  for (y = 0; y < padded_h; ++y) {
    for (x = 0; x < padded_w; ++x) {
      padded[y * padded_w + x] = randn(random, 1.0);
    }
  }
  for (y = lag; y < padded_h; ++y) {
    for (x = lag; x < padded_w; ++x) {
      double sum = 0;
      int i = 0;
      for (i = 0; i < n; ++i) {
        const int dx = coords[i][0];
        const int dy = coords[i][1];
        sum += padded[(y + dy) * padded_w + (x + dx)] * coeffs[i];
      }
      padded[y * padded_w + x] += sum;
    }
  }
  // Copy over the padded rows to the output
  for (y = 0; y < h; ++y) {
    memcpy(data + y * w, &padded[0] + y * padded_w, sizeof(*data) * w);
  }
}

std::vector<float> get_noise_psd(double *noise, int width, int height,
                                 int block_size) {
  float *block =
      (float *)aom_memalign(32, block_size * block_size * sizeof(block));
  std::vector<float> psd(block_size * block_size);
  if (block == nullptr) {
    EXPECT_NE(block, nullptr);
    return psd;
  }
  int num_blocks = 0;
  struct aom_noise_tx_t *tx = aom_noise_tx_malloc(block_size);
  if (tx == nullptr) {
    EXPECT_NE(tx, nullptr);
    return psd;
  }
  for (int y = 0; y <= height - block_size; y += block_size / 2) {
    for (int x = 0; x <= width - block_size; x += block_size / 2) {
      for (int yy = 0; yy < block_size; ++yy) {
        for (int xx = 0; xx < block_size; ++xx) {
          block[yy * block_size + xx] = (float)noise[(y + yy) * width + x + xx];
        }
      }
      aom_noise_tx_forward(tx, &block[0]);
      aom_noise_tx_add_energy(tx, &psd[0]);
      num_blocks++;
    }
  }
  for (int yy = 0; yy < block_size; ++yy) {
    for (int xx = 0; xx <= block_size / 2; ++xx) {
      psd[yy * block_size + xx] /= num_blocks;
    }
  }
  // Fill in the data that is missing due to symmetries
  for (int xx = 1; xx < block_size / 2; ++xx) {
    psd[(block_size - xx)] = psd[xx];
  }
  for (int yy = 1; yy < block_size; ++yy) {
    for (int xx = 1; xx < block_size / 2; ++xx) {
      psd[(block_size - yy) * block_size + (block_size - xx)] =
          psd[yy * block_size + xx];
    }
  }
  aom_noise_tx_free(tx);
  aom_free(block);
  return psd;
}

}  // namespace


// A container template class to hold a data type and extra arguments.
// All of these args are bundled into one struct so that we can use
// parameterized tests on combinations of supported data types
// (uint8_t and uint16_t) and bit depths (8, 10, 12).
template <typename T, int bit_depth, bool use_highbd>
struct BitDepthParams {
  typedef T data_type_t;
  static const int kBitDepth = bit_depth;
  static const bool kUseHighBD = use_highbd;
};

template <typename T>
class FlatBlockEstimatorTest : public ::testing::Test, public T {
 public:
  virtual void SetUp() { random_.Reset(171); }
  typedef std::vector<typename T::data_type_t> VecType;
  VecType data_;
  libaom_test::ACMRandom random_;
};

TYPED_TEST_SUITE_P(FlatBlockEstimatorTest);

TYPED_TEST_P(FlatBlockEstimatorTest, FindFlatBlocks) {
  const int kBlockSize = 32;
  aom_flat_block_finder_t flat_block_finder;
  ASSERT_EQ(1, aom_flat_block_finder_init(&flat_block_finder, kBlockSize,
                                          this->kBitDepth, this->kUseHighBD));

  const int num_blocks_w = 8;
  const int h = kBlockSize;
  const int w = kBlockSize * num_blocks_w;
  const int stride = w;
  this->data_.resize(h * stride, 128);
  std::vector<uint8_t> flat_blocks(num_blocks_w, 0);

  const int shift = this->kBitDepth - 8;
  for (int y = 0; y < kBlockSize; ++y) {
    for (int x = 0; x < kBlockSize; ++x) {
      // Block 0 (not flat): constant doesn't have enough variance to qualify
      this->data_[y * stride + x + 0 * kBlockSize] = 128 << shift;

      // Block 1 (not flat): too high of variance is hard to validate as flat
      this->data_[y * stride + x + 1 * kBlockSize] =
          ((uint8_t)(128 + randn(&this->random_, 5))) << shift;

      // Block 2 (flat): slight checkerboard added to constant
      const int check = (x % 2 + y % 2) % 2 ? -2 : 2;
      this->data_[y * stride + x + 2 * kBlockSize] = (128 + check) << shift;

      // Block 3 (flat): planar block with checkerboard pattern is also flat
      this->data_[y * stride + x + 3 * kBlockSize] =
          (y * 2 - x / 2 + 128 + check) << shift;

      // Block 4 (flat): gaussian random with standard deviation 1.
      this->data_[y * stride + x + 4 * kBlockSize] =
          ((uint8_t)(randn(&this->random_, 1) + x + 128.0)) << shift;

      // Block 5 (flat): gaussian random with standard deviation 2.
      this->data_[y * stride + x + 5 * kBlockSize] =
          ((uint8_t)(randn(&this->random_, 2) + y + 128.0)) << shift;

      // Block 6 (not flat): too high of directional gradient.
      const int strong_edge = x > kBlockSize / 2 ? 64 : 0;
      this->data_[y * stride + x + 6 * kBlockSize] =
          ((uint8_t)(randn(&this->random_, 1) + strong_edge + 128.0)) << shift;

      // Block 7 (not flat): too high gradient.
      const int big_check = ((x >> 2) % 2 + (y >> 2) % 2) % 2 ? -16 : 16;
      this->data_[y * stride + x + 7 * kBlockSize] =
          ((uint8_t)(randn(&this->random_, 1) + big_check + 128.0)) << shift;
    }
  }

  EXPECT_EQ(4, aom_flat_block_finder_run(&flat_block_finder,
                                         (uint8_t *)&this->data_[0], w, h,
                                         stride, &flat_blocks[0]));

  // First two blocks are not flat
  EXPECT_EQ(0, flat_blocks[0]);
  EXPECT_EQ(0, flat_blocks[1]);

  // Next 4 blocks are flat.
  EXPECT_EQ(255, flat_blocks[2]);
  EXPECT_EQ(255, flat_blocks[3]);
  EXPECT_EQ(255, flat_blocks[4]);
  EXPECT_EQ(255, flat_blocks[5]);

  // Last 2 are not flat by threshold
  EXPECT_EQ(0, flat_blocks[6]);
  EXPECT_EQ(0, flat_blocks[7]);

  // Add the noise from non-flat block 1 to every block.
  for (int y = 0; y < kBlockSize; ++y) {
    for (int x = 0; x < kBlockSize * num_blocks_w; ++x) {
      this->data_[y * stride + x] +=
          (this->data_[y * stride + x % kBlockSize + kBlockSize] -
           (128 << shift));
    }
  }
  // Now the scored selection will pick the one that is most likely flat (block
  // 0)
  EXPECT_EQ(1, aom_flat_block_finder_run(&flat_block_finder,
                                         (uint8_t *)&this->data_[0], w, h,
                                         stride, &flat_blocks[0]));
  EXPECT_EQ(1, flat_blocks[0]);
  EXPECT_EQ(0, flat_blocks[1]);
  EXPECT_EQ(0, flat_blocks[2]);
  EXPECT_EQ(0, flat_blocks[3]);
  EXPECT_EQ(0, flat_blocks[4]);
  EXPECT_EQ(0, flat_blocks[5]);
  EXPECT_EQ(0, flat_blocks[6]);
  EXPECT_EQ(0, flat_blocks[7]);

  aom_flat_block_finder_free(&flat_block_finder);
}

REGISTER_TYPED_TEST_SUITE_P(FlatBlockEstimatorTest, FindFlatBlocks);

typedef ::testing::Types<BitDepthParams<uint8_t, 8, false>,   // lowbd
                         BitDepthParams<uint16_t, 8, true>,   // lowbd in 16-bit
                         BitDepthParams<uint16_t, 10, true>,  // highbd data
                         BitDepthParams<uint16_t, 12, true> >
    AllBitDepthParams;
INSTANTIATE_TYPED_TEST_SUITE_P(FlatBlockInstatiation, FlatBlockEstimatorTest,
                               AllBitDepthParams);
