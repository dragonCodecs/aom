/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AOM_DSP_NOISE_MODEL_H_
#define AOM_AOM_DSP_NOISE_MODEL_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stdint.h>
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/grain_params.h"
#include "aom_ports/mem.h"
#include "aom_scale/yv12config.h"
#include "aom_mem/aom_mem.h"

/*!\brief Wrapper of data required to represent linear system of eqns and soln.
 */
typedef struct {
  double *A;
  double *b;
  double *x;
  int n;
} aom_equation_system_t;

/*!\brief Representation of a piecewise linear curve
 *
 * Holds n points as (x, y) pairs, that store the curve.
 */
typedef struct {
  double (*points)[2];
  int num_points;
} aom_noise_strength_lut_t;

/*!\brief Init the noise strength lut with the given number of points*/
int aom_noise_strength_lut_init(aom_noise_strength_lut_t *lut, int num_points);

/*!\brief Frees the noise strength lut. */
void aom_noise_strength_lut_free(aom_noise_strength_lut_t *lut);

/*!\brief Evaluate the lut at the point x.
 *
 * \param[in] lut  The lut data.
 * \param[in] x    The coordinate to evaluate the lut.
 */
double aom_noise_strength_lut_eval(const aom_noise_strength_lut_t *lut,
                                   double x);

/*!\brief Helper struct to model noise strength as a function of intensity.
 *
 * Internally, this structure holds a representation of a linear system
 * of equations that models noise strength (standard deviation) as a
 * function of intensity. The mapping is initially stored using a
 * piecewise representation with evenly spaced bins that cover the entire
 * domain from [min_intensity, max_intensity]. Each observation (x,y) gives a
 * constraint of the form:
 *   y_{i} (1 - a) + y_{i+1} a = y
 * where y_{i} is the value of bin i and x_{i} <= x <= x_{i+1} and
 * a = x/(x_{i+1} - x{i}). The equation system holds the corresponding
 * normal equations.
 *
 * As there may be missing data, the solution is regularized to get a
 * complete set of values for the bins. A reduced representation after
 * solving can be obtained by getting the corresponding noise_strength_lut_t.
 */
typedef struct {
  aom_equation_system_t eqns;
  double min_intensity;
  double max_intensity;
  int num_bins;
  int num_equations;
  double total;
} aom_noise_strength_solver_t;

/*!\brief Initializes the noise solver with the given number of bins.
 *
 * Returns 0 if initialization fails.
 *
 * \param[in]  solver    The noise solver to be initialized.
 * \param[in]  num_bins  Number of bins to use in the internal representation.
 * \param[in]  bit_depth The bit depth used to derive {min,max}_intensity.
 */
int aom_noise_strength_solver_init(aom_noise_strength_solver_t *solver,
                                   int num_bins, int bit_depth);
void aom_noise_strength_solver_free(aom_noise_strength_solver_t *solver);

/*!\brief Gets the x coordinate of bin i.
 *
 * \param[in]  i  The bin whose coordinate to query.
 */
double aom_noise_strength_solver_get_center(
    const aom_noise_strength_solver_t *solver, int i);

/*!\brief Add an observation of the block mean intensity to its noise strength.
 *
 * \param[in]  block_mean  The average block intensity,
 * \param[in]  noise_std   The observed noise strength.
 */
void aom_noise_strength_solver_add_measurement(
    aom_noise_strength_solver_t *solver, double block_mean, double noise_std);

/*!\brief Solves the current set of equations for the noise strength. */
int aom_noise_strength_solver_solve(aom_noise_strength_solver_t *solver);

/*!\brief Fits a reduced piecewise linear lut to the internal solution
 *
 * \param[in] max_num_points  The maximum number of output points
 * \param[out] lut  The output piecewise linear lut.
 */
int aom_noise_strength_solver_fit_piecewise(
    const aom_noise_strength_solver_t *solver, int max_num_points,
    aom_noise_strength_lut_t *lut);

/*!\brief Helper for holding precomputed data for finding flat blocks.
 *
 * Internally a block is modeled with a low-order polynomial model. A
 * planar model would be a bunch of equations like:
 * <[y_i x_i 1], [a_1, a_2, a_3]>  = b_i
 * for each point in the block. The system matrix A with row i as [y_i x_i 1]
 * is maintained as is the inverse, inv(A'*A), so that the plane parameters
 * can be fit for each block.
 */
typedef struct {
  double *AtA_inv;
  double *A;
  int num_params;  // The number of parameters used for internal low-order model
  int block_size;  // The block size the finder was initialized with
  double normalization;  // Normalization factor (1 / (2^(bit_depth) - 1))
  int use_highbd;        // Whether input data should be interpreted as uint16
} aom_flat_block_finder_t;

/*!\brief Init the block_finder with the given block size, bit_depth */
int aom_flat_block_finder_init(aom_flat_block_finder_t *block_finder,
                               int block_size, int bit_depth, int use_highbd);
void aom_flat_block_finder_free(aom_flat_block_finder_t *block_finder);

/*!\brief Helper to extract a block and low order "planar" model. */
void aom_flat_block_finder_extract_block(
    const aom_flat_block_finder_t *block_finder, const uint8_t *const data,
    int w, int h, int stride, int offsx, int offsy, double *plane,
    double *block);

static AOM_FORCE_INLINE void aom_update_mean_var(const int block_size, double block[], double *Gxx, double *Gxy, double *Gyy, double *mean, double *var) {
  for (int yi = 1; yi < block_size - 1; ++yi) {
    for (int xi = 1; xi < block_size - 1; ++xi) {
      const double gx = (block[yi * block_size + xi + 1] -
                        block[yi * block_size + xi - 1]) /
                        2;
      const double gy = (block[yi * block_size + xi + block_size] -
                        block[yi * block_size + xi - block_size]) /
                        2;
      *Gxx += gx * gx;
      *Gxy += gx * gy;
      *Gyy += gy * gy;

      // MSFT: uncomment the following to "fix" the issue
      // const double value = block[yi * block_size + xi];
      // *mean += value;
      // *var += value * value;
      *mean += block[yi * block_size + xi];
      *var += block[yi * block_size + xi] * block[yi * block_size + xi];
    }
  }
}

typedef struct {
  int index;
  float score;
} index_and_score_t;

static AOM_INLINE int compare_scores(const void *a, const void *b) {
  const float diff =
      ((index_and_score_t *)a)->score - ((index_and_score_t *)b)->score;
  if (diff < 0)
    return -1;
  else if (diff > 0)
    return 1;
  return 0;
}

/*!\brief Runs the flat block finder on the input data.
 *
 * Find flat blocks in the input image data. Returns a map of
 * flat_blocks, where the value of flat_blocks map will be non-zero
 * when a block is determined to be flat. A higher value indicates a bigger
 * confidence in the decision.
 */
static int aom_flat_block_finder_run(const aom_flat_block_finder_t *block_finder,
                              const uint8_t *const data, int w, int h,
                              int stride, uint8_t *flat_blocks) {
  // The gradient-based features used in this code are based on:
  //  A. Kokaram, D. Kelly, H. Denman and A. Crawford, "Measuring noise
  //  correlation for improved video denoising," 2012 19th, ICIP.
  // The thresholds are more lenient to allow for correct grain modeling
  // if extreme cases.
  const int block_size = block_finder->block_size;
  const int n = block_size * block_size;
  const double kTraceThreshold = 0.15 / (32 * 32);
  const double kRatioThreshold = 1.25;
  const double kNormThreshold = 0.08 / (32 * 32);
  const double kVarThreshold = 0.005 / (double)n;
  const int num_blocks_w = (w + block_size - 1) / block_size;
  const int num_blocks_h = (h + block_size - 1) / block_size;
  int num_flat = 0;
  int bx = 0, by = 0;
  double *plane = (double *)aom_malloc(n * sizeof(*plane));
  double *block = (double *)aom_malloc(n * sizeof(*block));
  index_and_score_t *scores = (index_and_score_t *)aom_malloc(
      num_blocks_w * num_blocks_h * sizeof(*scores));
  if (plane == NULL || block == NULL || scores == NULL) {
    fprintf(stderr, "Failed to allocate memory for block of size %d\n", n);
    aom_free(plane);
    aom_free(block);
    aom_free(scores);
    return -1;
  }

#ifdef NOISE_MODEL_LOG_SCORE
  fprintf(stderr, "score = [");
#endif
  for (by = 0; by < num_blocks_h; ++by) {
    for (bx = 0; bx < num_blocks_w; ++bx) {
      // Compute gradient covariance matrix.
      double Gxx = 0, Gxy = 0, Gyy = 0;
      double var = 0;
      double mean = 0;
      aom_flat_block_finder_extract_block(block_finder, data, w, h, stride,
                                          bx * block_size, by * block_size,
                                          plane, block);

      aom_update_mean_var(block_size, block, &Gxx, &Gxy, &Gyy, &mean, &var);

      mean /= (block_size - 2) * (block_size - 2);

      // Normalize gradients by block_size.
      Gxx /= ((block_size - 2) * (block_size - 2));
      Gxy /= ((block_size - 2) * (block_size - 2));
      Gyy /= ((block_size - 2) * (block_size - 2));
      var = var / ((block_size - 2) * (block_size - 2)) - mean * mean;

      {
        const double trace = Gxx + Gyy;
        const double det = Gxx * Gyy - Gxy * Gxy;
        const double e1 = (trace + sqrt(trace * trace - 4 * det)) / 2.;
        const double e2 = (trace - sqrt(trace * trace - 4 * det)) / 2.;
        const double norm = e1;  // Spectral norm
        const double ratio = (e1 / AOMMAX(e2, 1e-6));
        const int is_flat = (trace < kTraceThreshold) &&
                            (ratio < kRatioThreshold) &&
                            (norm < kNormThreshold) && (var > kVarThreshold);
        // The following weights are used to combine the above features to give
        // a sigmoid score for flatness. If the input was normalized to [0,100]
        // the magnitude of these values would be close to 1 (e.g., weights
        // corresponding to variance would be a factor of 10000x smaller).
        // The weights are given in the following order:
        //    [{var}, {ratio}, {trace}, {norm}, offset]
        // with one of the most discriminative being simply the variance.
        const double weights[5] = { -6682, -0.2056, 13087, -12434, 2.5694 };
        double sum_weights = weights[0] * var + weights[1] * ratio +
                             weights[2] * trace + weights[3] * norm +
                             weights[4];
        // clamp the value to [-25.0, 100.0] to prevent overflow
        sum_weights = fclamp(sum_weights, -25.0, 100.0);
        const float score = (float)(1.0 / (1 + exp(-sum_weights)));
        flat_blocks[by * num_blocks_w + bx] = is_flat ? 255 : 0;
        scores[by * num_blocks_w + bx].score = var > kVarThreshold ? score : 0;
        scores[by * num_blocks_w + bx].index = by * num_blocks_w + bx;
#ifdef NOISE_MODEL_LOG_SCORE
        fprintf(stderr, "%g %g %g %g %g %d ", score, var, ratio, trace, norm,
                is_flat);
#endif
        num_flat += is_flat;
      }
    }
#ifdef NOISE_MODEL_LOG_SCORE
    fprintf(stderr, "\n");
#endif
  }
#ifdef NOISE_MODEL_LOG_SCORE
  fprintf(stderr, "];\n");
#endif
  // Find the top-scored blocks (most likely to be flat) and set the flat blocks
  // be the union of the thresholded results and the top 10th percentile of the
  // scored results.
  qsort(scores, num_blocks_w * num_blocks_h, sizeof(*scores), &compare_scores);
  const int top_nth_percentile = num_blocks_w * num_blocks_h * 90 / 100;
  const float score_threshold = scores[top_nth_percentile].score;
  for (int i = 0; i < num_blocks_w * num_blocks_h; ++i) {
    if (scores[i].score >= score_threshold) {
      num_flat += flat_blocks[scores[i].index] == 0;
      flat_blocks[scores[i].index] |= 1;
    }
  }
  aom_free(block);
  aom_free(plane);
  aom_free(scores);
  return num_flat;
}

// The noise shape indicates the allowed coefficients in the AR model.
enum {
  AOM_NOISE_SHAPE_DIAMOND = 0,
  AOM_NOISE_SHAPE_SQUARE = 1
} UENUM1BYTE(aom_noise_shape);

// The parameters of the noise model include the shape type, lag, the
// bit depth of the input images provided, and whether the input images
// will be using uint16 (or uint8) representation.
typedef struct {
  aom_noise_shape shape;
  int lag;
  int bit_depth;
  int use_highbd;
} aom_noise_model_params_t;

/*!\brief State of a noise model estimate for a single channel.
 *
 * This contains a system of equations that can be used to solve
 * for the auto-regressive coefficients as well as a noise strength
 * solver that can be used to model noise strength as a function of
 * intensity.
 */
typedef struct {
  aom_equation_system_t eqns;
  aom_noise_strength_solver_t strength_solver;
  int num_observations;  // The number of observations in the eqn system
  double ar_gain;        // The gain of the current AR filter
} aom_noise_state_t;

/*!\brief Complete model of noise for a planar video
 *
 * This includes a noise model for the latest frame and an aggregated
 * estimate over all previous frames that had similar parameters.
 */
typedef struct {
  aom_noise_model_params_t params;
  aom_noise_state_t combined_state[3];  // Combined state per channel
  aom_noise_state_t latest_state[3];    // Latest state per channel
  int (*coords)[2];  // Offsets (x,y) of the coefficient samples
  int n;             // Number of parameters (size of coords)
  int bit_depth;
} aom_noise_model_t;

/*!\brief Result of a noise model update. */
enum {
  AOM_NOISE_STATUS_OK = 0,
  AOM_NOISE_STATUS_INVALID_ARGUMENT,
  AOM_NOISE_STATUS_INSUFFICIENT_FLAT_BLOCKS,
  AOM_NOISE_STATUS_DIFFERENT_NOISE_TYPE,
  AOM_NOISE_STATUS_INTERNAL_ERROR,
} UENUM1BYTE(aom_noise_status_t);

/*!\brief Initializes a noise model with the given parameters.
 *
 * Returns 0 on failure.
 */
int aom_noise_model_init(aom_noise_model_t *model,
                         const aom_noise_model_params_t params);
void aom_noise_model_free(aom_noise_model_t *model);

/*!\brief Updates the noise model with a new frame observation.
 *
 * Updates the noise model with measurements from the given input frame and a
 * denoised variant of it. Noise is sampled from flat blocks using the flat
 * block map.
 *
 * Returns a noise_status indicating if the update was successful. If the
 * Update was successful, the combined_state is updated with measurements from
 * the provided frame. If status is OK or DIFFERENT_NOISE_TYPE, the latest noise
 * state will be updated with measurements from the provided frame.
 *
 * \param[in,out] noise_model     The noise model to be updated
 * \param[in]     data            Raw frame data
 * \param[in]     denoised        Denoised frame data.
 * \param[in]     w               Frame width
 * \param[in]     h               Frame height
 * \param[in]     strides         Stride of the planes
 * \param[in]     chroma_sub_log2 Chroma subsampling for planes != 0.
 * \param[in]     flat_blocks     A map to blocks that have been determined flat
 * \param[in]     block_size      The size of blocks.
 */
aom_noise_status_t aom_noise_model_update(
    aom_noise_model_t *const noise_model, const uint8_t *const data[3],
    const uint8_t *const denoised[3], int w, int h, int strides[3],
    int chroma_sub_log2[2], const uint8_t *const flat_blocks, int block_size);

/*\brief Save the "latest" estimate into the "combined" estimate.
 *
 * This is meant to be called when the noise modeling detected a change
 * in parameters (or for example, if a user wanted to reset estimation at
 * a shot boundary).
 */
void aom_noise_model_save_latest(aom_noise_model_t *noise_model);

/*!\brief Converts the noise_model parameters to the corresponding
 *    grain_parameters.
 *
 * The noise structs in this file are suitable for estimation (e.g., using
 * floats), but the grain parameters in the bitstream are quantized. This
 * function does the conversion by selecting the correct quantization levels.
 */
int aom_noise_model_get_grain_parameters(aom_noise_model_t *const noise_model,
                                         aom_film_grain_t *film_grain);

/*!\brief Perform a Wiener filter denoising in 2D using the provided noise psd.
 *
 * \param[in]     data            Raw frame data
 * \param[out]    denoised        Denoised frame data
 * \param[in]     w               Frame width
 * \param[in]     h               Frame height
 * \param[in]     stride          Stride of the planes
 * \param[in]     chroma_sub_log2 Chroma subsampling for planes != 0.
 * \param[in]     noise_psd       The power spectral density of the noise
 * \param[in]     block_size      The size of blocks
 * \param[in]     bit_depth       Bit depth of the image
 * \param[in]     use_highbd      If true, uint8 pointers are interpreted as
 *                                uint16 and stride is measured in uint16.
 *                                This must be true when bit_depth >= 10.
 */
int aom_wiener_denoise_2d(const uint8_t *const data[3], uint8_t *denoised[3],
                          int w, int h, int stride[3], int chroma_sub_log2[2],
                          float *noise_psd[3], int block_size, int bit_depth,
                          int use_highbd);

struct aom_denoise_and_model_t;

/*!\brief Denoise the buffer and model the residual noise.
 *
 * This is meant to be called sequentially on input frames. The input buffer
 * is denoised and the residual noise is modelled. The current noise estimate
 * is populated in film_grain. Returns true on success. The grain.apply_grain
 * parameter will be true when the input buffer was successfully denoised and
 * grain was modelled. Returns false on error.
 *
 * \param[in]      ctx          Struct allocated with
 *                              aom_denoise_and_model_alloc that holds some
 *                              buffers for denoising and the current noise
 *                              estimate.
 * \param[in/out]   buf         The raw input buffer to be denoised.
 * \param[out]    grain         Output film grain parameters
 * \param[out]    apply_denoise Whether or not to apply the denoising to the
 *                              frame that will be encoded
 */
int aom_denoise_and_model_run(struct aom_denoise_and_model_t *ctx,
                              YV12_BUFFER_CONFIG *buf, aom_film_grain_t *grain,
                              int apply_denoise);

/*!\brief Allocates a context that can be used for denoising and noise modeling.
 *
 * \param[in]  bit_depth   Bit depth of buffers this will be run on.
 * \param[in]  block_size  Block size for noise modeling and flat block
 *                         estimation
 * \param[in]  noise_level The noise_level (2.5 for moderate noise, and 5 for
 *                         higher levels of noise)
 */
struct aom_denoise_and_model_t *aom_denoise_and_model_alloc(int bit_depth,
                                                            int block_size,
                                                            float noise_level);

/*!\brief Frees the denoise context allocated with aom_denoise_and_model_alloc
 */
void aom_denoise_and_model_free(struct aom_denoise_and_model_t *denoise_model);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // AOM_AOM_DSP_NOISE_MODEL_H_
