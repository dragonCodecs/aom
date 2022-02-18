/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <math.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"
#include "av1/common/av1_common_int.h"
#include "av1/common/av1_loopfilter.h"
#include "av1/common/reconinter.h"
#include "av1/common/seg_common.h"

static const SEG_LVL_FEATURES seg_lvl_lf_lut[MAX_MB_PLANE][2] = {
  { SEG_LVL_ALT_LF_Y_V, SEG_LVL_ALT_LF_Y_H },
  { SEG_LVL_ALT_LF_U, SEG_LVL_ALT_LF_U },
  { SEG_LVL_ALT_LF_V, SEG_LVL_ALT_LF_V }
};

static const int delta_lf_id_lut[MAX_MB_PLANE][2] = { { 0, 1 },
                                                      { 2, 2 },
                                                      { 3, 3 } };

static const int mode_lf_lut[] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // INTRA_MODES
  1, 1, 0, 1,                             // INTER_MODES (GLOBALMV == 0)
  1, 1, 1, 1, 1, 1, 0, 1  // INTER_COMPOUND_MODES (GLOBAL_GLOBALMV == 0)
};

static void update_sharpness(loop_filter_info_n *lfi, int sharpness_lvl) {
  int lvl;

  // For each possible value for the loop filter fill out limits
  for (lvl = 0; lvl <= MAX_LOOP_FILTER; lvl++) {
    // Set loop filter parameters that control sharpness.
    int block_inside_limit = lvl >> ((sharpness_lvl > 0) + (sharpness_lvl > 4));

    if (sharpness_lvl > 0) {
      if (block_inside_limit > (9 - sharpness_lvl))
        block_inside_limit = (9 - sharpness_lvl);
    }

    if (block_inside_limit < 1) block_inside_limit = 1;

    memset(lfi->lfthr[lvl].lim, block_inside_limit, SIMD_WIDTH);
    memset(lfi->lfthr[lvl].mblim, (2 * (lvl + 2) + block_inside_limit),
           SIMD_WIDTH);
  }
}

uint8_t av1_get_filter_level(const AV1_COMMON *cm,
                             const loop_filter_info_n *lfi_n, const int dir_idx,
                             int plane, const MB_MODE_INFO *mbmi) {
  const int segment_id = mbmi->segment_id;
  if (cm->delta_q_info.delta_lf_present_flag) {
    int8_t delta_lf;
    if (cm->delta_q_info.delta_lf_multi) {
      const int delta_lf_idx = delta_lf_id_lut[plane][dir_idx];
      delta_lf = mbmi->delta_lf[delta_lf_idx];
    } else {
      delta_lf = mbmi->delta_lf_from_base;
    }
    int base_level;
    if (plane == 0)
      base_level = cm->lf.filter_level[dir_idx];
    else if (plane == 1)
      base_level = cm->lf.filter_level_u;
    else
      base_level = cm->lf.filter_level_v;
    int lvl_seg = clamp(delta_lf + base_level, 0, MAX_LOOP_FILTER);
    assert(plane >= 0 && plane <= 2);
    const int seg_lf_feature_id = seg_lvl_lf_lut[plane][dir_idx];
    if (segfeature_active(&cm->seg, segment_id, seg_lf_feature_id)) {
      const int data = get_segdata(&cm->seg, segment_id, seg_lf_feature_id);
      lvl_seg = clamp(lvl_seg + data, 0, MAX_LOOP_FILTER);
    }

    if (cm->lf.mode_ref_delta_enabled) {
      const int scale = 1 << (lvl_seg >> 5);
      lvl_seg += cm->lf.ref_deltas[mbmi->ref_frame[0]] * scale;
      if (mbmi->ref_frame[0] > INTRA_FRAME)
        lvl_seg += cm->lf.mode_deltas[mode_lf_lut[mbmi->mode]] * scale;
      lvl_seg = clamp(lvl_seg, 0, MAX_LOOP_FILTER);
    }
    return lvl_seg;
  } else {
    return lfi_n->lvl[plane][segment_id][dir_idx][mbmi->ref_frame[0]]
                     [mode_lf_lut[mbmi->mode]];
  }
}

void av1_loop_filter_init(AV1_COMMON *cm) {
  assert(MB_MODE_COUNT == NELEMENTS(mode_lf_lut));
  loop_filter_info_n *lfi = &cm->lf_info;
  struct loopfilter *lf = &cm->lf;
  int lvl;

  // init limits for given sharpness
  update_sharpness(lfi, lf->sharpness_level);

  // init hev threshold const vectors
  for (lvl = 0; lvl <= MAX_LOOP_FILTER; lvl++)
    memset(lfi->lfthr[lvl].hev_thr, (lvl >> 4), SIMD_WIDTH);
}

// Update the loop filter for the current frame.
// This should be called before loop_filter_rows(),
// av1_loop_filter_frame() calls this function directly.
void av1_loop_filter_frame_init(AV1_COMMON *cm, int plane_start,
                                int plane_end) {
  int filt_lvl[MAX_MB_PLANE], filt_lvl_r[MAX_MB_PLANE];
  int plane;
  int seg_id;
  // n_shift is the multiplier for lf_deltas
  // the multiplier is 1 for when filter_lvl is between 0 and 31;
  // 2 when filter_lvl is between 32 and 63
  loop_filter_info_n *const lfi = &cm->lf_info;
  struct loopfilter *const lf = &cm->lf;
  const struct segmentation *const seg = &cm->seg;

  // update sharpness limits
  update_sharpness(lfi, lf->sharpness_level);

  filt_lvl[0] = cm->lf.filter_level[0];
  filt_lvl[1] = cm->lf.filter_level_u;
  filt_lvl[2] = cm->lf.filter_level_v;

  filt_lvl_r[0] = cm->lf.filter_level[1];
  filt_lvl_r[1] = cm->lf.filter_level_u;
  filt_lvl_r[2] = cm->lf.filter_level_v;

  assert(plane_start >= AOM_PLANE_Y);
  assert(plane_end <= MAX_MB_PLANE);

  for (plane = plane_start; plane < plane_end; plane++) {
    if (plane == 0 && !filt_lvl[0] && !filt_lvl_r[0])
      break;
    else if (plane == 1 && !filt_lvl[1])
      continue;
    else if (plane == 2 && !filt_lvl[2])
      continue;

    for (seg_id = 0; seg_id < MAX_SEGMENTS; seg_id++) {
      for (int dir = 0; dir < 2; ++dir) {
        int lvl_seg = (dir == 0) ? filt_lvl[plane] : filt_lvl_r[plane];
        const int seg_lf_feature_id = seg_lvl_lf_lut[plane][dir];
        if (segfeature_active(seg, seg_id, seg_lf_feature_id)) {
          const int data = get_segdata(&cm->seg, seg_id, seg_lf_feature_id);
          lvl_seg = clamp(lvl_seg + data, 0, MAX_LOOP_FILTER);
        }

        if (!lf->mode_ref_delta_enabled) {
          // we could get rid of this if we assume that deltas are set to
          // zero when not in use; encoder always uses deltas
          memset(lfi->lvl[plane][seg_id][dir], lvl_seg,
                 sizeof(lfi->lvl[plane][seg_id][dir]));
        } else {
          int ref, mode;
          const int scale = 1 << (lvl_seg >> 5);
          const int intra_lvl = lvl_seg + lf->ref_deltas[INTRA_FRAME] * scale;
          lfi->lvl[plane][seg_id][dir][INTRA_FRAME][0] =
              clamp(intra_lvl, 0, MAX_LOOP_FILTER);

          for (ref = LAST_FRAME; ref < REF_FRAMES; ++ref) {
            for (mode = 0; mode < MAX_MODE_LF_DELTAS; ++mode) {
              const int inter_lvl = lvl_seg + lf->ref_deltas[ref] * scale +
                                    lf->mode_deltas[mode] * scale;
              lfi->lvl[plane][seg_id][dir][ref][mode] =
                  clamp(inter_lvl, 0, MAX_LOOP_FILTER);
            }
          }
        }
      }
    }
  }
}

static TX_SIZE get_transform_size(const MACROBLOCKD *const xd,
                                  const MB_MODE_INFO *const mbmi,
                                  const int mi_row, const int mi_col,
                                  const int plane,
                                  const struct macroblockd_plane *plane_ptr) {
  assert(mbmi != NULL);
  if (xd && xd->lossless[mbmi->segment_id]) return TX_4X4;

  TX_SIZE tx_size =
      (plane == AOM_PLANE_Y)
          ? mbmi->tx_size
          : av1_get_max_uv_txsize(mbmi->bsize, plane_ptr->subsampling_x,
                                  plane_ptr->subsampling_y);
  assert(tx_size < TX_SIZES_ALL);
  if ((plane == AOM_PLANE_Y) && is_inter_block(mbmi) && !mbmi->skip_txfm) {
    const BLOCK_SIZE sb_type = mbmi->bsize;
    const int blk_row = mi_row & (mi_size_high[sb_type] - 1);
    const int blk_col = mi_col & (mi_size_wide[sb_type] - 1);
    const TX_SIZE mb_tx_size =
        mbmi->inter_tx_size[av1_get_txb_size_index(sb_type, blk_row, blk_col)];
    assert(mb_tx_size < TX_SIZES_ALL);
    tx_size = mb_tx_size;
  }

  return tx_size;
}

// Return TX_SIZE from get_transform_size(), so it is plane and direction
// aware
static TX_SIZE set_lpf_parameters(
    AV1_DEBLOCKING_PARAMETERS *const params, const ptrdiff_t mode_step,
    const AV1_COMMON *const cm, const MACROBLOCKD *const xd,
    const EDGE_DIR edge_dir, const uint32_t x, const uint32_t y,
    const int plane, const struct macroblockd_plane *const plane_ptr) {
  // reset to initial values
  params->filter_length = 0;

  // no deblocking is required
  const uint32_t width = plane_ptr->dst.width;
  const uint32_t height = plane_ptr->dst.height;
  if ((width <= x) || (height <= y)) {
    // just return the smallest transform unit size
    return TX_4X4;
  }

  const uint32_t scale_horz = plane_ptr->subsampling_x;
  const uint32_t scale_vert = plane_ptr->subsampling_y;
  // for sub8x8 block, chroma prediction mode is obtained from the bottom/right
  // mi structure of the co-located 8x8 luma block. so for chroma plane, mi_row
  // and mi_col should map to the bottom/right mi structure, i.e, both mi_row
  // and mi_col should be odd number for chroma plane.
  const int mi_row = scale_vert | ((y << scale_vert) >> MI_SIZE_LOG2);
  const int mi_col = scale_horz | ((x << scale_horz) >> MI_SIZE_LOG2);
  MB_MODE_INFO **mi =
      cm->mi_params.mi_grid_base + mi_row * cm->mi_params.mi_stride + mi_col;
  const MB_MODE_INFO *mbmi = mi[0];
  // If current mbmi is not correctly setup, return an invalid value to stop
  // filtering. One example is that if this tile is not coded, then its mbmi
  // it not set up.
  if (mbmi == NULL) return TX_INVALID;

  const TX_SIZE ts =
      get_transform_size(xd, mi[0], mi_row, mi_col, plane, plane_ptr);

  {
    const uint32_t coord = (VERT_EDGE == edge_dir) ? (x) : (y);
    const uint32_t transform_masks =
        edge_dir == VERT_EDGE ? tx_size_wide[ts] - 1 : tx_size_high[ts] - 1;
    const int32_t tu_edge = (coord & transform_masks) ? (0) : (1);

    if (!tu_edge) return ts;

    // prepare outer edge parameters. deblock the edge if it's an edge of a TU
    {
      const uint32_t curr_level =
          av1_get_filter_level(cm, &cm->lf_info, edge_dir, plane, mbmi);
      const int curr_skipped = mbmi->skip_txfm && is_inter_block(mbmi);
      uint32_t level = curr_level;
      if (coord) {
        {
          const MB_MODE_INFO *const mi_prev = *(mi - mode_step);
          if (mi_prev == NULL) return TX_INVALID;
          const int pv_row =
              (VERT_EDGE == edge_dir) ? (mi_row) : (mi_row - (1 << scale_vert));
          const int pv_col =
              (VERT_EDGE == edge_dir) ? (mi_col - (1 << scale_horz)) : (mi_col);
          const TX_SIZE pv_ts =
              get_transform_size(xd, mi_prev, pv_row, pv_col, plane, plane_ptr);

          const uint32_t pv_lvl =
              av1_get_filter_level(cm, &cm->lf_info, edge_dir, plane, mi_prev);

          const int pv_skip_txfm =
              mi_prev->skip_txfm && is_inter_block(mi_prev);
          const BLOCK_SIZE bsize = get_plane_block_size(
              mbmi->bsize, plane_ptr->subsampling_x, plane_ptr->subsampling_y);
          assert(bsize < BLOCK_SIZES_ALL);
          const int prediction_masks = edge_dir == VERT_EDGE
                                           ? block_size_wide[bsize] - 1
                                           : block_size_high[bsize] - 1;
          const int32_t pu_edge = !(coord & prediction_masks);
          // if the current and the previous blocks are skipped,
          // deblock the edge if the edge belongs to a PU's edge only.
          if ((curr_level || pv_lvl) &&
              (!pv_skip_txfm || !curr_skipped || pu_edge)) {
            const int dim = (VERT_EDGE == edge_dir)
                                ? AOMMIN(tx_size_wide_unit_log2[ts],
                                         tx_size_wide_unit_log2[pv_ts])
                                : AOMMIN(tx_size_high_unit_log2[ts],
                                         tx_size_high_unit_log2[pv_ts]);
            if (plane) {
              params->filter_length = (dim == 0) ? 4 : 6;
            } else {
              static const int tx_dim_to_filter_length[TX_SIZES] = { 4, 8, 14,
                                                                     14, 14 };
              assert(dim < TX_SIZES);
              assert(dim >= 0);
              params->filter_length = tx_dim_to_filter_length[dim];
            }

            // update the level if the current block is skipped,
            // but the previous one is not
            level = (curr_level) ? (curr_level) : (pv_lvl);
          }
        }
      }
      // prepare common parameters
      if (params->filter_length) {
        const loop_filter_thresh *const limits = cm->lf_info.lfthr + level;
        params->lfthr = limits;
      }
    }
  }

  return ts;
}

// Similar to set_lpf_parameters, but does so one row/col at a time to reduce
// calls to \ref get_transform_size and \ref av1_get_filter_level
static AOM_FORCE_INLINE void set_lpf_parameters_for_line_luma(
    AV1_DEBLOCKING_PARAMETERS *const params_buf, TX_SIZE *tx_buf,
    const AV1_COMMON *const cm, const MACROBLOCKD *const xd,
    const EDGE_DIR edge_dir, uint32_t mi_col, uint32_t mi_row,
    const struct macroblockd_plane *const plane_ptr, const uint32_t mi_range) {
  AV1_DEBLOCKING_PARAMETERS *params = params_buf;
  TX_SIZE *tx_size = tx_buf;

  TX_SIZE prev_tx_size = TX_INVALID;

  const int is_vert = edge_dir == VERT_EDGE;
  const ptrdiff_t mode_step = is_vert ? 1 : cm->mi_params.mi_stride;

  uint32_t *counter_ptr = is_vert ? &mi_col : &mi_row;
  // Unroll the first iteration
  {
    assert(mi_row << MI_SIZE_LOG2 < (uint32_t)plane_ptr->dst.width &&
           mi_col << MI_SIZE_LOG2 < (uint32_t)plane_ptr->dst.height);
    // reset to initial values
    params->filter_length = 0;

    MB_MODE_INFO **mi =
        cm->mi_params.mi_grid_base + mi_row * cm->mi_params.mi_stride + mi_col;
    const MB_MODE_INFO *mbmi = mi[0];
    assert(mbmi);

    const TX_SIZE ts =
        get_transform_size(xd, mi[0], mi_row, mi_col, AOM_PLANE_Y, plane_ptr);
    *tx_size = ts;

    const int advance_units =
        is_vert ? tx_size_wide_unit[ts] : tx_size_high_unit[ts];
#ifndef NDEBUG
    const uint32_t transform_masks =
        is_vert ? tx_size_wide[ts] - 1 : tx_size_high[ts] - 1;
    const int32_t tu_edge = (*counter_ptr & transform_masks) ? (0) : (1);
    assert(tu_edge);
#endif  // NDEBUG

    // prepare outer edge parameters. deblock the edge if it's an edge of a TU
    if (*counter_ptr) {
      const MB_MODE_INFO *const mi_prev = *(mi - mode_step);
      const int pv_row = is_vert ? mi_row : (mi_row - 1);
      const int pv_col = is_vert ? (mi_col - 1) : mi_col;
      const TX_SIZE pv_ts = get_transform_size(xd, mi_prev, pv_row, pv_col,
                                               AOM_PLANE_Y, plane_ptr);
      assert(mi_prev);
      uint32_t level =
          av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_Y, mbmi);
      if (!level) {
        level = av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_Y,
                                     mi_prev);
      }

      const int curr_skipped = mbmi->skip_txfm && is_inter_block(mbmi);
      const int32_t pu_edge = mi_prev != mbmi;
      if (level && (!curr_skipped || pu_edge)) {
        const int dim = is_vert ? AOMMIN(tx_size_wide_unit_log2[ts],
                                         tx_size_wide_unit_log2[pv_ts])
                                : AOMMIN(tx_size_high_unit_log2[ts],
                                         tx_size_high_unit_log2[pv_ts]);
        static const int tx_dim_to_filter_length[TX_SIZES] = { 4, 8, 14, 14,
                                                               14 };
        assert(dim < TX_SIZES);
        assert(dim >= 0);
        params->filter_length = tx_dim_to_filter_length[dim];

        // prepare common parameters
        const loop_filter_thresh *const limits = cm->lf_info.lfthr + level;
        params->lfthr = limits;
      }
    }

    // Advance
    *counter_ptr += advance_units;
    params += advance_units;
    tx_size += advance_units;

    prev_tx_size = ts;
  }

  while (*counter_ptr < mi_range) {
    assert(mi_row << MI_SIZE_LOG2 < (uint32_t)plane_ptr->dst.width &&
           mi_col << MI_SIZE_LOG2 < (uint32_t)plane_ptr->dst.height);
    // reset to initial values
    params->filter_length = 0;

    MB_MODE_INFO **mi =
        cm->mi_params.mi_grid_base + mi_row * cm->mi_params.mi_stride + mi_col;
    const MB_MODE_INFO *mbmi = mi[0];
    assert(mbmi);

    const TX_SIZE ts =
        get_transform_size(xd, mi[0], mi_row, mi_col, AOM_PLANE_Y, plane_ptr);
    *tx_size = ts;

    const int advance_units =
        is_vert ? tx_size_wide_unit[ts] : tx_size_high_unit[ts];
#ifndef NDEBUG
    const uint32_t transform_masks =
        is_vert ? tx_size_wide[ts] - 1 : tx_size_high[ts] - 1;
    const int32_t tu_edge = (*counter_ptr & transform_masks) ? (0) : (1);
    assert(tu_edge);
#endif  // NDEBUG

    // prepare outer edge parameters. deblock the edge if it's an edge of a TU
    const MB_MODE_INFO *const mi_prev = *(mi - mode_step);
    const TX_SIZE pv_ts = prev_tx_size;
    assert(mi_prev);
    uint32_t level =
        av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_Y, mbmi);
    if (!level) {
      level = av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_Y,
                                   mi_prev);
    }

    const int curr_skipped = mbmi->skip_txfm && is_inter_block(mbmi);
    const int32_t pu_edge = mi_prev != mbmi;
    if (level && (!curr_skipped || pu_edge)) {
      const int dim = is_vert ? AOMMIN(tx_size_wide_unit_log2[ts],
                                       tx_size_wide_unit_log2[pv_ts])
                              : AOMMIN(tx_size_high_unit_log2[ts],
                                       tx_size_high_unit_log2[pv_ts]);
      static const int tx_dim_to_filter_length[TX_SIZES] = { 4, 8, 14, 14, 14 };
      assert(dim < TX_SIZES);
      assert(dim >= 0);
      params->filter_length = tx_dim_to_filter_length[dim];

      // prepare common parameters
      const loop_filter_thresh *const limits = cm->lf_info.lfthr + level;
      params->lfthr = limits;
    }

    // Advance
    *counter_ptr += advance_units;
    params += advance_units;
    tx_size += advance_units;

    prev_tx_size = ts;
  }
}

static AOM_FORCE_INLINE void set_lpf_parameters_for_line_chroma(
    AV1_DEBLOCKING_PARAMETERS *const params_buf, TX_SIZE *tx_buf,
    const AV1_COMMON *const cm, const MACROBLOCKD *const xd,
    const EDGE_DIR edge_dir, uint32_t x, uint32_t y,
    const struct macroblockd_plane *const plane_ptr, const uint32_t range) {
  AV1_DEBLOCKING_PARAMETERS *params = params_buf;
  TX_SIZE *tx_size = tx_buf;

  TX_SIZE prev_tx_size = TX_INVALID;

  const int is_vert = edge_dir == VERT_EDGE;
  const uint32_t scale_horz = plane_ptr->subsampling_x;
  const uint32_t scale_vert = plane_ptr->subsampling_y;
  const ptrdiff_t mode_step =
      is_vert ? (1 << scale_horz) : (cm->mi_params.mi_stride << scale_vert);
  const loop_filter_thresh *const limits = cm->lf_info.lfthr;
  uint32_t *counter_ptr = is_vert ? &x : &y;

  {
    assert(x < (uint32_t)plane_ptr->dst.width &&
           y < (uint32_t)plane_ptr->dst.height);
    // reset to initial values
    params->filter_length = 0;

    // for sub8x8 block, chroma prediction mode is obtained from the
    // bottom/right mi structure of the co-located 8x8 luma block. so for chroma
    // plane, mi_row and mi_col should map to the bottom/right mi structure,
    // i.e, both mi_row and mi_col should be odd number for chroma plane.
    const int mi_row = scale_vert | ((y << scale_vert) >> MI_SIZE_LOG2);
    const int mi_col = scale_horz | ((x << scale_horz) >> MI_SIZE_LOG2);
    MB_MODE_INFO **mi =
        cm->mi_params.mi_grid_base + mi_row * cm->mi_params.mi_stride + mi_col;
    const MB_MODE_INFO *mbmi = mi[0];
    assert(mbmi);

    const TX_SIZE ts =
        get_transform_size(xd, mi[0], mi_row, mi_col, AOM_PLANE_U, plane_ptr);
    *tx_size = ts;

    const int advance_units =
        is_vert ? tx_size_wide_unit[ts] : tx_size_high_unit[ts];
#ifndef NDEBUG
    const uint32_t transform_masks =
        is_vert ? tx_size_wide[ts] - 1 : tx_size_high[ts] - 1;
    const int32_t tu_edge = (*counter_ptr & transform_masks) ? (0) : (1);
    assert(tu_edge);
#endif  // NDEBUG

    // prepare outer edge parameters. deblock the edge if it's an edge of a TU
    if (*counter_ptr) {
      const MB_MODE_INFO *const mi_prev = *(mi - mode_step);
      assert(mi_prev);
      const int pv_row = is_vert ? (mi_row) : (mi_row - (1 << scale_vert));
      const int pv_col = is_vert ? (mi_col - (1 << scale_horz)) : (mi_col);
      const TX_SIZE pv_ts = get_transform_size(xd, mi_prev, pv_row, pv_col,
                                               AOM_PLANE_U, plane_ptr);
      const int curr_skipped = mbmi->skip_txfm && is_inter_block(mbmi);
      const int32_t pu_edge = mi_prev != mbmi;
      const int dim = is_vert ? AOMMIN(tx_size_wide_unit_log2[ts],
                                       tx_size_wide_unit_log2[pv_ts])
                              : AOMMIN(tx_size_high_unit_log2[ts],
                                       tx_size_high_unit_log2[pv_ts]);

      uint32_t u_level =
          av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_U, mbmi);
      if (!u_level) {
        u_level = av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_U,
                                       mi_prev);
      }
#ifndef NDEBUG
      uint32_t v_level =
          av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_V, mbmi);
      if (!v_level) {
        v_level = av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_V,
                                       mi_prev);
      }
      assert(u_level == v_level);
#endif  // NDEBUG
      // For realtime mode, u and v have the same level
      if (u_level && (!curr_skipped || pu_edge)) {
        params->filter_length = (dim == 0) ? 4 : 6;
      }

      params->uv_lfthr[0] = limits + u_level;
      params->uv_lfthr[1] = limits + u_level;
    }

    // Advance
    *counter_ptr += MIN_TX_SIZE * advance_units;
    params += advance_units;
    tx_size += advance_units;

    prev_tx_size = ts;
  }

  while (*counter_ptr < range) {
    assert(x < (uint32_t)plane_ptr->dst.width &&
           y < (uint32_t)plane_ptr->dst.height);
    // reset to initial values
    params->filter_length = 0;

    // for sub8x8 block, chroma prediction mode is obtained from the
    // bottom/right mi structure of the co-located 8x8 luma block. so for chroma
    // plane, mi_row and mi_col should map to the bottom/right mi structure,
    // i.e, both mi_row and mi_col should be odd number for chroma plane.
    const int mi_row = scale_vert | ((y << scale_vert) >> MI_SIZE_LOG2);
    const int mi_col = scale_horz | ((x << scale_horz) >> MI_SIZE_LOG2);
    MB_MODE_INFO **mi =
        cm->mi_params.mi_grid_base + mi_row * cm->mi_params.mi_stride + mi_col;
    const MB_MODE_INFO *mbmi = mi[0];
    assert(mbmi);

    const TX_SIZE ts =
        get_transform_size(xd, mi[0], mi_row, mi_col, AOM_PLANE_U, plane_ptr);
    *tx_size = ts;

    const int advance_units =
        is_vert ? tx_size_wide_unit[ts] : tx_size_high_unit[ts];
#ifndef NDEBUG
    const uint32_t transform_masks =
        is_vert ? tx_size_wide[ts] - 1 : tx_size_high[ts] - 1;
    const int32_t tu_edge = (*counter_ptr & transform_masks) ? (0) : (1);
    assert(tu_edge);
#endif  // NDEBUG

    // prepare outer edge parameters. deblock the edge if it's an edge of a TU
    const MB_MODE_INFO *const mi_prev = *(mi - mode_step);
    assert(mi_prev);
    const TX_SIZE pv_ts = prev_tx_size;
    const int curr_skipped = mbmi->skip_txfm && is_inter_block(mbmi);
    const int32_t pu_edge = mi_prev != mbmi;
    const int dim =
        is_vert
            ? AOMMIN(tx_size_wide_unit_log2[ts], tx_size_wide_unit_log2[pv_ts])
            : AOMMIN(tx_size_high_unit_log2[ts], tx_size_high_unit_log2[pv_ts]);

    uint32_t u_level =
        av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_U, mbmi);
    if (!u_level) {
      u_level = av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_U,
                                     mi_prev);
    }
#ifndef NDEBUG
    uint32_t v_level =
        av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_V, mbmi);
    if (!v_level) {
      v_level = av1_get_filter_level(cm, &cm->lf_info, edge_dir, AOM_PLANE_V,
                                     mi_prev);
    }
    assert(u_level == v_level);
#endif  // NDEBUG
    // For realtime mode, u and v have the same level
    if (u_level && (!curr_skipped || pu_edge)) {
      params->filter_length = (dim == 0) ? 4 : 6;
    }

    params->uv_lfthr[0] = limits + u_level;
    params->uv_lfthr[1] = limits + u_level;

    // Advance
    *counter_ptr += MIN_TX_SIZE * advance_units;
    params += advance_units;
    tx_size += advance_units;

    prev_tx_size = ts;
  }
}

static AOM_INLINE int get_min_tx_height(const TX_SIZE *tx_buf,
                                        const int x_range) {
  int min_dim = INT_MAX;

  for (int x = 0; x < x_range;) {
    const TX_SIZE ts = *tx_buf;
    if (ts == TX_INVALID) {
      x++;
      continue;
    }
    tx_buf += tx_size_wide_unit[ts];
    x += tx_size_wide_unit[ts];

    min_dim = AOMMIN(min_dim, tx_size_high[ts]);
  }
  return min_dim;
}

static AOM_INLINE int get_min_tx_width(const TX_SIZE *tx_buf,
                                       const int y_range) {
  int min_dim = INT_MAX;

  for (int y = 0; y < y_range;) {
    const TX_SIZE ts = *tx_buf;
    if (ts == TX_INVALID) {
      y++;
      continue;
    }
    tx_buf += tx_size_high_unit[ts];
    y += tx_size_high_unit[ts];

    min_dim = AOMMIN(min_dim, tx_size_wide[ts]);
  }
  return min_dim;
}

static AOM_INLINE void filter_vert(uint8_t *dst, int dst_stride,
                                   const AV1_DEBLOCKING_PARAMETERS *params,
                                   bool use_dual) {
  const loop_filter_thresh *limits = params->lfthr;
  if (use_dual) {
    switch (params->filter_length) {
      // apply 4-tap filtering
      case 4:
        aom_lpf_vertical_4_dual(dst, dst_stride, limits->mblim, limits->lim,
                                limits->hev_thr, limits->mblim, limits->lim,
                                limits->hev_thr);
        break;
      case 6:  // apply 6-tap filter for chroma plane only
        aom_lpf_vertical_6_dual(dst, dst_stride, limits->mblim, limits->lim,
                                limits->hev_thr, limits->mblim, limits->lim,
                                limits->hev_thr);
        break;
      // apply 8-tap filtering
      case 8:
        aom_lpf_vertical_8_dual(dst, dst_stride, limits->mblim, limits->lim,
                                limits->hev_thr, limits->mblim, limits->lim,
                                limits->hev_thr);
        break;
      // apply 14-tap filtering
      case 14:
        aom_lpf_vertical_14_dual(dst, dst_stride, limits->mblim, limits->lim,
                                 limits->hev_thr, limits->mblim, limits->lim,
                                 limits->hev_thr);
        break;
      // no filtering
      default: break;
    }
  } else {
    switch (params->filter_length) {
      // apply 4-tap filtering
      case 4:
        aom_lpf_vertical_4(dst, dst_stride, limits->mblim, limits->lim,
                           limits->hev_thr);
        break;
      case 6:  // apply 6-tap filter for chroma plane only
        aom_lpf_vertical_6(dst, dst_stride, limits->mblim, limits->lim,
                           limits->hev_thr);
        break;
      // apply 8-tap filtering
      case 8:
        aom_lpf_vertical_8(dst, dst_stride, limits->mblim, limits->lim,
                           limits->hev_thr);
        break;
      // apply 14-tap filtering
      case 14:
        aom_lpf_vertical_14(dst, dst_stride, limits->mblim, limits->lim,
                            limits->hev_thr);
        break;
      // no filtering
      default: break;
    }
  }
}

static AOM_INLINE void filter_vert_chroma(
    uint8_t *u_dst, uint8_t *v_dst, int dst_stride,
    const AV1_DEBLOCKING_PARAMETERS *params, bool use_dual) {
  const loop_filter_thresh *u_limits = params->uv_lfthr[0];
  const loop_filter_thresh *v_limits = params->uv_lfthr[1];
  if (use_dual) {
    switch (params->filter_length) {
      // apply 4-tap filtering
      case 4:
        aom_lpf_vertical_4_dual(u_dst, dst_stride, u_limits->mblim,
                                u_limits->lim, u_limits->hev_thr,
                                u_limits->mblim, u_limits->lim,
                                u_limits->hev_thr);
        aom_lpf_vertical_4_dual(v_dst, dst_stride, v_limits->mblim,
                                v_limits->lim, v_limits->hev_thr,
                                v_limits->mblim, v_limits->lim,
                                v_limits->hev_thr);
        break;
      case 6:  // apply 6-tap filter for chroma plane only
        aom_lpf_vertical_6_dual(u_dst, dst_stride, u_limits->mblim,
                                u_limits->lim, u_limits->hev_thr,
                                u_limits->mblim, u_limits->lim,
                                u_limits->hev_thr);
        aom_lpf_vertical_6_dual(v_dst, dst_stride, v_limits->mblim,
                                v_limits->lim, v_limits->hev_thr,
                                v_limits->mblim, v_limits->lim,
                                v_limits->hev_thr);
        break;
      case 8:
      case 14: assert(0);
      // no filtering
      default: break;
    }
  } else {
    switch (params->filter_length) {
      // apply 4-tap filtering
      case 4:
        aom_lpf_vertical_4(u_dst, dst_stride, u_limits->mblim, u_limits->lim,
                           u_limits->hev_thr);
        aom_lpf_vertical_4(v_dst, dst_stride, v_limits->mblim, v_limits->lim,
                           u_limits->hev_thr);
        break;
      case 6:  // apply 6-tap filter for chroma plane only
        aom_lpf_vertical_6(u_dst, dst_stride, u_limits->mblim, u_limits->lim,
                           u_limits->hev_thr);
        aom_lpf_vertical_6(v_dst, dst_stride, v_limits->mblim, v_limits->lim,
                           v_limits->hev_thr);
        break;
      case 8:
      case 14: assert(0); break;
      // no filtering
      default: break;
    }
  }
}

void av1_filter_block_plane_vert(const AV1_COMMON *const cm,
                                 const MACROBLOCKD *const xd, const int plane,
                                 const MACROBLOCKD_PLANE *const plane_ptr,
                                 const uint32_t mi_row, const uint32_t mi_col) {
  const uint32_t scale_horz = plane_ptr->subsampling_x;
  const uint32_t scale_vert = plane_ptr->subsampling_y;
  uint8_t *const dst_ptr = plane_ptr->dst.buf;
  const int dst_stride = plane_ptr->dst.stride;
  const int plane_mi_rows =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_rows, scale_vert);
  const int plane_mi_cols =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, scale_horz);
  const int y_range = AOMMIN((int)(plane_mi_rows - (mi_row >> scale_vert)),
                             (MAX_MIB_SIZE >> scale_vert));
  const int x_range = AOMMIN((int)(plane_mi_cols - (mi_col >> scale_horz)),
                             (MAX_MIB_SIZE >> scale_horz));

  for (int y = 0; y < y_range; y++) {
    uint8_t *p = dst_ptr + y * MI_SIZE * dst_stride;
    for (int x = 0; x < x_range;) {
      // inner loop always filter vertical edges in a MI block. If MI size
      // is 8x8, it will filter the vertical edge aligned with a 8x8 block.
      // If 4x4 transform is used, it will then filter the internal edge
      //  aligned with a 4x4 block
      const uint32_t curr_x = ((mi_col * MI_SIZE) >> scale_horz) + x * MI_SIZE;
      const uint32_t curr_y = ((mi_row * MI_SIZE) >> scale_vert) + y * MI_SIZE;
      uint32_t advance_units;
      TX_SIZE tx_size;
      AV1_DEBLOCKING_PARAMETERS params;
      memset(&params, 0, sizeof(params));

      tx_size =
          set_lpf_parameters(&params, ((ptrdiff_t)1 << scale_horz), cm, xd,
                             VERT_EDGE, curr_x, curr_y, plane, plane_ptr);
      if (tx_size == TX_INVALID) {
        params.filter_length = 0;
        tx_size = TX_4X4;
      }

#if CONFIG_AV1_HIGHBITDEPTH
      const int use_highbitdepth = cm->seq_params->use_highbitdepth;
      const aom_bit_depth_t bit_depth = cm->seq_params->bit_depth;
      const loop_filter_thresh *limits = params.lfthr;
      switch (params.filter_length) {
        // apply 4-tap filtering
        case 4:
          if (use_highbitdepth)
            aom_highbd_lpf_vertical_4(CONVERT_TO_SHORTPTR(p), dst_stride,
                                      limits->mblim, limits->lim,
                                      limits->hev_thr, bit_depth);
          else
            aom_lpf_vertical_4(p, dst_stride, limits->mblim, limits->lim,
                               limits->hev_thr);
          break;
        case 6:  // apply 6-tap filter for chroma plane only
          assert(plane != 0);
          if (use_highbitdepth)
            aom_highbd_lpf_vertical_6(CONVERT_TO_SHORTPTR(p), dst_stride,
                                      limits->mblim, limits->lim,
                                      limits->hev_thr, bit_depth);
          else
            aom_lpf_vertical_6(p, dst_stride, limits->mblim, limits->lim,
                               limits->hev_thr);
          break;
        // apply 8-tap filtering
        case 8:
          if (use_highbitdepth)
            aom_highbd_lpf_vertical_8(CONVERT_TO_SHORTPTR(p), dst_stride,
                                      limits->mblim, limits->lim,
                                      limits->hev_thr, bit_depth);
          else
            aom_lpf_vertical_8(p, dst_stride, limits->mblim, limits->lim,
                               limits->hev_thr);
          break;
        // apply 14-tap filtering
        case 14:
          if (use_highbitdepth)
            aom_highbd_lpf_vertical_14(CONVERT_TO_SHORTPTR(p), dst_stride,
                                       limits->mblim, limits->lim,
                                       limits->hev_thr, bit_depth);
          else
            aom_lpf_vertical_14(p, dst_stride, limits->mblim, limits->lim,
                                limits->hev_thr);
          break;
        // no filtering
        default: break;
      }
#else
      filter_vert(p, dst_stride, &params, false);
#endif  // CONFIG_AV1_HIGHBITDEPTH
      // advance the destination pointer
      advance_units = tx_size_wide_unit[tx_size];
      x += advance_units;
      p += advance_units * MI_SIZE;
    }
  }
}

void av1_filter_block_plane_vert_rt(const AV1_COMMON *const cm,
                                    const MACROBLOCKD *const xd,
                                    const MACROBLOCKD_PLANE *const plane_ptr,
                                    const uint32_t mi_row,
                                    const uint32_t mi_col,
                                    AV1_DEBLOCKING_PARAMETERS *params_buf,
                                    TX_SIZE *tx_buf) {
  uint8_t *const dst_ptr = plane_ptr->dst.buf;
  const int dst_stride = plane_ptr->dst.stride;
  const int plane_mi_rows = ROUND_POWER_OF_TWO(cm->mi_params.mi_rows, 0);
  const int plane_mi_cols = ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, 0);
  const int y_range = AOMMIN((int)(plane_mi_rows - mi_row), MAX_MIB_SIZE);
  const int x_range = AOMMIN((int)(plane_mi_cols - mi_col), MAX_MIB_SIZE);
  assert(!(y_range % 2));
  for (int y = 0; y < y_range; y += 2) {
    const uint32_t curr_y = mi_row + y;
    const uint32_t x_start = mi_col;
    const uint32_t x_end = mi_col + x_range;
    set_lpf_parameters_for_line_luma(params_buf, tx_buf, cm, xd, VERT_EDGE,
                                     x_start, curr_y, plane_ptr, x_end);

    AV1_DEBLOCKING_PARAMETERS *params = params_buf;
    TX_SIZE *tx_size = tx_buf;

    uint8_t *p = dst_ptr + y * MI_SIZE * dst_stride;
    for (int x = 0; x < x_range;) {
      if (*tx_size == TX_INVALID) {
        params->filter_length = 0;
        *tx_size = TX_4X4;
      }

      filter_vert(p, dst_stride, params, true);

      // advance the destination pointer
      const uint32_t advance_units = tx_size_wide_unit[*tx_size];
      x += advance_units;
      p += advance_units * MI_SIZE;
      params += advance_units;
      tx_size += advance_units;
    }
  }
}

void av1_filter_block_plane_vert_rt_chroma(
    const AV1_COMMON *const cm, const MACROBLOCKD *const xd,
    const MACROBLOCKD_PLANE *const plane_ptr, const uint32_t mi_row,
    const uint32_t mi_col, AV1_DEBLOCKING_PARAMETERS *params_buf,
    TX_SIZE *tx_buf) {
  const uint32_t scale_horz = plane_ptr->subsampling_x;
  const uint32_t scale_vert = plane_ptr->subsampling_y;
  uint8_t *const u_dst_ptr = plane_ptr[0].dst.buf;
  uint8_t *const v_dst_ptr = plane_ptr[1].dst.buf;
  const int dst_stride = plane_ptr->dst.stride;
  const int plane_mi_rows =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_rows, scale_vert);
  const int plane_mi_cols =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, scale_horz);
  const int y_range = AOMMIN((int)(plane_mi_rows - (mi_row >> scale_vert)),
                             (MAX_MIB_SIZE >> scale_vert));
  const int x_range = AOMMIN((int)(plane_mi_cols - (mi_col >> scale_horz)),
                             (MAX_MIB_SIZE >> scale_horz));

  int min_height = 0;
  for (int y = 0; y < y_range; y++) {
    const uint32_t curr_y = ((mi_row * MI_SIZE) >> scale_vert) + y * MI_SIZE;
    const uint32_t x_start = ((mi_col * MI_SIZE) >> scale_horz) + 0 * MI_SIZE;
    const uint32_t x_end =
        ((mi_col * MI_SIZE) >> scale_horz) + x_range * MI_SIZE;
    set_lpf_parameters_for_line_chroma(params_buf, tx_buf, cm, xd, VERT_EDGE,
                                       x_start, curr_y, plane_ptr, x_end);

    AV1_DEBLOCKING_PARAMETERS *params = params_buf;
    TX_SIZE *tx_size = tx_buf;

    if (y % 2 == 0 && (y + 1) < y_range) {
      // If we are on an even row, and the minimum height is 8 pixels, then the
      // current and below rows must contain the same tx block. This is because
      // dim 4 can only happen every unit of 2**0, and 8 every unit of 2**1,
      // etc.
      min_height = get_min_tx_height(tx_buf, x_range);
    }
    uint8_t *u_dst = u_dst_ptr + y * MI_SIZE * dst_stride;
    uint8_t *v_dst = v_dst_ptr + y * MI_SIZE * dst_stride;
    for (int x = 0; x < x_range;) {
      // inner loop always filter vertical edges in a MI block. If MI size
      // is 8x8, it will filter the vertical edge aligned with a 8x8 block.
      // If 4x4 transform is used, it will then filter the internal edge
      //  aligned with a 4x4 block
      if (*tx_size == TX_INVALID) {
        params->filter_length = 0;
        *tx_size = TX_4X4;
      }

      filter_vert_chroma(u_dst, v_dst, dst_stride, params, min_height >= 8);

      // advance the destination pointer
      const uint32_t advance_units = tx_size_wide_unit[*tx_size];
      x += advance_units;
      u_dst += advance_units * MI_SIZE;
      v_dst += advance_units * MI_SIZE;
      params += advance_units;
      tx_size += advance_units;
    }
    if (min_height >= 8) {
      y++;
    }
  }
}

static AOM_INLINE void filter_horz(uint8_t *dst, int dst_stride,
                                   const AV1_DEBLOCKING_PARAMETERS *params,
                                   bool use_dual) {
  const loop_filter_thresh *limits = params->lfthr;
  if (use_dual) {
    switch (params->filter_length) {
      // apply 4-tap filtering
      case 4:
        aom_lpf_horizontal_4_dual(dst, dst_stride, limits->mblim, limits->lim,
                                  limits->hev_thr, limits->mblim, limits->lim,
                                  limits->hev_thr);
        break;
      case 6:  // apply 6-tap filter for chroma plane only
        aom_lpf_horizontal_6_dual(dst, dst_stride, limits->mblim, limits->lim,
                                  limits->hev_thr, limits->mblim, limits->lim,
                                  limits->hev_thr);
        break;
      // apply 8-tap filtering
      case 8:
        aom_lpf_horizontal_8_dual(dst, dst_stride, limits->mblim, limits->lim,
                                  limits->hev_thr, limits->mblim, limits->lim,
                                  limits->hev_thr);
        break;
      // apply 14-tap filtering
      case 14:
        aom_lpf_horizontal_14_dual(dst, dst_stride, limits->mblim, limits->lim,
                                   limits->hev_thr, limits->mblim, limits->lim,
                                   limits->hev_thr);
        break;
      // no filtering
      default: break;
    }
  } else {
    switch (params->filter_length) {
      // apply 4-tap filtering
      case 4:
        aom_lpf_horizontal_4(dst, dst_stride, limits->mblim, limits->lim,
                             limits->hev_thr);
        break;
      case 6:  // apply 6-tap filter for chroma plane only
        aom_lpf_horizontal_6(dst, dst_stride, limits->mblim, limits->lim,
                             limits->hev_thr);
        break;
      // apply 8-tap filtering
      case 8:
        aom_lpf_horizontal_8(dst, dst_stride, limits->mblim, limits->lim,
                             limits->hev_thr);
        break;
      // apply 14-tap filtering
      case 14:
        aom_lpf_horizontal_14(dst, dst_stride, limits->mblim, limits->lim,
                              limits->hev_thr);
        break;
      // no filtering
      default: break;
    }
  }
}

static AOM_INLINE void filter_horz_chroma(
    uint8_t *u_dst, uint8_t *v_dst, int dst_stride,
    const AV1_DEBLOCKING_PARAMETERS *params, bool use_dual) {
  const loop_filter_thresh *u_limits = params->uv_lfthr[0];
  const loop_filter_thresh *v_limits = params->uv_lfthr[1];
  if (use_dual) {
    switch (params->filter_length) {
      // apply 4-tap filtering
      case 4:
        aom_lpf_horizontal_4_dual(u_dst, dst_stride, u_limits->mblim,
                                  u_limits->lim, u_limits->hev_thr,
                                  u_limits->mblim, u_limits->lim,
                                  u_limits->hev_thr);
        aom_lpf_horizontal_4_dual(v_dst, dst_stride, v_limits->mblim,
                                  v_limits->lim, v_limits->hev_thr,
                                  v_limits->mblim, v_limits->lim,
                                  v_limits->hev_thr);
        break;
      case 6:  // apply 6-tap filter for chroma plane only
        aom_lpf_horizontal_6_dual(u_dst, dst_stride, u_limits->mblim,
                                  u_limits->lim, u_limits->hev_thr,
                                  u_limits->mblim, u_limits->lim,
                                  u_limits->hev_thr);
        aom_lpf_horizontal_6_dual(v_dst, dst_stride, v_limits->mblim,
                                  v_limits->lim, v_limits->hev_thr,
                                  v_limits->mblim, v_limits->lim,
                                  v_limits->hev_thr);
        break;
      case 8:
      case 14: assert(0);
      // no filtering
      default: break;
    }
  } else {
    switch (params->filter_length) {
      // apply 4-tap filtering
      case 4:
        aom_lpf_horizontal_4(u_dst, dst_stride, u_limits->mblim, u_limits->lim,
                             u_limits->hev_thr);
        aom_lpf_horizontal_4(v_dst, dst_stride, v_limits->mblim, v_limits->lim,
                             v_limits->hev_thr);
        break;
      case 6:  // apply 6-tap filter for chroma plane only
        aom_lpf_horizontal_6(u_dst, dst_stride, u_limits->mblim, u_limits->lim,
                             u_limits->hev_thr);
        aom_lpf_horizontal_6(v_dst, dst_stride, v_limits->mblim, v_limits->lim,
                             v_limits->hev_thr);
        break;
      case 8:
      case 14: assert(0);
      // no filtering
      default: break;
    }
  }
}

void av1_filter_block_plane_horz(const AV1_COMMON *const cm,
                                 const MACROBLOCKD *const xd, const int plane,
                                 const MACROBLOCKD_PLANE *const plane_ptr,
                                 const uint32_t mi_row, const uint32_t mi_col) {
  const uint32_t scale_horz = plane_ptr->subsampling_x;
  const uint32_t scale_vert = plane_ptr->subsampling_y;
  uint8_t *const dst_ptr = plane_ptr->dst.buf;
  const int dst_stride = plane_ptr->dst.stride;
  const int plane_mi_rows =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_rows, scale_vert);
  const int plane_mi_cols =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, scale_horz);
  const int y_range = AOMMIN((int)(plane_mi_rows - (mi_row >> scale_vert)),
                             (MAX_MIB_SIZE >> scale_vert));
  const int x_range = AOMMIN((int)(plane_mi_cols - (mi_col >> scale_horz)),
                             (MAX_MIB_SIZE >> scale_horz));
  for (int x = 0; x < x_range; x++) {
    uint8_t *p = dst_ptr + x * MI_SIZE;
    for (int y = 0; y < y_range;) {
      // inner loop always filter vertical edges in a MI block. If MI size
      // is 8x8, it will first filter the vertical edge aligned with a 8x8
      // block. If 4x4 transform is used, it will then filter the internal
      // edge aligned with a 4x4 block
      const uint32_t curr_x = ((mi_col * MI_SIZE) >> scale_horz) + x * MI_SIZE;
      const uint32_t curr_y = ((mi_row * MI_SIZE) >> scale_vert) + y * MI_SIZE;
      uint32_t advance_units;
      TX_SIZE tx_size;
      AV1_DEBLOCKING_PARAMETERS params;
      memset(&params, 0, sizeof(params));

      tx_size = set_lpf_parameters(
          &params, (cm->mi_params.mi_stride << scale_vert), cm, xd, HORZ_EDGE,
          curr_x, curr_y, plane, plane_ptr);
      if (tx_size == TX_INVALID) {
        params.filter_length = 0;
        tx_size = TX_4X4;
      }

#if CONFIG_AV1_HIGHBITDEPTH
      const int use_highbitdepth = cm->seq_params->use_highbitdepth;
      const aom_bit_depth_t bit_depth = cm->seq_params->bit_depth;
      const loop_filter_thresh *limits = params.lfthr;
      switch (params.filter_length) {
        // apply 4-tap filtering
        case 4:
          if (use_highbitdepth)
            aom_highbd_lpf_horizontal_4(CONVERT_TO_SHORTPTR(p), dst_stride,
                                        limits->mblim, limits->lim,
                                        limits->hev_thr, bit_depth);
          else
            aom_lpf_horizontal_4(p, dst_stride, limits->mblim, limits->lim,
                                 limits->hev_thr);
          break;
        // apply 6-tap filtering
        case 6:
          assert(plane != 0);
          if (use_highbitdepth)
            aom_highbd_lpf_horizontal_6(CONVERT_TO_SHORTPTR(p), dst_stride,
                                        limits->mblim, limits->lim,
                                        limits->hev_thr, bit_depth);
          else
            aom_lpf_horizontal_6(p, dst_stride, limits->mblim, limits->lim,
                                 limits->hev_thr);
          break;
        // apply 8-tap filtering
        case 8:
          if (use_highbitdepth)
            aom_highbd_lpf_horizontal_8(CONVERT_TO_SHORTPTR(p), dst_stride,
                                        limits->mblim, limits->lim,
                                        limits->hev_thr, bit_depth);
          else
            aom_lpf_horizontal_8(p, dst_stride, limits->mblim, limits->lim,
                                 limits->hev_thr);
          break;
        // apply 14-tap filtering
        case 14:
          if (use_highbitdepth)
            aom_highbd_lpf_horizontal_14(CONVERT_TO_SHORTPTR(p), dst_stride,
                                         limits->mblim, limits->lim,
                                         limits->hev_thr, bit_depth);
          else
            aom_lpf_horizontal_14(p, dst_stride, limits->mblim, limits->lim,
                                  limits->hev_thr);
          break;
        // no filtering
        default: break;
      }
#else
      filter_horz(p, dst_stride, &params, false);
#endif  // CONFIG_AV1_HIGHBITDEPTH

      // advance the destination pointer
      advance_units = tx_size_high_unit[tx_size];
      y += advance_units;
      p += advance_units * dst_stride * MI_SIZE;
    }
  }
}

void av1_filter_block_plane_horz_rt(const AV1_COMMON *const cm,
                                    const MACROBLOCKD *const xd,
                                    const MACROBLOCKD_PLANE *const plane_ptr,
                                    const uint32_t mi_row,
                                    const uint32_t mi_col,
                                    AV1_DEBLOCKING_PARAMETERS *params_buf,
                                    TX_SIZE *tx_buf) {
  uint8_t *const dst_ptr = plane_ptr->dst.buf;
  const int dst_stride = plane_ptr->dst.stride;
  const int plane_mi_rows = ROUND_POWER_OF_TWO(cm->mi_params.mi_rows, 0);
  const int plane_mi_cols = ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, 0);
  const int y_range = AOMMIN((int)(plane_mi_rows - mi_row), MAX_MIB_SIZE);
  const int x_range = AOMMIN((int)(plane_mi_cols - mi_col), MAX_MIB_SIZE);
  for (int x = 0; x < x_range; x += 2) {
    const uint32_t curr_x = mi_col + x;
    const uint32_t y_start = mi_row;
    const uint32_t y_end = mi_row + y_range;
    set_lpf_parameters_for_line_luma(params_buf, tx_buf, cm, xd, HORZ_EDGE,
                                     curr_x, y_start, plane_ptr, y_end);

    AV1_DEBLOCKING_PARAMETERS *params = params_buf;
    TX_SIZE *tx_size = tx_buf;

    uint8_t *p = dst_ptr + x * MI_SIZE;
    for (int y = 0; y < y_range;) {
      if (*tx_size == TX_INVALID) {
        params->filter_length = 0;
        *tx_size = TX_4X4;
      }

      filter_horz(p, dst_stride, params, true);

      // advance the destination pointer
      const uint32_t advance_units = tx_size_high_unit[*tx_size];
      y += advance_units;
      p += advance_units * dst_stride * MI_SIZE;
      params += advance_units;
      tx_size += advance_units;
    }
  }
}

void av1_filter_block_plane_horz_rt_chroma(
    const AV1_COMMON *const cm, const MACROBLOCKD *const xd,
    const MACROBLOCKD_PLANE *const plane_ptr, const uint32_t mi_row,
    const uint32_t mi_col, AV1_DEBLOCKING_PARAMETERS *params_buf,
    TX_SIZE *tx_buf) {
  const uint32_t scale_horz = plane_ptr->subsampling_x;
  const uint32_t scale_vert = plane_ptr->subsampling_y;
  uint8_t *const u_dst_ptr = plane_ptr[0].dst.buf;
  uint8_t *const v_dst_ptr = plane_ptr[1].dst.buf;
  const int dst_stride = plane_ptr->dst.stride;
  const int plane_mi_rows =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_rows, scale_vert);
  const int plane_mi_cols =
      ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, scale_horz);
  const int y_range = AOMMIN((int)(plane_mi_rows - (mi_row >> scale_vert)),
                             (MAX_MIB_SIZE >> scale_vert));
  const int x_range = AOMMIN((int)(plane_mi_cols - (mi_col >> scale_horz)),
                             (MAX_MIB_SIZE >> scale_horz));
  int min_width = 0;
  for (int x = 0; x < x_range; x++) {
    const uint32_t curr_x = ((mi_col * MI_SIZE) >> scale_horz) + x * MI_SIZE;
    const uint32_t y_start = ((mi_row * MI_SIZE) >> scale_vert) + 0 * MI_SIZE;
    const uint32_t y_end =
        ((mi_row * MI_SIZE) >> scale_vert) + y_range * MI_SIZE;
    set_lpf_parameters_for_line_chroma(params_buf, tx_buf, cm, xd, HORZ_EDGE,
                                       curr_x, y_start, plane_ptr, y_end);

    AV1_DEBLOCKING_PARAMETERS *params = params_buf;
    TX_SIZE *tx_size = tx_buf;

    if (x % 2 == 0 && (x + 1) < x_range) {
      // If we are on an even col, and the minimum width is 8 pixels, then the
      // current and left cols must contain the same tx block. This is because
      // dim 4 can only happen every unit of 2**0, and 8 every unit of 2**1,
      // etc.
      min_width = get_min_tx_width(tx_buf, y_range);
    }
    uint8_t *u_dst = u_dst_ptr + x * MI_SIZE;
    uint8_t *v_dst = v_dst_ptr + x * MI_SIZE;
    for (int y = 0; y < y_range;) {
      // inner loop always filter vertical edges in a MI block. If MI size
      // is 8x8, it will first filter the vertical edge aligned with a 8x8
      // block. If 4x4 transform is used, it will then filter the internal
      // edge aligned with a 4x4 block
      if (*tx_size == TX_INVALID) {
        params->filter_length = 0;
        *tx_size = TX_4X4;
      }

      filter_horz_chroma(u_dst, v_dst, dst_stride, params, min_width >= 8);

      // advance the destination pointer
      const int advance_units = tx_size_high_unit[*tx_size];
      y += advance_units;
      u_dst += advance_units * dst_stride * MI_SIZE;
      v_dst += advance_units * dst_stride * MI_SIZE;
      params += advance_units;
      tx_size += advance_units;
    }
    if (min_width >= 8) {
      x++;
    }
  }
}
