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

#include <assert.h>

#include "av1/common/cfl.h"
#include "av1/common/common.h"
#include "av1/common/entropy.h"
#include "av1/common/entropymode.h"
#include "av1/common/entropymv.h"
#include "av1/common/mvref_common.h"
#include "av1/common/pred_common.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/common/seg_common.h"
#include "av1/common/warped_motion.h"

#include "av1/decoder/decodeframe.h"
#include "av1/decoder/decodemv.h"

#include "aom_dsp/aom_dsp_common.h"

#define ACCT_STR __func__

#define DEC_MISMATCH_DEBUG 0

static PREDICTION_MODE read_intra_mode(aom_reader *r, aom_cdf_prob *cdf) {
  return (PREDICTION_MODE)aom_read_symbol(r, cdf, INTRA_MODES, ACCT_STR);
}

static void read_cdef(AV1_COMMON *cm, aom_reader *r, MACROBLOCKD *const xd) {
  const int skip_txfm = xd->mi[0]->skip_txfm;
  if (cm->features.coded_lossless) return;
  if (cm->features.allow_intrabc) {
    assert(cm->cdef_info.cdef_bits == 0);
    return;
  }

  // At the start of a superblock, mark that we haven't yet read CDEF strengths
  // for any of the CDEF units contained in this superblock.
  const int sb_mask = (cm->seq_params.mib_size - 1);
  const int mi_row_in_sb = (xd->mi_row & sb_mask);
  const int mi_col_in_sb = (xd->mi_col & sb_mask);
  if (mi_row_in_sb == 0 && mi_col_in_sb == 0) {
    xd->cdef_transmitted[0] = xd->cdef_transmitted[1] =
        xd->cdef_transmitted[2] = xd->cdef_transmitted[3] = false;
  }

  // CDEF unit size is 64x64 irrespective of the superblock size.
  const int cdef_size = 1 << (6 - MI_SIZE_LOG2);

  // Find index of this CDEF unit in this superblock.
  const int index_mask = cdef_size;
  const int cdef_unit_row_in_sb = ((xd->mi_row & index_mask) != 0);
  const int cdef_unit_col_in_sb = ((xd->mi_col & index_mask) != 0);
  const int index = (cm->seq_params.sb_size == BLOCK_128X128)
                        ? cdef_unit_col_in_sb + 2 * cdef_unit_row_in_sb
                        : 0;

  // Read CDEF strength from the first non-skip coding block in this CDEF unit.
  if (!xd->cdef_transmitted[index] && !skip_txfm) {
    // CDEF strength for this CDEF unit needs to be read into the MB_MODE_INFO
    // of the 1st block in this CDEF unit.
    const int first_block_mask = ~(cdef_size - 1);
    CommonModeInfoParams *const mi_params = &cm->mi_params;
    const int grid_idx =
        get_mi_grid_idx(mi_params, xd->mi_row & first_block_mask,
                        xd->mi_col & first_block_mask);
    MB_MODE_INFO *const mbmi = mi_params->mi_grid_base[grid_idx];
    mbmi->cdef_strength =
        aom_read_literal(r, cm->cdef_info.cdef_bits, ACCT_STR);
    xd->cdef_transmitted[index] = true;
  }
}

static int read_delta_qindex(AV1_COMMON *cm, const MACROBLOCKD *xd,
                             aom_reader *r, MB_MODE_INFO *const mbmi) {
  int sign, abs, reduced_delta_qindex = 0;
  BLOCK_SIZE bsize = mbmi->sb_type;
  const int b_col = xd->mi_col & (cm->seq_params.mib_size - 1);
  const int b_row = xd->mi_row & (cm->seq_params.mib_size - 1);
  const int read_delta_q_flag = (b_col == 0 && b_row == 0);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  if ((bsize != cm->seq_params.sb_size || mbmi->skip_txfm == 0) &&
      read_delta_q_flag) {
    abs = aom_read_symbol(r, ec_ctx->delta_q_cdf, DELTA_Q_PROBS + 1, ACCT_STR);
    const int smallval = (abs < DELTA_Q_SMALL);

    if (!smallval) {
      const int rem_bits = aom_read_literal(r, 3, ACCT_STR) + 1;
      const int thr = (1 << rem_bits) + 1;
      abs = aom_read_literal(r, rem_bits, ACCT_STR) + thr;
    }

    if (abs) {
      sign = aom_read_bit(r, ACCT_STR);
    } else {
      sign = 1;
    }

    reduced_delta_qindex = sign ? -abs : abs;
  }
  return reduced_delta_qindex;
}
static int read_delta_lflevel(const AV1_COMMON *const cm, aom_reader *r,
                              aom_cdf_prob *const cdf,
                              const MB_MODE_INFO *const mbmi, int mi_col,
                              int mi_row) {
  int reduced_delta_lflevel = 0;
  const BLOCK_SIZE bsize = mbmi->sb_type;
  const int b_col = mi_col & (cm->seq_params.mib_size - 1);
  const int b_row = mi_row & (cm->seq_params.mib_size - 1);
  const int read_delta_lf_flag = (b_col == 0 && b_row == 0);

  if ((bsize != cm->seq_params.sb_size || mbmi->skip_txfm == 0) &&
      read_delta_lf_flag) {
    int abs = aom_read_symbol(r, cdf, DELTA_LF_PROBS + 1, ACCT_STR);
    const int smallval = (abs < DELTA_LF_SMALL);
    if (!smallval) {
      const int rem_bits = aom_read_literal(r, 3, ACCT_STR) + 1;
      const int thr = (1 << rem_bits) + 1;
      abs = aom_read_literal(r, rem_bits, ACCT_STR) + thr;
    }
    const int sign = abs ? aom_read_bit(r, ACCT_STR) : 1;
    reduced_delta_lflevel = sign ? -abs : abs;
  }
  return reduced_delta_lflevel;
}

static UV_PREDICTION_MODE read_intra_mode_uv(FRAME_CONTEXT *ec_ctx,
                                             aom_reader *r,
#if CONFIG_DERIVED_INTRA_MODE
                                             const MACROBLOCKD *const xd,
#endif  // CONFIG_DERIVED_INTRA_MODE
                                             CFL_ALLOWED_TYPE cfl_allowed,
                                             PREDICTION_MODE y_mode) {
#if CONFIG_DERIVED_INTRA_MODE
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const BLOCK_SIZE bsize = mbmi->sb_type;
  if (av1_enable_derived_intra_mode(xd, bsize)) {
    mbmi->use_derived_intra_mode[1] = aom_read_symbol(
        r, ec_ctx->uv_derived_intra_mode_cdf[mbmi->use_derived_intra_mode[0]],
        2, ACCT_STR);
  }
  if (mbmi->use_derived_intra_mode[1]) {
    // TODO(anyone): if derived mode has already been calculated for luma, we
    // do not need to repeat it.
    return av1_get_derived_intra_mode(xd, bsize, &mbmi->derived_angle);
  }
#if DERIVED_INTRA_MODE_NOPD
  if (mbmi->use_derived_intra_mode[0]) y_mode = DC_PRED;
#endif  // DERIVED_INTRA_MODE_NOPD
#endif  // CONFIG_DERIVED_INTRA_MODE
  const UV_PREDICTION_MODE uv_mode =
      aom_read_symbol(r, ec_ctx->uv_mode_cdf[cfl_allowed][y_mode],
                      UV_INTRA_MODES - !cfl_allowed, ACCT_STR);
  return uv_mode;
}

static uint8_t read_cfl_alphas(FRAME_CONTEXT *const ec_ctx, aom_reader *r,
                               int8_t *signs_out) {
  const int8_t joint_sign =
      aom_read_symbol(r, ec_ctx->cfl_sign_cdf, CFL_JOINT_SIGNS, "cfl:signs");
  uint8_t idx = 0;
  // Magnitudes are only coded for nonzero values
  if (CFL_SIGN_U(joint_sign) != CFL_SIGN_ZERO) {
    aom_cdf_prob *cdf_u = ec_ctx->cfl_alpha_cdf[CFL_CONTEXT_U(joint_sign)];
    idx = (uint8_t)aom_read_symbol(r, cdf_u, CFL_ALPHABET_SIZE, "cfl:alpha_u")
          << CFL_ALPHABET_SIZE_LOG2;
  }
  if (CFL_SIGN_V(joint_sign) != CFL_SIGN_ZERO) {
    aom_cdf_prob *cdf_v = ec_ctx->cfl_alpha_cdf[CFL_CONTEXT_V(joint_sign)];
    idx += (uint8_t)aom_read_symbol(r, cdf_v, CFL_ALPHABET_SIZE, "cfl:alpha_v");
  }
  *signs_out = joint_sign;
  return idx;
}

static INTERINTRA_MODE read_interintra_mode(MACROBLOCKD *xd, aom_reader *r,
                                            int size_group) {
  const INTERINTRA_MODE ii_mode = (INTERINTRA_MODE)aom_read_symbol(
      r, xd->tile_ctx->interintra_mode_cdf[size_group], INTERINTRA_MODES,
      ACCT_STR);
  return ii_mode;
}

static PREDICTION_MODE read_inter_mode(FRAME_CONTEXT *ec_ctx, aom_reader *r,
                                       int16_t ctx) {
#if CONFIG_NEW_INTER_MODES
  const int16_t ismode_ctx = inter_single_mode_ctx(ctx);
  return SINGLE_INTER_MODE_START +
         aom_read_symbol(r, ec_ctx->inter_single_mode_cdf[ismode_ctx],
                         INTER_SINGLE_MODES, ACCT_STR);
#else
  int16_t mode_ctx = ctx & NEWMV_CTX_MASK;
  int is_newmv, is_zeromv, is_refmv;
  is_newmv = aom_read_symbol(r, ec_ctx->newmv_cdf[mode_ctx], 2, ACCT_STR) == 0;
  if (is_newmv) return NEWMV;

  mode_ctx = (ctx >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK;
  is_zeromv =
      aom_read_symbol(r, ec_ctx->zeromv_cdf[mode_ctx], 2, ACCT_STR) == 0;
  if (is_zeromv) return GLOBALMV;

  mode_ctx = (ctx >> REFMV_OFFSET) & REFMV_CTX_MASK;
  is_refmv = aom_read_symbol(r, ec_ctx->refmv_cdf[mode_ctx], 2, ACCT_STR) == 0;
  if (is_refmv)
    return NEARESTMV;
  else
    return NEARMV;
#endif  // CONFIG_NEW_INTER_MODES
}

#if CONFIG_NEW_INTER_MODES
static void read_drl_idx(int max_drl_bits, const int16_t mode_ctx,
                         FRAME_CONTEXT *ec_ctx, DecoderCodingBlock *dcb,
                         MB_MODE_INFO *mbmi, aom_reader *r) {
  MACROBLOCKD *const xd = &dcb->xd;
#if CONFIG_NEW_REF_SIGNALING
  uint8_t ref_frame_type = av1_ref_frame_type_nrs(mbmi->ref_frame_nrs);
#else
  uint8_t ref_frame_type = av1_ref_frame_type(mbmi->ref_frame);
#endif  // CONFIG_NEW_REF_SIGNALING
  mbmi->ref_mv_idx = 0;
  assert(!mbmi->skip_mode);
  const int range = av1_drl_range(dcb->ref_mv_count[ref_frame_type],
                                  mode_ctx >> 8, max_drl_bits);
  for (int idx = 0; idx < range; ++idx) {
    aom_cdf_prob *drl_cdf =
        av1_get_drl_cdf(ec_ctx, xd->weight[ref_frame_type], mode_ctx, idx);
    int drl_idx = aom_read_symbol(r, drl_cdf, 2, ACCT_STR);
    mbmi->ref_mv_idx = idx + drl_idx;
    if (!drl_idx) break;
  }
  assert(mbmi->ref_mv_idx < max_drl_bits + 1);
}
#else
static void read_drl_idx(FRAME_CONTEXT *ec_ctx, DecoderCodingBlock *dcb,
                         MB_MODE_INFO *mbmi, aom_reader *r) {
  MACROBLOCKD *const xd = &dcb->xd;
#if CONFIG_NEW_REF_SIGNALING
  uint8_t ref_frame_type = av1_ref_frame_type_nrs(mbmi->ref_frame_nrs);
#else
  uint8_t ref_frame_type = av1_ref_frame_type(mbmi->ref_frame);
#endif  // CONFIG_NEW_REF_SIGNALING
  mbmi->ref_mv_idx = 0;
  if (mbmi->mode == NEWMV || mbmi->mode == NEW_NEWMV) {
    for (int idx = 0; idx < MAX_DRL_BITS; ++idx) {
      if (dcb->ref_mv_count[ref_frame_type] > idx + 1) {
        uint8_t drl_ctx = av1_drl_ctx(xd->weight[ref_frame_type], idx);
        int drl_idx = aom_read_symbol(r, ec_ctx->drl_cdf[drl_ctx], 2, ACCT_STR);
        mbmi->ref_mv_idx = idx + drl_idx;
        if (!drl_idx) return;
      }
    }
  }
  if (have_nearmv_in_inter_mode(mbmi->mode)) {
    // Offset the NEARESTMV mode.
    // TODO(jingning): Unify the two syntax decoding loops after the NEARESTMV
    // mode is factored in.
    for (int idx = 1; idx < MAX_DRL_BITS + 1; ++idx) {
      if (dcb->ref_mv_count[ref_frame_type] > idx + 1) {
        uint8_t drl_ctx = av1_drl_ctx(xd->weight[ref_frame_type], idx);
        int drl_idx = aom_read_symbol(r, ec_ctx->drl_cdf[drl_ctx], 2, ACCT_STR);
        mbmi->ref_mv_idx = idx + drl_idx - 1;
        if (!drl_idx) return;
      }
    }
  }
}
#endif  // CONFIG_NEW_INTER_MODES

static MOTION_MODE read_motion_mode(AV1_COMMON *cm, MACROBLOCKD *xd,
                                    MB_MODE_INFO *mbmi, aom_reader *r) {
  if (cm->features.switchable_motion_mode == 0) return SIMPLE_TRANSLATION;
  if (mbmi->skip_mode) return SIMPLE_TRANSLATION;

#if CONFIG_NEW_REF_SIGNALING
  const MOTION_MODE last_motion_mode_allowed = motion_mode_allowed_nrs(
      xd->global_motion_nrs, xd, mbmi, cm->features.allow_warped_motion);
#else
  const MOTION_MODE last_motion_mode_allowed = motion_mode_allowed(
      xd->global_motion, xd, mbmi, cm->features.allow_warped_motion);
#endif  // CONFIG_NEW_REF_SIGNALING
  int motion_mode;

  if (last_motion_mode_allowed == SIMPLE_TRANSLATION) return SIMPLE_TRANSLATION;

  if (last_motion_mode_allowed == OBMC_CAUSAL) {
    motion_mode =
        aom_read_symbol(r, xd->tile_ctx->obmc_cdf[mbmi->sb_type], 2, ACCT_STR);
    return (MOTION_MODE)(SIMPLE_TRANSLATION + motion_mode);
  } else {
    motion_mode =
        aom_read_symbol(r, xd->tile_ctx->motion_mode_cdf[mbmi->sb_type],
                        MOTION_MODES, ACCT_STR);
#if CONFIG_EXT_ROTATION
    if (motion_mode == WARPED_CAUSAL) {
      mbmi->rot_flag = aom_read_symbol(
          r, xd->tile_ctx->warp_rotation_flag_cdf[mbmi->sb_type], 2, ACCT_STR);
      if (mbmi->rot_flag) {
        mbmi->rotation =
            aom_read_symbol(r, xd->tile_ctx->warp_rotation_degree_cdf,
                            ROTATION_COUNT, ACCT_STR) *
                ROTATION_STEP -
            ROTATION_RANGE;
      }
    }
#endif  // CONFIG_EXT_ROTATION
    return (MOTION_MODE)(SIMPLE_TRANSLATION + motion_mode);
  }
}

#if CONFIG_EXT_ROTATION
static void read_warp_rotation(MACROBLOCKD *xd, MB_MODE_INFO *mbmi,
                               aom_reader *r) {
  if (globalmv_rotation_allowed(xd)) {
    mbmi->rot_flag = aom_read_symbol(
        r, xd->tile_ctx->globalmv_rotation_flag_cdf[mbmi->sb_type], 2,
        ACCT_STR);
    if (mbmi->rot_flag) {
      mbmi->rotation =
          aom_read_symbol(r, xd->tile_ctx->globalmv_rotation_degree_cdf,
                          ROTATION_COUNT, ACCT_STR) *
              ROTATION_STEP -
          ROTATION_RANGE;
    }
  } else if (simple_translation_rotation_allowed(mbmi)) {
    mbmi->rot_flag = aom_read_symbol(
        r, xd->tile_ctx->translation_rotation_flag_cdf[mbmi->sb_type], 2,
        ACCT_STR);
    if (mbmi->rot_flag) {
      mbmi->rotation =
          aom_read_symbol(r, xd->tile_ctx->translation_rotation_degree_cdf,
                          ROTATION_COUNT, ACCT_STR) *
              ROTATION_STEP -
          ROTATION_RANGE;
    }
  }
}
#endif  // CONFIG_EXT_ROTATION

static PREDICTION_MODE read_inter_compound_mode(MACROBLOCKD *xd, aom_reader *r,
                                                int16_t ctx) {
  const int mode =
      aom_read_symbol(r, xd->tile_ctx->inter_compound_mode_cdf[ctx],
                      INTER_COMPOUND_MODES, ACCT_STR);
#if CONFIG_NEW_INTER_MODES
  assert(is_inter_compound_mode(NEAR_NEARMV + mode));
  return NEAR_NEARMV + mode;
#else
  assert(is_inter_compound_mode(NEAREST_NEARESTMV + mode));
  return NEAREST_NEARESTMV + mode;
#endif  // CONFIG_NEW_INTER_MODES
}

int av1_neg_deinterleave(int diff, int ref, int max) {
  if (!ref) return diff;
  if (ref >= (max - 1)) return max - diff - 1;
  if (2 * ref < max) {
    if (diff <= 2 * ref) {
      if (diff & 1)
        return ref + ((diff + 1) >> 1);
      else
        return ref - (diff >> 1);
    }
    return diff;
  } else {
    if (diff <= 2 * (max - ref - 1)) {
      if (diff & 1)
        return ref + ((diff + 1) >> 1);
      else
        return ref - (diff >> 1);
    }
    return max - (diff + 1);
  }
}

static int read_segment_id(AV1_COMMON *const cm, const MACROBLOCKD *const xd,
                           aom_reader *r, int skip) {
  int cdf_num;
  const int pred = av1_get_spatial_seg_pred(cm, xd, &cdf_num);
  if (skip) return pred;

  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  struct segmentation *const seg = &cm->seg;
  struct segmentation_probs *const segp = &ec_ctx->seg;
  aom_cdf_prob *pred_cdf = segp->spatial_pred_seg_cdf[cdf_num];
  const int coded_id = aom_read_symbol(r, pred_cdf, MAX_SEGMENTS, ACCT_STR);
  const int segment_id =
      av1_neg_deinterleave(coded_id, pred, seg->last_active_segid + 1);

  if (segment_id < 0 || segment_id > seg->last_active_segid) {
    aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                       "Corrupted segment_ids");
  }
  return segment_id;
}

static int dec_get_segment_id(const AV1_COMMON *cm, const uint8_t *segment_ids,
                              int mi_offset, int x_mis, int y_mis) {
  int segment_id = INT_MAX;

  for (int y = 0; y < y_mis; y++)
    for (int x = 0; x < x_mis; x++)
      segment_id = AOMMIN(
          segment_id, segment_ids[mi_offset + y * cm->mi_params.mi_cols + x]);

  assert(segment_id >= 0 && segment_id < MAX_SEGMENTS);
  return segment_id;
}

static void set_segment_id(AV1_COMMON *cm, int mi_offset, int x_mis, int y_mis,
                           int segment_id) {
  assert(segment_id >= 0 && segment_id < MAX_SEGMENTS);

  for (int y = 0; y < y_mis; y++)
    for (int x = 0; x < x_mis; x++)
      cm->cur_frame->seg_map[mi_offset + y * cm->mi_params.mi_cols + x] =
          segment_id;
}

static int read_intra_segment_id(AV1_COMMON *const cm,
                                 const MACROBLOCKD *const xd, int bsize,
                                 aom_reader *r, int skip) {
  struct segmentation *const seg = &cm->seg;
  if (!seg->enabled) return 0;  // Default for disabled segmentation
  assert(seg->update_map && !seg->temporal_update);

  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int mi_offset = mi_row * mi_params->mi_cols + mi_col;
  const int bw = mi_size_wide[bsize];
  const int bh = mi_size_high[bsize];
  const int x_mis = AOMMIN(mi_params->mi_cols - mi_col, bw);
  const int y_mis = AOMMIN(mi_params->mi_rows - mi_row, bh);
  const int segment_id = read_segment_id(cm, xd, r, skip);
  set_segment_id(cm, mi_offset, x_mis, y_mis, segment_id);
  return segment_id;
}

static void copy_segment_id(const CommonModeInfoParams *const mi_params,
                            const uint8_t *last_segment_ids,
                            uint8_t *current_segment_ids, int mi_offset,
                            int x_mis, int y_mis) {
  for (int y = 0; y < y_mis; y++)
    for (int x = 0; x < x_mis; x++)
      current_segment_ids[mi_offset + y * mi_params->mi_cols + x] =
          last_segment_ids
              ? last_segment_ids[mi_offset + y * mi_params->mi_cols + x]
              : 0;
}

static int get_predicted_segment_id(AV1_COMMON *const cm, int mi_offset,
                                    int x_mis, int y_mis) {
  return cm->last_frame_seg_map ? dec_get_segment_id(cm, cm->last_frame_seg_map,
                                                     mi_offset, x_mis, y_mis)
                                : 0;
}

static int read_inter_segment_id(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                                 int preskip, aom_reader *r) {
  struct segmentation *const seg = &cm->seg;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  const int mi_offset = mi_row * mi_params->mi_cols + mi_col;
  const int bw = mi_size_wide[mbmi->sb_type];
  const int bh = mi_size_high[mbmi->sb_type];

  // TODO(slavarnway): move x_mis, y_mis into xd ?????
  const int x_mis = AOMMIN(mi_params->mi_cols - mi_col, bw);
  const int y_mis = AOMMIN(mi_params->mi_rows - mi_row, bh);

  if (!seg->enabled) return 0;  // Default for disabled segmentation

  if (!seg->update_map) {
    copy_segment_id(mi_params, cm->last_frame_seg_map, cm->cur_frame->seg_map,
                    mi_offset, x_mis, y_mis);
    return get_predicted_segment_id(cm, mi_offset, x_mis, y_mis);
  }

  int segment_id;
  if (preskip) {
    if (!seg->segid_preskip) return 0;
  } else {
    if (mbmi->skip_txfm) {
      if (seg->temporal_update) {
        mbmi->seg_id_predicted = 0;
      }
      segment_id = read_segment_id(cm, xd, r, 1);
      set_segment_id(cm, mi_offset, x_mis, y_mis, segment_id);
      return segment_id;
    }
  }

  if (seg->temporal_update) {
    const int ctx = av1_get_pred_context_seg_id(xd);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    struct segmentation_probs *const segp = &ec_ctx->seg;
    aom_cdf_prob *pred_cdf = segp->pred_cdf[ctx];
    mbmi->seg_id_predicted = aom_read_symbol(r, pred_cdf, 2, ACCT_STR);
    if (mbmi->seg_id_predicted) {
      segment_id = get_predicted_segment_id(cm, mi_offset, x_mis, y_mis);
    } else {
      segment_id = read_segment_id(cm, xd, r, 0);
    }
  } else {
    segment_id = read_segment_id(cm, xd, r, 0);
  }
  set_segment_id(cm, mi_offset, x_mis, y_mis, segment_id);
  return segment_id;
}

static int read_skip_mode(AV1_COMMON *cm, const MACROBLOCKD *xd, int segment_id,
                          aom_reader *r) {
  if (!cm->current_frame.skip_mode_info.skip_mode_flag) return 0;

  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP)) {
    return 0;
  }

  if (!is_comp_ref_allowed(xd->mi[0]->sb_type)) return 0;

#if CONFIG_NEW_REF_SIGNALING
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
#else
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME) ||
      segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
#endif  // CONFIG_NEW_REF_SIGNALING
    // These features imply single-reference mode, while skip mode implies
    // compound reference. Hence, the two are mutually exclusive.
    // In other words, skip_mode is implicitly 0 here.
    return 0;
  }

  const int ctx = av1_get_skip_mode_context(xd);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const int skip_mode =
      aom_read_symbol(r, ec_ctx->skip_mode_cdfs[ctx], 2, ACCT_STR);
  return skip_mode;
}

static int read_skip_txfm(AV1_COMMON *cm, const MACROBLOCKD *xd, int segment_id,
                          aom_reader *r) {
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP)) {
    return 1;
  } else {
    const int ctx = av1_get_skip_txfm_context(xd);
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    const int skip_txfm =
        aom_read_symbol(r, ec_ctx->skip_txfm_cdfs[ctx], 2, ACCT_STR);
    return skip_txfm;
  }
}

// Merge the sorted list of cached colors(cached_colors[0...n_cached_colors-1])
// and the sorted list of transmitted colors(colors[n_cached_colors...n-1]) into
// one single sorted list(colors[...]).
static void merge_colors(uint16_t *colors, uint16_t *cached_colors,
                         int n_colors, int n_cached_colors) {
  if (n_cached_colors == 0) return;
  int cache_idx = 0, trans_idx = n_cached_colors;
  for (int i = 0; i < n_colors; ++i) {
    if (cache_idx < n_cached_colors &&
        (trans_idx >= n_colors ||
         cached_colors[cache_idx] <= colors[trans_idx])) {
      colors[i] = cached_colors[cache_idx++];
    } else {
      assert(trans_idx < n_colors);
      colors[i] = colors[trans_idx++];
    }
  }
}

static void read_palette_colors_y(MACROBLOCKD *const xd, int bit_depth,
                                  PALETTE_MODE_INFO *const pmi, aom_reader *r) {
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  uint16_t cached_colors[PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 0, color_cache);
  const int n = pmi->palette_size[0];
  int idx = 0;
  for (int i = 0; i < n_cache && idx < n; ++i)
    if (aom_read_bit(r, ACCT_STR)) cached_colors[idx++] = color_cache[i];
  if (idx < n) {
    const int n_cached_colors = idx;
    pmi->palette_colors[idx++] = aom_read_literal(r, bit_depth, ACCT_STR);
    if (idx < n) {
      const int min_bits = bit_depth - 3;
      int bits = min_bits + aom_read_literal(r, 2, ACCT_STR);
      int range = (1 << bit_depth) - pmi->palette_colors[idx - 1] - 1;
      for (; idx < n; ++idx) {
        assert(range >= 0);
        const int delta = aom_read_literal(r, bits, ACCT_STR) + 1;
        pmi->palette_colors[idx] = clamp(pmi->palette_colors[idx - 1] + delta,
                                         0, (1 << bit_depth) - 1);
        range -= (pmi->palette_colors[idx] - pmi->palette_colors[idx - 1]);
        bits = AOMMIN(bits, av1_ceil_log2(range));
      }
    }
    merge_colors(pmi->palette_colors, cached_colors, n, n_cached_colors);
  } else {
    memcpy(pmi->palette_colors, cached_colors, n * sizeof(cached_colors[0]));
  }
}

static void read_palette_colors_uv(MACROBLOCKD *const xd, int bit_depth,
                                   PALETTE_MODE_INFO *const pmi,
                                   aom_reader *r) {
  const int n = pmi->palette_size[1];
  // U channel colors.
  uint16_t color_cache[2 * PALETTE_MAX_SIZE];
  uint16_t cached_colors[PALETTE_MAX_SIZE];
  const int n_cache = av1_get_palette_cache(xd, 1, color_cache);
  int idx = 0;
  for (int i = 0; i < n_cache && idx < n; ++i)
    if (aom_read_bit(r, ACCT_STR)) cached_colors[idx++] = color_cache[i];
  if (idx < n) {
    const int n_cached_colors = idx;
    idx += PALETTE_MAX_SIZE;
    pmi->palette_colors[idx++] = aom_read_literal(r, bit_depth, ACCT_STR);
    if (idx < PALETTE_MAX_SIZE + n) {
      const int min_bits = bit_depth - 3;
      int bits = min_bits + aom_read_literal(r, 2, ACCT_STR);
      int range = (1 << bit_depth) - pmi->palette_colors[idx - 1];
      for (; idx < PALETTE_MAX_SIZE + n; ++idx) {
        assert(range >= 0);
        const int delta = aom_read_literal(r, bits, ACCT_STR);
        pmi->palette_colors[idx] = clamp(pmi->palette_colors[idx - 1] + delta,
                                         0, (1 << bit_depth) - 1);
        range -= (pmi->palette_colors[idx] - pmi->palette_colors[idx - 1]);
        bits = AOMMIN(bits, av1_ceil_log2(range));
      }
    }
    merge_colors(pmi->palette_colors + PALETTE_MAX_SIZE, cached_colors, n,
                 n_cached_colors);
  } else {
    memcpy(pmi->palette_colors + PALETTE_MAX_SIZE, cached_colors,
           n * sizeof(cached_colors[0]));
  }

  // V channel colors.
  if (aom_read_bit(r, ACCT_STR)) {  // Delta encoding.
    const int min_bits_v = bit_depth - 4;
    const int max_val = 1 << bit_depth;
    int bits = min_bits_v + aom_read_literal(r, 2, ACCT_STR);
    pmi->palette_colors[2 * PALETTE_MAX_SIZE] =
        aom_read_literal(r, bit_depth, ACCT_STR);
    for (int i = 1; i < n; ++i) {
      int delta = aom_read_literal(r, bits, ACCT_STR);
      if (delta && aom_read_bit(r, ACCT_STR)) delta = -delta;
      int val = (int)pmi->palette_colors[2 * PALETTE_MAX_SIZE + i - 1] + delta;
      if (val < 0) val += max_val;
      if (val >= max_val) val -= max_val;
      pmi->palette_colors[2 * PALETTE_MAX_SIZE + i] = val;
    }
  } else {
    for (int i = 0; i < n; ++i) {
      pmi->palette_colors[2 * PALETTE_MAX_SIZE + i] =
          aom_read_literal(r, bit_depth, ACCT_STR);
    }
  }
}

static void read_palette_mode_info(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                                   aom_reader *r) {
  const int num_planes = av1_num_planes(cm);
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const BLOCK_SIZE bsize = mbmi->sb_type;
  assert(av1_allow_palette(cm->features.allow_screen_content_tools, bsize));
  PALETTE_MODE_INFO *const pmi = &mbmi->palette_mode_info;
  const int bsize_ctx = av1_get_palette_bsize_ctx(bsize);

  if (mbmi->mode == DC_PRED) {
    const int palette_mode_ctx = av1_get_palette_mode_ctx(xd);
    const int modev = aom_read_symbol(
        r, xd->tile_ctx->palette_y_mode_cdf[bsize_ctx][palette_mode_ctx], 2,
        ACCT_STR);
    if (modev) {
      pmi->palette_size[0] =
          aom_read_symbol(r, xd->tile_ctx->palette_y_size_cdf[bsize_ctx],
                          PALETTE_SIZES, ACCT_STR) +
          2;
      read_palette_colors_y(xd, cm->seq_params.bit_depth, pmi, r);
    }
  }
  if (num_planes > 1 && mbmi->uv_mode == UV_DC_PRED && xd->is_chroma_ref) {
    const int palette_uv_mode_ctx = (pmi->palette_size[0] > 0);
    const int modev = aom_read_symbol(
        r, xd->tile_ctx->palette_uv_mode_cdf[palette_uv_mode_ctx], 2, ACCT_STR);
    if (modev) {
      pmi->palette_size[1] =
          aom_read_symbol(r, xd->tile_ctx->palette_uv_size_cdf[bsize_ctx],
                          PALETTE_SIZES, ACCT_STR) +
          2;
      read_palette_colors_uv(xd, cm->seq_params.bit_depth, pmi, r);
    }
  }
}

static int read_angle_delta(aom_reader *r, aom_cdf_prob *cdf) {
  const int sym = aom_read_symbol(r, cdf, 2 * MAX_ANGLE_DELTA + 1, ACCT_STR);
  return sym - MAX_ANGLE_DELTA;
}

static void read_filter_intra_mode_info(const AV1_COMMON *const cm,
                                        MACROBLOCKD *const xd, aom_reader *r) {
  MB_MODE_INFO *const mbmi = xd->mi[0];
  FILTER_INTRA_MODE_INFO *filter_intra_mode_info =
      &mbmi->filter_intra_mode_info;

  if (av1_filter_intra_allowed(cm, mbmi)) {
    filter_intra_mode_info->use_filter_intra = aom_read_symbol(
        r, xd->tile_ctx->filter_intra_cdfs[mbmi->sb_type], 2, ACCT_STR);
    if (filter_intra_mode_info->use_filter_intra) {
      filter_intra_mode_info->filter_intra_mode = aom_read_symbol(
          r, xd->tile_ctx->filter_intra_mode_cdf, FILTER_INTRA_MODES, ACCT_STR);
    }
  } else {
    filter_intra_mode_info->use_filter_intra = 0;
  }
}

void av1_read_tx_type(const AV1_COMMON *const cm, MACROBLOCKD *xd, int blk_row,
                      int blk_col, TX_SIZE tx_size, aom_reader *r) {
  MB_MODE_INFO *mbmi = xd->mi[0];
  uint8_t *tx_type =
      &xd->tx_type_map[blk_row * xd->tx_type_map_stride + blk_col];
  *tx_type = DCT_DCT;

  // No need to read transform type if block is skipped.
  if (mbmi->skip_txfm ||
      segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP))
    return;

  // No need to read transform type for lossless mode(qindex==0).
  const int qindex = xd->qindex[mbmi->segment_id];
  if (qindex == 0) return;

  const int inter_block = is_inter_block(mbmi);
  if (get_ext_tx_types(tx_size, inter_block, cm->features.reduced_tx_set_used) >
      1) {
    const TxSetType tx_set_type = av1_get_ext_tx_set_type(
        tx_size, inter_block, cm->features.reduced_tx_set_used);
    const int eset =
        get_ext_tx_set(tx_size, inter_block, cm->features.reduced_tx_set_used);
    // eset == 0 should correspond to a set with only DCT_DCT and
    // there is no need to read the tx_type
    assert(eset != 0);

    const TX_SIZE square_tx_size = txsize_sqr_map[tx_size];
    FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
    if (inter_block) {
      *tx_type = av1_ext_tx_inv[tx_set_type][aom_read_symbol(
          r, ec_ctx->inter_ext_tx_cdf[eset][square_tx_size],
          av1_num_ext_tx_set[tx_set_type], ACCT_STR)];
    } else {
      const PREDICTION_MODE intra_mode =
          mbmi->filter_intra_mode_info.use_filter_intra
              ? fimode_to_intradir[mbmi->filter_intra_mode_info
                                       .filter_intra_mode]
              : mbmi->mode;
      *tx_type = av1_ext_tx_inv[tx_set_type][aom_read_symbol(
          r, ec_ctx->intra_ext_tx_cdf[eset][square_tx_size][intra_mode],
          av1_num_ext_tx_set[tx_set_type], ACCT_STR)];
    }
  }
}

static INLINE void read_mv(aom_reader *r, MV *mv, MV ref, nmv_context *ctx,
                           MvSubpelPrecision precision);

static INLINE int is_mv_valid(const MV *mv);

static INLINE int assign_dv(AV1_COMMON *cm, MACROBLOCKD *xd, int_mv *mv,
                            const int_mv *ref_mv, int mi_row, int mi_col,
                            BLOCK_SIZE bsize, aom_reader *r) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  read_mv(r, &mv->as_mv, ref_mv->as_mv, &ec_ctx->ndvc, MV_SUBPEL_NONE);
  // DV should not have sub-pel.
  assert((mv->as_mv.col & 7) == 0);
  assert((mv->as_mv.row & 7) == 0);
  mv->as_mv.col = (mv->as_mv.col >> 3) * 8;
  mv->as_mv.row = (mv->as_mv.row >> 3) * 8;
  int valid = is_mv_valid(&mv->as_mv) &&
              av1_is_dv_valid(mv->as_mv, cm, xd, mi_row, mi_col, bsize,
                              cm->seq_params.mib_size_log2);
  return valid;
}

static void read_intrabc_info(AV1_COMMON *const cm, DecoderCodingBlock *dcb,
                              aom_reader *r) {
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  mbmi->use_intrabc = aom_read_symbol(r, ec_ctx->intrabc_cdf, 2, ACCT_STR);
  if (mbmi->use_intrabc) {
    BLOCK_SIZE bsize = mbmi->sb_type;
    mbmi->mode = DC_PRED;
    mbmi->uv_mode = UV_DC_PRED;
#if CONFIG_REMOVE_DUAL_FILTER
    mbmi->interp_fltr = BILINEAR;
#else
    mbmi->interp_filters = av1_broadcast_interp_filter(BILINEAR);
#endif  // CONFIG_REMOVE_DUAL_FILTER
    mbmi->motion_mode = SIMPLE_TRANSLATION;
    av1_set_mbmi_mv_precision(mbmi, MV_SUBPEL_NONE);

    int16_t inter_mode_ctx[MODE_CTX_REF_FRAMES];
    int_mv ref_mvs[INTRA_FRAME + 1][MAX_MV_REF_CANDIDATES];

#if CONFIG_NEW_REF_SIGNALING
    av1_find_mv_refs_nrs(cm, xd, mbmi, INTRA_FRAME_NRS, dcb->ref_mv_count,
                         xd->ref_mv_stack, xd->weight, ref_mvs,
                         /*global_mvs_nrs=*/NULL, inter_mode_ctx);
#else
    av1_find_mv_refs(cm, xd, mbmi, INTRA_FRAME, dcb->ref_mv_count,
                     xd->ref_mv_stack, xd->weight, ref_mvs, /*global_mvs=*/NULL,
                     inter_mode_ctx);
#endif  // CONFIG_NEW_REF_SIGNALING

    int_mv nearestmv, nearmv;

    av1_find_best_ref_mvs(ref_mvs[INTRA_FRAME], &nearestmv, &nearmv,
                          cm->features.fr_mv_precision);
    assert(cm->features.fr_mv_precision == MV_SUBPEL_NONE &&
           mbmi->max_mv_precision == MV_SUBPEL_NONE);
    int_mv dv_ref = nearestmv.as_int == 0 ? nearmv : nearestmv;
    if (dv_ref.as_int == 0)
      av1_find_ref_dv(&dv_ref, &xd->tile, cm->seq_params.mib_size, xd->mi_row);
    // Ref DV should not have sub-pel.
    int valid_dv = (dv_ref.as_mv.col & 7) == 0 && (dv_ref.as_mv.row & 7) == 0;
    dv_ref.as_mv.col = (dv_ref.as_mv.col >> 3) * 8;
    dv_ref.as_mv.row = (dv_ref.as_mv.row >> 3) * 8;
    valid_dv = valid_dv && assign_dv(cm, xd, &mbmi->mv[0], &dv_ref, xd->mi_row,
                                     xd->mi_col, bsize, r);
    if (!valid_dv) {
      // Intra bc motion vectors are not valid - signal corrupt frame
      aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                         "Invalid intrabc dv");
    }
  }
}

// If delta q is present, reads delta_q index.
// Also reads delta_q loop filter levels, if present.
static void read_delta_q_params(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                                aom_reader *r) {
  DeltaQInfo *const delta_q_info = &cm->delta_q_info;

  if (delta_q_info->delta_q_present_flag) {
    MB_MODE_INFO *const mbmi = xd->mi[0];
    xd->current_base_qindex +=
        read_delta_qindex(cm, xd, r, mbmi) * delta_q_info->delta_q_res;
    /* Normative: Clamp to [1,MAXQ] to not interfere with lossless mode */
#if CONFIG_EXTQUANT
    xd->current_base_qindex = clamp(
        xd->current_base_qindex, 1,
        cm->seq_params.bit_depth == AOM_BITS_8
            ? MAXQ_8_BITS
            : cm->seq_params.bit_depth == AOM_BITS_10 ? MAXQ_10_BITS : MAXQ);
#else
    xd->current_base_qindex = clamp(xd->current_base_qindex, 1, MAXQ);
#endif
    FRAME_CONTEXT *const ec_ctx = xd->tile_ctx;
    if (delta_q_info->delta_lf_present_flag) {
      const int mi_row = xd->mi_row;
      const int mi_col = xd->mi_col;
      if (delta_q_info->delta_lf_multi) {
        const int frame_lf_count =
            av1_num_planes(cm) > 1 ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2;
        for (int lf_id = 0; lf_id < frame_lf_count; ++lf_id) {
          const int tmp_lvl =
              xd->delta_lf[lf_id] +
              read_delta_lflevel(cm, r, ec_ctx->delta_lf_multi_cdf[lf_id], mbmi,
                                 mi_col, mi_row) *
                  delta_q_info->delta_lf_res;
          mbmi->delta_lf[lf_id] = xd->delta_lf[lf_id] =
              clamp(tmp_lvl, -MAX_LOOP_FILTER, MAX_LOOP_FILTER);
        }
      } else {
        const int tmp_lvl = xd->delta_lf_from_base +
                            read_delta_lflevel(cm, r, ec_ctx->delta_lf_cdf,
                                               mbmi, mi_col, mi_row) *
                                delta_q_info->delta_lf_res;
        mbmi->delta_lf_from_base = xd->delta_lf_from_base =
            clamp(tmp_lvl, -MAX_LOOP_FILTER, MAX_LOOP_FILTER);
      }
    }
  }
}

static void read_intra_frame_mode_info(AV1_COMMON *const cm,
                                       DecoderCodingBlock *dcb, aom_reader *r) {
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  const MB_MODE_INFO *above_mi = xd->above_mbmi;
  const MB_MODE_INFO *left_mi = xd->left_mbmi;
  const BLOCK_SIZE bsize = mbmi->sb_type;
  struct segmentation *const seg = &cm->seg;

  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  if (seg->segid_preskip)
    mbmi->segment_id = read_intra_segment_id(cm, xd, bsize, r, 0);

  mbmi->skip_txfm = read_skip_txfm(cm, xd, mbmi->segment_id, r);

  if (!seg->segid_preskip)
    mbmi->segment_id = read_intra_segment_id(cm, xd, bsize, r, mbmi->skip_txfm);

  read_cdef(cm, r, xd);

  read_delta_q_params(cm, xd, r);

  mbmi->current_qindex = xd->current_base_qindex;

#if CONFIG_NEW_REF_SIGNALING
  mbmi->ref_frame_nrs[0] = INTRA_FRAME_NRS;
  mbmi->ref_frame_nrs[1] = INVALID_IDX;
#endif  // CONFIG_NEW_REF_SIGNALING
  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->ref_frame[1] = NONE_FRAME;
  mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;
  mbmi->filter_intra_mode_info.use_filter_intra = 0;
#if CONFIG_DERIVED_INTRA_MODE
  mbmi->use_derived_intra_mode[0] = 0;
  mbmi->use_derived_intra_mode[1] = 0;
#endif  // CONFIG_DERIVED_INTRA_MODE

  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;
  xd->above_txfm_context = cm->above_contexts.txfm[xd->tile.tile_row] + mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (mi_row & MAX_MIB_MASK);

  if (av1_allow_intrabc(cm)) {
    read_intrabc_info(cm, dcb, r);
    if (is_intrabc_block(mbmi)) return;
  }

#if CONFIG_DERIVED_INTRA_MODE
  if (av1_enable_derived_intra_mode(xd, mbmi->sb_type)) {
    aom_cdf_prob *cdf =
        get_derived_intra_mode_cdf(ec_ctx, xd->above_mbmi, xd->left_mbmi);
    mbmi->use_derived_intra_mode[0] = aom_read_symbol(r, cdf, 2, ACCT_STR);
  }
  if (mbmi->use_derived_intra_mode[0]) {
    mbmi->mode = av1_get_derived_intra_mode(xd, bsize, &mbmi->derived_angle);
  } else {
    mbmi->mode = read_intra_mode(r, get_y_mode_cdf(ec_ctx, above_mi, left_mi));
  }
#else
  mbmi->mode = read_intra_mode(r, get_y_mode_cdf(ec_ctx, above_mi, left_mi));
#endif  // CONFIG_DERIVED_INTRA_MODE

  const int use_angle_delta = av1_use_angle_delta(bsize);
  mbmi->angle_delta[PLANE_TYPE_Y] =
      (use_angle_delta &&
#if CONFIG_DERIVED_INTRA_MODE
       !mbmi->use_derived_intra_mode[0] &&
#endif  // CONFIG_DERIVED_INTRA_MODE
       av1_is_directional_mode(mbmi->mode))
          ? read_angle_delta(r, ec_ctx->angle_delta_cdf[mbmi->mode - V_PRED])
          : 0;

  if (!cm->seq_params.monochrome && xd->is_chroma_ref) {
    mbmi->uv_mode = read_intra_mode_uv(ec_ctx, r,
#if CONFIG_DERIVED_INTRA_MODE
                                       xd,
#endif  // CONFIG_DERIVED_INTRA_MODE
                                       is_cfl_allowed(xd), mbmi->mode);
    if (mbmi->uv_mode == UV_CFL_PRED) {
      mbmi->cfl_alpha_idx = read_cfl_alphas(ec_ctx, r, &mbmi->cfl_alpha_signs);
    }
    mbmi->angle_delta[PLANE_TYPE_UV] =
        (use_angle_delta &&
#if CONFIG_DERIVED_INTRA_MODE
         !mbmi->use_derived_intra_mode[1] &&
#endif  // CONFIG_DERIVED_INTRA_MODE
         av1_is_directional_mode(get_uv_mode(mbmi->uv_mode)))
            ? read_angle_delta(r,
                               ec_ctx->angle_delta_cdf[mbmi->uv_mode - V_PRED])
            : 0;
  } else {
    // Avoid decoding angle_info if there is is no chroma prediction
    mbmi->uv_mode = UV_DC_PRED;
  }
  xd->cfl.store_y = store_cfl_required(cm, xd);

  if (av1_allow_palette(cm->features.allow_screen_content_tools, bsize))
    read_palette_mode_info(cm, xd, r);

  read_filter_intra_mode_info(cm, xd, r);
}

static int read_mv_component(aom_reader *r, nmv_component *mvcomp,
                             MvSubpelPrecision precision) {
  int mag, d, fr, hp;
  const int sign = aom_read_symbol(r, mvcomp->sign_cdf, 2, ACCT_STR);
  const int mv_class =
      aom_read_symbol(r, mvcomp->classes_cdf, MV_CLASSES, ACCT_STR);
  const int class0 = mv_class == MV_CLASS_0;

  // Integer part
  if (class0) {
    d = aom_read_symbol(r, mvcomp->class0_cdf, CLASS0_SIZE, ACCT_STR);
    mag = 0;
  } else {
    const int n = mv_class + CLASS0_BITS - 1;  // number of bits
    d = 0;
    for (int i = 0; i < n; ++i)
      d |= aom_read_symbol(r, mvcomp->bits_cdf[i], 2, ACCT_STR) << i;
    mag = CLASS0_SIZE << (mv_class + 2);
  }

  if (precision > MV_SUBPEL_NONE) {
    // 1/2 and 1/4 pel parts
#if CONFIG_FLEX_MVRES
    fr = aom_read_symbol(
             r, class0 ? mvcomp->class0_fp_cdf[d][0] : mvcomp->fp_cdf[0], 2,
             ACCT_STR)
         << 1;
    fr += precision > MV_SUBPEL_HALF_PRECISION
              ? aom_read_symbol(r,
                                class0 ? mvcomp->class0_fp_cdf[d][1 + (fr >> 1)]
                                       : mvcomp->fp_cdf[1 + (fr >> 1)],
                                2, ACCT_STR)
              : 1;
#else
    fr = aom_read_symbol(r, class0 ? mvcomp->class0_fp_cdf[d] : mvcomp->fp_cdf,
                         MV_FP_SIZE, ACCT_STR);
#endif  // CONFIG_FLEX_MVRES

    // 1/8 pel part (if hp is not used, the default value of the hp is 1)
    hp = (precision > MV_SUBPEL_QTR_PRECISION)
             ? aom_read_symbol(r,
                               class0 ? mvcomp->class0_hp_cdf : mvcomp->hp_cdf,
                               2, ACCT_STR)
             : 1;
  } else {
    fr = 3;
    hp = 1;
  }

  // Result
  mag += ((d << 3) | (fr << 1) | hp) + 1;
  return sign ? -mag : mag;
}

static INLINE void read_mv(aom_reader *r, MV *mv, MV ref, nmv_context *ctx,
                           MvSubpelPrecision precision) {
  MV diff = kZeroMv;
  const MV_JOINT_TYPE joint_type =
      (MV_JOINT_TYPE)aom_read_symbol(r, ctx->joints_cdf, MV_JOINTS, ACCT_STR);
  if (mv_joint_vertical(joint_type))
    diff.row = read_mv_component(r, &ctx->comps[0], precision);

  if (mv_joint_horizontal(joint_type))
    diff.col = read_mv_component(r, &ctx->comps[1], precision);

#if CONFIG_FLEX_MVRES
  lower_mv_precision(&ref, precision);
#endif  // CONFIG_FLEX_MVRES

  mv->row = ref.row + diff.row;
  mv->col = ref.col + diff.col;
}

static REFERENCE_MODE read_block_reference_mode(AV1_COMMON *cm,
                                                const MACROBLOCKD *xd,
                                                aom_reader *r) {
  if (!is_comp_ref_allowed(xd->mi[0]->sb_type)) return SINGLE_REFERENCE;
  if (cm->current_frame.reference_mode == REFERENCE_MODE_SELECT) {
    const int ctx = av1_get_reference_mode_context(xd);
    const REFERENCE_MODE mode = (REFERENCE_MODE)aom_read_symbol(
        r, xd->tile_ctx->comp_inter_cdf[ctx], 2, ACCT_STR);
    return mode;  // SINGLE_REFERENCE or COMPOUND_REFERENCE
  } else {
    assert(cm->current_frame.reference_mode == SINGLE_REFERENCE);
    return cm->current_frame.reference_mode;
  }
}

#define READ_REF_BIT(pname) \
  aom_read_symbol(r, av1_get_pred_cdf_##pname(xd), 2, ACCT_STR)

#if !CONFIG_NEW_REF_SIGNALING
static COMP_REFERENCE_TYPE read_comp_reference_type(const MACROBLOCKD *xd,
                                                    aom_reader *r) {
  const int ctx = av1_get_comp_reference_type_context(xd);
  const COMP_REFERENCE_TYPE comp_ref_type =
      (COMP_REFERENCE_TYPE)aom_read_symbol(
          r, xd->tile_ctx->comp_ref_type_cdf[ctx], 2, ACCT_STR);
  return comp_ref_type;  // UNIDIR_COMP_REFERENCE or BIDIR_COMP_REFERENCE
}
#endif  // !CONFIG_NEW_REF_SIGNALING

#if CONFIG_NEW_REF_SIGNALING
static void set_ref_frames_for_skip_mode_nrs(
    AV1_COMMON *const cm, MV_REFERENCE_FRAME_NRS ref_frame_nrs[2]) {
  ref_frame_nrs[0] = cm->current_frame.skip_mode_info.ref_frame_idx_0;
  ref_frame_nrs[1] = cm->current_frame.skip_mode_info.ref_frame_idx_1;
}
#else
static void set_ref_frames_for_skip_mode(AV1_COMMON *const cm,
                                         MV_REFERENCE_FRAME ref_frame[2]) {
  ref_frame[0] = LAST_FRAME + cm->current_frame.skip_mode_info.ref_frame_idx_0;
  ref_frame[1] = LAST_FRAME + cm->current_frame.skip_mode_info.ref_frame_idx_1;
}
#endif  // CONFIG_NEW_REF_SIGNALING

#if CONFIG_NEW_REF_SIGNALING
static AOM_INLINE void read_single_ref_nrs(
    MACROBLOCKD *const xd, MV_REFERENCE_FRAME_NRS ref_frame_nrs[2],
    const NewRefFramesData *const new_ref_frame_data, aom_reader *r) {
  const int n_refs = new_ref_frame_data->n_total_refs;
  for (int i = 0; i < n_refs - 1; i++) {
    const int bit = aom_read_symbol(
        r, av1_get_pred_cdf_single_ref_nrs(xd, i, n_refs), 2, ACCT_STR);
    if (bit) {
      ref_frame_nrs[0] = i;
      return;
    }
  }
  ref_frame_nrs[0] = n_refs - 1;
}

static AOM_INLINE void read_compound_ref_nrs(
    const MACROBLOCKD *xd, MV_REFERENCE_FRAME_NRS ref_frame_nrs[2],
    const NewRefFramesData *const new_ref_frame_data, aom_reader *r) {
  const int n_refs = new_ref_frame_data->n_total_refs;
  assert(n_refs >= 2);
  int n_bits = 0;
  for (int i = 0; i < n_refs + n_bits - 2; i++) {
    const int bit = aom_read_symbol(
        r, av1_get_pred_cdf_compound_ref_nrs(xd, i, n_refs), 2, ACCT_STR);
    if (bit) {
      ref_frame_nrs[n_bits++] = i;
      if (n_bits == 2) break;
    }
  }
  if (n_bits < 2) ref_frame_nrs[1] = n_refs - 1;
  if (n_bits < 1) ref_frame_nrs[0] = n_refs - 2;
#if !PURE_NEW_REF_SIGNALING
  const int swap_refs =
      skip_compound_search(convert_ranked_ref_to_named_ref_index(
                               new_ref_frame_data, ref_frame_nrs[0]),
                           convert_ranked_ref_to_named_ref_index(
                               new_ref_frame_data, ref_frame_nrs[1]));
  if (swap_refs) {
    MV_REFERENCE_FRAME_NRS tmp = ref_frame_nrs[0];
    ref_frame_nrs[0] = ref_frame_nrs[1];
    ref_frame_nrs[1] = tmp;
  }
#endif  // !USE_NEW_REF_SIGNALING
}
#endif  // CONFIG_NEW_REF_SIGNALING

// Read the referncence frame
static void read_ref_frames(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                            aom_reader *r, int segment_id,
#if CONFIG_NEW_REF_SIGNALING
                            MV_REFERENCE_FRAME_NRS ref_frame_nrs[2],
#endif  // CONFIG_NEW_REF_SIGNALING
                            MV_REFERENCE_FRAME ref_frame[2]) {
  if (xd->mi[0]->skip_mode) {
#if CONFIG_NEW_REF_SIGNALING
    set_ref_frames_for_skip_mode_nrs(cm, ref_frame_nrs);
    // TODO(sarahparker, debargha): When the entire function converts
    // to the new framework, remove the conversion to named refs below.
    ref_frame[0] = convert_ranked_ref_to_named_ref_index(
        &cm->new_ref_frame_data, ref_frame_nrs[0]);
    ref_frame[1] = convert_ranked_ref_to_named_ref_index(
        &cm->new_ref_frame_data, ref_frame_nrs[1]);
#else
    set_ref_frames_for_skip_mode(cm, ref_frame);
#endif  // CONFIG_NEW_REF_SIGNALING
    return;
  }

#if CONFIG_NEW_REF_SIGNALING
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP) ||
      segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
    ref_frame_nrs[0] = get_closest_pastcur_ref_index(cm);
    ref_frame_nrs[1] = INVALID_IDX;
    ref_frame[0] = convert_ranked_ref_to_named_ref_index(
        &cm->new_ref_frame_data, ref_frame_nrs[0]);
    ref_frame[1] = NONE_FRAME;
#else
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    ref_frame[0] = (MV_REFERENCE_FRAME)get_segdata(&cm->seg, segment_id,
                                                   SEG_LVL_REF_FRAME);
    ref_frame[1] = NONE_FRAME;
  } else if (segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP) ||
             segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
    ref_frame[0] = LAST_FRAME;
    ref_frame[1] = NONE_FRAME;
#endif  // CONFIG_NEW_REF_SIGNALING
  } else {
    const REFERENCE_MODE mode = read_block_reference_mode(cm, xd, r);
    if (mode == COMPOUND_REFERENCE) {
#if CONFIG_NEW_REF_SIGNALING
      read_compound_ref_nrs(xd, ref_frame_nrs, &cm->new_ref_frame_data, r);
      ref_frame[0] = convert_ranked_ref_to_named_ref_index(
          &cm->new_ref_frame_data, ref_frame_nrs[0]);
      ref_frame[1] = convert_ranked_ref_to_named_ref_index(
          &cm->new_ref_frame_data, ref_frame_nrs[1]);
#else
      const COMP_REFERENCE_TYPE comp_ref_type = read_comp_reference_type(xd, r);

      if (comp_ref_type == UNIDIR_COMP_REFERENCE) {
        const int bit = READ_REF_BIT(uni_comp_ref_p);
        if (bit) {
          ref_frame[0] = BWDREF_FRAME;
          ref_frame[1] = ALTREF_FRAME;
        } else {
          const int bit1 = READ_REF_BIT(uni_comp_ref_p1);
          if (bit1) {
            const int bit2 = READ_REF_BIT(uni_comp_ref_p2);
            if (bit2) {
              ref_frame[0] = LAST_FRAME;
              ref_frame[1] = GOLDEN_FRAME;
            } else {
              ref_frame[0] = LAST_FRAME;
              ref_frame[1] = LAST3_FRAME;
            }
          } else {
            ref_frame[0] = LAST_FRAME;
            ref_frame[1] = LAST2_FRAME;
          }
        }

        return;
      }

      assert(comp_ref_type == BIDIR_COMP_REFERENCE);

      const int idx = 1;
      const int bit = READ_REF_BIT(comp_ref_p);
      // Decode forward references.
      if (!bit) {
        const int bit1 = READ_REF_BIT(comp_ref_p1);
        ref_frame[!idx] = bit1 ? LAST2_FRAME : LAST_FRAME;
      } else {
        const int bit2 = READ_REF_BIT(comp_ref_p2);
        ref_frame[!idx] = bit2 ? GOLDEN_FRAME : LAST3_FRAME;
      }

      // Decode backward references.
      const int bit_bwd = READ_REF_BIT(comp_bwdref_p);
      if (!bit_bwd) {
        const int bit1_bwd = READ_REF_BIT(comp_bwdref_p1);
        ref_frame[idx] = bit1_bwd ? ALTREF2_FRAME : BWDREF_FRAME;
      } else {
        ref_frame[idx] = ALTREF_FRAME;
      }
#endif  // CONFIG_NEW_REF_SIGNALING
    } else if (mode == SINGLE_REFERENCE) {
#if CONFIG_NEW_REF_SIGNALING
      read_single_ref_nrs(xd, ref_frame_nrs, &cm->new_ref_frame_data, r);
      ref_frame_nrs[1] = INVALID_IDX;
      ref_frame[0] = convert_ranked_ref_to_named_ref_index(
          &cm->new_ref_frame_data, ref_frame_nrs[0]);
      ref_frame[1] = NONE_FRAME;
#else
      const int bit0 = READ_REF_BIT(single_ref_p1);
      if (bit0) {
        const int bit1 = READ_REF_BIT(single_ref_p2);
        if (!bit1) {
          const int bit5 = READ_REF_BIT(single_ref_p6);
          ref_frame[0] = bit5 ? ALTREF2_FRAME : BWDREF_FRAME;
        } else {
          ref_frame[0] = ALTREF_FRAME;
        }
      } else {
        const int bit2 = READ_REF_BIT(single_ref_p3);
        if (bit2) {
          const int bit4 = READ_REF_BIT(single_ref_p5);
          ref_frame[0] = bit4 ? GOLDEN_FRAME : LAST3_FRAME;
        } else {
          const int bit3 = READ_REF_BIT(single_ref_p4);
          ref_frame[0] = bit3 ? LAST2_FRAME : LAST_FRAME;
        }
      }

      ref_frame[1] = NONE_FRAME;
#endif  // CONFIG_NEW_REF_SIGNALING
    } else {
      assert(0 && "Invalid prediction mode.");
    }
  }
}

static INLINE void read_mb_interp_filter(const MACROBLOCKD *const xd,
                                         InterpFilter interp_filter,
#if !CONFIG_REMOVE_DUAL_FILTER
                                         bool enable_dual_filter,
#endif  // !CONFIG_REMOVE_DUAL_FILTER
                                         MB_MODE_INFO *const mbmi,
                                         aom_reader *r) {
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  if (!av1_is_interp_needed(xd)) {
    set_default_interp_filters(mbmi, interp_filter);
    return;
  }

  if (interp_filter != SWITCHABLE) {
#if CONFIG_REMOVE_DUAL_FILTER
    mbmi->interp_fltr = interp_filter;
#else
    mbmi->interp_filters = av1_broadcast_interp_filter(interp_filter);
#endif  // CONFIG_REMOVE_DUAL_FILTER
  } else {
#if CONFIG_REMOVE_DUAL_FILTER
    const int ctx = av1_get_pred_context_switchable_interp(xd, 0);
    const InterpFilter filter = (InterpFilter)aom_read_symbol(
        r, ec_ctx->switchable_interp_cdf[ctx], SWITCHABLE_FILTERS, ACCT_STR);
    mbmi->interp_fltr = filter;
#else
    InterpFilter ref0_filter[2] = { EIGHTTAP_REGULAR, EIGHTTAP_REGULAR };
    for (int dir = 0; dir < 2; ++dir) {
      const int ctx = av1_get_pred_context_switchable_interp(xd, dir);
      ref0_filter[dir] = (InterpFilter)aom_read_symbol(
          r, ec_ctx->switchable_interp_cdf[ctx], SWITCHABLE_FILTERS, ACCT_STR);
      if (!enable_dual_filter) {
        ref0_filter[1] = ref0_filter[0];
        break;
      }
    }
    // The index system works as: (0, 1) -> (vertical, horizontal) filter types
    mbmi->interp_filters.as_filters.x_filter = ref0_filter[1];
    mbmi->interp_filters.as_filters.y_filter = ref0_filter[0];
#endif  // CONFIG_REMOVE_DUAL_FILTER
  }
}

static void read_intra_block_mode_info(AV1_COMMON *const cm,
                                       MACROBLOCKD *const xd,
                                       MB_MODE_INFO *const mbmi,
                                       aom_reader *r) {
  const BLOCK_SIZE bsize = mbmi->sb_type;
  const int use_angle_delta = av1_use_angle_delta(bsize);

  mbmi->ref_frame[0] = INTRA_FRAME;
  mbmi->ref_frame[1] = NONE_FRAME;
#if CONFIG_NEW_REF_SIGNALING
  mbmi->ref_frame_nrs[0] = INTRA_FRAME_NRS;
  mbmi->ref_frame_nrs[1] = INVALID_IDX;
#endif  // CONFIG_NEW_REF_SIGNALING
#if CONFIG_DERIVED_INTRA_MODE
  mbmi->use_derived_intra_mode[0] = 0;
  mbmi->use_derived_intra_mode[1] = 0;
#endif  // CONFIG_DERIVED_INTRA_MODE

  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

#if CONFIG_DERIVED_INTRA_MODE
  if (av1_enable_derived_intra_mode(xd, mbmi->sb_type)) {
    aom_cdf_prob *cdf =
        get_derived_intra_mode_cdf(ec_ctx, xd->above_mbmi, xd->left_mbmi);
    mbmi->use_derived_intra_mode[0] = aom_read_symbol(r, cdf, 2, ACCT_STR);
  }
  if (mbmi->use_derived_intra_mode[0]) {
    mbmi->mode = av1_get_derived_intra_mode(xd, bsize, &mbmi->derived_angle);
  } else {
    mbmi->mode =
        read_intra_mode(r, ec_ctx->y_mode_cdf[size_group_lookup[bsize]]);
  }
#else
  mbmi->mode = read_intra_mode(r, ec_ctx->y_mode_cdf[size_group_lookup[bsize]]);
#endif  // CONFIG_DERIVED_INTRA_MODE

  mbmi->angle_delta[PLANE_TYPE_Y] =
      use_angle_delta &&
#if CONFIG_DERIVED_INTRA_MODE
              !mbmi->use_derived_intra_mode[0] &&
#endif  // CONFIG_DERIVED_INTRA_MODE
              av1_is_directional_mode(mbmi->mode)
          ? read_angle_delta(r, ec_ctx->angle_delta_cdf[mbmi->mode - V_PRED])
          : 0;
  if (!cm->seq_params.monochrome && xd->is_chroma_ref) {
    mbmi->uv_mode = read_intra_mode_uv(ec_ctx, r,
#if CONFIG_DERIVED_INTRA_MODE
                                       xd,
#endif  // CONFIG_DERIVED_INTRA_MODE
                                       is_cfl_allowed(xd), mbmi->mode);
    if (mbmi->uv_mode == UV_CFL_PRED) {
      mbmi->cfl_alpha_idx =
          read_cfl_alphas(xd->tile_ctx, r, &mbmi->cfl_alpha_signs);
    }
    mbmi->angle_delta[PLANE_TYPE_UV] =
        use_angle_delta &&
#if CONFIG_DERIVED_INTRA_MODE
                !mbmi->use_derived_intra_mode[1] &&
#endif  // CONFIG_DERIVED_INTRA_MODE
                av1_is_directional_mode(get_uv_mode(mbmi->uv_mode))
            ? read_angle_delta(r,
                               ec_ctx->angle_delta_cdf[mbmi->uv_mode - V_PRED])
            : 0;
  } else {
    // Avoid decoding angle_info if there is is no chroma prediction
    mbmi->uv_mode = UV_DC_PRED;
  }
  xd->cfl.store_y = store_cfl_required(cm, xd);

  mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;
  if (av1_allow_palette(cm->features.allow_screen_content_tools, bsize))
    read_palette_mode_info(cm, xd, r);

  read_filter_intra_mode_info(cm, xd, r);
}

static INLINE int is_mv_valid(const MV *mv) {
  return mv->row > MV_LOW && mv->row < MV_UPP && mv->col > MV_LOW &&
         mv->col < MV_UPP;
}

static INLINE int assign_mv(AV1_COMMON *cm, MACROBLOCKD *xd,
                            PREDICTION_MODE mode,
#if CONFIG_NEW_REF_SIGNALING
                            MV_REFERENCE_FRAME_NRS ref_frame_nrs[2],
#else
                            MV_REFERENCE_FRAME ref_frame[2],
#endif  // CONFIG_NEW_REF_SIGNALING
                            int_mv mv[2], int_mv ref_mv[2],
                            int_mv nearest_mv[2], int_mv near_mv[2],
                            int is_compound, MvSubpelPrecision precision,
                            aom_reader *r) {
#if CONFIG_NEW_INTER_MODES
  (void)nearest_mv;
#endif  // CONFIG_NEW_INTER_MODES

  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  MB_MODE_INFO *mbmi = xd->mi[0];
  BLOCK_SIZE bsize = mbmi->sb_type;
  FeatureFlags *const features = &cm->features;
  assert(IMPLIES(features->cur_frame_force_integer_mv,
                 precision == MV_SUBPEL_NONE));
  switch (mode) {
    case NEWMV: {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      read_mv(r, &mv[0].as_mv, ref_mv[0].as_mv, nmvc, precision);
      break;
    }
#if !CONFIG_NEW_INTER_MODES
    case NEARESTMV: {
      mv[0].as_int = nearest_mv[0].as_int;
      break;
    }
#endif  // !CONFIG_NEW_INTER_MODES
    case NEARMV: {
      mv[0].as_int = near_mv[0].as_int;
      break;
    }
    case GLOBALMV: {
#if CONFIG_NEW_REF_SIGNALING
      // TODO(sarahparker) Temporary assert, see aomedia:3060
      mv[0].as_int =
          gm_get_motion_vector(&cm->global_motion_nrs[ref_frame_nrs[0]],
                               features->fr_mv_precision, bsize, xd->mi_col,
                               xd->mi_row)
              .as_int;
#else
      mv[0].as_int = gm_get_motion_vector(&cm->global_motion[ref_frame[0]],
                                          features->fr_mv_precision, bsize,
                                          xd->mi_col, xd->mi_row)
                         .as_int;
#endif  // CONFIG_NEW_REF_SIGNALING
      break;
    }
    case NEW_NEWMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case NEW_NEWMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      assert(is_compound);
      for (int i = 0; i < 2; ++i) {
        nmv_context *const nmvc = &ec_ctx->nmvc;
        read_mv(r, &mv[i].as_mv, ref_mv[i].as_mv, nmvc, precision);
      }
      break;
    }
#if !CONFIG_NEW_INTER_MODES
    case NEAREST_NEARESTMV: {
      assert(is_compound);
      mv[0].as_int = nearest_mv[0].as_int;
      mv[1].as_int = nearest_mv[1].as_int;
      break;
    }
#endif  // !CONFIG_NEW_INTER_MODES
    case NEAR_NEARMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case NEAR_NEARMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      assert(is_compound);
      mv[0].as_int = near_mv[0].as_int;
      mv[1].as_int = near_mv[1].as_int;
      break;
    }
#if !CONFIG_NEW_INTER_MODES
    case NEW_NEARESTMV: {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      read_mv(r, &mv[0].as_mv, ref_mv[0].as_mv, nmvc, precision);
      assert(is_compound);
      mv[1].as_int = nearest_mv[1].as_int;
      break;
    }
    case NEAREST_NEWMV: {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      mv[0].as_int = nearest_mv[0].as_int;
      read_mv(r, &mv[1].as_mv, ref_mv[1].as_mv, nmvc, precision);
      assert(is_compound);
      break;
    }
#endif  // !CONFIG_NEW_INTER_MODES
    case NEAR_NEWMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case NEAR_NEWMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      mv[0].as_int = near_mv[0].as_int;
      read_mv(r, &mv[1].as_mv, ref_mv[1].as_mv, nmvc, precision);
      assert(is_compound);
      break;
    }
    case NEW_NEARMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case NEW_NEARMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      nmv_context *const nmvc = &ec_ctx->nmvc;
      read_mv(r, &mv[0].as_mv, ref_mv[0].as_mv, nmvc, precision);
      assert(is_compound);
      mv[1].as_int = near_mv[1].as_int;
      break;
    }
    case GLOBAL_GLOBALMV:
#if CONFIG_OPTFLOW_REFINEMENT
    case GLOBAL_GLOBALMV_OPTFLOW:
#endif  // CONFIG_OPTFLOW_REFINEMENT
    {
      assert(is_compound);
#if CONFIG_NEW_REF_SIGNALING
      // TODO(sarahparker) Temporary assert, see aomedia:3060
      mv[0].as_int =
          gm_get_motion_vector(&cm->global_motion_nrs[ref_frame_nrs[0]],
                               features->fr_mv_precision, bsize, xd->mi_col,
                               xd->mi_row)
              .as_int;
      mv[1].as_int =
          gm_get_motion_vector(&cm->global_motion_nrs[ref_frame_nrs[1]],
                               features->fr_mv_precision, bsize, xd->mi_col,
                               xd->mi_row)
              .as_int;
#else
      mv[0].as_int = gm_get_motion_vector(&cm->global_motion[ref_frame[0]],
                                          features->fr_mv_precision, bsize,
                                          xd->mi_col, xd->mi_row)
                         .as_int;
      mv[1].as_int = gm_get_motion_vector(&cm->global_motion[ref_frame[1]],
                                          features->fr_mv_precision, bsize,
                                          xd->mi_col, xd->mi_row)
                         .as_int;
#endif  // CONFIG_NEW_REF_SIGNALING
      break;
    }
    default: { return 0; }
  }

  int ret = is_mv_valid(&mv[0].as_mv);
  if (is_compound) {
    ret = ret && is_mv_valid(&mv[1].as_mv);
  }
  return ret;
}

static int read_is_inter_block(AV1_COMMON *const cm, MACROBLOCKD *const xd,
                               int segment_id, aom_reader *r) {
#if !CONFIG_NEW_REF_SIGNALING
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_REF_FRAME)) {
    const int frame = get_segdata(&cm->seg, segment_id, SEG_LVL_REF_FRAME);
    if (frame < LAST_FRAME) return 0;
    return frame != INTRA_FRAME;
  }
#endif  // !CONFIG_NEW_REF_SIGNALING
  if (segfeature_active(&cm->seg, segment_id, SEG_LVL_GLOBALMV)) {
    return 1;
  }
  const int ctx = av1_get_intra_inter_context(xd);
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;
  const int is_inter =
      aom_read_symbol(r, ec_ctx->intra_inter_cdf[ctx], 2, ACCT_STR);
  return is_inter;
}

#if DEC_MISMATCH_DEBUG
static void dec_dump_logs(AV1_COMMON *cm, MB_MODE_INFO *const mbmi, int mi_row,
                          int mi_col, int16_t mode_ctx) {
  int_mv mv[2] = { { 0 } };
  for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref)
    mv[ref].as_mv = mbmi->mv[ref].as_mv;

  const int16_t newmv_ctx = mode_ctx & NEWMV_CTX_MASK;
  int16_t zeromv_ctx = -1;
  int16_t refmv_ctx = -1;
  if (mbmi->mode != NEWMV) {
    zeromv_ctx = (mode_ctx >> GLOBALMV_OFFSET) & GLOBALMV_CTX_MASK;
    if (mbmi->mode != GLOBALMV)
      refmv_ctx = (mode_ctx >> REFMV_OFFSET) & REFMV_CTX_MASK;
  }

#define FRAME_TO_CHECK 11
  if (cm->current_frame.frame_number == FRAME_TO_CHECK && cm->show_frame == 1) {
    printf(
        "=== DECODER ===: "
        "Frame=%d, (mi_row,mi_col)=(%d,%d), skip_mode=%d, mode=%d, bsize=%d, "
        "show_frame=%d, mv[0]=(%d,%d), mv[1]=(%d,%d), ref[0]=%d, "
        "ref[1]=%d, motion_mode=%d, mode_ctx=%d, "
        "newmv_ctx=%d, zeromv_ctx=%d, refmv_ctx=%d, tx_size=%d\n",
        cm->current_frame.frame_number, mi_row, mi_col, mbmi->skip_mode,
        mbmi->mode, mbmi->sb_type, cm->show_frame, mv[0].as_mv.row,
        mv[0].as_mv.col, mv[1].as_mv.row, mv[1].as_mv.col,
#if CONFIG_NEW_REF_SIGNALING
        mbmi->ref_frame_nrs[0], mbmi->ref_frame_nrs[1],
#else
        mbmi->ref_frame[0], mbmi->ref_frame[1],
#endif  // CONFIG_NEW_REF_SIGNALING
        mbmi->motion_mode, mode_ctx, newmv_ctx, zeromv_ctx, refmv_ctx,
        mbmi->tx_size);
  }
}
#endif  // DEC_MISMATCH_DEBUG

static void read_inter_block_mode_info(AV1Decoder *const pbi,
                                       DecoderCodingBlock *dcb,
                                       MB_MODE_INFO *const mbmi,
                                       aom_reader *r) {
  AV1_COMMON *const cm = &pbi->common;
  FeatureFlags *const features = &cm->features;
  const BLOCK_SIZE bsize = mbmi->sb_type;
  const MvSubpelPrecision fr_mv_precision = features->fr_mv_precision;
  int_mv nearestmv[2], nearmv[2];
  int_mv ref_mvs[MODE_CTX_REF_FRAMES][MAX_MV_REF_CANDIDATES] = { { { 0 } } };
  int16_t inter_mode_ctx[MODE_CTX_REF_FRAMES];
  int pts[SAMPLES_ARRAY_SIZE], pts_inref[SAMPLES_ARRAY_SIZE];
  MACROBLOCKD *const xd = &dcb->xd;
  SB_INFO *sbi = xd->sbi;
  FRAME_CONTEXT *ec_ctx = xd->tile_ctx;

  mbmi->uv_mode = UV_DC_PRED;
  mbmi->palette_mode_info.palette_size[0] = 0;
  mbmi->palette_mode_info.palette_size[1] = 0;

#if CONFIG_NEW_REF_SIGNALING
  av1_collect_neighbors_ref_counts_nrs(cm, xd);
#else
  av1_collect_neighbors_ref_counts(xd);
#endif  // CONFIG_NEW_REF_SIGNALING

  read_ref_frames(cm, xd, r, mbmi->segment_id,
#if CONFIG_NEW_REF_SIGNALING
                  mbmi->ref_frame_nrs,
#endif  // CONFIG_NEW_REF_SIGNALING
                  mbmi->ref_frame);
  const int is_compound = has_second_ref(mbmi);

  const MV_REFERENCE_FRAME ref_frame = av1_ref_frame_type(mbmi->ref_frame);
  (void)ref_frame;
#if CONFIG_NEW_REF_SIGNALING
  MV_REFERENCE_FRAME_NRS ref_frame_nrs =
      av1_ref_frame_type_nrs(mbmi->ref_frame_nrs);
  av1_find_mv_refs_nrs(cm, xd, mbmi, ref_frame_nrs, dcb->ref_mv_count,
                       xd->ref_mv_stack, xd->weight, ref_mvs,
                       /*global_mvs_nrs=*/NULL, inter_mode_ctx);
#else
  av1_find_mv_refs(cm, xd, mbmi, ref_frame, dcb->ref_mv_count, xd->ref_mv_stack,
                   xd->weight, ref_mvs, /*global_mvs=*/NULL, inter_mode_ctx);
#endif  // CONFIG_NEW_REF_SIGNALING

  mbmi->ref_mv_idx = 0;

  if (mbmi->skip_mode) {
    assert(is_compound);
#if CONFIG_NEW_INTER_MODES
    mbmi->mode = NEAR_NEARMV;
    mbmi->ref_mv_idx = 0;
#else
    mbmi->mode = NEAREST_NEARESTMV;
#endif  // !CONFIG_NEW_INTER_MODES
  } else {
    if (segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_SKIP) ||
        segfeature_active(&cm->seg, mbmi->segment_id, SEG_LVL_GLOBALMV)) {
      mbmi->mode = GLOBALMV;
    } else {
      const int16_t mode_ctx = av1_mode_context_analyzer(inter_mode_ctx,
#if CONFIG_NEW_REF_SIGNALING
                                                         mbmi->ref_frame_nrs
#else
                                                         mbmi->ref_frame
#endif  // CONFIG_NEW_REF_SIGNALING
      );
      if (is_compound)
        mbmi->mode = read_inter_compound_mode(xd, r, mode_ctx);
      else
        mbmi->mode = read_inter_mode(ec_ctx, r, mode_ctx);
      // TODO(chiyotsai@google.com): Remove the following line
      av1_set_default_mbmi_mv_precision(mbmi, sbi);
      if (have_drl_index(mbmi->mode))
        read_drl_idx(
#if CONFIG_NEW_INTER_MODES
            cm->features.max_drl_bits,
            av1_mode_context_pristine(inter_mode_ctx,
#if CONFIG_NEW_REF_SIGNALING
                                      mbmi->ref_frame_nrs
#else
                                      mbmi->ref_frame
#endif  // CONFIG_NEW_REF_SIGNALING
                                      ),
#endif  // CONFIG_NEW_INTER_MODES
            ec_ctx, dcb, mbmi, r);
    }
  }
  av1_set_default_mbmi_mv_precision(mbmi, sbi);

  if (is_compound != is_inter_compound_mode(mbmi->mode)) {
    aom_internal_error(xd->error_info, AOM_CODEC_CORRUPT_FRAME,
                       "Prediction mode %d invalid with ref frame %d %d",
                       mbmi->mode,
#if CONFIG_NEW_REF_SIGNALING
                       mbmi->ref_frame_nrs[0], mbmi->ref_frame_nrs[1]
#else
                       mbmi->ref_frame[0], mbmi->ref_frame[1]
#endif  // CONFIG_NEW_REF_SIGNALING
    );
  }

  // Note: the ref_mvs are constructed at frame level precision, but we may
  // additionally down scale the precision later during assign_mv call.
  if (!is_compound && mbmi->mode != GLOBALMV) {
#if CONFIG_NEW_REF_SIGNALING
    av1_find_best_ref_mvs(ref_mvs[mbmi->ref_frame_nrs[0]], &nearestmv[0],
                          &nearmv[0], fr_mv_precision);
#else
    av1_find_best_ref_mvs(ref_mvs[mbmi->ref_frame[0]], &nearestmv[0],
                          &nearmv[0], fr_mv_precision);
#endif  // CONFIG_NEW_REF_SIGNALING
  }

  if (is_compound && mbmi->mode != GLOBAL_GLOBALMV) {
#if CONFIG_NEW_INTER_MODES
    int ref_mv_idx = mbmi->ref_mv_idx;
#else
    int ref_mv_idx = mbmi->ref_mv_idx + 1;
#endif  // CONFIG_NEW_INTER_MODES
#if CONFIG_NEW_REF_SIGNALING
    nearestmv[0] = xd->ref_mv_stack[ref_frame_nrs][0].this_mv;
    nearestmv[1] = xd->ref_mv_stack[ref_frame_nrs][0].comp_mv;
    nearmv[0] = xd->ref_mv_stack[ref_frame_nrs][ref_mv_idx].this_mv;
    nearmv[1] = xd->ref_mv_stack[ref_frame_nrs][ref_mv_idx].comp_mv;
#else
    nearestmv[0] = xd->ref_mv_stack[ref_frame][0].this_mv;
    nearestmv[1] = xd->ref_mv_stack[ref_frame][0].comp_mv;
    nearmv[0] = xd->ref_mv_stack[ref_frame][ref_mv_idx].this_mv;
    nearmv[1] = xd->ref_mv_stack[ref_frame][ref_mv_idx].comp_mv;
#endif  // CONFIG_NEW_REF_SIGNALING
    lower_mv_precision(&nearestmv[0].as_mv, fr_mv_precision);
    lower_mv_precision(&nearestmv[1].as_mv, fr_mv_precision);
    lower_mv_precision(&nearmv[0].as_mv, fr_mv_precision);
    lower_mv_precision(&nearmv[1].as_mv, fr_mv_precision);
#if CONFIG_NEW_INTER_MODES
  } else if (mbmi->mode == NEARMV) {
#if CONFIG_NEW_REF_SIGNALING
    nearmv[0] =
        xd->ref_mv_stack[mbmi->ref_frame_nrs[0]][mbmi->ref_mv_idx].this_mv;
#else
    nearmv[0] = xd->ref_mv_stack[mbmi->ref_frame[0]][mbmi->ref_mv_idx].this_mv;
#endif  // CONFIG_NEW_REF_SIGNALING
  }
#else
  } else if (mbmi->ref_mv_idx > 0 && mbmi->mode == NEARMV) {
#if CONFIG_NEW_REF_SIGNALING
    nearmv[0] =
        xd->ref_mv_stack[mbmi->ref_frame_nrs[0]][1 + mbmi->ref_mv_idx].this_mv;
#else
    nearmv[0] =
        xd->ref_mv_stack[mbmi->ref_frame[0]][1 + mbmi->ref_mv_idx].this_mv;
#endif  // CONFIG_NEW_REF_SIGNALING
  }
#endif  // CONFIG_NEW_INTER_MODES

  int_mv ref_mv[2] = { nearestmv[0], nearestmv[1] };

  if (is_compound) {
    int ref_mv_idx = mbmi->ref_mv_idx;
#if !CONFIG_NEW_INTER_MODES
    // Special case: NEAR_NEWMV and NEW_NEARMV modes use
    // 1 + mbmi->ref_mv_idx (like NEARMV) instead of
    // mbmi->ref_mv_idx (like NEWMV)
    if (mbmi->mode == NEAR_NEWMV || mbmi->mode == NEW_NEARMV)
      ref_mv_idx = 1 + mbmi->ref_mv_idx;
#endif  // !CONFIG_NEW_INTER_MODES
    // TODO(jingning, yunqing): Do we need a lower_mv_precision() call here?
    if (compound_ref0_mode(mbmi->mode) == NEWMV)
#if CONFIG_NEW_REF_SIGNALING
      ref_mv[0] = xd->ref_mv_stack[ref_frame_nrs][ref_mv_idx].this_mv;
#else
      ref_mv[0] = xd->ref_mv_stack[ref_frame][ref_mv_idx].this_mv;
#endif  // CONFIG_NEW_REF_SIGNALING

    if (compound_ref1_mode(mbmi->mode) == NEWMV)
#if CONFIG_NEW_REF_SIGNALING
      ref_mv[1] = xd->ref_mv_stack[ref_frame_nrs][ref_mv_idx].comp_mv;
#else
      ref_mv[1] = xd->ref_mv_stack[ref_frame][ref_mv_idx].comp_mv;
#endif  // CONFIG_NEW_REF_SIGNALING
  } else {
    if (mbmi->mode == NEWMV) {
#if CONFIG_NEW_REF_SIGNALING
      if (dcb->ref_mv_count[ref_frame_nrs] > 1)
        ref_mv[0] = xd->ref_mv_stack[ref_frame_nrs][mbmi->ref_mv_idx].this_mv;
#else
      if (dcb->ref_mv_count[ref_frame] > 1)
        ref_mv[0] = xd->ref_mv_stack[ref_frame][mbmi->ref_mv_idx].this_mv;
#endif  // CONFIG_NEW_REF_SIGNALING
    }
  }
#if CONFIG_NEW_INTER_MODES
  if (mbmi->skip_mode) {
    assert(mbmi->mode == NEAR_NEARMV);
    assert(mbmi->ref_mv_idx == 0);
  }
#else
  if (mbmi->skip_mode) assert(mbmi->mode == NEAREST_NEARESTMV);
#endif  // CONFIG_NEW_INTER_MODES

  const int mv_corrupted_flag =
      !assign_mv(cm, xd, mbmi->mode,
#if CONFIG_NEW_REF_SIGNALING
                 mbmi->ref_frame_nrs,
#else
                 mbmi->ref_frame,
#endif  // CONFIG_NEW_REF_SIGNALING
                 mbmi->mv, ref_mv, nearestmv, nearmv, is_compound,
                 mbmi->pb_mv_precision, r);
  aom_merge_corrupted_flag(&dcb->corrupted, mv_corrupted_flag);

  mbmi->use_wedge_interintra = 0;
  if (cm->seq_params.enable_interintra_compound && !mbmi->skip_mode &&
      is_interintra_allowed(mbmi)) {
    const int bsize_group = size_group_lookup[bsize];
    const int interintra =
        aom_read_symbol(r, ec_ctx->interintra_cdf[bsize_group], 2, ACCT_STR);
#if CONFIG_NEW_REF_SIGNALING
    assert(mbmi->ref_frame_nrs[1] == INVALID_IDX);
#else
    assert(mbmi->ref_frame[1] == NONE_FRAME);
#endif  // CONFIG_NEW_REF_SIGNALING
    if (interintra) {
      const INTERINTRA_MODE interintra_mode =
          read_interintra_mode(xd, r, bsize_group);
      mbmi->ref_frame[1] = INTRA_FRAME;
#if CONFIG_NEW_REF_SIGNALING
      mbmi->ref_frame_nrs[1] = INTRA_FRAME_NRS;
#endif  // CONFIG_NEW_REF_SIGNALING
      mbmi->interintra_mode = interintra_mode;
      mbmi->angle_delta[PLANE_TYPE_Y] = 0;
      mbmi->angle_delta[PLANE_TYPE_UV] = 0;
      mbmi->filter_intra_mode_info.use_filter_intra = 0;
      if (av1_is_wedge_used(bsize)) {
        mbmi->use_wedge_interintra = aom_read_symbol(
            r, ec_ctx->wedge_interintra_cdf[bsize], 2, ACCT_STR);
        if (mbmi->use_wedge_interintra) {
          mbmi->interintra_wedge_index = (int8_t)aom_read_symbol(
              r, ec_ctx->wedge_idx_cdf[bsize], MAX_WEDGE_TYPES, ACCT_STR);
        }
      }
    }
  }

  for (int ref = 0; ref < 1 + has_second_ref(mbmi); ++ref) {
#if CONFIG_NEW_REF_SIGNALING
    const MV_REFERENCE_FRAME frame = mbmi->ref_frame_nrs[ref];
    xd->block_ref_scale_factors[ref] =
        get_ref_scale_factors_const_nrs(cm, frame);
#else
    const MV_REFERENCE_FRAME frame = mbmi->ref_frame[ref];
    xd->block_ref_scale_factors[ref] = get_ref_scale_factors_const(cm, frame);
#endif  // CONFIG_NEW_REF_SIGNALING
  }

  mbmi->motion_mode = SIMPLE_TRANSLATION;
  if (is_motion_variation_allowed_bsize(mbmi->sb_type, xd->mi_row,
                                        xd->mi_col) &&
      !mbmi->skip_mode && !has_second_ref(mbmi)) {
    mbmi->num_proj_ref = av1_findSamples(cm, xd, pts, pts_inref);
  }
  av1_count_overlappable_neighbors(cm, xd);

#if CONFIG_NEW_REF_SIGNALING
  if (mbmi->ref_frame_nrs[1] != INTRA_FRAME_NRS)
#else
  if (mbmi->ref_frame[1] != INTRA_FRAME)
#endif  // CONFIG_NEW_REF_SIGNALING
    mbmi->motion_mode = read_motion_mode(cm, xd, mbmi, r);
#if CONFIG_EXT_ROTATION
  if (simple_translation_rotation_allowed(mbmi) ||
      globalmv_rotation_allowed(xd)) {
    read_warp_rotation(xd, mbmi, r);
  }
#endif  // CONFIG_EXT_ROTATION

  // init
  mbmi->comp_group_idx = 0;
  mbmi->compound_idx = 1;
  mbmi->interinter_comp.type = COMPOUND_AVERAGE;

  if (has_second_ref(mbmi) &&
#if CONFIG_OPTFLOW_REFINEMENT
      mbmi->mode <= NEW_NEWMV &&
#endif  // CONFIG_OPTFLOW_REFINEMENT
      !mbmi->skip_mode) {
    // Read idx to indicate current compound inter prediction mode group
    const int masked_compound_used = is_any_masked_compound_used(bsize) &&
                                     cm->seq_params.enable_masked_compound;

    if (masked_compound_used) {
      const int ctx_comp_group_idx = get_comp_group_idx_context(cm, xd);
      mbmi->comp_group_idx = (uint8_t)aom_read_symbol(
          r, ec_ctx->comp_group_idx_cdf[ctx_comp_group_idx], 2, ACCT_STR);
    }

    if (mbmi->comp_group_idx == 0) {
#if !CONFIG_REMOVE_DIST_WTD_COMP
      if (cm->seq_params.order_hint_info.enable_dist_wtd_comp) {
        const int comp_index_ctx = get_comp_index_context(cm, xd);
        mbmi->compound_idx = (uint8_t)aom_read_symbol(
            r, ec_ctx->compound_index_cdf[comp_index_ctx], 2, ACCT_STR);
        mbmi->interinter_comp.type =
            mbmi->compound_idx ? COMPOUND_AVERAGE : COMPOUND_DISTWTD;
      } else {
#endif  // !CONFIG_REMOVE_DIST_WTD_COMP
        // Distance-weighted compound is disabled, so always use average
        mbmi->compound_idx = 1;
        mbmi->interinter_comp.type = COMPOUND_AVERAGE;
#if !CONFIG_REMOVE_DIST_WTD_COMP
      }
#endif  // !CONFIG_REMOVE_DIST_WTD_COMP
    } else {
      assert(cm->current_frame.reference_mode != SINGLE_REFERENCE &&
             is_inter_compound_mode(mbmi->mode) &&
             mbmi->motion_mode == SIMPLE_TRANSLATION);
      assert(masked_compound_used);

      // compound_diffwtd, wedge
      if (is_interinter_compound_used(COMPOUND_WEDGE, bsize)) {
        mbmi->interinter_comp.type =
            COMPOUND_WEDGE + aom_read_symbol(r,
                                             ec_ctx->compound_type_cdf[bsize],
                                             MASKED_COMPOUND_TYPES, ACCT_STR);
      } else {
        mbmi->interinter_comp.type = COMPOUND_DIFFWTD;
      }

      if (mbmi->interinter_comp.type == COMPOUND_WEDGE) {
        assert(is_interinter_compound_used(COMPOUND_WEDGE, bsize));
        mbmi->interinter_comp.wedge_index = (int8_t)aom_read_symbol(
            r, ec_ctx->wedge_idx_cdf[bsize], MAX_WEDGE_TYPES, ACCT_STR);
        mbmi->interinter_comp.wedge_sign = (int8_t)aom_read_bit(r, ACCT_STR);
      } else {
        assert(mbmi->interinter_comp.type == COMPOUND_DIFFWTD);
        mbmi->interinter_comp.mask_type =
            aom_read_literal(r, MAX_DIFFWTD_MASK_BITS, ACCT_STR);
      }
    }
  }

  read_mb_interp_filter(xd, features->interp_filter,
#if !CONFIG_REMOVE_DUAL_FILTER
                        cm->seq_params.enable_dual_filter,
#endif  // !CONFIG_REMOVE_DUAL_FILTER
                        mbmi, r);

  const int mi_row = xd->mi_row;
  const int mi_col = xd->mi_col;

  if (mbmi->motion_mode == WARPED_CAUSAL) {
    mbmi->wm_params.wmtype = DEFAULT_WMTYPE;
    mbmi->wm_params.invalid = 0;

    if (mbmi->num_proj_ref > 1) {
      mbmi->num_proj_ref = av1_selectSamples(&mbmi->mv[0].as_mv, pts, pts_inref,
                                             mbmi->num_proj_ref, bsize);
    }

    if (av1_find_projection(mbmi->num_proj_ref, pts, pts_inref, bsize,
                            mbmi->mv[0].as_mv.row, mbmi->mv[0].as_mv.col,
                            &mbmi->wm_params,
#if CONFIG_EXT_ROTATION
                            xd,
#endif  // CONFIG_EXT_ROTATION
                            mi_row, mi_col)) {
#if WARPED_MOTION_DEBUG
      printf("Warning: unexpected warped model from aomenc\n");
#endif
      mbmi->wm_params.invalid = 1;
    }
  }

  xd->cfl.store_y = store_cfl_required(cm, xd);

#if CONFIG_REF_MV_BANK
  av1_update_ref_mv_bank(cm, xd, mbmi);
#endif  // CONFIG_REF_MV_BANK

#if DEC_MISMATCH_DEBUG
  dec_dump_logs(cm, mi, mi_row, mi_col, mode_ctx);
#endif  // DEC_MISMATCH_DEBUG
}

static void read_inter_frame_mode_info(AV1Decoder *const pbi,
                                       DecoderCodingBlock *dcb, aom_reader *r) {
  AV1_COMMON *const cm = &pbi->common;
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mbmi = xd->mi[0];
  int inter_block = 1;

  mbmi->mv[0].as_int = 0;
  mbmi->mv[1].as_int = 0;
  mbmi->segment_id = read_inter_segment_id(cm, xd, 1, r);

  mbmi->skip_mode = read_skip_mode(cm, xd, mbmi->segment_id, r);

  if (mbmi->skip_mode)
    mbmi->skip_txfm = 1;
  else
    mbmi->skip_txfm = read_skip_txfm(cm, xd, mbmi->segment_id, r);

  if (!cm->seg.segid_preskip)
    mbmi->segment_id = read_inter_segment_id(cm, xd, 0, r);

  read_cdef(cm, r, xd);

  read_delta_q_params(cm, xd, r);

  if (!mbmi->skip_mode)
    inter_block = read_is_inter_block(cm, xd, mbmi->segment_id, r);

  mbmi->current_qindex = xd->current_base_qindex;

  xd->above_txfm_context =
      cm->above_contexts.txfm[xd->tile.tile_row] + xd->mi_col;
  xd->left_txfm_context =
      xd->left_txfm_context_buffer + (xd->mi_row & MAX_MIB_MASK);

  if (inter_block)
    read_inter_block_mode_info(pbi, dcb, mbmi, r);
  else
    read_intra_block_mode_info(cm, xd, mbmi, r);
}

static void intra_copy_frame_mvs(AV1_COMMON *const cm, int mi_row, int mi_col,
                                 int x_mis, int y_mis) {
  const int frame_mvs_stride = ROUND_POWER_OF_TWO(cm->mi_params.mi_cols, 1);
  MV_REF *frame_mvs =
      cm->cur_frame->mvs + (mi_row >> 1) * frame_mvs_stride + (mi_col >> 1);
  x_mis = ROUND_POWER_OF_TWO(x_mis, 1);
  y_mis = ROUND_POWER_OF_TWO(y_mis, 1);

  for (int h = 0; h < y_mis; h++) {
    MV_REF *mv = frame_mvs;
    for (int w = 0; w < x_mis; w++) {
#if CONFIG_NEW_REF_SIGNALING
      mv->ref_frame = INVALID_IDX;
#else
      mv->ref_frame = NONE_FRAME;
#endif  // CONFIG_NEW_REF_SIGNALING
      mv++;
    }
    frame_mvs += frame_mvs_stride;
  }
}

void av1_read_mode_info(AV1Decoder *const pbi, DecoderCodingBlock *dcb,
                        aom_reader *r, int x_mis, int y_mis) {
  AV1_COMMON *const cm = &pbi->common;
  MACROBLOCKD *const xd = &dcb->xd;
  MB_MODE_INFO *const mi = xd->mi[0];
  mi->use_intrabc = 0;

  if (frame_is_intra_only(cm)) {
    read_intra_frame_mode_info(cm, dcb, r);
    if (cm->seq_params.order_hint_info.enable_ref_frame_mvs)
      intra_copy_frame_mvs(cm, xd->mi_row, xd->mi_col, x_mis, y_mis);
  } else {
    read_inter_frame_mode_info(pbi, dcb, r);
    if (cm->seq_params.order_hint_info.enable_ref_frame_mvs)
      av1_copy_frame_mvs(cm, mi, xd->mi_row, xd->mi_col, x_mis, y_mis);
  }
}
