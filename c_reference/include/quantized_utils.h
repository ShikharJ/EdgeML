// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_UTILS_H__
#define __QUANTIZED_UTILS_H__

#include <math.h>
#include "quantized_datatypes.h"

// Function for saturating the input to the required format.
// This function isn't used actively because of SeeDot generated scales
// ensuring the overflows aren't a possibility.
static inline Q15_T q15_saturate(Q31_T inp) {
    if (inp > Q15_TMAX){
        return (Q15_T)Q15_TMAX;
    } else if (inp < Q15_TMIN) {
        return (Q15_T)Q15_TMIN;
    } else {
        return (Q15_T)inp;
    }
}

// These functions are used to provide a truncation of input to a specific
// range within the ReLU operation.
static inline Q7_T q7_relu(Q7_T inp, Q7_T limit) {
    if (inp > limit){
        return limit;
    } else if (inp < 0) {
        return 0;
    } else {
        return inp;
    }
}

static inline Q15_T q15_relu(Q15_T inp, Q15_T limit) {
    if (inp > limit){
        return limit;
    } else if (inp < 0) {
        return 0;
    } else {
        return inp;
    }
}

static inline Q31_T q31_relu(Q31_T inp, Q31_T limit) {
    if (inp > limit){
        return limit;
    } else if (inp < 0) {
        return 0;
    } else {
        return inp;
    }
}

static const Q15_T exp_table_A[256] = {16384, 15391, 14459, 13583, 12760, 11987, 11261, 10578, 9937, 9335, 8770, 8238, 7739, 7270, 6830, 6416, 6027, 5662, 5319, 4997, 4694, 4410, 4143, 3892, 3656, 3434, 3226, 3031, 2847, 2675, 2513, 2360, 2217, 2083, 1957, 1838, 1727, 1622, 1524, 1432, 1345, 1263, 1187, 1115, 1047, 984, 924, 868, 816, 766, 720, 676, 635, 597, 561, 527, 495, 465, 437, 410, 385, 362, 340, 319, 300, 282, 265, 249, 234, 220, 206, 194, 182, 171, 161, 151, 142, 133, 125, 118, 110, 104, 97, 92, 86, 81, 76, 71, 67, 63, 59, 56, 52, 49, 46, 43, 41, 38, 36, 34, 32, 30, 28, 26, 25, 23, 22, 20, 19, 18, 17, 16, 15, 14, 13, 12, 12, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static const Q15_T exp_table_B[128] = {16384, 16376, 16368, 16360, 16352, 16344, 16336, 16328, 16320, 16312, 16304, 16296, 16288, 16280, 16272, 16264, 16256, 16249, 16241, 16233, 16225, 16217, 16209, 16201, 16193, 16185, 16177, 16169, 16162, 16154, 16146, 16138, 16130, 16122, 16114, 16106, 16099, 16091, 16083, 16075, 16067, 16059, 16051, 16044, 16036, 16028, 16020, 16012, 16004, 15997, 15989, 15981, 15973, 15965, 15958, 15950, 15942, 15934, 15927, 15919, 15911, 15903, 15895, 15888, 15880, 15872, 15864, 15857, 15849, 15841, 15833, 15826, 15818, 15810, 15803, 15795, 15787, 15779, 15772, 15764, 15756, 15749, 15741, 15733, 15726, 15718, 15710, 15703, 15695, 15687, 15680, 15672, 15664, 15657, 15649, 15641, 15634, 15626, 15618, 15611, 15603, 15596, 15588, 15580, 15573, 15565, 15558, 15550, 15542, 15535, 15527, 15520, 15512, 15504, 15497, 15489, 15482, 15474, 15467, 15459, 15452, 15444, 15437, 15429, 15421, 15414, 15406, 15399};

static inline Q15_T exp_base_16(Q15_T inp, Q15_T scale) {
  Q15_T val = (inp == -32768) ? 32767 : -inp;
  Q15_T val1 = val % 128;
  val >>= 7;
  Q31_T ret = (Q31_T)exp_table_A[val] * (Q31_T)exp_table_B[val1];
  return (Q15_T)((ret / scale) >> 14);
}

/**
 * @brief Compute the element-wise addition between two vectors.
 * @param[in]       vec1      pointer to the first input vector
 * @param[in]       vec2      pointer to the second input vector
 * @param[in]       len       length of the input vectors
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scvec1    scale factor of the first input vector
 * @param[in]       scvec2    scale factor of the second input vector
 * @param[in]       scret     scale factor of the output vector
 * @param[in]       demote    scale factor for output variable demotion
 * @return          none
 * @example         vec1      = {-425, -169, -3534, 524, -2739, 87, 52, 292}
 *                  vec2      = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933}
 *                  len       = 8
 *                  scvec1    = 1
 *                  scvec2    = 8
 *                  scret     = 1
 *                  demote    = 1
 *                  ret       = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699}
 */
static inline void q15_v_add(const Q15_T* vec1, const Q15_T* vec2, ITER_T len,
                             Q15_T* ret, SCALE_T scvec1, SCALE_T scvec2,
                             SCALE_T scret, SCALE_T demote) {
  #ifdef SHIFT
    SCALE_T scalevec1 = scvec1 + scret;
    SCALE_T scalevec2 = scvec2 + scret;
  #else
    SCALE_T scalevec1 = scvec1 * scret;
    SCALE_T scalevec2 = scvec2 * scret;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = (((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2)) >> demote);
        *ret++ = (((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2)) >> demote);
        *ret++ = (((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2)) >> demote);
        *ret++ = (((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2)) >> demote);
      #else
        *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2)) / demote;
        *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2)) / demote;
        *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2)) / demote;
        *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2)) / demote;
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = (((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2)) >> demote);
    #else
      *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2)) / demote;
    #endif
  }
}
/**
 * @brief Compute the element-wise subtraction between two vectors.
 * @param[in]       vec1      pointer to the first input vector
 * @param[in]       vec2      pointer to the second input vector
 * @param[in]       len       length of the input vectors
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scvec1    scale factor of the first input vector
 * @param[in]       scvec2    scale factor of the second input vector
 * @param[in]       scret     scale factor of the output vector
 * @return          none
 * @example         vec1      = {-425, -169, -3534, 524, -2739, 87, 52, 292}
 *                  vec2      = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933}
 *                  len       = 8
 *                  scvec1    = 1
 *                  scvec2    = 8
 *                  scret     = 1
 *                  ret       = {1922, 1020, -4040, 1437, -3812, 2244, 712, 1283}
 */
static inline void q7_v_sub(const Q7_T* vec1, const Q7_T* vec2, ITER_T len,
                            Q7_T* ret, SCALE_T scvec1, SCALE_T scvec2,
                            SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scalevec1 = scvec1 + scret;
    SCALE_T scalevec2 = scvec2 + scret;
  #else
    SCALE_T scalevec1 = scvec1 * scret;
    SCALE_T scalevec2 = scvec2 * scret;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
        *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
        *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
        *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
      #else
        *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
        *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
        *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
        *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
    #else
      *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
    #endif
  }
}

static inline void q15_v_sub(const Q15_T* vec1, const Q15_T* vec2, ITER_T len,
                             Q15_T* ret, SCALE_T scvec1, SCALE_T scvec2,
                             SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scalevec1 = scvec1 + scret;
    SCALE_T scalevec2 = scvec2 + scret;
  #else
    SCALE_T scalevec1 = scvec1 * scret;
    SCALE_T scalevec2 = scvec2 * scret;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
        *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
        *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
        *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
      #else
        *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
        *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
        *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
        *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = ((*vec1++ >> scalevec1) - (*vec2++ >> scalevec2));
    #else
      *ret++ = ((*vec1++ / scalevec1) - (*vec2++ / scalevec2));
    #endif
  }
}
/**
 * @brief Compute the element-wise product (also known as Hadamard product) between two vectors.
 * @param[in]       vec1      pointer to the first input vector
 * @param[in]       vec2      pointer to the second input vector
 * @param[in]       len       length of the input vectors
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scvec1    scale factor of the first input vector
 * @param[in]       scvec2    scale factor of the second input vector
 * @return          none
 * @example         vec1      = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018}
 *                  vec2      = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282}
 *                  len       = 8
 *                  scvec1    = 32
 *                  scvec2    = 64
 *                  ret       = {1423, 7085, -16378, 8209, -12067, 6805, 6475, 6897}
 */
static inline void q7_v_hadamard(const Q7_T* vec1, const Q7_T* vec2, ITER_T len,
                                 Q7_T* ret, SCALE_T scvec1, SCALE_T scvec2) {
  #ifdef SHIFT
    SCALE_T scalevec = scvec1 + scvec2;
  #else
    SCALE_T scalevec = scvec1 * scvec2;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = (((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) >> scalevec);
        *ret++ = (((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) >> scalevec);
        *ret++ = (((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) >> scalevec);
        *ret++ = (((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) >> scalevec);
      #else
        *ret++ = ((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) / scalevec;
        *ret++ = ((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) / scalevec;
        *ret++ = ((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) / scalevec;
        *ret++ = ((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) / scalevec;
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = (((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) >> scalevec);
    #else
      *ret++ = ((Q15_T)(*vec1++) * (Q15_T)(*vec2++)) / scalevec;
    #endif
  }
}

static inline void q15_v_hadamard(const Q15_T* vec1, const Q15_T* vec2,
                                  ITER_T len, Q15_T* ret, SCALE_T scvec1,
                                  SCALE_T scvec2) {
  #ifdef SHIFT
    SCALE_T scalevec = scvec1 + scvec2;
  #else
    SCALE_T scalevec = scvec1 * scvec2;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = (((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) >> scalevec);
        *ret++ = (((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) >> scalevec);
        *ret++ = (((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) >> scalevec);
        *ret++ = (((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) >> scalevec);
      #else
        *ret++ = ((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) / scalevec;
        *ret++ = ((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) / scalevec;
        *ret++ = ((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) / scalevec;
        *ret++ = ((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) / scalevec;
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = (((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) >> scalevec);
    #else
      *ret++ = ((Q31_T)(*vec1++) * (Q31_T)(*vec2++)) / scalevec;
    #endif
  }
}
/**
 * @brief Compute the element-wise Sigmoid activation on the input vector.
 * @param[in]       vec            pointer to the input vector
 * @param[in]       len            length of the input vector
 * @param[out]      ret            pointer to the vector storing the output
 * @param[in]       div            division factor of the input vector
 * @param[in]       add            addition offset of the input vector
 * @param[in]       sigmoid_limit  saturation limit for the Sigmoid activation
 * @param[in]       scale_in       scale factor of the input vector
 * @param[in]       scale_out      scale factor of the output vector
 * @param[in]       use_tables     flag for using pre-computed (base 16) exp tables for calculating Sigmoid on the input
 * @return          none
 * @example         formula        = saturate(0, (vec_{i} / div) + add, sigmoid_limit) * 2^{scale_out - scale_in} (use_tables set to 0)
 *                  vec            = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699}
 *                  len            = 8
 *                  div            = 2
 *                  add            = 1024
 *                  sigmoid_limit  = 2048
 *                  scale_in       = 11
 *                  scale_out      = 14
 *                  use_tables     = 0
 *                  ret            = {0, 2760, 0, 6640, 1528, 0, 5760, 5400}
 */
static inline void q15_v_sigmoid(const Q15_T* vec, ITER_T len, Q15_T* ret,
                                 Q15_T div, Q15_T add, Q15_T sigmoid_limit,
                                 SCALE_T scale_in, SCALE_T scale_out,
                                 ITER_T use_tables) {
  if (use_tables) {
    #ifdef LOOP_UNROLL
      ITER_T len_unroll = len >> 2;
      len = len % 4;
      while (len_unroll--) {
        Q15_T w = *vec++;
        Q15_T x = *vec++;
        Q15_T y = *vec++;
        Q15_T z = *vec++;

        *ret++ = (w <= 0) ? (Q15_T)((((Q31_T)exp_base_16(w, 1)) << 14) /
                                    ((Q31_T)exp_base_16(w, 1) + (Q31_T)16384)) :
                            (Q15_T)(((Q31_T)267943936L) /
                                    ((Q31_T)16384 + (Q31_T)exp_base_16(-w, 1)));
        *ret++ = (x <= 0) ? (Q15_T)((((Q31_T)exp_base_16(x, 1)) << 14) /
                                    ((Q31_T)exp_base_16(x, 1) + (Q31_T)16384)) :
                            (Q15_T)(((Q31_T)267943936L) /
                                    ((Q31_T)16384 + (Q31_T)exp_base_16(-x, 1)));
        *ret++ = (y <= 0) ? (Q15_T)((((Q31_T)exp_base_16(y, 1)) << 14) /
                                    ((Q31_T)exp_base_16(y, 1) + (Q31_T)16384)) :
                            (Q15_T)(((Q31_T)267943936L) /
                                    ((Q31_T)16384 + (Q31_T)exp_base_16(-y, 1)));
        *ret++ = (z <= 0) ? (Q15_T)((((Q31_T)exp_base_16(z, 1)) << 14) /
                                    ((Q31_T)exp_base_16(z, 1) + (Q31_T)16384)) :
                            (Q15_T)(((Q31_T)267943936L) /
                                    ((Q31_T)16384 + (Q31_T)exp_base_16(-z, 1)));
      }
    #endif

    while (len--) {
      Q15_T w = *vec++;
      *ret++ = (w <= 0) ? (Q15_T)((((Q31_T)exp_base_16(w, 1)) << 14) /
                                  ((Q31_T)exp_base_16(w, 1) + (Q31_T)16384)) :
                          (Q15_T)(((Q31_T)267943936L) /
                                  ((Q31_T)16384 + (Q31_T)exp_base_16(-w, 1)));
    }
  } else {
    SCALE_T scaleout = (scale_out - scale_in);

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = len >> 2;
      len = len % 4;
      while (len_unroll--) {
        Q15_T w = (*vec++ / div) + add;
        Q15_T x = (*vec++ / div) + add;
        Q15_T y = (*vec++ / div) + add;
        Q15_T z = (*vec++ / div) + add;

        *ret++ = (w <= 0) ? 0 : (((w >= sigmoid_limit) ? sigmoid_limit : w) << scaleout);
        *ret++ = (x <= 0) ? 0 : (((x >= sigmoid_limit) ? sigmoid_limit : x) << scaleout);
        *ret++ = (y <= 0) ? 0 : (((y >= sigmoid_limit) ? sigmoid_limit : y) << scaleout);
        *ret++ = (z <= 0) ? 0 : (((z >= sigmoid_limit) ? sigmoid_limit : z) << scaleout);
      }
    #endif

    while (len--) {
      Q15_T w = (*vec++ / div) + add;
      *ret++ = (w <= 0) ? 0 : (((w >= sigmoid_limit) ? sigmoid_limit : w) << scaleout);
    }
  }
}
/**
 * @brief Compute the element-wise TanHyperbolic activation on the input vector.
 * @param[in]       vec            pointer to the input vector
 * @param[in]       len            length of the input vector
 * @param[out]      ret            pointer to the vector storing the output
 * @param[in]       scale_in       scale factor of the input vector
 * @param[in]       scale_out      scale factor of the output vector
 * @param[in]       use_tables     flag for using pre-computed (base 16) exp tables for calculating TanH on the input
 * @return          none
 * @example         formula        = saturate(-2^{scale_in}, vec_{i}, 2^{scale_in}) * 2^{scale_out - scale_in} (use_tables set to 0)
 *                  vec            = {178, 1064, -4162, 1718, -1663, 851, 1244, 1282}
 *                  len            = 8
 *                  scale_in       = 11
 *                  scale_out      = 11
 *                  use_tables     = 0
 *                  ret            = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282}
 */
static inline void q15_v_tanh(const Q15_T* vec, ITER_T len, Q15_T* ret,
                              SCALE_T scale_in, SCALE_T scale_out,
                              ITER_T use_tables) {
  if (use_tables) {
    #ifdef LOOP_UNROLL
      ITER_T len_unroll = len >> 2;
      len = len % 4;
      while (len_unroll--) {
        Q15_T w = q15_saturate(2 * (*vec++));
        Q15_T x = q15_saturate(2 * (*vec++));
        Q15_T y = q15_saturate(2 * (*vec++));
        Q15_T z = q15_saturate(2 * (*vec++));

        *ret++ = (w <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(w, 1) - 16384)) << 14) /
                                    (exp_base_16(w, 1) + 16384)) :
                            (Q15_T)((((Q31_T)(16384 - exp_base_16(-w, 1))) << 14) /
                                    (exp_base_16(-w, 1) + 16384));
        *ret++ = (x <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(x, 1) - 16384)) << 14) /
                                    (exp_base_16(x, 1) + 16384)) :
                            (Q15_T)((((Q31_T)(16384 - exp_base_16(-x, 1))) << 14) /
                                    (exp_base_16(-x, 1) + 16384));
        *ret++ = (y <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(y, 1) - 16384)) << 14) /
                                    (exp_base_16(y, 1) + 16384)) :
                            (Q15_T)((((Q31_T)(16384 - exp_base_16(-y, 1))) << 14) /
                                    (exp_base_16(-y, 1) + 16384));
        *ret++ = (z <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(z, 1) - 16384)) << 14) /
                                    (exp_base_16(z, 1) + 16384)) :
                            (Q15_T)((((Q31_T)(16384 - exp_base_16(-z, 1))) << 14) /
                                    (exp_base_16(-z, 1) + 16384));
      }
    #endif

    while (len--) {
      Q15_T w = q15_saturate(2 * (*vec++));
      *ret++ = (w <= 0) ? (Q15_T)((((Q31_T)(exp_base_16(w, 1) - 16384)) << 14) /
                                  (exp_base_16(w, 1) + 16384)) :
                          (Q15_T)((((Q31_T)(16384 - exp_base_16(-w, 1))) << 14) /
                                  (exp_base_16(-w, 1) + 16384));
    }
  } else {
    SCALE_T scalein = (1 << scale_in);
    SCALE_T scaleout = scale_out - scale_in;

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = len >> 2;
      len = len % 4;
      while (len_unroll--) {
        Q15_T w = *vec++;
        Q15_T x = *vec++;
        Q15_T y = *vec++;
        Q15_T z = *vec++;

        *ret++ = ((w >= scalein) ? scalein : ((w <= -scalein) ? (-scalein) : w)) << scaleout;
        *ret++ = ((x >= scalein) ? scalein : ((x <= -scalein) ? (-scalein) : x)) << scaleout;
        *ret++ = ((y >= scalein) ? scalein : ((y <= -scalein) ? (-scalein) : y)) << scaleout;
        *ret++ = ((z >= scalein) ? scalein : ((z <= -scalein) ? (-scalein) : z)) << scaleout;
      }
    #endif

    while (len--) {
      Q15_T w = *vec++;
      *ret++ = ((w >= scalein) ? scalein : ((w <= -scalein) ? (-scalein) : w)) << scaleout;
    }
  }
}
/**
 * @brief Compute the addition of a scalar to every element of a vector.
 * @param[in]       scalar    the input scalar to be added to a vector
 * @param[in]       vec       pointer to the input vector
 * @param[in]       len       length of the input vector
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scscalar  scale factor of the input scalar
 * @param[in]       scvec     scale factor of the input vector
 * @param[in]       scret     scale factor of the output vector
 * @return          none
 * @example         scalar    = 30111
 *                  vec       = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901}
 *                  len       = 8
 *                  scscalar  = 256
 *                  scvec     = 1
 *                  scret     = 1
 *                  ret       = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018}
 */
static inline void q15_v_scalar_add(Q15_T scalar, const Q15_T* vec, ITER_T len,
                                    Q15_T* ret, SCALE_T scscalar, SCALE_T scvec,
                                    SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scaledscalar = scalar >> (scscalar + scret);
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scaledscalar = scalar / (scscalar * scret);
    SCALE_T scalevec = scvec * scret;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = (scaledscalar + (*vec++ >> scalevec));
        *ret++ = (scaledscalar + (*vec++ >> scalevec));
        *ret++ = (scaledscalar + (*vec++ >> scalevec));
        *ret++ = (scaledscalar + (*vec++ >> scalevec));
      #else
        *ret++ = (scaledscalar + (*vec++ / scalevec));
        *ret++ = (scaledscalar + (*vec++ / scalevec));
        *ret++ = (scaledscalar + (*vec++ / scalevec));
        *ret++ = (scaledscalar + (*vec++ / scalevec));
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = (scaledscalar + (*vec++ >> scalevec));
    #else
      *ret++ = (scaledscalar + (*vec++ / scalevec));
    #endif
  }
}
/**
 * @brief Compute the subtraction of every element of a vector (B) from a scalar (a). The resultant vector has elements C_{i} = a - B_{i}.
 * @param[in]       scalar    the input scalar
 * @param[in]       vec       pointer to the input vector to be subtracted
 * @param[in]       len       length of the input vector
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scscalar  scale factor of the input scalar
 * @param[in]       scvec     scale factor of the input vector
 * @param[in]       scret     scale factor of the output vector
 * @return          none
 * @example         scalar    = 16384
 *                  vec       = {0, 2760, 0, 6640, 1528, 0, 5760, 5400}
 *                  len       = 8
 *                  scscalar  = 1
 *                  scvec     = 1
 *                  scret     = 1
 *                  ret       = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984}
 */
static inline void q15_v_scalar_sub(Q15_T scalar, const Q15_T* vec, ITER_T len,
                                    Q15_T* ret, SCALE_T scscalar, SCALE_T scvec,
                                    SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scaledscalar = scalar >> (scscalar + scret);
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scaledscalar = scalar / (scscalar * scret);
    SCALE_T scalevec = scvec * scret;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = (scaledscalar - (*vec++ >> scalevec));
        *ret++ = (scaledscalar - (*vec++ >> scalevec));
        *ret++ = (scaledscalar - (*vec++ >> scalevec));
        *ret++ = (scaledscalar - (*vec++ >> scalevec));
      #else
        *ret++ = (scaledscalar - (*vec++ / scalevec));
        *ret++ = (scaledscalar - (*vec++ / scalevec));
        *ret++ = (scaledscalar - (*vec++ / scalevec));
        *ret++ = (scaledscalar - (*vec++ / scalevec));
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = (scaledscalar - (*vec++ >> scalevec));
    #else
      *ret++ = (scaledscalar - (*vec++ / scalevec));
    #endif
  }
}
/**
 * @brief Compute the multiplication of a scalar to every element of a vector.
 * @param[in]       scalar    the input scalar to be multiplied
 * @param[in]       vec       pointer to the input vector
 * @param[in]       len       length of the input vector
 * @param[out]      ret       pointer to the vector storing the output
 * @param[in]       scscalar  scale factor of the input scalar
 * @param[in]       scvec     scale factor of the input vector
 * @return          none
 * @example         scalar    = 32522
 *                  vec       = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984}
 *                  len       = 8
 *                  scscalar  = 128
 *                  scvec     = 256
 *                  ret       = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901}
 */
static inline void q15_v_scalar_mul(Q15_T scalar, const Q15_T* vec, ITER_T len,
                                    Q15_T* ret, SCALE_T scscalar,
                                    SCALE_T scvec) {
  SCALE_T upscalar = scalar;
  #ifdef SHIFT
    SCALE_T scale = scscalar + scvec;
  #else
    SCALE_T scale = scscalar * scvec;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = (upscalar * (Q31_T)(*vec++)) >> scale;
        *ret++ = (upscalar * (Q31_T)(*vec++)) >> scale;
        *ret++ = (upscalar * (Q31_T)(*vec++)) >> scale;
        *ret++ = (upscalar * (Q31_T)(*vec++)) >> scale;
      #else
        *ret++ = (upscalar * (Q31_T)(*vec++)) / scale;
        *ret++ = (upscalar * (Q31_T)(*vec++)) / scale;
        *ret++ = (upscalar * (Q31_T)(*vec++)) / scale;
        *ret++ = (upscalar * (Q31_T)(*vec++)) / scale;
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = (upscalar * (Q31_T)(*vec++)) >> scale;
    #else
      *ret++ = (upscalar * (Q31_T)(*vec++)) / scale;
    #endif
  }
}
/**
 * @brief Finds the index of largest element in a vector.
 * @param[in]       vec       pointer to input vector
 * @param[in]       len       length of the vector
 * @param[out]      ret       pointer variable storing the index of the largest element in the vector
 * @return          none
 * @example         vec       = {12, 24, 54, 1, 2, 10}
 *                  len       = 6
 *                  *ret      = 2
 */
void q15_v_argmax(const Q15_T* const vec, ITER_T len, ITER_T* const ret);
/**
 * @brief Performs element-wise up-scaling on a vector.
 * @param[in]       vec       pointer to the vector on which up-scaling is to be performed
 * @param[in]       len       length of the vector
 * @param[out]      ret       pointer to the output vector
 * @param[in]       scvec     scaling factor of the vector
 * @return          none
 * @example         vec       = {423, -987, -2342, 1232}
 *                  len       = 4
 *                  scvec     = 10
 *                  ret       = {4230, -9870, -23420, 12320}
 */
void q15_v_scale_up(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scvec);
/**
 * @brief Performs element-wise down-scaling on a vector.
 * @param[in]       vec       pointer to the vector on which down-scaling is to be performed
 * @param[in]       len       length of the vector
 * @param[out]      ret       pointer to the output vector
 * @param[in]       scvec     scaling factor of the vector
 * @return          none
 * @example         vec       = {4232, -9879, -2342, 1232}
 *                  len       = 4
 *                  scvec     = 37
 *                  ret       = {114, -267, -63, 33}
 */
void q15_v_scale_down(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scvec);

/**
 * @brief Performs the matrix multiplication of a matrix and a vector.
 * @param[in]       mat       pointer to input matrix in row-major order
 * @param[in]       vec       pointer to the input vector
 * @param[in]       nrows     number of rows of the input matrix
 * @param[in]       ncols     number of columns of the input matrix
 * @param[out]      ret       pointer to the output vector
 * @param[in]       scmat     scale factor of the input matrix
 * @param[in]       scvec     scale factor of the input vector
 * @param[in]       H1        depth parameter for division-by-two used in TreeSum
 * @param[in]       H2        depth parameter for direct sum used in TreeSum

 * @return          none
 * @example         mat       = { {7069, -10389, 1562, -1992},
 *                                {3262, -37, -1143, -995},
 *                                {5513, -17035, -14615, -6636},
 *                                {4733, -403, 4106, -1104},
 *                                {-2707, -1287, -18128, -1832},
 *                                {-10108, -137, 2064, 1207},
 *                                {5233, 226, 831, -1909},
 *                                {4489, -1099, 2845, -1261} }
 *                  vec       = {1040, 1919, 4254, 4024}
 *                  nrows     = 8
 *                  ncols     = 4
 *                  scmat     = 128
 *                  scvec     = 64
 *                  H1        = 2
 *                  H2        = 0
 *                  ret       = {-425, -169, -3534, 524, -2739, 87, 52, 292}
 */
void q15xq7_q15_m_mulvec(const Q15_T* mat, const Q7_T* const vec, ITER_T nrows,
                         ITER_T ncols, Q15_T* ret, SCALE_T scmat,
                         SCALE_T scvec, SCALE_T H1, SCALE_T H2);
void q15_m_mulvec(const Q15_T* mat, const Q15_T* const vec, ITER_T nrows,
                  ITER_T ncols, Q15_T* ret, SCALE_T scmat, SCALE_T scvec,
                  SCALE_T H1, SCALE_T H2);
/**
 * @brief Performs sparse matrix multiplication of a matrix and a vector.
 * row_indices and mat_values combined are a sparse representation; dim(vec) = [ncols].
 * mat_values[i] is the i^th non-zero value of the input matrix, and row_indices[i] encodes the (1-indexed) row location of mat_values[i].
 * If number of zeroes before row_indices[i] is l, then l is the column location of the (i-l)th matrix value.
 * @param[in]       row_indices  pointer to input matrix which stores the row indices of non-zero values of matrix A
 * @param[in]       mat_values   pointer to input matrix which stores the non-zero values of matrix A
 * @param[in]       vec          pointer to the input vector
 * @param[in]       nrows        number of rows of the input matrix
 * @param[in]       ncols        number of columns of the input matrix
 * @param[out]      ret          pointer to the output vector
 * @param[in]       scmat        scale factor of the input matrix
 * @param[in]       scvec        scale factor of the input vector
 * @param[in]       H1           depth parameter for division-by-two used in TreeSum
 * @param[in]       H2           depth parameter for direct sum used in TreeSum
 * @return          none
 * @example         mat          = { {23, 32, 0},
 *                                   {0, 0, 1},
 *                                   {48, 0, 0}}
 *                  row_indices  = {1, 3, 0, 1, 0, 2, 0}
 *                  mat_values   = {23, 48, 32, 1}
 *                  vec          = {1, 2, 3}
 *                  nrows        = 3
 *                  ncols        = 3
 *                  scmat        = 1
 *                  scvec        = 1
 *                  H1           = 1
 *                  H2           = 0
 *                  ret          = {87, 3, 48}
 */
void q15xq7_q15_m_sparse_mulvec(const ITER_T* row_indices,
                                const Q15_T* mat_values, const Q7_T* vec,
                                ITER_T nrows, ITER_T ncols, Q15_T* ret,
                                SCALE_T scmat, SCALE_T scvec, SCALE_T H1,
                                SCALE_T H2);
void q15_m_sparse_mulvec(const ITER_T* row_indices, const Q15_T* mat_values,
                         const Q15_T* vec, ITER_T nrows, ITER_T ncols,
                         Q15_T* ret, SCALE_T scmat, SCALE_T scvec, SCALE_T H1,
                         SCALE_T H2);

static inline void q15xq7_q15_m_sparse_hardcode_mulvec_w(const Q7_T* vec,
                                                         Q15_T* ret,
                                                         SCALE_T scmat,
                                                         SCALE_T scvec,
                                                         SCALE_T H1) {
  Q31_T a = *vec++;
  Q31_T b = *vec++;
  Q31_T c = *vec++;
  Q31_T d = *vec++;

  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
    *ret++ = ((5115 * a) + (9914 * c) + (-9619 * d)) >> scale;
    *ret++ = ((-10312 * a) + (10136 * b)) >> scale;
    *ret++ = ((12851 * c)) >> scale;
    *ret++ = ((-576 * a) + (-802 * b) + (663 * c) + (-1917 * d)) >> scale;
    *ret++ = ((-2829 * a) + (2234 * b) + (8805 * c) + (-11879 * d)) >> scale;
    *ret++ = ((1891 * a) + (4935 * b) + (-20961 * c) + (27435 * d)) >> scale;
    *ret++ = ((8029 * d)) >> scale;
    *ret++ = ((-1021 * b)) >> scale;
    *ret++ = ((10418 * a)) >> scale;
    *ret++ = ((1144 * a) + (17792 * c) + (-22498 * d)) >> scale;
    *ret++ = ((-643 * c)) >> scale;
    *ret++ = ((-14259 * a) + (19156 * b)) >> scale;
    *ret++ = ((-440 * a)) >> scale;
    *ret++ = ((-2244 * b) + (-8755 * c)) >> scale;
    *ret++ = ((-12238 * b)) >> scale;
    *ret++ = ((6572 * a)) >> scale;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of Q31_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * H1;
    *ret++ = ((5115 * a) + (9914 * c) + (-9619 * d)) / scale;
    *ret++ = ((-10312 * a) + (10136 * b)) / scale;
    *ret++ = ((12851 * c)) / scale;
    *ret++ = ((-576 * a) + (-802 * b) + (663 * c) + (-1917 * d)) / scale;
    *ret++ = ((-2829 * a) + (2234 * b) + (8805 * c) + (-11879 * d)) / scale;
    *ret++ = ((1891 * a) + (4935 * b) + (-20961 * c) + (27435 * d)) / scale;
    *ret++ = ((8029 * d)) / scale;
    *ret++ = ((-1021 * b)) / scale;
    *ret++ = ((10418 * a)) / scale;
    *ret++ = ((1144 * a) + (17792 * c) + (-22498 * d)) / scale;
    *ret++ = ((-643 * c)) / scale;
    *ret++ = ((-14259 * a) + (19156 * b)) / scale;
    *ret++ = ((-440 * a)) / scale;
    *ret++ = ((-2244 * b) + (-8755 * c)) / scale;
    *ret++ = ((-12238 * b)) / scale;
    *ret++ = ((6572 * a)) / scale;
  #endif
}

static inline void q15xq7_q15_m_sparse_hardcode_mulvec_u(const Q15_T* vec,
                                                         Q15_T* ret,
                                                         SCALE_T scmat,
                                                         SCALE_T scvec,
                                                         SCALE_T H1) {
  Q31_T a = *vec++;
  Q31_T b = *vec++;
  Q31_T c = *vec++;
  Q31_T d = *vec++;
  Q31_T e = *vec++;
  Q31_T f = *vec++;
  Q31_T g = *vec++;
  Q31_T h = *vec++;
  Q31_T i = *vec++;
  Q31_T j = *vec++;
  Q31_T k = *vec++;
  Q31_T l = *vec++;
  Q31_T m = *vec++;
  Q31_T n = *vec++;
  Q31_T o = *vec++;
  Q31_T p = *vec++;

  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
    *ret++ = ((8953 * a) + (-26241 * e) + (-10713 * f) + (10523 * g) + (9994 * j) + (-6916 * k) + (-2193 * p)) >> scale;
    *ret++ = ((5151 * g) + (-10210 * k) + (-12062 * p)) >> scale;
    *ret++ = ((11915 * b) + (5412 * k) + (4310 * o) + (6230 * p)) >> scale;
    *ret++ = ((6435 * a) + (-9015 * c) + (-6637 * e) + (-8736 * g) + (-19550 * k)) >> scale;
    *ret++ = ((10735 * c)) >> scale;
    *ret++ = ((-17159 * c) + (10213 * d) + (25321 * e) + (-10076 * f) + (-20389 * h) + (29157 * j) + (6749 * k) + (7074 * m) + (-22597 * n)) >> scale;
    *ret++ = ((-17935 * b) + (11239 * d) + (15771 * e) + (7358 * g) + (20463 * i) + (12893 * o)) >> scale;
    *ret++ = ((-7508 * c) + (10577 * f) + (-7728 * k) + (7597 * n) + (-6135 * p)) >> scale;
    *ret++ = ((24570 * b) + (-16464 * f) + (6168 * j) + (-12736 * n)) >> scale;
    *ret++ = ((-17711 * e) + (17371 * f) + (-11251 * j) + (-9761 * k) + (28019 * n) + (-12281 * p)) >> scale;
    *ret++ = ((-4866 * a) + (17486 * d) + (6405 * l)) >> scale;
    *ret++ = ((-20050 * b) + (-2780 * g) + (12651 * i) + (-7660 * l) + (29316 * o)) >> scale;
    *ret++ = ((-8711 * c) + (5609 * e) + (-6933 * f) + (-9216 * j) + (-4419 * k) + (-7222 * l) + (12369 * m) + (3576 * n) + (-2898 * o) + (-8376 * p)) >> scale;
    *ret++ = ((-4524 * a) + (12069 * e) + (-13209 * j)) >> scale;
    *ret++ = ((2766 * b) + (-4524 * i) + (985 * l) + (-8617 * o)) >> scale;
    *ret++ = ((1291 * d) + (-6256 * h)) >> scale;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of Q31_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * H1;
    *ret++ = ((8953 * a) + (-26241 * e) + (-10713 * f) + (10523 * g) + (9994 * j) + (-6916 * k) + (-2193 * p)) / scale;
    *ret++ = ((5151 * g) + (-10210 * k) + (-12062 * p)) / scale;
    *ret++ = ((11915 * b) + (5412 * k) + (4310 * o) + (6230 * p)) / scale;
    *ret++ = ((6435 * a) + (-9015 * c) + (-6637 * e) + (-8736 * g) + (-19550 * k)) / scale;
    *ret++ = ((10735 * c)) / scale;
    *ret++ = ((-17159 * c) + (10213 * d) + (25321 * e) + (-10076 * f) + (-20389 * h) + (29157 * j) + (6749 * k) + (7074 * m) + (-22597 * n)) / scale;
    *ret++ = ((-17935 * b) + (11239 * d) + (15771 * e) + (7358 * g) + (20463 * i) + (12893 * o)) / scale;
    *ret++ = ((-7508 * c) + (10577 * f) + (-7728 * k) + (7597 * n) + (-6135 * p)) / scale;
    *ret++ = ((24570 * b) + (-16464 * f) + (6168 * j) + (-12736 * n)) / scale;
    *ret++ = ((-17711 * e) + (17371 * f) + (-11251 * j) + (-9761 * k) + (28019 * n) + (-12281 * p)) / scale;
    *ret++ = ((-4866 * a) + (17486 * d) + (6405 * l)) / scale;
    *ret++ = ((-20050 * b) + (-2780 * g) + (12651 * i) + (-7660 * l) + (29316 * o)) / scale;
    *ret++ = ((-8711 * c) + (5609 * e) + (-6933 * f) + (-9216 * j) + (-4419 * k) + (-7222 * l) + (12369 * m) + (3576 * n) + (-2898 * o) + (-8376 * p)) / scale;
    *ret++ = ((-4524 * a) + (12069 * e) + (-13209 * j)) / scale;
    *ret++ = ((2766 * b) + (-4524 * i) + (985 * l) + (-8617 * o)) / scale;
    *ret++ = ((1291 * d) + (-6256 * h)) / scale;
  #endif
}

static inline void q15_m_sparse_hardcode_mulvec_w(const Q15_T* vec, Q15_T* ret,
                                                  SCALE_T scmat, SCALE_T scvec,
                                                  SCALE_T H1) {
  Q31_T a = *vec++;
  vec++;
  Q31_T c = *vec++;
  Q31_T d = *vec++;
  Q31_T e = *vec++;
  Q31_T f = *vec++;
  Q31_T g = *vec++;
  Q31_T h = *vec++;
  vec++;
  Q31_T j = *vec++;
  Q31_T k = *vec++;
  Q31_T l = *vec++;
  Q31_T m = *vec++;
  Q31_T n = *vec++;
  Q31_T o = *vec++;
  Q31_T p = *vec++;

  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
    *ret++ = ((-8196 * c) + (-11399 * h) + (-14351 * m)) >> scale;
    *ret++ = ((-15989 * d) + (10915 * g) + (-2935 * h) + (17708 * m)) >> scale;
    *ret++ = ((19682 * c) + (18093 * e) + (8585 * f) + (8279 * j) + (10240 * m) + (12059 * p)) >> scale;
    *ret++ = ((13664 * a) + (7592 * c) + (-7837 * e) + (6740 * g) + (-4843 * j) + (-21691 * l) + (12301 * m) + (-14347 * o) + (9897 * p)) >> scale;
    *ret++ = ((17420 * c) + (-8762 * m) + (-17601 * p)) >> scale;
    *ret++ = ((10075 * a) + (-13146 * d) + (2591 * e) + (-13854 * g) + (-22534 * k) + (5013 * m) + (7177 * n)) >> scale;
    *ret++ = ((10095 * d) + (11074 * k) + (17713 * m) + (3119 * p)) >> scale;
    *ret++ = ((-9882 * c) + (11674 * m) + (-24534 * p)) >> scale;
    *ret++ = ((9482 * c)) >> scale;
    *ret++ = ((8105 * c) + (-7406 * e) + (13297 * g) + (-26392 * m) + (-11338 * n) + (24616 * p)) >> scale;
    *ret++ = ((-17187 * d) + (-12527 * e) + (-5868 * h) + (-14600 * k) + (-10216 * n)) >> scale;
    *ret++ = ((12061 * c) + (-23490 * d) + (8211 * e) + (7285 * k) + (-19634 * m) + (14401 * p)) >> scale;
    *ret++ = ((-8136 * a) + (9304 * c) + (3690 * e) + (-8786 * f) + (-8288 * h) + (-5691 * j) + (12173 * m) + (5844 * p)) >> scale;
    *ret++ = ((15839 * c) + (-16882 * h) + (17992 * k) + (9481 * l) + (7225 * m) + (9236 * n) + (7722 * p)) >> scale;
    *ret++ = ((9716 * c) + (-18332 * d) + (16173 * k) + (5626 * p)) >> scale;
    *ret++ = ((10633 * h)) >> scale;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of Q31_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * H1;
    *ret++ = ((-8196 * c) + (-11399 * h) + (-14351 * m)) / scale;
    *ret++ = ((-15989 * d) + (10915 * g) + (-2935 * h) + (17708 * m)) / scale;
    *ret++ = ((19682 * c) + (18093 * e) + (8585 * f) + (8279 * j) + (10240 * m) + (12059 * p)) / scale;
    *ret++ = ((13664 * a) + (7592 * c) + (-7837 * e) + (6740 * g) + (-4843 * j) + (-21691 * l) + (12301 * m) + (-14347 * o) + (9897 * p)) / scale;
    *ret++ = ((17420 * c) + (-8762 * m) + (-17601 * p)) / scale;
    *ret++ = ((10075 * a) + (-13146 * d) + (2591 * e) + (-13854 * g) + (-22534 * k) + (5013 * m) + (7177 * n)) / scale;
    *ret++ = ((10095 * d) + (11074 * k) + (17713 * m) + (3119 * p)) / scale;
    *ret++ = ((-9882 * c) + (11674 * m) + (-24534 * p)) / scale;
    *ret++ = ((9482 * c)) / scale;
    *ret++ = ((8105 * c) + (-7406 * e) + (13297 * g) + (-26392 * m) + (-11338 * n) + (24616 * p)) / scale;
    *ret++ = ((-17187 * d) + (-12527 * e) + (-5868 * h) + (-14600 * k) + (-10216 * n)) / scale;
    *ret++ = ((12061 * c) + (-23490 * d) + (8211 * e) + (7285 * k) + (-19634 * m) + (14401 * p)) / scale;
    *ret++ = ((-8136 * a) + (9304 * c) + (3690 * e) + (-8786 * f) + (-8288 * h) + (-5691 * j) + (12173 * m) + (5844 * p)) / scale;
    *ret++ = ((15839 * c) + (-16882 * h) + (17992 * k) + (9481 * l) + (7225 * m) + (9236 * n) + (7722 * p)) / scale;
    *ret++ = ((9716 * c) + (-18332 * d) + (16173 * k) + (5626 * p)) / scale;
    *ret++ = ((10633 * h)) / scale;
  #endif
}

static inline void q15_m_sparse_hardcode_mulvec_u(const Q15_T* vec, Q15_T* ret,
                                                  SCALE_T scmat, SCALE_T scvec,
                                                  SCALE_T H1) {
  Q31_T a = *vec++;
  Q31_T b = *vec++;
  Q31_T c = *vec++;
  Q31_T d = *vec++;
  Q31_T e = *vec++;
  Q31_T f = *vec++;
  Q31_T g = *vec++;
  Q31_T h = *vec++;
  Q31_T i = *vec++;
  Q31_T j = *vec++;
  Q31_T k = *vec++;
  Q31_T l = *vec++;
  Q31_T m = *vec++;
  Q31_T n = *vec++;
  Q31_T o = *vec++;
  Q31_T p = *vec++;

  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
    *ret++ = ((3526 * a) + (3527 * h) + (-13454 * i) + (-6020 * m) + (7062 * p)) >> scale;
    *ret++ = ((-8879 * i) + (-6760 * m) + (-6458 * n) + (8387 * p)) >> scale;
    *ret++ = 0;
    *ret++ = ((4471 * j) + (7221 * n)) >> scale;
    *ret++ = ((3745 * b) + (5464 * f) + (12124 * g) + (7620 * h) + (5083 * k) + (7550 * m)) >> scale;
    *ret++ = ((-7819 * e) + (14245 * f) + (-6172 * g) + (14408 * h) + (6579 * p)) >> scale;
    *ret++ = ((-7411 * c) + (-6447 * e) + (-6362 * i) + (6985 * j) + (-6077 * k) + (-7037 * m)) >> scale;
    *ret++ = ((-7566 * b) + (-11012 * f) + (4536 * h) + (-8462 * i) + (7994 * j) + (-3968 * k) + (4739 * l)) >> scale;
    *ret++ = ((18699 * a) + (-8845 * c) + (13114 * i) + (-5723 * n) + (11032 * p)) >> scale;
    *ret++ = ((9152 * d) + (11232 * e) + (5191 * h) + (-14378 * i) + (-8078 * j) + (5689 * l) + (5458 * o) + (-6078 * p)) >> scale;
    *ret++ = ((-8189 * b) + (6066 * f) + (11913 * g) + (8255 * h) + (-6820 * i)) >> scale;
    *ret++ = ((5444 * c) + (-6389 * d) + (7254 * f) + (-7959 * i) + (-4719 * j) + (12197 * k) + (8632 * p)) >> scale;
    *ret++ = ((11737 * a) + (11763 * m)) >> scale;
    *ret++ = ((-5826 * f) + (7696 * g) + (-4719 * p)) >> scale;
    *ret++ = ((-5149 * b) + (13139 * g) + (6609 * h) + (-4058 * l) + (5413 * o) + (7630 * p)) >> scale;
    *ret++ = ((-5371 * a) + (-10029 * b) + (-6352 * g) + (-15090 * i) + (-8446 * l) + (-6262 * m)) >> scale;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of Q31_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * H1;
    *ret++ = ((3526 * a) + (3527 * h) + (-13454 * i) + (-6020 * m) + (7062 * p)) / scale;
    *ret++ = ((-8879 * i) + (-6760 * m) + (-6458 * n) + (8387 * p)) / scale;
    *ret++ = 0;
    *ret++ = ((4471 * j) + (7221 * n)) / scale;
    *ret++ = ((3745 * b) + (5464 * f) + (12124 * g) + (7620 * h) + (5083 * k) + (7550 * m)) / scale;
    *ret++ = ((-7819 * e) + (14245 * f) + (-6172 * g) + (14408 * h) + (6579 * p)) / scale;
    *ret++ = ((-7411 * c) + (-6447 * e) + (-6362 * i) + (6985 * j) + (-6077 * k) + (-7037 * m)) / scale;
    *ret++ = ((-7566 * b) + (-11012 * f) + (4536 * h) + (-8462 * i) + (7994 * j) + (-3968 * k) + (4739 * l)) / scale;
    *ret++ = ((18699 * a) + (-8845 * c) + (13114 * i) + (-5723 * n) + (11032 * p)) / scale;
    *ret++ = ((9152 * d) + (11232 * e) + (5191 * h) + (-14378 * i) + (-8078 * j) + (5689 * l) + (5458 * o) + (-6078 * p)) / scale;
    *ret++ = ((-8189 * b) + (6066 * f) + (11913 * g) + (8255 * h) + (-6820 * i)) / scale;
    *ret++ = ((5444 * c) + (-6389 * d) + (7254 * f) + (-7959 * i) + (-4719 * j) + (12197 * k) + (8632 * p)) / scale;
    *ret++ = ((11737 * a) + (11763 * m)) / scale;
    *ret++ = ((-5826 * f) + (7696 * g) + (-4719 * p)) / scale;
    *ret++ = ((-5149 * b) + (13139 * g) + (6609 * h) + (-4058 * l) + (5413 * o) + (7630 * p)) / scale;
    *ret++ = ((-5371 * a) + (-10029 * b) + (-6352 * g) + (-15090 * i) + (-8446 * l) + (-6262 * m)) / scale;
  #endif
}

/**
 * @brief Performs the element-wise addition of two input tensors.
 * dim(ten1) = dim(ten2) = [nbatches][nrows][ncols][nchannels]
 * @param[in]       ten1      pointer to the first input tensor
 * @param[in]       ten2      pointer to the second input tensor
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       scten1    scaling factor for the first input tensor
 * @param[in]       scten2    scaling factor for the second input tensor
 * @param[in]       scret     scaling factor for the output tensor
 * @return          none
 * @example         ten1      = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  ten2      = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  nbatches  = 2
 *                  nrows     = 2
 *                  ncols     = 2
 *                  nchannels = 2
 *                  scten1    = 2
 *                  scten2    = 2
 *                  scret     = 1
 *                  ret       = { { {1324, 5452}, {3454, 3434} },
 *                                { {8788, 3410}, {5412, 8934} } },
 *                              { { {6894, 1210}, {6790, 5424} },
 *                                { {8976, 4538}, {9348, 9320} } }
 */
void q7_t_add(const Q7_T* ten1, const Q7_T* ten2, ITER_T nbatches,
              ITER_T nrows, ITER_T ncols, ITER_T nchannels, Q7_T* ret,
              SCALE_T scten1, SCALE_T scten2, SCALE_T scret);
void q15_t_add(const Q15_T* ten1, const Q15_T* ten2, ITER_T nbatches,
               ITER_T nrows, ITER_T ncols, ITER_T nchannels, Q15_T* ret,
               SCALE_T scten1, SCALE_T scten2, SCALE_T scret);
/**
 * @brief Performs the channel-wise addition of a bias term to the input tensor.
 * dim(ten) = dim(ret) = [nbatches][nrows][ncols][nchannels]; dim(vec) = [nchannels].
 * @param[in]       ten       pointer to the input tensor on which addition is to be performed
 * @param[in]       vec       pointer to the bias vector which is to be added
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       scten     scaling factor for the input tensor
 * @param[in]       scvec     scaling factor for the bias vector
 * @param[in]       scret     scaling factor for the output tensor
 * @return          none
 * @example         ten       = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  vec       = {8452, 2341}
 *                  nbatches  = 2
 *                  nrows     = 2
 *                  ncols     = 2
 *                  nchannels = 2
 *                  scten     = 1
 *                  scvec     = 2
 *                  scret     = 2
 *                  ret       = { { {2775, 3311}, {4072, 2305} },
 *                                { {6507, 2290}, {5051, 5055} } },
 *                              { { {5560, 1190}, {5740, 3300} },
 *                                { {6601, 2854}, {7019, 5248} } }
 */
void q7xq15_q7_t_add_vec(const Q7_T* ten, const Q15_T* const vec,
                         ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                         ITER_T nchannels, Q7_T* ret, SCALE_T scmat,
                         SCALE_T scvec, SCALE_T scret);
void q15_t_add_vec(const Q15_T* ten, const Q15_T* const vec, ITER_T nbatches,
                   ITER_T nrows, ITER_T ncols, ITER_T nchannels, Q15_T* ret,
                   SCALE_T scmat, SCALE_T scvec, SCALE_T scret);
/**
 * @brief Replace any negative element present in the tensor with zero and clips positive elements to the limit.
 * @param[in]       ten       pointer to tensor on which element-wise ReLU6 operation is to be applied
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       limit     upper threshold of the ReLU operation
 * @param[in]       div       scaling factor for the input tensor
 * @param[in]
 * @return          none
 * @example         ten       = { { {124, 53}, {45, 35} },
 *                                { {87, -11}, {54, 89} } },
 *                              { { {95, -12}, {90, 42} },
 *                                { {76, 39}, {93, 21} } }
 *                  nbatches  = 2
 *                  nrows     = 2
 *                  ncols     = 2
 *                  nchannels = 2
 *                  limit     = 64
 *                  div       = 1
 *                  ret       = { { {64, 53}, {45, 35} },
 *                                { {64, 0}, {54, 64} } },
 *                              { { {64, 0}, {64, 42} },
 *                                { {64, 39}, {64, 21} } }
 */
void q7_t_relu(const Q7_T* ten, ITER_T nbatches, ITER_T nrows, ITER_T ncols,
               ITER_T nchannels, Q7_T* ret, Q7_T limit, Q7_T div);
/**
 * @brief Computes the L2-Norm for each channel of the input tensor, and divides each number in that channel by it.
 * dim(ten) = dim(ret) = [nbatches][nrows][ncols][nchannels].
 * @param[in]       ten       pointer to tensor on which channel-wise L2-Norm operation is to be applied
 * @param[in]       nbatches  number of batches of the input tensor
 * @param[in]       nrows     number of rows of the input tensor
 * @param[in]       ncols     number of columns of the input tensor
 * @param[in]       nchannels number of channels of the input tensor
 * @param[out]      ret       pointer to the output tensor
 * @param[in]       scale_in  scale factor of the input tensor
 * @param[in]       scale_out scale factor of the output tensor
 * @return          none
 * @example         ten       = { { {1324, 5453}, {3454, 3435} },
 *                                { {8789, 3411}, {5412, 8934} } },
 *                              { { {6895, 1211}, {6790, 5425} },
 *                                { {8976, 4539}, {9348, 9321} } }
 *                  nbatches  = 2
 *                  nrows     = 2
 *                  ncols     = 2
 *                  nchannels = 2
 *                  scale_in  = 1
 *                  scale_out = 1
 *                  ret       = { { {662, 2726}, {1727, 1717} },
 *                                { {4394, 1705}, {2706, 4467} } },
 *                              { { {3447, 605}, {3395, 2712} },
 *                                { {4488, 2269}, {4674, 4660} } }
 */
void q15_t_l2_norm(const Q15_T* ten, ITER_T nbatches, ITER_T nrows,
                   ITER_T ncols, ITER_T nchannels, Q15_T* ret,
                   SCALE_T scale_in, SCALE_T scale_out);

/**
 * @brief Computes the maxpool operation on the input tensor with the given parameters.
 * @param[in]       input          pointer to the tensor on which convolution is to be performed
 * @param[in]       filter         pointer to the convolutional filter tensor
 * @param[out]      output         pointer to the output tensor
 * @param[in]       N              number of batches of the input tensor
 * @param[in]       H              number of rows of the input tensor
 * @param[in]       W              number of columns of the input tensor
 * @param[in]       CIn            number of channels of the input tensor
 * @param[in]       HF             number of rows of the convolutional filter
 * @param[in]       WF             number of columns of the convolutional filter
 * @param[in]       CF             number of channels of the convolutional filter
 * @param[in]       COut           number of channels of the output tensor
 * @param[in]       HOut           number of rows of the output tensor
 * @param[in]       WOut           number of columns of the output tensor
 * @param[in]       G              number of groups of convolutional filters
 * @param[in]       HPadU          padding over the top row
 * @param[in]       HPadD          padding under the bottom row
 * @param[in]       WPadL          padding before the leftmost column
 * @param[in]       WPadR          padding after the rightmost column
 * @param[in]       HStride        stride of the convolution filter along the rows, used for moving the receptive field horizontally within the larger image
 * @param[in]       WStride        stride of the convolution filter along the columns, used for moving the receptive field vertically within the larger image
 * @param[in]       HDilation      dilation of the convolution filter along the rows (number of skipped input rows between two consecutive filter rows is HDilation - 1)
 * @param[in]       WDilation      dilation of the convolution filter along the columns (number of skipped input columns between two consecutive filter rows is WDilation - 1)
 * @param[in]       scinput        scale of the input tensor
 * @param[in]       scoutput       scale of the output tensor
 * @param[in]       demote         scale factor for output variable demotion
 * @return          none
 * @example         Please refer the test-case: test_q15_convolution() in file: c_reference/tests/utils/test_quantized_utils.c
 */
void q7xq15_q7_convolution(const Q7_T* const input, const Q15_T* const filter,
  Q7_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote);
void q7xq15_q15_convolution(const Q7_T* const input, const Q15_T* const filter,
  Q15_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote);
void q15_convolution(const Q15_T* const input, const Q15_T* const filter,
  Q15_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote);

#endif
