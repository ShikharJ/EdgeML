// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"
#ifdef CMSISDSP
  #include <stdlib.h>
  #include "arm_math.h"
#endif

void v_q_add(const INT_T* vec1, const INT_T* vec2, ITER_T len,
             INT_T* ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  #ifdef CMSISDSP
    INT_T *ret2 = malloc(len * sizeof(INT_T));
    arm_shift_q15(vec1, -(scvec1 + scret), ret, len);
    arm_shift_q15(vec2, -(scvec2 + scret), ret2, len);
    arm_add_q15(ret, ret2, ret, len);
    free(ret2);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((vec1[i] >> (scvec1 + scret)) + (vec2[i] >> (scvec2 + scret)));
      #else
        ret[i] = ((vec1[i] / scvec1) / scret) + ((vec2[i] / scvec2) / scret);
      #endif
    }
  #endif
}

void v_q_sub(const INT_T* vec1, const INT_T* vec2, ITER_T len,
             INT_T* ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  #ifdef CMSISDSP
    INT_T *ret2 = malloc(len * sizeof(INT_T));
    arm_shift_q15(vec1, -(scvec1 + scret), ret, len);
    arm_shift_q15(vec2, -(scvec2 + scret), ret2, len);
    arm_sub_q15(ret, ret2, ret, len);
    free(ret2);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((vec1[i] >> (scvec1 + scret)) - (vec2[i] >> (scvec2 + scret)));
      #else
        ret[i] = ((vec1[i] / scvec1) / scret) - ((vec2[i] / scvec2) / scret);
      #endif
    }
  #endif
}

void v_q_hadamard(const INT_T* vec1, const INT_T* vec2, ITER_T len,
                  INT_T* ret, SCALE_T scvec1, SCALE_T scvec2) {
  #ifdef CMSISDSP
    // Need to test the variation with scaling first as well.
    arm_mult_q15(vec1, vec2, ret, len);
    arm_shift_q15(ret, -(scvec1 + scvec2), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((INTM_T)vec1[i] * (INTM_T)vec2[i]) >> (scvec1 + scvec2);
      #else
        ret[i] = ((((INTM_T)vec1[i] * (INTM_T)vec2[i]) / scvec1) / scvec2);
      #endif
    }
  #endif
}

void v_q_sigmoid(const INT_T* const vec, ITER_T len, INT_T* const ret, INT_T div,
                 INT_T add, INT_T sigmoid_limit, SCALE_T scale_in,
                 SCALE_T scale_out) {
  for (ITER_T i = 0; i < len; i++) {
    INT_T x = (vec[i] / div) + add;

    if (x >= sigmoid_limit) {
      ret[i] = sigmoid_limit << (scale_out - scale_in);
    } else if (x <= 0) {
      ret[i] = 0;
    } else {
      ret[i] = x << (scale_out - scale_in);
    }
  }
}

void v_q_tanh(const INT_T* const vec, ITER_T len, INT_T* const ret,
              SCALE_T scale_in, SCALE_T scale_out) {
  INT_T scale = (1 << scale_in);
  for (ITER_T i = 0; i < len; i++) {
    if (vec[i] >= scale) {
      ret[i] = scale;
    } else if (vec[i] <= -scale) {
      ret[i] = (-scale);
    } else {
      ret[i] = vec[i];
    }
    ret[i] <<= (scale_out - scale_in);
  }
}

void v_q_scalar_add(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  #ifdef CMSISDSP
    arm_shift_q15(vec, -(scvec + scret), ret, len);
    arm_offset_q15(ret, (scalar >> (scscalar + scret)), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((scalar >> (scscalar + scret)) + (vec[i] >> (scvec + scret)));
      #else
        ret[i] = ((scalar / scscalar) / scret) + ((vec[i] / scvec) / scret);
      #endif
    }
  #endif
}

void v_q_scalar_sub(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  #ifdef CMSISDSP
    arm_shift_q15(vec, -(scvec + scret), ret, len);
    arm_negate_q15(ret, ret, len);
    arm_offset_q15(ret, (scalar >> (scscalar + scret)), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((scalar >> (scscalar + scret)) - (vec[i] >> (scvec + scret)));
      #else
        ret[i] = ((scalar / scscalar) / scret) - ((vec[i] / scvec) / scret);
      #endif
    }
  #endif
}

void v_q_scalar_mul(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec) {
  #ifdef CMSISDSP
    arm_scale_q15(vec, scalar, -(scscalar + scvec), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((INTM_T)scalar * (INTM_T)vec[i]) >> (scscalar + scvec);
      #else
        ret[i] = ((((INTM_T)scalar * (INTM_T)vec[i]) / scscalar) / scvec);
      #endif
    }
  #endif
}

void m_q_mulvec(INT_T* mat, INT_T* vec, ITER_T nrows,
                ITER_T ncols, INT_T* ret, SCALE_T scmat, SCALE_T scvec,
                ITER_T H1, ITER_T H2) {
  #ifdef CMSISDSP
    INT_T tmp[ncols];
    arm_matrix_instance_q15 A, B, C;
    arm_mat_init_q15(&A, nrows, ncols, mat);
    arm_mat_init_q15(&B, ncols, 1, vec);
    arm_mat_init_q15(&C, nrows, 1, ret);
    arm_mat_mult_fast_q15(&A, &B, &C, &tmp);
  #else
    INTM_T tmp[ncols];
    for (ITER_T row = 0; row < nrows; row++) {
      INT_T* mat_offset = (INT_T*)mat + row * ncols;

      for (ITER_T col = 0; col < ncols; col++) {
        tmp[col] = ((INTM_T)(*mat_offset++) * (INTM_T)vec[col]);
      }

      ITER_T count = ncols, depth = 0;
      int divbytwo = 1;

      while (depth < (H1 + H2)) {
        if (depth >= H1)
          divbytwo = 0;

        for (ITER_T p = 0; p < ((ncols >> 1) + 1); p++) {
          INTM_T sum;
          if (p < (count >> 1)) {
            if (divbytwo == 1) {
              #ifdef SHIFT
                sum = (tmp[2 * p] >> 1) + (tmp[(2 * p) + 1] >> 1);
              #else
                sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
              #endif
            } else {
              sum = tmp[2 * p] + tmp[(2 * p) + 1];
            }
          } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
            if (divbytwo == 1) {
              #ifdef SHIFT
                sum = (tmp[2 * p] >> 1);
              #else
                sum = tmp[2 * p] / 2;
              #endif
            } else {
              sum = tmp[2 * p];
            }
          } else {
            sum = 0;
          }

          tmp[p] = sum;
        }
        count = (count + 1) >> 1;
        depth++;
      }
      #ifdef SHIFT
        ret[row] = (tmp[0] >> (scmat + scvec));
      #else
        ret[row] = ((tmp[0] / scmat) / scvec);
      #endif
    }
  #endif
}
