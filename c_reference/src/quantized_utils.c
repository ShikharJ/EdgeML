// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stddef.h>
#include <string.h>
#include "quantized_utils.h"

void q15_v_add(const Q15_T* vec1, const Q15_T* vec2, ITER_T len, Q15_T* ret,
               SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret, SCALE_T demote) {
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

void q7_v_sub(const Q7_T* vec1, const Q7_T* vec2, ITER_T len, Q7_T* ret,
              SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
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

void q15_v_sub(const Q15_T* vec1, const Q15_T* vec2, ITER_T len, Q15_T* ret,
               SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
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


void q7_v_hadamard(const Q7_T* vec1, const Q7_T* vec2, ITER_T len, Q7_T* ret,
                   SCALE_T scvec1, SCALE_T scvec2) {
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

void q15_v_hadamard(const Q15_T* vec1, const Q15_T* vec2, ITER_T len,
                    Q15_T* ret, SCALE_T scvec1, SCALE_T scvec2) {
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

void q15_v_sigmoid(const Q15_T* vec, ITER_T len, Q15_T* ret, Q15_T div,
                   Q15_T add, Q15_T sigmoid_limit, SCALE_T scale_in,
                   SCALE_T scale_out, ITER_T use_tables) {
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

void q15_v_tanh(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scale_in,
                SCALE_T scale_out, ITER_T use_tables) {
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

void q15_v_scalar_add(Q15_T scalar, const Q15_T* vec, ITER_T len, Q15_T* ret,
                      SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
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

void q15_v_scalar_sub(Q15_T scalar, const Q15_T* vec, ITER_T len, Q15_T* ret,
                      SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
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

void q15_v_scalar_mul(Q15_T scalar, const Q15_T* vec, ITER_T len, Q15_T* ret,
                      SCALE_T scscalar, SCALE_T scvec) {
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

void q15_v_argmax(const Q15_T* const vec, ITER_T len, ITER_T* const ret) {
  Q15_T max_value = vec[0];
  ITER_T max_index = 0;

  for (ITER_T i = 1; i < len; i++) {
    if (max_value < vec[i]) {
      max_index = i;
      max_value = vec[i];
    }
  }

  *ret = max_index;
}

void q15_v_scale_up(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scvec) {
  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = ((*vec++) << scvec);
        *ret++ = ((*vec++) << scvec);
        *ret++ = ((*vec++) << scvec);
        *ret++ = ((*vec++) << scvec);
      #else
        *ret++ = ((*vec++) * scvec);
        *ret++ = ((*vec++) * scvec);
        *ret++ = ((*vec++) * scvec);
        *ret++ = ((*vec++) * scvec);
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = ((*vec++) << scvec);
    #else
      *ret++ = ((*vec++) * scvec);
    #endif
  }
}

void q15_v_scale_down(const Q15_T* vec, ITER_T len, Q15_T* ret, SCALE_T scvec) {
  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = ((*vec++) >> scvec);
        *ret++ = ((*vec++) >> scvec);
        *ret++ = ((*vec++) >> scvec);
        *ret++ = ((*vec++) >> scvec);
      #else
        *ret++ = ((*vec++) / scvec);
        *ret++ = ((*vec++) / scvec);
        *ret++ = ((*vec++) / scvec);
        *ret++ = ((*vec++) / scvec);
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = ((*vec++) >> scvec);
    #else
      *ret++ = ((*vec++) / scvec);
    #endif
  }
}

void q15xq7_q15_m_mulvec(const Q15_T* mat, const Q7_T* const vec, ITER_T nrows,
                         ITER_T ncols, Q15_T* ret, SCALE_T scmat,
                         SCALE_T scvec, SCALE_T H1, SCALE_T H2) {
  Q31_T sum;
  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
  #else
    SCALE_T scale = scmat * scvec * H1;
  #endif

  while (nrows--) {
    sum = 0;
    ITER_T cols = ncols;
    const Q7_T* vec_offset = (const Q7_T*)vec;

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = cols >> 2;
      cols = cols % 4;
      while (len_unroll--) {
        sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
        sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
        sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
        sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
      }
    #endif

    while (cols--) {
      sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
    }

    #ifdef SHIFT
      *ret++ = (sum >> scale);
    #else
      *ret++ = (sum / scale);
    #endif
  }
}

void q15_m_mulvec(const Q15_T* mat, const Q15_T* const vec, ITER_T nrows,
                  ITER_T ncols, Q15_T* ret, SCALE_T scmat, SCALE_T scvec,
                  SCALE_T H1, SCALE_T H2) {
  Q63_T sum;
  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of Q31_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * H1;
  #endif

  while (nrows--) {
    sum = 0;
    ITER_T cols = ncols;
    const Q15_T* vec_offset = (const Q15_T*)vec;

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = cols >> 2;
      cols = cols % 4;
      while (len_unroll--) {
        sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
        sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
        sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
        sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
      }
    #endif

    while (cols--) {
      sum += (Q31_T)(*mat++) * (Q31_T)(*vec_offset++);
    }

    #ifdef SHIFT
      *ret++ = (sum >> scale);
    #else
      *ret++ = (sum / scale);
    #endif
  }
}

void q15xq7_q15_m_sparse_mulvec(const ITER_T* row_indices,
                                const Q15_T* mat_values, const Q7_T* vec,
                                ITER_T nrows, ITER_T ncols, Q15_T* ret,
                                SCALE_T scmat, SCALE_T scvec, SCALE_T H1,
                                SCALE_T H2) {
  ITER_T index;
  Q31_T vec_offset;
  memset(ret, 0, nrows * sizeof(Q15_T));
  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of Q31_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * H1;
  #endif

  while (ncols--) {
    index = *row_indices++;
    vec_offset = *vec++;

    while (index != 0) {
      #ifdef SHIFT
        ret[index - 1] += ((*mat_values++) * vec_offset) >> scale;
      #else
        ret[index - 1] += ((*mat_values++) * vec_offset) / scale;
      #endif
      index = *row_indices++;
    }
  }
}

void q15_m_sparse_mulvec(const ITER_T* row_indices, const Q15_T* mat_values,
                         const Q15_T* vec, ITER_T nrows, ITER_T ncols,
                         Q15_T* ret, SCALE_T scmat, SCALE_T scvec, SCALE_T H1,
                         SCALE_T H2) {
  ITER_T index;
  Q31_T vec_offset;
  memset(ret, 0, nrows * sizeof(Q15_T));
  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of Q31_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * H1;
  #endif

  while (ncols--) {
    index = *row_indices++;
    vec_offset = *vec++;

    while (index != 0) {
      #ifdef SHIFT
        ret[index - 1] += ((*mat_values++) * vec_offset) >> scale;
      #else
        ret[index - 1] += ((*mat_values++) * vec_offset) / scale;
      #endif
      index = *row_indices++;
    }
  }
}

void q15xq7_q15_m_sparse_hardcode_mulvec_w(const Q7_T* vec, Q15_T* ret,
                                           SCALE_T scmat, SCALE_T scvec,
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

void q15xq7_q15_m_sparse_hardcode_mulvec_u(const Q15_T* vec, Q15_T* ret,
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

void q15_m_sparse_hardcode_mulvec_w(const Q15_T* vec, Q15_T* ret, SCALE_T scmat,
                                    SCALE_T scvec, SCALE_T H1) {
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

void q15_m_sparse_hardcode_mulvec_u(const Q15_T* vec, Q15_T* ret, SCALE_T scmat,
                                    SCALE_T scvec, SCALE_T H1) {
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

void q7_t_add(const Q7_T* ten1, const Q7_T* ten2, ITER_T nbatches, ITER_T nrows,
              ITER_T ncols, ITER_T nchannels, Q7_T* ret, SCALE_T scten1,
              SCALE_T scten2, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;

  #ifdef SHIFT
    SCALE_T scaleten1 = scten1 + scret;
    SCALE_T scaleten2 = scten2 + scret;
  #else
    SCALE_T scaleten1 = scten1 * scret;
    SCALE_T scaleten2 = scten2 * scret;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
        *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
        *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
        *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
      #else
        *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
        *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
        *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
        *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
    #else
      *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
    #endif
  }
}

void q15_t_add(const Q15_T* ten1, const Q15_T* ten2, ITER_T nbatches,
               ITER_T nrows, ITER_T ncols, ITER_T nchannels, Q15_T* ret,
               SCALE_T scten1, SCALE_T scten2, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  #ifdef SHIFT
    SCALE_T scaleten1 = scten1 + scret;
    SCALE_T scaleten2 = scten2 + scret;
  #else
    SCALE_T scaleten1 = scten1 * scret;
    SCALE_T scaleten2 = scten2 * scret;
  #endif

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      #ifdef SHIFT
        *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
        *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
        *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
        *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
      #else
        *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
        *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
        *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
        *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
      #endif
    }
  #endif

  while (len--) {
    #ifdef SHIFT
      *ret++ = ((*ten1++ >> scaleten1) + (*ten2++ >> scaleten2));
    #else
      *ret++ = ((*ten1++ / scaleten1) + (*ten2++ / scaleten2));
    #endif
  }
}

void q7xq15_q7_t_add_vec(const Q7_T* ten, const Q15_T* const vec,
                         ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                         ITER_T nchannels, Q7_T* ret, SCALE_T scten,
                         SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols;
  #ifdef SHIFT
    SCALE_T scaleten = scten + scret;
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scaleten = scten * scret;
    SCALE_T scalevec = scvec * scret;
  #endif

  while (len--) {
    ITER_T channels = nchannels;
    const Q15_T* vec_offset = (const Q15_T*)vec;

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = channels >> 2;
      channels = channels % 4;
      while (len_unroll--) {
        #ifdef SHIFT
          *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
          *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
          *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
          *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
        #else
          *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
          *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
          *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
          *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
        #endif
      }
    #endif

    while (channels--) {
      #ifdef SHIFT
        *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
      #else
        *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
      #endif
    }
  }
}

void q15_t_add_vec(const Q15_T* ten, const Q15_T* const vec,
                   ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                   ITER_T nchannels, Q15_T* ret, SCALE_T scten,
                   SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols;
  #ifdef SHIFT
    SCALE_T scaleten = scten + scret;
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scaleten = scten * scret;
    SCALE_T scalevec = scvec * scret;
  #endif

  while (len--) {
    ITER_T channels = nchannels;
    const Q15_T* vec_offset = (const Q15_T*)vec;

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = channels >> 2;
      channels = channels % 4;
      while (len_unroll--) {
        #ifdef SHIFT
          *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
          *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
          *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
          *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
        #else
          *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
          *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
          *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
          *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
        #endif
      }
    #endif

    while (channels--) {
      #ifdef SHIFT
        *ret++ = ((*ten++ >> scaleten) + (*vec_offset++ >> scalevec));
      #else
        *ret++ = ((*ten++ / scaleten) + (*vec_offset++ / scalevec));
      #endif
    }
  }
}

void q7_t_relu(const Q7_T* ten, ITER_T nbatches, ITER_T nrows, ITER_T ncols,
               ITER_T nchannels, Q7_T* ret, Q7_T limit, Q7_T div) {
  ITER_T len = nbatches * nrows * ncols * nchannels;

  #ifdef LOOP_UNROLL
    ITER_T len_unroll = len >> 2;
    len = len % 4;
    while (len_unroll--) {
      *ret++ = q7_relu(*ten++, limit) / div;
      *ret++ = q7_relu(*ten++, limit) / div;
      *ret++ = q7_relu(*ten++, limit) / div;
      *ret++ = q7_relu(*ten++, limit) / div;
    }
  #endif

  while (len--) {
    *ret++ = q7_relu(*ten++, limit) / div;
  }
}

void q15_t_l2_norm(const Q15_T* ten, ITER_T nbatches, ITER_T nrows,
                   ITER_T ncols, ITER_T nchannels, Q15_T* ret,
                   SCALE_T scale_in, SCALE_T scale_out) {
  ITER_T len = nbatches * nrows * ncols;
  #ifndef SHIFT
    SCALE_T scdiv = (1 << scale_out);
  #endif

  for (ITER_T i = 0; i < len; i++) {
    Q31_T sum_square = 0;
    ITER_T channels = nchannels;
    const Q15_T* ten_offset = ten;

    #ifdef LOOP_UNROLL
      ITER_T len_unroll = channels >> 2;
      channels = channels % 4;

      while (len_unroll--) {
        Q31_T w = *ten_offset++;
        Q31_T x = *ten_offset++;
        Q31_T y = *ten_offset++;
        Q31_T z = *ten_offset++;

        sum_square += ((w * w) >> (2 * scale_out));
        sum_square += ((x * x) >> (2 * scale_out));
        sum_square += ((y * y) >> (2 * scale_out));
        sum_square += ((z * z) >> (2 * scale_out));
      }
    #endif

    while (channels--) {
      Q31_T w = *ten_offset++;
      sum_square += ((w * w) >> (2 * scale_out));
    }

    Q15_T inverse_norm_low = 1;
    Q15_T inverse_norm_high = (1 << (scale_out - 1));
    Q31_T one = (1 << (-(2 * scale_in + 2)));

    while (inverse_norm_low + 1 < inverse_norm_high) {
      Q15_T mid = ((inverse_norm_high + inverse_norm_low) >> 1);

      if ((Q63_T)sum_square * mid * mid > one) {
        inverse_norm_high = mid;
      } else {
        inverse_norm_low = mid;
      }
    }

    channels = nchannels;
    #ifdef LOOP_UNROLL
      len_unroll = channels >> 2;
      channels = channels % 4;

      while (len_unroll--) {
        #ifdef SHIFT
          *ret++  = ((*ten++) >> scale_out) * inverse_norm_low;
          *ret++  = ((*ten++) >> scale_out) * inverse_norm_low;
          *ret++  = ((*ten++) >> scale_out) * inverse_norm_low;
          *ret++  = ((*ten++) >> scale_out) * inverse_norm_low;
        #else
          *ret++  = ((*ten++) / scdiv) * inverse_norm_low;
          *ret++  = ((*ten++) / scdiv) * inverse_norm_low;
          *ret++  = ((*ten++) / scdiv) * inverse_norm_low;
          *ret++  = ((*ten++) / scdiv) * inverse_norm_low;
        #endif
      }
    #endif

    while (channels--) {
      #ifdef SHIFT
        *ret++  = ((*ten++) >> scale_out) * inverse_norm_low;
      #else
        *ret++  = ((*ten++) / scdiv) * inverse_norm_low;
      #endif
    }
  }
}

void q7xq15_q7_convolution(const Q7_T* const input, const Q15_T* const filter,
  Q7_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote) {
  S_ITER_T HOffsetFL = ((HF - 1) >> 1);
  S_ITER_T HOffsetFR = (HF >> 1);
  S_ITER_T WOffsetFL = ((WF - 1) >> 1);
  S_ITER_T WOffsetFR = (WF >> 1);

  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * HOffsetFL) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * WOffsetFL) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * HOffsetFR) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * WOffsetFR) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T GOffsetF = HF * HOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  Q31_T sum;
  #ifdef SHIFT
    SCALE_T scale = scinput + scoutput + demote;
  #else
    SCALE_T scale = scinput * scoutput * demote;
  #endif
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut + HIndexOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF + NIndexIn;
          ITER_T GIndexF = g * GOffsetF;
          Q7_T* output_offset = ((Q7_T*)output) + g * COut + WIndexOut;
          for (ITER_T c = 0; c < COut; c++) {

            sum = 0;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              if ((hoffset < 0) || (hoffset >= (S_ITER_T)H)) {
                continue;
              }
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn + CIndexIn;
              ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * HOffsetF + GIndexF + c;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                if ((woffset < 0) || (woffset >= (S_ITER_T)W)) {
                  continue;
                }
                const Q7_T* input_offset = ((const Q7_T*)input) + ((ITER_T)woffset) * CIn + HIndexIn;
                const Q15_T* filter_offset = ((const Q15_T*)filter) + ((ITER_T)(wf + WOffsetFL)) * WOffsetF + HIndexF;
                ITER_T channels = CF;

                #ifdef LOOP_UNROLL
                  ITER_T len_unroll = CF >> 2;
                  channels = CF % 4;
                  while (len_unroll--) {
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                  }
                #endif

                while (channels--) {
                  sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                  filter_offset += COut;
                }
              }
            }

            #ifdef SHIFT
              *output_offset++ = (sum >> scale);
            #else
              *output_offset++ = (sum / scale);
            #endif
          }
        }
      }
    }
  }
}

void q7xq15_q15_convolution(const Q7_T* const input, const Q15_T* const filter,
  Q15_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote) {
  S_ITER_T HOffsetFL = ((HF - 1) >> 1);
  S_ITER_T HOffsetFR = (HF >> 1);
  S_ITER_T WOffsetFL = ((WF - 1) >> 1);
  S_ITER_T WOffsetFR = (WF >> 1);

  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * HOffsetFL) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * WOffsetFL) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * HOffsetFR) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * WOffsetFR) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T GOffsetF = HF * HOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  Q31_T sum;
  #ifdef SHIFT
    SCALE_T scale = scinput + scoutput + demote;
  #else
    SCALE_T scale = scinput * scoutput * demote;
  #endif
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut + HIndexOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF + NIndexIn;
          ITER_T GIndexF = g * GOffsetF;
          Q15_T* output_offset = ((Q15_T*)output) + g * COut + WIndexOut;
          for (ITER_T c = 0; c < COut; c++) {

            sum = 0;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              if ((hoffset < 0) || (hoffset >= (S_ITER_T)H)) {
                continue;
              }
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn + CIndexIn;
              ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * HOffsetF + GIndexF + c;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                if ((woffset < 0) || (woffset >= (S_ITER_T)W)) {
                  continue;
                }
                const Q7_T* input_offset = ((const Q7_T*)input) + ((ITER_T)woffset) * CIn + HIndexIn;
                const Q15_T* filter_offset = ((const Q15_T*)filter) + ((ITER_T)(wf + WOffsetFL)) * WOffsetF + HIndexF;
                ITER_T channels = CF;

                #ifdef LOOP_UNROLL
                  ITER_T len_unroll = CF >> 2;
                  channels = CF % 4;
                  while (len_unroll--) {
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                  }
                #endif

                while (channels--) {
                  sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                  filter_offset += COut;
                }
              }
            }

            #ifdef SHIFT
              *output_offset++ = (sum >> scale);
            #else
              *output_offset++ = (sum / scale);
            #endif
          }
        }
      }
    }
  }
}

void q15_convolution(const Q15_T* const input, const Q15_T* const filter,
  Q15_T* const output, ITER_T N, ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF,
  ITER_T WF, ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
  S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
  ITER_T HStride, ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
  SCALE_T scinput, SCALE_T scoutput, SCALE_T demote) {
  S_ITER_T HOffsetFL = ((HF - 1) >> 1);
  S_ITER_T HOffsetFR = (HF >> 1);
  S_ITER_T WOffsetFL = ((WF - 1) >> 1);
  S_ITER_T WOffsetFR = (WF >> 1);

  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * HOffsetFL) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * WOffsetFL) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * HOffsetFR) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * WOffsetFR) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T GOffsetF = HF * HOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  Q63_T sum;
  #ifdef SHIFT
    SCALE_T scale = scinput + scoutput + demote;
  #else
    SCALE_T scale = scinput * scoutput * demote;
  #endif
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut + HIndexOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF + NIndexIn;
          ITER_T GIndexF = g * GOffsetF;
          Q15_T* output_offset = ((Q15_T*)output) + g * COut + WIndexOut;
          for (ITER_T c = 0; c < COut; c++) {

            sum = 0;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              if ((hoffset < 0) || (hoffset >= (S_ITER_T)H)) {
                continue;
              }
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn + CIndexIn;
              ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * HOffsetF + GIndexF + c;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                if ((woffset < 0) || (woffset >= (S_ITER_T)W)) {
                  continue;
                }
                const Q15_T* input_offset = ((const Q15_T*)input) + ((ITER_T)woffset) * CIn + HIndexIn;
                const Q15_T* filter_offset = ((const Q15_T*)filter) + ((ITER_T)(wf + WOffsetFL)) * WOffsetF + HIndexF;
                ITER_T channels = CF;

                #ifdef LOOP_UNROLL
                  ITER_T len_unroll = CF >> 2;
                  channels = CF % 4;
                  while (len_unroll--) {
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                    sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                    filter_offset += COut;
                  }
                #endif

                while (channels--) {
                  sum += ((Q31_T)(*input_offset++)) * ((Q31_T)(*filter_offset));
                  filter_offset += COut;
                }
              }
            }

            #ifdef SHIFT
              *output_offset++ = (sum >> scale);
            #else
              *output_offset++ = (sum / scale);
            #endif
          }
        }
      }
    }
  }
}
