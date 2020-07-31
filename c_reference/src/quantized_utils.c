// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"

void v_q_treesum(INTM_T* const vec, ITER_T len, SCALE_T H1, SCALE_T H2) {
  ITER_T count = len, depth = 0;
  int divbytwo = 1;

  while (depth < (H1 + H2)) {
    if (depth >= H1) {
      divbytwo = 0;
    }

    for (ITER_T p = 0; p < ((len >> 1) + 1); p++) {
      if (p < (count >> 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1) + (vec[(2 * p) + 1] >> 1);
          #else
            vec[p] = vec[2 * p] / 2 + vec[(2 * p) + 1] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p] + vec[(2 * p) + 1];
        }
      } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1);
          #else
            vec[p] = vec[2 * p] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p];
        }
      } else {
        vec[p] = 0;
      }
    }
    count = (count + 1) >> 1;
    depth++;
  }
}

void v_q_add(const INT_T* vec1, const INT_T* vec2, ITER_T len, INT_T* ret,
             SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scalevec1 = scvec1 + scret;
    SCALE_T scalevec2 = scvec2 + scret;
  #else
    SCALE_T scalevec1 = scvec1 * scret;
    SCALE_T scalevec2 = scvec2 * scret;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
    #ifdef SHIFT
      *ret++ = ((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2));
      *ret++ = ((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2));
      *ret++ = ((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2));
      *ret++ = ((*vec1++ >> scalevec1) + (*vec2++ >> scalevec2));
    #else
      *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2));
      *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2));
      *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2));
      *ret++ = ((*vec1++ / scalevec1) + (*vec2++ / scalevec2));
    #endif
  }
}

void v_q_sub(const INT_T* vec1, const INT_T* vec2, ITER_T len, INT_T* ret,
             SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scalevec1 = scvec1 + scret;
    SCALE_T scalevec2 = scvec2 + scret;
  #else
    SCALE_T scalevec1 = scvec1 * scret;
    SCALE_T scalevec2 = scvec2 * scret;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
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
}

void v_q_hadamard(const INT_T* vec1, const INT_T* vec2, ITER_T len, INT_T* ret,
                  SCALE_T scvec1, SCALE_T scvec2) {
  #ifdef SHIFT
    SCALE_T scalevec = scvec1 + scvec2;
  #else
    SCALE_T scalevec = scvec1 * scvec2;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
    #ifdef SHIFT
      *ret++ = (((INTM_T)(*vec1++) * (INTM_T)(*vec2++)) >> scalevec);
      *ret++ = (((INTM_T)(*vec1++) * (INTM_T)(*vec2++)) >> scalevec);
      *ret++ = (((INTM_T)(*vec1++) * (INTM_T)(*vec2++)) >> scalevec);
      *ret++ = (((INTM_T)(*vec1++) * (INTM_T)(*vec2++)) >> scalevec);
    #else
      *ret++ = ((INTM_T)(*vec1++) * (INTM_T)(*vec2++)) / scalevec;
      *ret++ = ((INTM_T)(*vec1++) * (INTM_T)(*vec2++)) / scalevec;
      *ret++ = ((INTM_T)(*vec1++) * (INTM_T)(*vec2++)) / scalevec;
      *ret++ = ((INTM_T)(*vec1++) * (INTM_T)(*vec2++)) / scalevec;
    #endif
  }
}

void v_q_sigmoid(const INT_T* vec, ITER_T len, INT_T* ret, INT_T div,
                 INT_T add, INT_T sigmoid_limit, SCALE_T scale_in,
                 SCALE_T scale_out) {
  SCALE_T scaleout = (scale_out - scale_in);
  for (ITER_T i = 0; i < len; i += 4) {
    INT_T w = (*vec++ / div) + add;
    INT_T x = (*vec++ / div) + add;
    INT_T y = (*vec++ / div) + add;
    INT_T z = (*vec++ / div) + add;

    *ret++ = (w <= 0) ? 0 : (((w >= sigmoid_limit) ? sigmoid_limit : w) << scaleout);
    *ret++ = (x <= 0) ? 0 : (((x >= sigmoid_limit) ? sigmoid_limit : x) << scaleout);
    *ret++ = (y <= 0) ? 0 : (((y >= sigmoid_limit) ? sigmoid_limit : y) << scaleout);
    *ret++ = (z <= 0) ? 0 : (((z >= sigmoid_limit) ? sigmoid_limit : z) << scaleout);
  }
}

void v_q_tanh(const INT_T* vec, ITER_T len, INT_T* ret, SCALE_T scale_in,
              SCALE_T scale_out) {
  SCALE_T scalein = (1 << scale_in);
  SCALE_T scaleout = scale_out - scale_in;
  for (ITER_T i = 0; i < len; i += 4) {
    INT_T w = *vec++;
    INT_T x = *vec++;
    INT_T y = *vec++;
    INT_T z = *vec++;

    *ret++ = ((w >= scalein) ? scalein : ((w <= -scalein) ? (-scalein) : w)) << scaleout;
    *ret++ = ((x >= scalein) ? scalein : ((x <= -scalein) ? (-scalein) : x)) << scaleout;
    *ret++ = ((y >= scalein) ? scalein : ((y <= -scalein) ? (-scalein) : y)) << scaleout;
    *ret++ = ((z >= scalein) ? scalein : ((z <= -scalein) ? (-scalein) : z)) << scaleout;
  }
}

void v_q_scalar_add(INT_T scalar, const INT_T* vec, ITER_T len, INT_T* ret,
                    SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scaledscalar = scalar >> (scscalar + scret);
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scaledscalar = scalar / (scscalar * scret);
    SCALE_T scalevec = scvec * scret;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
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
}

void v_q_scalar_sub(INT_T scalar, const INT_T* vec, ITER_T len, INT_T* ret,
                    SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scaledscalar = scalar >> (scscalar + scret);
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scaledscalar = scalar / (scscalar * scret);
    SCALE_T scalevec = scvec * scret;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
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
}

void v_q_sub_scalar(const INT_T* vec, INT_T scalar, ITER_T len, INT_T* ret,
                    SCALE_T scvec, SCALE_T scscalar, SCALE_T scret) {
  #ifdef SHIFT
    SCALE_T scaledscalar = scalar >> (scscalar + scret);
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scaledscalar = scalar / (scscalar * scret);
    SCALE_T scalevec = scvec * scret;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
    #ifdef SHIFT
      *ret++ = ((*vec++ >> scalevec) - scaledscalar);
      *ret++ = ((*vec++ >> scalevec) - scaledscalar);
      *ret++ = ((*vec++ >> scalevec) - scaledscalar);
      *ret++ = ((*vec++ >> scalevec) - scaledscalar);
    #else
      *ret++ = ((*vec++ / scalevec) - scaledscalar);
      *ret++ = ((*vec++ / scalevec) - scaledscalar);
      *ret++ = ((*vec++ / scalevec) - scaledscalar);
      *ret++ = ((*vec++ / scalevec) - scaledscalar);
    #endif
  }
}

void v_q_scalar_mul(INT_T scalar, const INT_T* vec, ITER_T len, INT_T* ret,
                    SCALE_T scscalar, SCALE_T scvec) {
  SCALE_T upscalar = scalar;
  #ifdef SHIFT
    SCALE_T scale = scscalar + scvec;
  #else
    SCALE_T scale = scscalar * scvec;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
    #ifdef SHIFT
      *ret++ = (upscalar * (INTM_T)(*vec++)) >> scale;
      *ret++ = (upscalar * (INTM_T)(*vec++)) >> scale;
      *ret++ = (upscalar * (INTM_T)(*vec++)) >> scale;
      *ret++ = (upscalar * (INTM_T)(*vec++)) >> scale;
    #else
      *ret++ = (upscalar * (INTM_T)(*vec++)) / scale;
      *ret++ = (upscalar * (INTM_T)(*vec++)) / scale;
      *ret++ = (upscalar * (INTM_T)(*vec++)) / scale;
      *ret++ = (upscalar * (INTM_T)(*vec++)) / scale;
    #endif
  }
}

void v_q_argmax(const INT_T* const vec, ITER_T len, ITER_T* const ret) {
  INT_T max_value = vec[0];
  ITER_T max_index = 0;

  for (ITER_T i = 1; i < len; i++) {
    if (max_value < vec[i]) {
      max_index = i;
      max_value = vec[i];
    }
  }

  *ret = max_index;
}

void v_q_relu(INT_T* vec, ITER_T len) {
  for (ITER_T i = 0; i < len; i += 4) {
    *vec = (*vec < 0) ? 0 : *vec;
    vec++;
    *vec = (*vec < 0) ? 0 : *vec;
    vec++;
    *vec = (*vec < 0) ? 0 : *vec;
    vec++;
    *vec = (*vec < 0) ? 0 : *vec;
    vec++;
  }
}

void v_q_exp(const INT_T* vec, ITER_T len, INT_T* ret,
             SCALE_T scvec, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i += 4) {
    *ret++ = ((INT_T)(exp(((float)*vec++) / scvec) * scret));
    *ret++ = ((INT_T)(exp(((float)*vec++) / scvec) * scret));
    *ret++ = ((INT_T)(exp(((float)*vec++) / scvec) * scret));
    *ret++ = ((INT_T)(exp(((float)*vec++) / scvec) * scret));
  }
}

void v_q_scale_up(INT_T* vec, ITER_T len, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i += 4) {
    #ifdef SHIFT
      *vec++ <<= scvec;
      *vec++ <<= scvec;
      *vec++ <<= scvec;
      *vec++ <<= scvec;
    #else
      *vec++ *= scvec;
      *vec++ *= scvec;
      *vec++ *= scvec;
      *vec++ *= scvec;
    #endif
  }
}

void v_q_scale_down(INT_T* vec, ITER_T len, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i += 4) {
    #ifdef SHIFT
      *vec++ >>= scvec;
      *vec++ >>= scvec;
      *vec++ >>= scvec;
      *vec++ >>= scvec;
    #else
      *vec++ /= scvec;
      *vec++ /= scvec;
      *vec++ /= scvec;
      *vec++ /= scvec;
    #endif
  }
}

void m_q_transpose(const INT_T* const mat, ITER_T nrows, ITER_T ncols,
                   INT_T* const ret) {
  ITER_T len = nrows * ncols, counter = 0;
  for (ITER_T i = 0; i < len; i++) {
    if (counter >= len) {
      counter -= len - 1;
    }

    ret[i] = mat[counter];
    counter += nrows;
  }
}

void m_q_reverse(const INT_T* const mat, ITER_T nrows, ITER_T ncols,
                 ITER_T axis, INT_T* ret) {
  ITER_T len = nrows * ncols;

  if (axis == 0) {
    ITER_T col_counter = 0, row_index = len - ncols;

    for (ITER_T i = 0; i < len; i += 4) {
      if (col_counter >= ncols) {
        col_counter = 0;
        row_index -= ncols;
      }

      *ret++ = mat[row_index + col_counter++];
      *ret++ = mat[row_index + col_counter++];
      *ret++ = mat[row_index + col_counter++];
      *ret++ = mat[row_index + col_counter++];
    }
  } else {
    S_ITER_T row_counter = ncols - 1;
    ITER_T col_index = 0;

    for (ITER_T i = 0; i < len; i += 4) {
      if (row_counter < 0) {
        row_counter = ncols - 1;
        col_index += ncols;
      }

      *ret++ = mat[col_index + (ITER_T)row_counter--];
      *ret++ = mat[col_index + (ITER_T)row_counter--];
      *ret++ = mat[col_index + (ITER_T)row_counter--];
      *ret++ = mat[col_index + (ITER_T)row_counter--];
    }
  }
}

void m_q_add_vec(const INT_T* mat, const INT_T* const vec, ITER_T nrows,
                 ITER_T ncols, INT_T* ret, SCALE_T scmat, SCALE_T scvec,
                 SCALE_T scret) {
  ITER_T len = nrows * ncols;
  INT_T* vec_offset = NULL;
  #ifdef SHIFT
    SCALE_T scalemat = scmat + scret;
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scalemat = scmat * scret;
    SCALE_T scalevec = scvec * scret;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
    if (i % ncols == 0) {
      vec_offset = (INT_T *)vec;
    }

    #ifdef SHIFT
      *ret++ = ((*mat++ >> scalemat) + (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) + (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) + (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) + (*vec_offset++ >> scalevec));
    #else
      *ret++ = ((*mat++ / scalemat) + (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) + (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) + (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) + (*vec_offset++ / scalevec));
    #endif
  }
}

void m_q_sub_vec(const INT_T* mat, const INT_T* const vec, ITER_T nrows,
                 ITER_T ncols, INT_T* ret, SCALE_T scmat, SCALE_T scvec,
                 SCALE_T scret) {
  ITER_T len = nrows * ncols;
  INT_T* vec_offset = NULL;
  #ifdef SHIFT
    SCALE_T scalemat = scmat + scret;
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scalemat = scmat * scret;
    SCALE_T scalevec = scvec * scret;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
    if (i % ncols == 0) {
      vec_offset = (INT_T *)vec;
    }

    #ifdef SHIFT
      *ret++ = ((*mat++ >> scalemat) - (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) - (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) - (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) - (*vec_offset++ >> scalevec));
    #else
      *ret++ = ((*mat++ / scalemat) - (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) - (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) - (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) - (*vec_offset++ / scalevec));
    #endif
  }
}

void m_q_mulvec(const INT_T* mat, const INT_T* const vec, ITER_T nrows,
                ITER_T ncols, INT_T* const ret, SCALE_T scmat, SCALE_T scvec,
                SCALE_T H1, SCALE_T H2) {
  int64_t sum;
  INT_T* vec_offset = NULL;
  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + H1;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of INTM_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * H1;
  #endif
  for (ITER_T row = 0; row < nrows; row++) {
    sum = 0;
    vec_offset = (INT_T*)vec;
    for (ITER_T col = 0; col < ncols; col += 4) {
      sum += (INTM_T)(*mat++) * (INTM_T)(*vec_offset++);
      sum += (INTM_T)(*mat++) * (INTM_T)(*vec_offset++);
      sum += (INTM_T)(*mat++) * (INTM_T)(*vec_offset++);
      sum += (INTM_T)(*mat++) * (INTM_T)(*vec_offset++);
    }

    #ifdef SHIFT
      ret[row] = sum >> scale;
    #else
      ret[row] = sum / scale;
    #endif
  }
}

void m_q_sparse_mulvec(const ITER_T* const col_indices, const INT_T* const mat_values,
                       const INT_T* const vec, ITER_T ndims, INT_T* const ret,
                       SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T iter_index = 0, iter_value = 0;
  #ifdef SHIFT
    SCALE_T scale = scmat + scvec + scret;
  #else
    // Be careful, the below implementation would not work if the denominator
    // exceeds the range of INTM_T range. In such a case, cast the denominator
    // to int64_t.
    SCALE_T scale = scmat * scvec * scret;
  #endif
  for (ITER_T k = 0; k < ndims; k++) {
    ITER_T index = col_indices[iter_index];

    while (index != 0) {
      #ifdef SHIFT
        ret[index - 1] += (((INTM_T)mat_values[iter_value] * (INTM_T)vec[k]) >> scale);
      #else
        ret[index - 1] += (((INTM_T)mat_values[iter_value] * (INTM_T)vec[k]) / scale);
      #endif
      iter_index++;
      iter_value++;
      index = col_indices[iter_index];
    }

    iter_index++;
  }
}

void t_q_add_vec(const INT_T* mat, const INT_T* const vec, ITER_T nbatches,
                 ITER_T nrows, ITER_T ncols, ITER_T nchannels, INT_T* ret,
                 SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  INT_T* vec_offset = NULL;
  #ifdef SHIFT
    SCALE_T scalemat = scmat + scret;
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scalemat = scmat * scret;
    SCALE_T scalevec = scvec * scret;
  #endif
  for (ITER_T i = 0; i < len; i += 4) {
    if (i % nchannels == 0) {
      vec_offset = (INT_T*)vec;
    }

    #ifdef SHIFT
      *ret++ = ((*mat++ >> scalemat) + (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) + (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) + (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) + (*vec_offset++ >> scalevec));
    #else
      *ret++ = ((*mat++ / scalemat) + (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) + (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) + (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) + (*vec_offset++ / scalevec));
    #endif
  }
}

void t_q_sub_vec(const INT_T* mat, const INT_T* const vec, ITER_T nbatches,
                 ITER_T nrows, ITER_T ncols, ITER_T nchannels, INT_T* ret,
                 SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  INT_T* vec_offset = NULL;
  #ifdef SHIFT
    SCALE_T scalemat = scmat + scret;
    SCALE_T scalevec = scvec + scret;
  #else
    SCALE_T scalemat = scmat * scret;
    SCALE_T scalevec = scvec * scret;
  #endif
  for (ITER_T i = 0, c = 0; i < len; i += 4, c += 4) {
    if (i % nchannels == 0) {
      vec_offset = (INT_T*)vec;
    }

    #ifdef SHIFT
      *ret++ = ((*mat++ >> scalemat) - (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) - (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) - (*vec_offset++ >> scalevec));
      *ret++ = ((*mat++ >> scalemat) - (*vec_offset++ >> scalevec));
    #else
      *ret++ = ((*mat++ / scalemat) - (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) - (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) - (*vec_offset++ / scalevec));
      *ret++ = ((*mat++ / scalemat) - (*vec_offset++ / scalevec));
    #endif
  }
}

void q_maxpool(const INT_T* const input, INT_T* const output, ITER_T N,
               ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF,
               ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G, S_ITER_T HPadU,
               S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride,
               ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
               SCALE_T scinput, SCALE_T scoutput) {
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
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  #ifdef SHIFT
    SCALE_T scale = scinput + scoutput;
  #else
    SCALE_T scale = scinput * scoutput;
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
          ITER_T CIndexOut = g * COut + WIndexOut;
          for (ITER_T c = 0; c < COut; c++) {

            INT_T max = INT_TMIN;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              if ((hoffset < 0) || (hoffset >= (S_ITER_T)H)) {
                if (max < 0) {
                  max = 0;
                }
                continue;
              }

              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn + CIndexIn;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                if ((woffset < 0) || (woffset >= (S_ITER_T)W)) {
                  if (max < 0) {
                    max = 0;
                  }
                  continue;
                }
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn + HIndexIn;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  INT_T a = input[cf + WIndexIn];
                  if (max < a) {
                    max = a;
                  }
                }
              }
            }

            #ifdef SHIFT
              output[c + CIndexOut] = (max >> scale);
            #else
              output[c + CIndexOut] = max / scale;
            #endif
          }
        }
      }
    }
  }
}

void q_convolution(const INT_T* const input, const INT_T* const filter,
                   INT_T* const output, ITER_T N, ITER_T H, ITER_T W,
                   ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF, ITER_T COut,
                   ITER_T HOut, ITER_T WOut, ITER_T G, S_ITER_T HPadU,
                   S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR,
                   ITER_T HStride, ITER_T WStride, ITER_T HDilation,
                   ITER_T WDilation, SCALE_T H1, SCALE_T H2, SCALE_T scinput,
                   SCALE_T scoutput) {
  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T HOffsetFR = (HF >> 1);
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T WOffsetFR = (WF >> 1);

  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * HOffsetFL) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * WOffsetFL) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * HOffsetFR) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * WOffsetFR) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;

  #ifdef SHIFT
    SCALE_T scale = scinput + scoutput + H1;
  #else
    SCALE_T scale = scinput * scoutput * H1;
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
          ITER_T CIndexOut = g * COut + WIndexOut;
          for (ITER_T c = 0; c < COut; c++) {

            int64_t sum = 0;
            for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              if ((hoffset < 0) || (hoffset >= (S_ITER_T)H)) {
                continue;
              }
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn + CIndexIn + c;
              ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * HOffsetF + c;
              for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                if ((woffset < 0) || (woffset >= (S_ITER_T)W)) {
                  continue;
                }
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn + HIndexIn;
                ITER_T WIndexF = ((ITER_T)(wf + WOffsetFL)) * WOffsetF + HIndexF;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  sum += ((INTM_T)input[WIndexIn + cf]) *
                         ((INTM_T)filter[WIndexF + (cf * COut)]);
                }
              }
            }

            #ifdef SHIFT
              output[c + CIndexOut] = (sum >> scale);
            #else
              output[c + CIndexOut] = (sum / scale);
            #endif
          }
        }
      }
    }
  }
}
