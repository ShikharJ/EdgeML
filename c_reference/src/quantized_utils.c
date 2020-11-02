// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stddef.h>
#include <string.h>
#include "quantized_utils.h"

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
