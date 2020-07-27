// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_mbconv.h"

void q_mbconv_block(const INT_T* const input, const INT_T* const filter1,
  const INT_T* const BN1W, const INT_T* const BN1B, const INT_T* const filter2,
  const INT_T* const BN2W, const INT_T* const BN2B, const INT_T* const filter3,
  const INT_T* const BN3W, const INT_T* const BN3B, INT_T* const output,
  INT_T* const convBuffer1, INT_T* const convBuffer2, ITER_T N, ITER_T H,
  ITER_T W, ITER_T CIn, ITER_T CTemp, ITER_T HF, ITER_T WF, ITER_T COut,
  ITER_T HOut, ITER_T WOut, S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
  S_ITER_T WPadR, ITER_T HStride, ITER_T WStride, INTM_T limit1, INTM_T limit2,
  L_SCALE_T shrU1, L_SCALE_T shrX1, L_SCALE_T shrU2, L_SCALE_T shrX2,
  L_SCALE_T shrU3, L_SCALE_T shrW3, L_SCALE_T shlU1, L_SCALE_T shlX1,
  L_SCALE_T shlU2, L_SCALE_T shlX2, L_SCALE_T shlU3, L_SCALE_T shlW3) {

  S_ITER_T HOffsetFL = (HF - 1) >> 1;
  S_ITER_T WOffsetFL = (WF - 1) >> 1;
  S_ITER_T HOffsetFR = HF >> 1;
  S_ITER_T WOffsetFR = WF >> 1;

  S_ITER_T HOffsetL = HOffsetFL - HPadU;
  S_ITER_T WOffsetL = WOffsetFL - WPadL;
  S_ITER_T HOffsetR = HOffsetFR - HPadD;
  S_ITER_T WOffsetR = WOffsetFR - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T HOffsetC1 = W * CTemp;
  ITER_T GOffsetF = HF * WF;
  ITER_T HOffsetOut = WOut * COut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    ITER_T margin = 0, nstart = 0;
    if ((S_ITER_T)HF - HPadU - (S_ITER_T)HStride > 0) {
      margin = (ITER_T)((S_ITER_T)HF - HPadU - (S_ITER_T)HStride);
    }
    if (HPadU < 0) {
      // nstart will always be zero unless HPadU is negative.
      nstart = (ITER_T)(-HPadU);
    }

    int64_t sum;
    for (ITER_T i = nstart; i < margin; i++) {
      ITER_T HIndexIn = i * HOffsetIn + NIndexIn;
      ITER_T HIndexC1 = i * HOffsetC1;
      for (ITER_T j = 0; j < W; j++) {
        ITER_T WIndexIn = j * CIn + HIndexIn;
        ITER_T WIndexC1 = j * CTemp + HIndexC1;
        for (ITER_T k = 0; k < CTemp; k++) {
          sum = 0;
          for (ITER_T l = 0; l < CIn; l++) {
            sum += ((INTM_T)input[WIndexIn + l]) *
                   ((INTM_T)filter1[l * CTemp + k]);
          }

          #ifdef SHIFT
            INTM_T x = (((INTM_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                        ((INTM_T)BN1W[k]));
          #else
            INTM_T x = (((INTM_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                        ((INTM_T)BN1W[k]));
          #endif
          x = q_relu(x, limit1);
          #ifdef SHIFT
            convBuffer1[WIndexC1 + k] = ((x << shlX1) >> shrX1);
          #else
            convBuffer1[WIndexC1 + k] = (x * shlX1) / shrX1;
          #endif
        }
      }
    }

    ITER_T hout = 0;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; hout++, h += (S_ITER_T)HStride) {
      ITER_T HIndexOut = hout * HOffsetOut + NIndexOut;
      for (ITER_T i = 0; i < HStride; i++) {
        ITER_T iRed = (i + margin + hout * HStride) % HF;
        ITER_T iFull = i + margin + hout * HStride;
        ITER_T HIndexC1 = iRed * HOffsetC1;
        ITER_T HIndexIn = iFull * HOffsetIn + NIndexIn;
        if (iFull < H) {
          for (ITER_T j = 0; j < W; j++) {
            ITER_T WIndexIn = j * CIn + HIndexIn;
            ITER_T WIndexC1 = j * CTemp + HIndexC1;
            for (ITER_T k = 0; k < CTemp; k++) {
              sum = 0;
              for (ITER_T l = 0; l < CIn; l++) {
                sum += ((INTM_T)input[WIndexIn + l]) *
                       ((INTM_T)filter1[l * CTemp + k]);
              }

              #ifdef SHIFT
                INTM_T x = (((INTM_T)(((sum << shlU1) >> shrU1) + BN1B[k])) *
                            ((INTM_T)BN1W[k]));
              #else
                INTM_T x = (((INTM_T)((sum * shlU1) / shrU1 + BN1B[k])) *
                            ((INTM_T)BN1W[k]));
              #endif
              x = q_relu(x, limit1);
              #ifdef SHIFT
                convBuffer1[WIndexC1 + k] = ((x << shlX1) >> shrX1);
              #else
                convBuffer1[WIndexC1 + k] = (x * shlX1) / shrX1;
              #endif
            }
          }
        } else {
          for (ITER_T j = 0; j < W; j++) {
            ITER_T WIndexC1 = j * CTemp + HIndexC1;
            for (ITER_T k = 0; k < CTemp; k++) {
              #ifdef SHIFT
                INTM_T x = (((INTM_T)BN1B[k]) * ((INTM_T)BN1W[k]));
              #else
                INTM_T x = (((INTM_T)BN1B[k]) * ((INTM_T)BN1W[k]));
              #endif
              x = q_relu(x, limit1);
              #ifdef SHIFT
                convBuffer1[WIndexC1 + k] = ((x << shlX1) >> shrX1);
              #else
                convBuffer1[WIndexC1 + k] = (x * shlX1) / shrX1;
              #endif
            }
          }
        }
      }

      ITER_T wout = 0;
      for (S_ITER_T w = WOffsetL; w < ((S_ITER_T)W) - WOffsetR; wout++, w += ((S_ITER_T)WStride)) {
        ITER_T WIndexOut = wout * COut + HIndexOut;
        for (ITER_T g = 0; g < CTemp; g++) {
          sum = 0;
          ITER_T GIndexF = g * GOffsetF;
          for (S_ITER_T hf = -HOffsetFL; hf <= HOffsetFR; hf++) {
            S_ITER_T hindex = h + hf;
            if ((hindex < 0) || (hindex >= (S_ITER_T)H)){
              continue;
            }
            ITER_T HIndexC1 = (((ITER_T)hindex) % HF) * HOffsetC1;
            ITER_T HIndexF = ((ITER_T)(hf + HOffsetFL)) * WF + GIndexF;
            for (S_ITER_T wf = -WOffsetFL; wf <= WOffsetFR; wf++) {
              S_ITER_T windex = w + wf;
              if ((windex < 0) || (windex >= (S_ITER_T)W)) {
                continue;
              } else {
                sum += ((INTM_T)convBuffer1[HIndexC1 + ((ITER_T)windex) * CTemp + g]) *
                       ((INTM_T)filter2[HIndexF + ((ITER_T)(wf + WOffsetFL))]);
              }
            }
          }

          #ifdef SHIFT
            INTM_T x = (((INTM_T)(((sum << shlU2) >> shrU2) + BN2B[g])) *
                        ((INTM_T)BN2W[g]));
          #else
            INTM_T x = (((INTM_T)((sum * shlU2) / shrU2 + BN2B[g])) *
                        ((INTM_T)BN2W[g]));
          #endif
          x = q_relu(x, limit2);
          #ifdef SHIFT
            convBuffer2[g] = ((x << shlX2) >> shrX2);
          #else
            convBuffer2[g] = (x * shlX2) / shrX2;
          #endif
        }

        for (ITER_T i = 0; i < COut; i++) {
          sum = 0;
          for (ITER_T g = 0; g < CTemp; g++) {
            sum += ((INTM_T)convBuffer2[g]) * ((INTM_T)filter3[g * COut + i]);
          }

          #ifdef SHIFT
            output[WIndexOut + i] =
              (((((INTM_T)(((sum << shlU3) >> shrU3) + BN3B[i])) *
                ((INTM_T) BN3W[i])) << shlW3) >> shrW3);
          #else
            output[WIndexOut + i] =
              ((((INTM_T)((sum * shlU3) / shrU3 + BN3B[i])) *
                ((INTM_T) BN3W[i])) * shlW3) / shrW3;
          #endif
        }
      }
    }
  }
}
