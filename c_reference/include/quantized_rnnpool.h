// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_RNNPOOL_H__
#define __QUANTIZED_RNNPOOL_H__

#include "quantized_utils.h"

typedef int (*q7xq15_q15_rnn_t)(Q15_T* const, ITER_T, const Q7_T* const, ITER_T, ITER_T, const void*, void*, const void*, const void*, int, int);
typedef int (*q15_rnn_t)(Q15_T* const, ITER_T, const Q15_T* const, ITER_T, ITER_T, const void*, void*, const void*, const void*, int, int);

/**
 * @brief Block implementation of RNNPool operator
 * @param[in]        patch           pointer to activation of patch (row, col, channel)
 * @param[in]        inputDims       dimension of each input pixel
 * @param[in]        patchDim        number of rows and columns in a square patch
 * @param[in]        stride          stride length in the larger image to get to next row
 * @param[in]        rnn1            function pointer to RNN1
 * @param[in]        hiddenDims1     dimension of the hidden state of RNN1
 * @param[in]        rnn1_params     pointer to parameters of RNN1
 * @param[in]        rnn1_buffers    pointer to buffers needed for RNN1
 * @param[in]        rnn1_scales     pointer to the scales needed for RNN1
 * @param[in]        rnn1_log_scales pointer to the log-scales needed for RNN1
 * @param[in]        rnn2            function pointer to RNN2
 * @param[in]        hiddenDims2     dimension of the hidden state of RNN2
 * @param[in]        rnn2_params     pointer to parameters of RNN2
 * @param[in]        rnn2_buffers    pointer to buffers needed for RNN2
 * @param[in]        rnn2_scales     pointer to the scales needed for RNN2
 * @param[in]        rnn2_log_scales pointer to the log-scales needed for RNN2
 * @param[out]       output          pointer to output, initialized to size 4 * hiddenDims2
 * @param[in]        buffer          pointer to buffer, initialized to size hiddenDims1 * max{nrows, cols}
 * @param[in]        ShR1            scale-down factor for intermediate rnnpool output
 * @param[in]        ShL1            scale-up factor for intermediate rnnpool output
 * @param[in]        ShR2            scale-down factor for final rnnpool output
 * @param[in]        ShL2            scale-up factor for final rnnpool output
 * @param[in]        LShR1           log-scale-down factor for intermediate rnnpool output
 * @param[in]        LShL1           log-scale-up factor for intermediate rnnpool output
 * @param[in]        LShR2           log-scale-down factor for final rnnpool output
 * @param[in]        LShL2           log-scale-up factor for final rnnpool output
 * @return none
 * @example          Please refer the file: c_reference/tests/rnnpool/test_quantized_rnnpool.c
 */
int q7xq15_q15_rnnpool_block(const Q7_T* const patch, ITER_T inputDims,
  ITER_T patchDim, ITER_T stride, q7xq15_q15_rnn_t rnn1, ITER_T hiddenDims1,
  const void* rnn1_params, void* rnn1_buffers, const void* rnn1_scales,
  const void* rnn1_log_scales, q15_rnn_t rnn2, ITER_T hiddenDims2,
  const void* rnn2_params, void* rnn2_buffers, const void* rnn2_scales,
  const void* rnn2_log_scales, Q15_T* const output, Q15_T* const buffer,
  SCALE_T ShR1, SCALE_T ShL1, SCALE_T ShR2, SCALE_T ShL2, SCALE_T LShR1,
  SCALE_T LShL1, SCALE_T LShR2, SCALE_T LShL2);
int q15_rnnpool_block(const Q15_T* const patch, ITER_T inputDims,
  ITER_T patchDim, ITER_T stride, q15_rnn_t rnn1, ITER_T hiddenDims1,
  const void* rnn1_params, void* rnn1_buffers, const void* rnn1_scales,
  const void* rnn1_log_scales, q15_rnn_t rnn2, ITER_T hiddenDims2,
  const void* rnn2_params, void* rnn2_buffers, const void* rnn2_scales,
  const void* rnn2_log_scales, Q15_T* const output, Q15_T* const buffer,
  SCALE_T ShR1, SCALE_T ShL1, SCALE_T ShR2, SCALE_T ShL2, SCALE_T LShR1,
  SCALE_T LShL1, SCALE_T LShR2, SCALE_T LShL2);

#endif
