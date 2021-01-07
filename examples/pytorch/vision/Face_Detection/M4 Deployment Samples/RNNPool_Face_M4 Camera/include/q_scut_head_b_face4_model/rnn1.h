// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stddef.h>

#define INPUT_CHANNELS 4
#define PATCH_DIM 8
#define HIDDEN_DIM1 16
#define SPARSE_FACTOR1 2

#ifdef SPARSE
  static ITER_T W1Ids[HIDDEN_DIM1 * INPUT_CHANNELS / SPARSE_FACTOR1 + INPUT_CHANNELS] = {1, 2, 4, 5, 6, 9, 10, 12, 13, 16, 0, 2, 4, 5, 6, 8, 12, 14, 15, 0, 1, 3, 4, 5, 6, 10, 11, 14, 0, 1, 4, 5, 6, 7, 10, 0};
  static Q15_T W1Vals[HIDDEN_DIM1 * INPUT_CHANNELS / SPARSE_FACTOR1] = {5115, -10312, -576, -2829, 1891, 10418, 1144, -14259, -440, 6572, 10136, -802, 2234, 4935, -1021, 19156, -2244, -12238, 9914, 12851, 663, 8805, -20961, 17792, -643, -8755, -9619, -1917, -11879, 27435, 8029, -22498};
  static ITER_T U1Ids[HIDDEN_DIM1 * HIDDEN_DIM1 / SPARSE_FACTOR1 + HIDDEN_DIM1] = {1, 4, 11, 14, 0, 3, 7, 9, 12, 15, 0, 4, 5, 6, 8, 13, 0, 6, 7, 11, 16, 0, 1, 4, 6, 7, 10, 13, 14, 0, 1, 6, 8, 9, 10, 13, 0, 1, 2, 4, 7, 12, 0, 6, 16, 0, 7, 12, 15, 0, 1, 6, 9, 10, 13, 14, 0, 1, 2, 3, 4, 6, 8, 10, 13, 0, 11, 12, 13, 15, 0, 6, 13, 0, 6, 8, 9, 10, 13, 0, 3, 7, 12, 13, 15, 0, 1, 2, 3, 8, 10, 13, 0};
  static Q15_T U1Vals[HIDDEN_DIM1 * HIDDEN_DIM1 / SPARSE_FACTOR1] = {8953, 6435, -4866, -4524, 11915, -17935, 24570, -20050, 2766, -9015, 10735, -17159, -7508, -8711, 10213, 11239, 17486, 1291, -26241, -6637, 25321, 15771, -17711, 5609, 12069, -10713, -10076, 10577, -16464, 17371, -6933, 10523, 5151, -8736, 7358, -2780, -20389, -6256, 20463, 12651, -4524, 9994, 29157, 6168, -11251, -9216, -13209, -6916, -10210, 5412, -19550, 6749, -7728, -9761, -4419, 6405, -7660, -7222, 985, 7074, 12369, -22597, 7597, -12736, 28019, 3576, 4310, 12893, 29316, -2898, -8617, -2193, -12062, 6230, -6135, -12281, -8376};
#else
  static Q15_T W1[HIDDEN_DIM1 * INPUT_CHANNELS] = {5115, 0, 9914, -9619, -10312, 10136, 0, 0, 0, 0, 12851, 0, -576, -802, 663, -1917, -2829, 2234, 8805, -11879, 1891, 4935, -20961, 27435, 0, 0, 0, 8029, 0, -1021, 0, 0, 10418, 0, 0, 0, 1144, 0, 17792, -22498, 0, 0, -643, 0, -14259, 19156, 0, 0, -440, 0, 0, 0, 0, -2244, -8755, 0, 0, -12238, 0, 0, 6572, 0, 0, 0};
  static Q15_T U1[HIDDEN_DIM1 * HIDDEN_DIM1] = {8953, 0, 0, 0, -26241, -10713, 10523, 0, 0, 9994, -6916, 0, 0, 0, 0, -2193, 0, 0, 0, 0, 0, 0, 5151, 0, 0, 0, -10210, 0, 0, 0, 0, -12062, 0, 11915, 0, 0, 0, 0, 0, 0, 0, 0, 5412, 0, 0, 0, 4310, 6230, 6435, 0, -9015, 0, -6637, 0, -8736, 0, 0, 0, -19550, 0, 0, 0, 0, 0, 0, 0, 10735, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -17159, 10213, 25321, -10076, 0, -20389, 0, 29157, 6749, 0, 7074, -22597, 0, 0, 0, -17935, 0, 11239, 15771, 0, 7358, 0, 20463, 0, 0, 0, 0, 0, 12893, 0, 0, 0, -7508, 0, 0, 10577, 0, 0, 0, 0, -7728, 0, 0, 7597, 0, -6135, 0, 24570, 0, 0, 0, -16464, 0, 0, 0, 6168, 0, 0, 0, -12736, 0, 0, 0, 0, 0, 0, -17711, 17371, 0, 0, 0, -11251, -9761, 0, 0, 28019, 0, -12281, -4866, 0, 0, 17486, 0, 0, 0, 0, 0, 0, 0, 6405, 0, 0, 0, 0, 0, -20050, 0, 0, 0, 0, -2780, 0, 12651, 0, 0, -7660, 0, 0, 29316, 0, 0, 0, -8711, 0, 5609, -6933, 0, 0, 0, -9216, -4419, -7222, 12369, 3576, -2898, -8376, -4524, 0, 0, 0, 12069, 0, 0, 0, 0, -13209, 0, 0, 0, 0, 0, 0, 0, 2766, 0, 0, 0, 0, 0, 0, -4524, 0, 0, 985, 0, 0, -8617, 0, 0, 0, 0, 1291, 0, 0, 0, -6256, 0, 0, 0, 0, 0, 0, 0, 0};
#endif

static Q15_T Bg1[HIDDEN_DIM1] = {13051, -3038, 13681, -4257, 5116, 2737, 17971, 8014, 175, -205, 5270, 6100, -4462, 12877, -10139,  7379};
static Q15_T Bh1[HIDDEN_DIM1] = {11728, 10111, 13481, 4598, -572, 9291, -2942, 8797, 6884, 8186, 10069, -3196, 17547, 13467, 10348, 10647};

#ifdef SPARSE
  static Q7xQ15_FastGRNN_Params RNN1_PARAMS = {
    .mean = NULL,
    .stdDev = NULL,
    .W = NULL,
    .Wids = W1Ids,
    .Wvals = W1Vals,
    .U = NULL,
    .Uids = U1Ids,
    .Uvals = U1Vals,
    .Bg = Bg1,
    .Bh = Bh1,
    .sigmoid_zeta = 16384,
    .sigmoid_nu = 23611
  };
#else
  static Q7xQ15_FastGRNN_Params RNN1_PARAMS = {
    .mean = NULL,
    .stdDev = NULL,
    .W = W1,
    .Wids = NULL,
    .Wvals = NULL,
    .U = U1,
    .Uids = NULL,
    .Uvals = NULL,
    .Bg = Bg1,
    .Bh = Bh1,
    .sigmoid_zeta = 16384,
    .sigmoid_nu = 23611
  };
#endif

static Q15_T preComp11[HIDDEN_DIM1] = {};
static Q15_T preComp12[HIDDEN_DIM1] = {};
static Q15_T preComp13[HIDDEN_DIM1] = {};
static Q7_T normFeatures1[INPUT_CHANNELS] = {};

static Q7xQ15_FastGRNN_Buffers RNN1_BUFFERS = {
  .preComp1 = preComp11,
  .preComp2 = preComp12,
  .preComp3 = preComp13,
  .normFeatures = normFeatures1
};

#ifdef SHIFT
  static Q15_FastGRNN_Scales RNN1_SCALES = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 3, //8
    .normFeaturesMVW = 2, //4
    .H1W = 1, //2
    .H2W = 0,
    .U = 8, //256
    .hiddenStateMVU = 5, //32
    .H1U = 2, //4
    .H2U = 0,
    .mV1AddMV2 = 0, //1
    .mV2AddMV1 = 1, //2
    .mV1AddMV2Out = 0, //1
    .mV1AddMV2Demote = 0, //1
    .pC1AddBg = 0, //1
    .Bg = 2, //4
    .pC1AddBgOut = 1, //2
    .pC1AddBgDemote = 0, //1
    .sigmoidLimit = 0,
    .sigmoidScaleIn = 0,
    .sigmoidScaleOut = 0,
    .pC1AddBh = 0, //1
    .Bh = 2, //4
    .pC1AddBhOut = 0, //1
    .pC1AddBhDemote = 1, //2
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 6, //64
    .hiddenStateHDGate = 7, //128
    .qOneScale = 0, //1
    .qOneSubGate = 0, //1
    .qOneSubGateOut = 0, //1
    .sigmoidZeta = 7, //128
    .sigmoidZetaMulQOneSubGate = 7, //128
    .sigmoidNu = 15, //32767
    .sigmoidNuAddQOneSubGate = 0, //1
    .sigmoidNuAddQOneSubGateOut = 0, //1
    .sigmoidNuAddQOneSubGateHDUpdate = 7, //128
    .updateHDSigmoidNuAddQOneSubGate = 7, //128
    .pC3AddPC1 = 1, //2
    .pC1AddPC3 = 0, //1
    .hiddenStateOut = 0, //1
    .hiddenStateDemote = 0, //1
    .div = 0,
    .add = 0,
    .qOne = 16384,
    .useTableSigmoid = 1,
    .useTableTanH = 1
  };
  static SCALE_T ShR1 = 0; //1
  static SCALE_T ShL1 = 0; //1
#else
  static Q15_FastGRNN_Scales RNN1_SCALES = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 8,
    .normFeaturesMVW = 4,
    .H1W = 2,
    .H2W = 0,
    .U = 256,
    .hiddenStateMVU = 32,
    .H1U = 4,
    .H2U = 0,
    .mV1AddMV2 = 1,
    .mV2AddMV1 = 2,
    .mV1AddMV2Out = 1,
    .mV1AddMV2Demote = 1,
    .pC1AddBg = 1,
    .Bg = 4,
    .pC1AddBgOut = 2,
    .pC1AddBgDemote = 1,
    .sigmoidLimit = 0,
    .sigmoidScaleIn = 0,
    .sigmoidScaleOut = 0,
    .pC1AddBh = 1,
    .Bh = 4,
    .pC1AddBhOut = 1,
    .pC1AddBhDemote = 2,
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 64,
    .hiddenStateHDGate = 128,
    .qOneScale = 1,
    .qOneSubGate = 1,
    .qOneSubGateOut = 1,
    .sigmoidZeta = 128,
    .sigmoidZetaMulQOneSubGate = 128,
    .sigmoidNu = 32767,
    .sigmoidNuAddQOneSubGate = 1,
    .sigmoidNuAddQOneSubGateOut = 1,
    .sigmoidNuAddQOneSubGateHDUpdate = 128,
    .updateHDSigmoidNuAddQOneSubGate = 128,
    .pC3AddPC1 = 2,
    .pC1AddPC3 = 1,
    .hiddenStateOut = 1,
    .hiddenStateDemote = 1,
    .div = 0,
    .add = 0,
    .qOne = 16384,
    .useTableSigmoid = 1,
    .useTableTanH = 1
  };
  static SCALE_T ShR1 = 1;
  static SCALE_T ShL1 = 1;
#endif
