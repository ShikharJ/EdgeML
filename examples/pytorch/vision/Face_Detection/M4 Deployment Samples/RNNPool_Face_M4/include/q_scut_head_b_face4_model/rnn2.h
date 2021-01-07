// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stddef.h>

#define HIDDEN_DIM2 16
#define SPARSE_FACTOR2 2

#ifdef SPARSE
  static ITER_T W2Ids[HIDDEN_DIM2 * HIDDEN_DIM2 / SPARSE_FACTOR2 + HIDDEN_DIM2] = {4, 6, 13, 0, 0, 1, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 0, 2, 6, 7, 11, 12, 15, 0, 3, 4, 6, 10, 11, 12, 13, 0, 3, 13, 0, 2, 4, 6, 10, 0, 1, 2, 11, 13, 14, 16, 0, 0, 3, 4, 13, 0, 6, 7, 11, 12, 14, 15, 0, 4, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 0, 6, 10, 11, 14, 0, 4, 0, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 0};
  static Q15_T W2Vals[HIDDEN_DIM2 * HIDDEN_DIM2 / SPARSE_FACTOR2] = {13664, 10075, -8136, -8196, 19682, 7592, 17420, -9882, 9482, 8105, 12061, 9304, 15839, 9716, -15989, -13146, 10095, -17187, -23490, -18332, 18093, -7837, 2591, -7406, -12527, 8211, 3690, 8585, -8786, 10915, 6740, -13854, 13297, -11399, -2935, -5868, -8288, -16882, 10633, 8279, -4843, -5691, -22534, 11074, -14600, 7285, 17992, 16173, -21691, 9481, -14351, 17708, 10240, 12301, -8762, 5013, 17713, 11674, -26392, -19634, 12173, 7225, 7177, -11338, -10216, 9236, -14347, 12059, 9897, -17601, 3119, -24534, 24616, 14401, 5844, 7722, 5626};
  static ITER_T U2Ids[HIDDEN_DIM2 * HIDDEN_DIM2 / SPARSE_FACTOR2 + HIDDEN_DIM2] = {1, 9, 13, 16, 0, 5, 8, 11, 15, 16, 0, 7, 9, 12, 0, 10, 12, 0, 6, 7, 10, 0, 5, 6, 8, 11, 12, 14, 0, 5, 6, 11, 14, 15, 16, 0, 1, 5, 6, 8, 10, 11, 15, 0, 1, 2, 7, 8, 9, 10, 11, 12, 16, 0, 4, 7, 8, 10, 12, 0, 5, 7, 8, 12, 0, 8, 10, 15, 16, 0, 1, 2, 5, 7, 13, 16, 0, 2, 4, 9, 0, 10, 15, 0, 1, 2, 6, 9, 10, 12, 14, 15, 0};
  static Q15_T U2Vals[HIDDEN_DIM2 * HIDDEN_DIM2 / SPARSE_FACTOR2] = {3526, 18699, 11737, -5371, 3745, -7566, -8189, -5149, -10029, -7411, -8845, 5444, 9152, -6389, -7819, -6447, 11232, 5464, 14245, -11012, 6066, 7254, -5826, 12124, -6172, 11913, 7696, 13139, -6352, 3527, 7620, 14408, 4536, 5191, 8255, 6609, -13454, -8879, -6362, -8462, 13114, -14378, -6820, -7959, -15090, 4471, 6985, 7994, -8078, -4719, 5083, -6077, -3968, 12197, 4739, 5689, -4058, -8446, -6020, -6760, 7550, -7037, 11763, -6262, -6458, 7221, -5723, 5458, 5413, 7062, 8387, 6579, 11032, -6078, 8632, -4719, 7630};
#else
  static Q15_T W2[HIDDEN_DIM2 * HIDDEN_DIM1] = {0, 0, -8196, 0, 0, 0, 0, -11399, 0, 0, 0, 0, -14351, 0, 0, 0, 0, 0, 0, -15989, 0, 0, 10915, -2935, 0, 0, 0, 0, 17708, 0, 0, 0, 0, 0, 19682, 0, 18093, 8585, 0, 0, 0, 8279, 0, 0, 10240, 0, 0, 12059, 13664, 0, 7592, 0, -7837, 0, 6740, 0, 0, -4843, 0, -21691, 12301, 0, -14347, 9897, 0, 0, 17420, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8762, 0, 0, -17601, 10075, 0, 0, -13146, 2591, 0, -13854, 0, 0, 0, -22534, 0, 5013, 7177, 0, 0, 0, 0, 0, 10095, 0, 0, 0, 0, 0, 0, 11074, 0, 17713, 0, 0, 3119, 0, 0, -9882, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11674, 0, 0, -24534, 0, 0, 9482, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8105, 0, -7406, 0, 13297, 0, 0, 0, 0, 0, -26392, -11338, 0, 24616, 0, 0, 0, -17187, -12527, 0, 0, -5868, 0, 0, -14600, 0, 0, -10216, 0, 0, 0, 0, 12061, -23490, 8211, 0, 0, 0, 0, 0, 7285, 0, -19634, 0, 0, 14401, -8136, 0, 9304, 0, 3690, -8786, 0, -8288, 0, -5691, 0, 0, 12173, 0, 0, 5844, 0, 0, 15839, 0, 0, 0, 0, -16882, 0, 0, 17992, 9481, 7225, 9236, 0, 7722, 0, 0, 9716, -18332, 0, 0, 0, 0, 0, 0, 16173, 0, 0, 0, 0, 5626, 0, 0, 0, 0, 0, 0, 0, 10633, 0, 0, 0, 0, 0, 0, 0, 0};
  static Q15_T U2[HIDDEN_DIM2 * HIDDEN_DIM2] = {3526, 0, 0, 0, 0, 0, 0, 3527, -13454, 0, 0, 0, -6020, 0, 0, 7062, 0, 0, 0, 0, 0, 0, 0, 0, -8879, 0, 0, 0, -6760, -6458, 0, 8387, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4471, 0, 0, 0, 7221, 0, 0, 0, 3745, 0, 0, 0, 5464, 12124, 7620, 0, 0, 5083, 0, 7550, 0, 0, 0, 0, 0, 0, 0, -7819, 14245, -6172, 14408, 0, 0, 0, 0, 0, 0, 0, 6579, 0, 0, -7411, 0, -6447, 0, 0, 0, -6362, 6985, -6077, 0, -7037, 0, 0, 0, 0, -7566, 0, 0, 0, -11012, 0, 4536, -8462, 7994, -3968, 4739, 0, 0, 0, 0, 18699, 0, -8845, 0, 0, 0, 0, 0, 13114, 0, 0, 0, 0, -5723, 0, 11032, 0, 0, 0, 9152, 11232, 0, 0, 5191, -14378, -8078, 0, 5689, 0, 0, 5458, -6078, 0, -8189, 0, 0, 0, 6066, 11913, 8255, -6820, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5444, -6389, 0, 7254, 0, 0, -7959, -4719, 12197, 0, 0, 0, 0, 8632, 11737, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11763, 0, 0, 0, 0, 0, 0, 0, 0, -5826, 7696, 0, 0, 0, 0, 0, 0, 0, 0, -4719, 0, -5149, 0, 0, 0, 0, 13139, 6609, 0, 0, 0, -4058, 0, 0, 5413, 7630, -5371, -10029, 0, 0, 0, 0, -6352, 0, -15090, 0, 0, -8446, -6262, 0, 0, 0};
#endif

static Q15_T Bg2[HIDDEN_DIM2] = {24543, 20796, 18386, 18971, 13113, 25124, 26109, 14899, 7760, 14659, 23204, 22477, 20851, 21492, 21867, 21741};
static Q15_T Bh2[HIDDEN_DIM2] = {19365, 6998, 32429, 23961, 4849, 14317, 4004, 10797, 21648, 19293, 18133, 16057, 10184, 13324, 2524, 8112};

#ifdef SPARSE
  static Q15_FastGRNN_Params RNN2_PARAMS = {
    .mean = NULL,
    .stdDev = NULL,
    .W = NULL,
    .Wids = W2Ids,
    .Wvals = W2Vals,
    .U = NULL,
    .Uids = U2Ids,
    .Uvals = U2Vals,
    .Bg = Bg2,
    .Bh = Bh2,
    .sigmoid_zeta = 16384,
    .sigmoid_nu = 23611
  };
#else
  static Q15_FastGRNN_Params RNN2_PARAMS = {
    .mean = NULL,
    .stdDev = NULL,
    .W = W2,
    .Wids = NULL,
    .Wvals = NULL,
    .U = U2,
    .Uids = NULL,
    .Uvals = NULL,
    .Bg = Bg2,
    .Bh = Bh2,
    .sigmoid_zeta = 16384,
    .sigmoid_nu = 23611
  };
#endif

static Q15_T preComp21[HIDDEN_DIM2] = {};
static Q15_T preComp22[HIDDEN_DIM2] = {};
static Q15_T preComp23[HIDDEN_DIM2] = {};
static Q15_T normFeatures2[HIDDEN_DIM1] = {};

static Q15_FastGRNN_Buffers RNN2_BUFFERS = {
  .preComp1 = preComp21,
  .preComp2 = preComp22,
  .preComp3 = preComp23,
  .normFeatures = normFeatures2
};

#ifdef SHIFT
  static Q15_FastGRNN_Scales RNN2_SCALES = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 8, //256
    .normFeaturesMVW = 5, //32
    .H1W = 2, //4
    .H2W = 0,
    .U = 8, //256
    .hiddenStateMVU = 5, //32
    .H1U = 2, //4
    .H2U = 0,
    .mV1AddMV2 = 0, //1
    .mV2AddMV1 = 0, //1
    .mV1AddMV2Out = 1, //2
    .mV1AddMV2Demote = 0, //1
    .pC1AddBg = 0, //1
    .Bg = 2, //4
    .pC1AddBgOut = 0, //1
    .pC1AddBgDemote = 2, //4
    .sigmoidLimit = 0,
    .sigmoidScaleIn = 0,
    .sigmoidScaleOut = 0,
    .pC1AddBh = 0, //1
    .Bh = 2, //4
    .pC1AddBhOut = 0, //1
    .pC1AddBhDemote = 2, //4
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 7, //128
    .hiddenStateHDGate = 7, //128
    .qOneScale = 0, //1
    .qOneSubGate = 0, //1
    .qOneSubGateOut = 0, //1
    .sigmoidZeta = 7, //128
    .sigmoidZetaMulQOneSubGate = 6, //64
    .sigmoidNu = 15, //32767
    .sigmoidNuAddQOneSubGate = 0, //1
    .sigmoidNuAddQOneSubGateOut = 0, //1
    .sigmoidNuAddQOneSubGateHDUpdate = 7, //128
    .updateHDSigmoidNuAddQOneSubGate = 7, //128
    .pC3AddPC1 = 0, //1
    .pC1AddPC3 = 0, //1
    .hiddenStateOut = 0, //1
    .hiddenStateDemote = 0, //1
    .div = 0,
    .add = 0,
    .qOne = 16384,
    .useTableSigmoid = 1,
    .useTableTanH = 1
  };
  static SCALE_T ShR2 = 8; //256
  static SCALE_T ShL2 = 0; //1
#else
  static Q15_FastGRNN_Scales RNN2_SCALES = {
    .input = 0,
    .mean = 0,
    .meanSub = 0,
    .stdDev = 0,
    .normFeaturesHDStdDev = 0,
    .W = 256,
    .normFeaturesMVW = 32,
    .H1W = 4,
    .H2W = 0,
    .U = 256,
    .hiddenStateMVU = 32,
    .H1U = 4,
    .H2U = 0,
    .mV1AddMV2 = 1,
    .mV2AddMV1 = 1,
    .mV1AddMV2Out = 2,
    .mV1AddMV2Demote = 1,
    .pC1AddBg = 1,
    .Bg = 4,
    .pC1AddBgOut = 1,
    .pC1AddBgDemote = 4,
    .sigmoidLimit = 0,
    .sigmoidScaleIn = 0,
    .sigmoidScaleOut = 0,
    .pC1AddBh = 1,
    .Bh = 4,
    .pC1AddBhOut = 1,
    .pC1AddBhDemote = 4,
    .tanhScaleIn = 0,
    .tanhScaleOut = 0,
    .gateHDHiddenState = 128,
    .hiddenStateHDGate = 128,
    .qOneScale = 1,
    .qOneSubGate = 1,
    .qOneSubGateOut = 1,
    .sigmoidZeta = 128,
    .sigmoidZetaMulQOneSubGate = 64,
    .sigmoidNu = 32767,
    .sigmoidNuAddQOneSubGate = 1,
    .sigmoidNuAddQOneSubGateOut = 1,
    .sigmoidNuAddQOneSubGateHDUpdate = 128,
    .updateHDSigmoidNuAddQOneSubGate = 128,
    .pC3AddPC1 = 1,
    .pC1AddPC3 = 1,
    .hiddenStateOut = 1,
    .hiddenStateDemote = 1,
    .div = 0,
    .add = 0,
    .qOne = 16384,
    .useTableSigmoid = 1,
    .useTableTanH = 1
  };
  static SCALE_T ShR2 = 256;
  static SCALE_T ShL2 = 1;
#endif
