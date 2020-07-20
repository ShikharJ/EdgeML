// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <stdio.h>
#include "quantized_utils.h"

// All values generated from Seedot on Wider Regression dataset.
// By default, all tests run without using bit-shifting operations.
// Function for matching the predicted and expected quantized outputs.
static int check_output(const INT_T* const pred, const INT_T* const expected,
                        unsigned len) {
  for (unsigned i = 0; i < len; i++)
  {
    if (pred[i] != expected[i]) {
      printf("Output: %d, Expected: %d at Index: %d\n", pred[i], expected[i], i);
      return 1;
    }
  }
  return 0;
}

// Test v_q_treesum() function.
int test_v_q_treesum() {
  INTM_T qvec_A[128] = {3038976, 0, 0, 1514478, 0, 0, 778261, 32670, -2619599, 0, 3849336, 5310900, 0, 0, 0, 0, 0, 0, 0, 0, -142898, 1510353, 0, -6888482, 0, -760720, 1296384, -6749490, -9687275, -686501, -743600, -2112105, 0, 8962408, 0, -17460547, -1477630, 0, 0, -2195694, -860184, -214912, 0, -1389548, 0, 0, 2081898, 0, 0, 23544, -3351768, 0, 0, 0, 3886614, 0, -5839384, 0, 842100, 4051917, 0, 0, 1459796, 2006850, 517867, 3044471, 0, 2578300, 0, 0, -1921101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4709000, 0, 0, 0, 0, -65664, 0, 0, 4313415, -1126680, 0, 0, -8524140, 0, 6248088, 0, 0, 0, 0, 2273558, 0, 0, -738913, 0, 0, -13912800, 0, 0, 0, -5329773, 5346088, 0, 113245, 0, 0, 8590397, 0, 2224368, 1020225, 489580, 0, -185584, -86475, 0, -2013258, -2417536, 0};

  #ifdef SHIFT
    INTM_T expected = {-135840};
  #else
    INTM_T expected = {-135837};
  #endif

  v_q_treesum(&qvec_A[0], 128, 7, 0);
  if (qvec_A[0] != expected) {
      printf("TreeSum Output: %d, Expected: %d\n", qvec_A[0], expected);
      return 1;
  }
  return 0;
}

// Test v_q_add() function.
int test_v_q_add() {
  const INT_T qvec_A[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
  const INT_T qvec_B[8] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
  INT_T pred[8];

  #ifdef SHIFT
    const INT_T expected[8] = {-2773, -1359, -3028, -390, -1666, -2071, -608, -700};
    v_q_add(&qvec_A[0], &qvec_B[0], 8, &pred[0], 0, 3, 0);
  #else
    const INT_T expected[8] = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699};
    v_q_add(&qvec_A[0], &qvec_B[0], 8, &pred[0], 1, 8, 1);
  #endif

  return check_output(pred, expected, 8);
}

// Test v_q_sub() function.
int test_v_q_sub() {
  const INT_T qvec_A[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
  const INT_T qvec_B[8] = {-18777, -9518, 4055, -7309, 8584, -17257, -5280, -7933};
  INT_T pred[8];

  #ifdef SHIFT
    const INT_T expected[8] = {1923, 1021, -4040, 1438, -3812, 2245, 712, 1284};
    v_q_sub(&qvec_A[0], &qvec_B[0], 8, &pred[0], 0, 3, 0);
  #else
    const INT_T expected[8] = {1922, 1020, -4040, 1437, -3812, 2244, 712, 1283};
    v_q_sub(&qvec_A[0], &qvec_B[0], 8, &pred[0], 1, 8, 1);
  #endif

  return check_output(pred, expected, 8);
}

// Test v_q_hadamard() function.
int test_v_q_hadamard() {
  const INT_T qvec_A[8] = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018};
  const INT_T qvec_B[8] = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282};
  INT_T pred[8];

  #ifdef SHIFT
    const INT_T expected[8] = {1423, 7085, -16378, 8209, -12068, 6805, 6475, 6897};
    v_q_hadamard(&qvec_A[0], &qvec_B[0], 8, &pred[0], 5, 6);
  #else
    const INT_T expected[8] = {1423, 7085, -16378, 8209, -12067, 6805, 6475, 6897};
    v_q_hadamard(&qvec_A[0], &qvec_B[0], 8, &pred[0], 32, 64);
  #endif

  return check_output(pred, expected, 8);
}

// Test v_q_sigmoid() function.
int test_v_q_sigmoid() {
  const INT_T qvec_A[8] = {-2772, -1358, -3028, -389, -1666, -2070, -608, -699};
  const INT_T expected[8] = {0, 2760, 0, 6640, 1528, 0, 5760, 5400};
  INT_T pred[8];

  v_q_sigmoid(&qvec_A[0], 8, &pred[0], 2, 1024, 2048, 11, 14);
  return check_output(pred, expected, 8);
}

// Test v_q_tanh() function.
int test_v_q_tanh() {
  const INT_T qvec_A[8] = {178, 1064, -4162, 1718, -1663, 851, 1244, 1282};
  const INT_T expected[8] = {178, 1064, -2048, 1718, -1663, 851, 1244, 1282};
  INT_T pred[8];

  v_q_tanh(&qvec_A[0], 8, &pred[0], 11, 11);
  return check_output(pred, expected, 8);
}

// Test v_q_scalar_add() function.
int test_v_q_scalar_add() {
  const INT_T qscalar_A = 30111;
  const INT_T qvec_B[8] = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901};
  const INT_T expected[8] = {16378, 13638, 16378, 9787, 14861, 16378, 10661, 11018};
  INT_T pred[8];

  #ifdef SHIFT
    v_q_scalar_add(qscalar_A, &qvec_B[0], 8, &pred[0], 8, 0, 0);
  #else
    v_q_scalar_add(qscalar_A, &qvec_B[0], 8, &pred[0], 256, 1, 1);
  #endif

  return check_output(pred, expected, 8);
}

// Test v_q_scalar_sub() function.
int test_v_q_scalar_sub() {
  const INT_T qscalar_A = 16384;
  const INT_T qvec_B[8] = {0, 2760, 0, 6640, 1528, 0, 5760, 5400};
  const INT_T expected[8] = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984};
  INT_T pred[8];

  #ifdef SHIFT
    v_q_scalar_sub(qscalar_A, &qvec_B[0], 8, &pred[0], 0, 0, 0);
  #else
    v_q_scalar_sub(qscalar_A, &qvec_B[0], 8, &pred[0], 1, 1, 1);
  #endif

  return check_output(pred, expected, 8);
}

// Test v_q_sub_scalar() function.
int test_v_q_sub_scalar() {
  const INT_T qvec_A[8] = {0, 2760, 0, 6640, 1528, 0, 5760, 5400};
  const INT_T qscalar_B = 16384;
  const INT_T expected[8] = {-16384, -13624, -16384, -9744, -14856, -16384, -10624, -10984};
  INT_T pred[8];

  #ifdef SHIFT
    v_q_sub_scalar(&qvec_A[0], qscalar_B, 8, &pred[0], 0, 0, 0);
  #else
    v_q_sub_scalar(&qvec_A[0], qscalar_B, 8, &pred[0], 1, 1, 1);
  #endif

  return check_output(pred, expected, 8);
}

// Test v_q_scalar_mul() function.
int test_v_q_scalar_mul() {
  const INT_T qscalar_A = 32522;
  const INT_T qvec_B[8] = {16384, 13624, 16384, 9744, 14856, 16384, 10624, 10984};
  const INT_T expected[8] = {16261, 13521, 16261, 9670, 14744, 16261, 10544, 10901};
  INT_T pred[8];

  #ifdef SHIFT
    v_q_scalar_mul(qscalar_A, &qvec_B[0], 8, &pred[0], 7, 8);
  #else
    v_q_scalar_mul(qscalar_A, &qvec_B[0], 8, &pred[0], 128, 256);
  #endif

  return check_output(pred, expected, 8);
}

// Test v_q_argmax() function.
int test_v_q_argmax() {
  const INT_T qvec_A[8] = {1675, 9870, -9876, -1234, 5674, 28765, 9876, 12654};
  const ITER_T expected[1] = {5};
  ITER_T pred[1];

  v_q_argmax(&qvec_A[0], 8, &pred[0]);
  return check_output((const INT_T*)pred, (const INT_T*)expected, 1);
}

// Test v_q_relu() function.
int test_v_q_relu() {
  INT_T qvec_A[16] = {-3648, 648, -2147, -2348, 1468, -4348, 3648, 3648, -648, 9648, 3778, 4743, 7483, -243, 8, -21};
  const INT_T expected[16] = {0, 648, 0, 0, 1468, 0, 3648, 3648, 0, 9648, 3778, 4743, 7483, 0, 8, 0};

  v_q_relu(&qvec_A[0], 16);
  return check_output(qvec_A, expected, 16);
}

// Test v_q_exp() function.
int test_v_q_exp() {
  const INT_T qvec_A[16] = {13, 54, 34, 35, 87, 11, 41, 93, 89, 11, 90, 25, 76, 39, 48, 93};
  const INT_T expected[16] = {40, 6832, 560, 635, 29493, 31, 1345, -22628, 18482, 31, 25215, 182, -24195, 1047, 3227, -22628};
  INT_T pred[16];

  v_q_exp(&qvec_A[0], 16, &pred[0], 8, 8);
  return check_output(pred, expected, 16);
}

// Test v_q_scale_up() function.
int test_v_q_scale_up() {
  INT_T qvec_A[16] = {423, -987, -2342, 1232, -324, 843, 982, 2342, 343, 654, 987, 654, 567, 2876, 987, 1265};
  INT_T expected[16] = {846, -1974, -4684, 2464, -648, 1686, 1964, 4684, 686, 1308, 1974, 1308, 1134, 5752, 1974, 2530};

  #ifdef SHIFT
    v_q_scale_up(&qvec_A[0], 16, 1);
  #else
    v_q_scale_up(&qvec_A[0], 16, 2);
  #endif

  return check_output(qvec_A, expected, 16);
}

// Test v_q_scale_down() function.
int test_v_q_scale_down() {
  INT_T qvec_A[16] = {4232, -9879, -2342, 1232, -3242, 8432, 9823, 2342, 343, 6543, 9876, 6542, 5674, 28765, 9876, 12654};

  #ifdef SHIFT
    const INT_T expected[16] = {2116, -4940, -1171, 616, -1621, 4216, 4911, 1171, 171, 3271, 4938, 3271, 2837, 14382, 4938, 6327};
    v_q_scale_down(&qvec_A[0], 16, 1);
  #else
    const INT_T expected[16] = {2116, -4939, -1171, 616, -1621, 4216, 4911, 1171, 171, 3271, 4938, 3271, 2837, 14382, 4938, 6327};
    v_q_scale_down(&qvec_A[0], 16, 2);
  #endif

  return check_output(qvec_A, expected, 16);
}

// Test m_q_transpose() function.
int test_m_q_transpose() {
  const INT_T qmat_A[4 * 3] = {1238, 5432, 1834, 6543, -5698, -2342, 9876, 5674, 8435, 6542, 7824, 3924};
  const INT_T expected_A[3 * 4] = {1238, 6543, 9876, 6542, 5432, -5698, 5674, 7824, 1834, -2342, 8435, 3924};
  INT_T pred_A[12];

  const INT_T qmat_B[3 * 3] = {1238, 5432, 1834, 6543, -5698, -2342, 9876, 5674, 8435};
  const INT_T expected_B[3 * 3] = {1238, 6543, 9876, 5432, -5698, 5674, 1834, -2342, 8435};
  INT_T pred_B[9];

  m_q_transpose(&qmat_A[0], 3, 4, &pred_A[0]);
  m_q_transpose(&qmat_B[0], 3, 3, &pred_B[0]);

  return (check_output(pred_A, expected_A, 12) || check_output(pred_B, expected_B, 9));
}

// Test m_q_reverse() function.
int test_m_q_reverse() {
  const INT_T qmat_A[4 * 4]= {4232, -9879, -2342, 1232, -3242, 8432, 9823, 2342, 343, 6543, 9876, 6542, 5674, 28765, 9876, 12654};
  const INT_T expected_A[4 * 4] = {1232, -2342, -9879, 4232, 2342, 9823, 8432, -3242, 6542, 9876, 6543, 343, 12654, 9876, 28765, 5674};
  const INT_T expected_B[4 * 4] = {5674, 28765, 9876, 12654, 343, 6543, 9876, 6542, -3242, 8432, 9823, 2342, 4232, -9879, -2342, 1232};
  INT_T pred_A[16];
  INT_T pred_B[16];

  m_q_reverse(&qmat_A[0], 4, 4, 1, &pred_A[0]);
  m_q_reverse(&qmat_A[0], 4, 4, 0, &pred_B[0]);

  return check_output(pred_A, expected_A, 16) || check_output(pred_B, expected_B, 16);
}

// Test m_q_add_vec() function.
int test_m_q_add_vec() {
  const INT_T qmat_A[4 * 4] = {1324, 5453, 3454, 3435, 8789, 3411, 5412, 8934, 6895, 1211, 6790, 5425, 8976, 4539, 9348, 9321};
  const INT_T qvec_B[4] = {8452, 2341, 9383, 2353};
  const INT_T expected[16] = {2775, 3311, 4072, 2305, 6507, 2290, 5051, 5055, 5560, 1190, 5740, 3300, 6601, 2854, 7019, 5248};
  INT_T pred[16];

  #ifdef SHIFT
    m_q_add_vec(&qmat_A[0], &qvec_B[0], 4, 4, &pred[0], 0, 1, 1);
  #else
    m_q_add_vec(&qmat_A[0], &qvec_B[0], 4, 4, &pred[0], 1, 2, 2);
  #endif

  return check_output(pred, expected, 16);
}

// Test m_q_sub_vec() function.
int test_m_q_sub_vec() {
  const INT_T qmat_A[4 * 4] = {1324, 5453, 3454, 3435, 8789, 3411, 5412, 8934, 6895, 1211, 6790, 5425, 8976, 4539, 9348, 9321};
  const INT_T qvec_B[4] = {8452, 2341, 9383, 2353};
  const INT_T expected[16] = {-1451, 2141, -618, 1129, 2281, 1120, 361, 3879, 1334, 20, 1050, 2124, 2375, 1684, 2329, 4072};
  INT_T pred[16];

  #ifdef SHIFT
    m_q_sub_vec(&qmat_A[0], &qvec_B[0], 4, 4, &pred[0], 0, 1, 1);
  #else
    m_q_sub_vec(&qmat_A[0], &qvec_B[0], 4, 4, &pred[0], 1, 2, 2);
  #endif

  return check_output(pred, expected, 16);
}

// Test m_q_mulvec() function.
int test_m_q_mulvec() {
  const INT_T qmat_A[8 * 4] = {7069, -10389, 1562, -1992, 3262, -37, -1143, -995, 5513, -17035, -14615, -6636, 4733, -403, 4106, -1104, -2707, -1287, -18128, -1832, -10108, -137, 2064, 1207, 5233, 226, 831, -1909, 4489, -1099, 2845, -1261};
  const INT_T qvec_B[4] = {1040, 1919, 4254, 4024};
  INT_T pred[8], temp[8];

  #ifdef SHIFT
    const INT_T expected[8] = {-426, -170, -3535, 524, -2740, 87, 52, 292};
    m_q_mulvec(&qmat_A[0], &qvec_B[0], 8, 4, &pred[0], &temp[0], 7, 6, 2, 0);
  #else
    const INT_T expected[8] = {-425, -169, -3534, 524, -2739, 87, 52, 292};
    m_q_mulvec(&qmat_A[0], &qvec_B[0], 8, 4, &pred[0], &temp[0], 128, 64, 2, 0);
  #endif

  return check_output(pred, expected, 8);
}

// Test m_q_sparse_mulvec() function.
int test_m_q_sparse_mulvec() {
  const ITER_T qmat_A[16] = {1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 8, 10, 12, 14, 0};
  const INT_T qmat_B[14] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140};
  const INT_T qvec_C[2] = {1, 2};
  const INT_T expected[14] = {1, 22, 3, 27, 6, 32, 8, 27, 0, 30, 0, 32, 0, 35};
  INT_T pred[14] = {};

  #ifdef SHIFT
    m_q_sparse_mulvec(&qmat_A[0], &qmat_B[0], &qvec_C[0], 2, &pred[0], 0, 1, 2);
  #else
    m_q_sparse_mulvec(&qmat_A[0], &qmat_B[0], &qvec_C[0], 2, &pred[0], 1, 2, 4);
  #endif

  return check_output(pred, expected, 14);
}

// Test t_q_add_vec() function.
int test_t_q_add_vec() {
  const INT_T qmat_A[2 * 2 * 2 * 2] = {1324, 5453,
                                       3454, 3435,

                                       8789, 3411,
                                       5412, 8934,


                                       6895, 1211,
                                       6790, 5425,

                                       8976, 4539,
                                       9348, 9321};
  const INT_T qvec_B[2] = {8452, 2341};
  const INT_T expected[2 * 2 * 2 * 2] = {2775, 3311,
                                         3840, 2302,

                                         6507, 2290,
                                         4819, 5052,


                                         5560, 1190,
                                         5508, 3297,

                                         6601, 2854,
                                         6787, 5245};
  INT_T pred[16];

  #ifdef SHIFT
    t_q_add_vec(&qmat_A[0], &qvec_B[0], 2, 2, 2, 2, &pred[0], 0, 1, 1);
  #else
    t_q_add_vec(&qmat_A[0], &qvec_B[0], 2, 2, 2, 2, &pred[0], 1, 2, 2);
  #endif

  return check_output(pred, expected, 16);
}

// Test t_q_sub_vec() function.
int test_t_q_sub_vec() {
  const INT_T qmat_A[2 * 2 * 2 * 2] = {1324, 5453,
                                       3454, 3435,

                                       8789, 3411,
                                       5412, 8934,


                                       6895, 1211,
                                       6790, 5425,

                                       8976, 4539,
                                       9348, 9321};
  const INT_T qvec_B[2] = {8452, 2341};
  const INT_T expected[2 * 2 * 2 * 2] = {-1451, 2141,
                                         -386, 1132,

                                         2281, 1120,
                                         593, 3882,


                                         1334, 20,
                                         1282, 2127,

                                         2375, 1684,
                                         2561, 4075};
  INT_T pred[16];

  #ifdef SHIFT
    t_q_sub_vec(&qmat_A[0], &qvec_B[0], 2, 2, 2, 2, &pred[0], 0, 1, 1);
  #else
    t_q_sub_vec(&qmat_A[0], &qvec_B[0], 2, 2, 2, 2, &pred[0], 1, 2, 2);
  #endif

  return check_output(pred, expected, 16);
}

// Test q_maxpool() function.
int test_q_maxpool() {
  const INT_T qmat_A[2 * 2 * 2 * 2] = {11, 220,
                                       130, 40,

                                       50, 60,
                                       66, 76,


                                       86, 910,
                                       411, 312,

                                       513, 514,
                                       715, 716};
  const INT_T qmat_B[2 * 2 * 2 * 2] = {100, 992,
                                       15, 26,

                                       27, 8,
                                       3, 4,


                                       5, 2,
                                       2, 2,

                                       7, 8,
                                       29, 140};
  const INT_T expected_A[2 * 1 * 1 * 2] = {32, 55,


                                           178, 227};
  const INT_T expected_B[2 * 3 * 3 * 2] = {100, 992,
                                           100, 992,
                                           15, 26,

                                           100, 992,
                                           100, 992,
                                           15, 26,

                                           27, 8,
                                           27, 8,
                                           3, 4,


                                           5, 2,
                                           5, 2,
                                           2, 2,

                                           7, 8,
                                           29, 140,
                                           29, 140,

                                           7, 8,
                                           29, 140,
                                           29, 140};
  const INT_T expected_C[2 * 2 * 2 * 2] = {100, 992,
                                           100, 992,

                                           100, 992,
                                           100, 992,


                                           29, 140,
                                           29, 140,

                                           29, 140,
                                           29, 140};
  const INT_T expected_D[2 * 3 * 3 * 2] = {16, 19,
                                           0, 0,
                                           12, 15,

                                           0, 0,
                                           0, 0,
                                           0, 0,

                                           32, 10,
                                           0, 0,
                                           2, 55,


                                           178, 179,
                                           0, 0,
                                           128, 128,

                                           0, 0,
                                           0, 0,
                                           0, 0,

                                           102, 78,
                                           0, 0,
                                           21, 227};
  INT_T pred_A[2 * 1 * 1 * 2], pred_B[2 * 3 * 3 * 2], pred_C[2 * 2 * 2 * 2], pred_D[2 * 3 * 3 * 2];

  #ifdef SHIFT
    q_maxpool(qmat_A, pred_A, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);
    q_maxpool(qmat_B, pred_B, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
    q_maxpool(qmat_B, pred_C, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0);
    q_maxpool(qmat_A, pred_D, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 1, 1);
  #else
    q_maxpool(qmat_A, pred_A, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2);
    q_maxpool(qmat_B, pred_B, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    q_maxpool(qmat_B, pred_C, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    q_maxpool(qmat_A, pred_D, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2);
  #endif
  return (check_output(pred_A, expected_A, 4) || check_output(pred_B, expected_B, 36) || check_output(pred_C, expected_C, 8) || check_output(pred_D, expected_D, 36));
}

// Test q_convolution() function.
int test_q_convolution() {
  const INT_T qmat_A[2 * 2 * 2 * 2] = {11, 220,
                                       130, 40,

                                       50, 60,
                                       66, 76,


                                       86, 910,
                                       411, 312,

                                       513, 514,
                                       715, 716};
  //Convolution Filters
  const INT_T qmat_B[2 * 2 * 1 * 1] = {0, 1,
                                       1, 0};
  const INT_T qmat_C[2 * 2 * 2 * 1] = {0, 1,
                                       1, 0,

                                       1, 0,
                                       0, 1};
  const INT_T qmat_D[3 * 3 * 1 * 1] = {0, 0, 1,
                                       0, 1, 0,
                                       1, 0, 0};

  const INT_T expected_A[2 * 1 * 1 * 2] = {44, 25,


                                           230, 206};
  const INT_T expected_B[2 * 1 * 1 * 1] = {58,


                                           317};
  const INT_T expected_C[2 * 2 * 2 * 2] = {1, 27,
                                           22, 12,

                                           22, 12,
                                           8, 9,


                                           10, 113,
                                           115, 103,

                                           115, 103,
                                           89, 89};
  const INT_T expected_D[2 * 3 * 3 * 2] = {0, 0,
                                           0, 0,
                                           1, 1,

                                           0, 0,
                                           0, 0,
                                           0, 0,

                                           4, 1,
                                           0, 0,
                                           0, 0,


                                           0, 0,
                                           0, 0,
                                           16, 16,

                                           0, 0,
                                           0, 0,
                                           0, 0,

                                           12, 9,
                                           0, 0,
                                           0, 0};
  INT_T pred_A[2 * 1 * 1 * 2], pred_B[2 * 1 * 1 * 1], pred_C[2 * 2 * 2 * 2], pred_D[2 * 3 * 3 * 2];
  INTM_T temp[16];

  #ifdef SHIFT
    q_convolution(qmat_A, qmat_B, pred_A, temp, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 0, 0);
    q_convolution(qmat_A, qmat_C, pred_B, temp, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 3, 0, 0, 0);
    q_convolution(qmat_A, qmat_D, pred_C, temp, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0);
    q_convolution(qmat_A, qmat_B, pred_D, temp, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 3, 0, 1, 1);
  #else
    q_convolution(qmat_A, qmat_B, pred_A, temp, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1);
    q_convolution(qmat_A, qmat_C, pred_B, temp, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 3, 0, 1, 1);
    q_convolution(qmat_A, qmat_D, pred_C, temp, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 1, 1);
    q_convolution(qmat_A, qmat_B, pred_D, temp, 2, 2, 2, 2, 2, 2, 1, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 3, 3, 3, 0, 2, 2);
  #endif
  return (check_output(pred_A, expected_A, 4) || check_output(pred_B, expected_B, 2) || check_output(pred_C, expected_C, 16) || check_output(pred_D, expected_D, 36));
}

int main() {
  if (test_v_q_treesum()) {
    printf("Test Failure for v_q_treesum()!\n");
  } else if (test_v_q_add()) {
    printf("Test Failure for v_q_add()!\n");
  } else if (test_v_q_sub()) {
    printf("Test Failure for v_q_sub()!\n");
  } else if (test_v_q_hadamard()) {
    printf("Test Failure for v_q_hadamard()!\n");
  } else if (test_v_q_sigmoid()) {
    printf("Test Failure for v_q_sigmoid()!\n");
  } else if (test_v_q_tanh()) {
    printf("Test Failure for v_q_tanh()!\n");
  } else if (test_v_q_scalar_add()) {
    printf("Test Failure for v_q_scalar_add()!\n");
  } else if (test_v_q_scalar_sub()) {
    printf("Test Failure for v_q_scalar_sub()!\n");
  } else if (test_v_q_sub_scalar()) {
    printf("Test Failure for v_q_sub_scalar()!\n");
  } else if (test_v_q_scalar_mul()) {
    printf("Test Failure for v_q_scalar_mul()!\n");
  } else if (test_v_q_argmax()) {
    printf("Test Failure for v_q_argmax()!\n");
  } else if (test_v_q_relu()) {
    printf("Test Failure for v_q_relu()!\n");
  } else if (test_v_q_exp()) {
    printf("Test Failure for v_q_exp()!\n");
  } else if (test_v_q_scale_up()) {
    printf("Test Failure for v_q_scale_up()!\n");
  } else if (test_v_q_scale_down()) {
    printf("Test Failure for v_q_scale_down()!\n");
  } else if (test_m_q_transpose()) {
    printf("Test Failure for m_q_transpose()!\n");
  } else if (test_m_q_reverse()) {
    printf("Test Failure for m_q_reverse()!\n");
  } else if (test_m_q_add_vec()) {
    printf("Test Failure for m_q_add_vec()!\n");
  } else if (test_m_q_sub_vec()) {
    printf("Test Failure for m_q_sub_vec()!\n");
  } else if (test_m_q_mulvec()) {
    printf("Test Failure for m_q_mulvec()!\n");
  } else if (test_m_q_sparse_mulvec()) {
    printf("Test Failure for m_q_sparse_mulvec()!\n");
  } else if (test_t_q_add_vec()) {
    printf("Test Failure for t_q_add_vec()!\n");
  } else if (test_t_q_sub_vec()) {
    printf("Test Failure for t_q_sub_vec()!\n");
  } else if (test_q_maxpool()) {
    printf("Test Failure for q_maxpool()!\n");
  } else if (test_q_convolution()) {
    printf("Test Failure for q_convolution()!\n");
  } else {
    printf("All Tests Passed!\n");
    return 0;
  }

  return -1;
}
