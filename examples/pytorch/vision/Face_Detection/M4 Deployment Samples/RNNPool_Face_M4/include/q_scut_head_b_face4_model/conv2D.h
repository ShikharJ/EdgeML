// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define CONV2D_N 1
#define CBR1F_H 240
#define CBR1F_W 320
#define CBR1F_CIN 1
#define CBR1F_HF 3
#define CBR1F_WF 3
#define CBR1F_CF 1
#define CONV2D_COUT 4
#define CONV2D_HOUT 120
#define CONV2D_WOUT 160
#define CBR1F_HPADL 1
#define CBR1F_HPADR 1
#define CBR1F_WPADL 1
#define CBR1F_WPADR 1
#define CBR1F_HSTRIDE 2
#define CBR1F_WSTRIDE 2
#define CBR1F_HDILATION 1
#define CBR1F_WDILATION 1
#define CBR1F_G 1

#define CBR1W_HF 1
#define CBR1W_WF 1
#define CBR1W_CF 1
#define CBR1W_COUT 1
#define CBR1W_HPADL 0
#define CBR1W_HPADR 0
#define CBR1W_WPADL 0
#define CBR1W_WPADR 0
#define CBR1W_HSTRIDE 1
#define CBR1W_WSTRIDE 1
#define CBR1W_HDILATION 1
#define CBR1W_WDILATION 1
#define CBR1W_G 4

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static const Q15_T CBR1F[CBR1F_G * CBR1F_HF * CBR1F_WF * CBR1F_CF * CONV2D_COUT] = {6919, -6351, 3878, -2797, 12889, -12067, 2371, -4121, 1680, 397, -4700, 6764, 12907, -11456, -859, -1194, 8803, 1413, 12533, -12806, -14022, 11965, -237, -1539, -4455, -3289, -7134, 5029, -17471, 10876, 752, -1939, -8215, 8503, 2901, -2826};
static const Q15_T CBR1W[CONV2D_COUT] = {31111, 29559, 27650, 31874};
static const Q15_T CBR1B[CONV2D_COUT] = {13471, 16541, 11369, -17805};

static const Q7_T CONV2D_Limit = 96;
static const Q7_T CONV2D_Div = 1;

#ifdef SHIFT
  static const SCALE_T CBR1F_Scinput = 5;  //32
  static const SCALE_T CBR1F_Scoutput = 2; //4
  static const SCALE_T CBR1F_Demote = 8;   //256
  static const SCALE_T CBR1B_Scten = 0;    //1
  static const SCALE_T CBR1B_Scvec = 11;   //2048
  static const SCALE_T CBR1B_Scret = 0;    //1
  static const SCALE_T CBR1W_Scinput = 3;  //8
  static const SCALE_T CBR1W_Scoutput = 4; //16
  static const SCALE_T CBR1W_Demote = 8;   //256
#else
  static const SCALE_T CBR1F_Scinput = 32;
  static const SCALE_T CBR1F_Scoutput = 4;
  static const SCALE_T CBR1F_Demote = 256;
  static const SCALE_T CBR1B_Scten = 1;
  static const SCALE_T CBR1B_Scvec = 2048;
  static const SCALE_T CBR1B_Scret = 1;
  static const SCALE_T CBR1W_Scinput = 8;
  static const SCALE_T CBR1W_Scoutput = 16;
  static const SCALE_T CBR1W_Demote = 256;
#endif
