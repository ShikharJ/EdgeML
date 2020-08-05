// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define L4_N 1
#define L4_H 30
#define L4_W 40
#define L4_CIN 32
#define L4_CTEMP 64
#define L4_HF 3
#define L4_WF 3
#define L4_COUT 32
#define L4_HOUT 30
#define L4_WOUT 40
#define L4_HPADL 1
#define L4_HPADR 1
#define L4_WPADL 1
#define L4_WPADR 1
#define L4_HSTRIDE 1
#define L4_WSTRIDE 1

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static INT_T L4_F1[1 * 1 * 1 * L4_CIN * L4_CTEMP] = {2964, 7856, 10397, 8552, -12235, 7373, 6919, 8764, 3506, -159, -1203, 2424, 3272, -4886, -98, -1976, 6658, 1980, 4726, -6369, 4594, 599, 4217, -187, 1495, -435, 865, -1008, 3410, -2481, 1810, -4582, 16079, 7947, -569, -4045, 2147, 561, -5237, 3870, 4381, 175, 1143, 4506, 6483, 457, 3844, -1718, -4444, 5942, -1103, 9702, -5007, -5626, 2093, 2539, -2424, 10445, 1758, 1165, -3395, -419, -5356, -3960, 112, -4153, 683, 2286, 9419, 5471, -4256, 1062, 3724, 1363, 6479, 1424, -6765, 14693, -10542, 401, -3891, 429, 4789, 14018, 7991, -10213, -2543, 107, -825, 251, 7127, 6380, 1934, -4206, -61, -633, -5141, -7123, -1508, 366, -3979, 2424, 7122, 4753, -5105, 8396, -14, -1311, -3236, -16252, -1828, 6034, 14280, 2303, -8170, -1626, 13052, 2394, -5798, -13559, 776, -9154, -215, 4699, 1863, 6034, 436, -2658, -529, 2616, -5269, 594, -3386, 2249, -280, -174, 1186, 3522, 3835, 191, -4010, 191, 6288, -1125, 3265, 1406, -1259, 6215, 7970, 9033, -55, -11931, -4616, 6852, 1206, 3575, -4977, -703, -1894, 4949, -5444, 8922, -3778, 6817, -2918, -277, 5652, 3905, -1506, 868, 2057, -5365, 9029, 5295, 2255, 2577, 3139, 7575, -2387, -2327, -8439, -4943, 4389, -961, 6904, 7625, 5862, -2332, -3977, -1276, 2884, 415, 7902, -2105, 2947, 3120, 7497, -4830, -745, -6730, -149, 723, 11230, 5992, 1949, 7058, -779, -5528, -4223, -539, 4709, 6089, -2408, 1853, -3989, 1160, -8815, -115, -2584, 2997, -5226, 4664, -1389, -4253, -4257, 6334, -536, -2998, 1948, -2288, 5976, -4391, 390, 1442, 2561, 3637, -2922, 6919, -1422, -14133, -193, 12174, 497, -4286, -6139, -2538, 865, -5127, -3263, -4735, -2870, -2378, -6165, -1859, -12814, -1019, -9536, 5710, 6914, -7315, 1222, 3185, -4511, 4174, 1764, 1161, 11433, 3980, 4618, 8851, 3938, 8377, 4123, 4102, -1470, -2471, -331, -7770, -3206, -1532, 9013, 2436, -1463, 1337, -970, 2208, 9708, -3118, -3589, 366, 2373, 3987, 859, -2014, 8931, 894, -9327, 5983, 529, 455, 5867, 4050, -10109, 4254, 7525, 10933, -5427, 1062, -2033, -13485, 1104, 6392, 5906, -3479, 3598, 10494, 5019, -3566, -7389, 13139, 3647, 6206, -3187, 6557, -2396, 1827, 6280, 8450, -9147, 3152, -2286, -1774, -6246, -3361, 7574, -7102, 4581, -1125, -918, -1585, -14839, 174, -4522, 5805, 6644, 2907, 2827, -3044, -3292, 2714, -1080, -4475, 4556, -2958, -4785, -8994, 8098, 1317, -205, 6819, -4179, -7692, -443, 2308, -7318, -7057, -3451, -1571, -9134, 4597, 88, -4979, -3843, -3916, -3954, 13129, 7899, -6964, 3190, 183, -9980, -3748, -705, -4002, -3129, 148, 6476, -1552, 1580, 8615, -319, -7739, 10133, 4101, 3228, -1858, -825, 3542, -553, -5017, 1272, -1058, 8506, -2747, 9721, -5118, 3180, -3437, -3258, 6294, 815, 4976, -4760, 4925, -9120, 4545, -963, -3847, -4437, 3832, -4651, -632, 572, 3988, -1314, -1416, -2330, 5304, 399, -5799, -2721, 7840, 9321, 2967, 7440, 151, -182, 7594, -1229, -24, 3483, 2002, 309, -3976, 3723, -1467, 11047, 675, 2368, -3993, -437, -442, 633, 640, 3895, 5520, 2197, 5705, -6499, -3067, -1709, 1525, 2435, 5711, 4114, -710, 9995, 2586, -7007, 4652, 341, -869, 176, 4029, 987, 5393, 3501, -5678, 9964, -992, -3379, -3959, 860, -5396, 1256, 3007, -7313, -1552, -3302, 4864, 1480, 10454, 7216, 2030, 11010, -1263, 3476, -6606, 2986, -8416, -5359, -17438, -3292, 4125, -504, 2149, 4344, 1203, 7153, 3126, -1061, -2385, 1475, -1525, 7044, 2962, -8735, 13437, -7158, 1149, 7742, -5544, -1312, -5959, 6223, -1231, -10270, -4340, -677, 4215, 1418, -9440, 8269, -490, -7928, -4083, 1019, 822, 4993, 7177, -5557, 8520, -15280, 6025, -6351, 7417, -2158, 2460, 552, 3449, 7187, 6322, 3378, -2752, -3198, -4454, 7681, 8018, -5968, 8236, 14027, -7955, 2920, -1258, 9901, -4994, 1119, -15064, 3733, -3284, -5375, -5161, 10011, 13861, 9951, 6573, 11341, 13241, -1216, 2251, -811, 70, 987, -11092, -5008, 3517, 4443, -1540, 3008, -8577, -5239, -7260, 439, 1121, -1522, -2036, 537, 5515, -5297, -5481, -1472, 7245, -10765, -894, -5197, -8245, -14060, -4403, -2356, -505, -9632, 3403, 2971, 2935, -153, 7709, -654, -2709, -519, 15816, 2372, 3435, -2069, 14348, 340, -1972, 2066, 5269, -9756, 4056, -5663, -4625, -7287, 5759, 7977, 4764, -7051, 8319, 5465, 9968, 1019, 2195, 6287, 1260, -4536, 1722, 4172, 285, -9431, 911, -2057, 3583, -3198, -4299, -9541, -5076, -6460, -446, -4881, 3316, 5439, 4301, -1211, -4169, -6749, 5348, 2974, -156, -7026, 4392, -2182, 9337, 2193, 1374, 887, 3114, 1241, -1822, 4008, 982, 3409, -5235, 6226, 2239, 6617, 4796, -1640, -4857, -3669, 6998, 8916, -10867, -9609, 14737, 728, -5196, -8351, 1235, 2434, -11012, 8892, 309, -3780, -1103, 3910, 4384, -5665, -4348, -2284, -1803, 3160, -6282, -742, 3442, -2749, -3922, 1193, -3306, -2079, -4436, 1188, 1257, -2253, 4185, -8154, 5435, 3013, -3140, -1725, -4043, 3844, -5155, 1453, -5188, -3927, -2316, 5572, 8, -3028, 696, 3129, -5595, -1943, 3552, -9378, 696, 6624, -2731, -4482, 8335, 2446, 27, 3316, 83, 5317, -4699, 4528, 4827, 1630, 2938, 1252, -869, -384, -1776, -5785, -24, -6333, 5785, 2873, -1295, 2733, -2086, -205, 3423, -9069, -12885, -448, -5709, -4561, -671, -1352, 9605, -4479, -6643, 8435, -5995, -1604, -7035, 3924, 6719, -3576, 2660, 724, 1459, -7821, 983, -97, -918, -5729, 3072, -1064, -4821, -1793, 608, 1924, -3416, -1672, 282, 3652, 2581, -4767, -866, -342, 7392, -1648, -2493, 2927, 2977, 3125, 6282, 772, 503, 8762, -9370, 833, 7010, 3470, -10396, -6170, 470, -3516, 986, 2997, 4381, -3081, 6969, 373, 375, 3488, -9240, 559, 4520, -3774, -4743, -950, -7148, 4160, -5804, 4106, 1070, -3743, -884, 6546, -1902, -5647, -2096, -3020, 770, 1498, -5701, 6517, 7388, 1002, -3709, 2652, 730, 179, -1258, 1613, -1271, 1210, 1806, -684, -121, -3885, 9114, 6786, 391, -5264, 1054, -2529, -7020, -3372, -2758, 6866, 11691, -8, 345, 9946, 2742, -18344, 2178, -2537, -1577, -1462, 5195, 6329, 9257, 7004, 8634, -8701, -2871, 8010, 12707, 4182, -189, 2624, -1932, 4365, -3927, 6330, 1602, -3246, 3353, -990, 331, 8476, -8876, -8067, -4314, 5245, 1818, 7574, 4946, -2442, -2114, 8293, 3538, -1120, -12948, -4170, -8501, -2861, 3029, 2364, 11504, -6123, -6433, -1805, -1204, 5693, 10835, 2026, -2354, -1006, 872, 4455, 6230, 5679, -8849, -5146, 4842, -5373, -2945, 440, -3890, 1164, -4427, 891, -8747, 6652, -494, -1393, 7573, -5217, 3225, -5429, -3023, 5236, 6937, 3601, -2190, -3201, 4773, -8830, 7184, -4438, -10983, -7405, -818, -214, 4728, 1539, -2210, -5109, -5130, -10529, -4984, -756, 2258, 9523, 962, -7472, 3523, 7642, -7298, 7340, -8053, 1466, -4585, -7466, 2712, 283, 7355, 4217, 9740, 2764, -3062, -3155, 1630, 3790, -3930, -4496, 10593, -3788, 212, 17959, -347, 1901, -1605, -942, 5645, 6781, 8922, -3957, 7815, 4977, 9933, -3436, 4997, 523, -2185, -7871, 2101, -214, 1357, -1816, -7668, 7794, 4734, 5330, 8136, 5184, -5470, -3627, -4870, -2787, 2118, 3336, 3874, -1825, 4175, 2169, -6630, -4591, -14790, 1737, -2368, 1338, 1229, -6248, 6111, -5652, 4913, 13547, -1041, 3701, -8676, 10634, -6378, -6160, 359, 10398, -5948, -2335, -1969, -1907, 5091, 4413, -7605, -3144, -10061, 5738, 2315, -2321, 7525, -1842, 1193, -7558, -11704, -9487, -5878, -261, -6038, -3996, 50, 448, -1857, -11562, 172, 1053, -134, 8951, 3572, 3403, -5456, -6533, -2701, -1112, 2971, 5045, 454, 2005, -7387, 1294, -7720, -358, -1069, 9733, -1773, 4194, 6686, 7520, 1848, -4734, -7329, -3092, -5653, 478, 5312, -1379, -1594, -3443, 7331, -9981, -5939, -6898, -9640, 4672, -2172, -9319, -1526, -7572, 6380, 5340, 16013, 8855, 2093, -4990, -8539, -7251, 4337, 2729, 6318, -8471, -247, 17744, -7429, 662, -61, -11781, -2259, -705, 5731, -8499, 790, 8638, 5712, 7109, -3511, -4343, 6067, -6159, -8140, -4163, -9552, 5633, -15017, 4553, 5091, 2080, 6506, 4187, -121, -1720, -10141, 9060, 1893, -7523, 1802, -15143, -6840, -376, -9268, 583, -2712, 335, -12303, 1920, 11639, 186, -4885, 3861, 5286, -4306, -2962, -15891, -7293, 3242, 9164, -3295, -264, 3424, -1846, -4212, -13239, -1342, 4446, 4306, 4933, 3682, -4946, 2310, 10215, -4300, 5862, 4753, -2461, -4669, -5869, 9102, 4293, 1862, 958, 5517, -1844, -5614, 9866, 3728, -4568, 8821, 3959, 2852, 6908, 8431, -3371, 4296, 1765, -1039, 1535, 8650, 4017, 377, 4190, -1619, -1088, 6339, -694, -3929, -1747, 6839, -1693, 9492, 395, 369, 2003, -5315, 1281, 5216, -5875, 1606, 6696, 5445, 7550, 743, -161, -2586, 3399, -8292, -3100, 6992, 3000, 10975, -972, -4495, 4450, 2336, -6535, -66, -7051, -9450, -5866, -7449, -1409, -10479, -5459, -5919, 3434, -1881, 2272, 21440, -3802, -2472, 3925, 412, 2260, -4238, -5816, -4094, 2504, -456, 8740, 951, -6143, 4515, -522, -349, -3229, 10984, -3176, 2044, -3180, 3607, 6090, 7610, 5780, 119, -1458, -10005, 3955, -5032, -9231, 5464, -517, -75, 8597, -1131, 7428, -362, -2083, 11881, 5538, 7800, 2762, 12272, 199, 3513, 285, 2988, -2361, -4751, -4576, 1402, 42, 4884, -4613, -6129, -6432, 3819, 7701, -1659, 9881, 1943, 842, -5756, 2038, 268, 3561, -1671, -2224, 5042, 2243, 2214, 4952, 4619, 4849, -962, -5355, 9266, 2309, -3033, -3347, -860, -3239, 3273, -435, 901, -510, -1912, 5268, -728, -768, 6932, 2308, -6593, -2276, -6836, 12834, 1321, 1670, -4558, -3749, -1536, 1597, -2957, 6292, -5989, -3113, 1396, 4568, -345, -3124, 10166, 6262, -4415, 5208, 4607, -3180, -11382, -5199, -2358, -1882, 2017, -1356, 705, 918, -4086, -7778, 7170, 19898, 7725, -5642, 6505, -3047, 6894, 1899, 3062, 7452, 519, 5233, 7754, -3729, 6007, -2180, 1647, 3083, -2967, 7846, -1774, -7839, 5389, 8213, 2515, 730, -6478, 4232, -3557, -2922, -4889, 4084, -2275, -4412, 4169, 13014, 5200, -5966, -300, 5968, 3047, -2312, 3890, -2265, -5072, 6171, 3744, -1058, 3485, -3544, 12497, -670, 4405, -579, -2229, 2045, -1386, 7960, -1411, -1532, -3791, 228, 6909, 5092, 7231, 2193, -7445, -2329, 2437, 2152, 7286, -2320, -1647, -2155, -2323, -273, -3506, -9176, 1847, -681, -3130, 2736, -2847, 1294, -7230, -550, -2762, -4892, -9670, 2176, 862, 2089, 1618, 2801, -6856, -2676, -661, 796, -5766, -5512, -11695, 977, 6424, -4645, 3181, -12734, -1887, -2646, 5062, 1277, 70, 8895, 10539, -1764, -1011, -10214, -752, 3291, -3767, -3988, 3701, 2978, 7249, 1758, -2091, 479, 1000, 6152, 6460, 1642, -5747, -6073, 5588, 8840, -8068, 2086, 11, 2392, 5099, -1491, -5753, -198, 6690, 4834, 546, 998, -6445, -5456, -7263, 3926, 2640, 9011, -7836, 2383, -198, -759, 3424, -202, 1653, 11900, -2441, 12388, 3789, 3893, 1147, 1677, -581, 1797, -3149, 493, -528, -11636, 1857, -2010, -19, 10721, -4217, -1824, 3874, -14853, -3624, 3583, 853, 4063, 5569, 1060, -3574, 2198, 594, 866, 285, 3608, -7682, 3681, -4198, -3088, 129, 5863, 4461, -9011, -3418, -2899, 6322, -2870, -1300, -398, -4976, -3975, -428, 3022, 2209, 2561, 6400, -4251, 4866, 4686, 768, 10184, -9351, -2560, -4046, 2642, -7001, -8425, 1261, 9409, -8204, 3221, 1217, -1307, -4264, -2324, -9635, -946, 7635, 4326, 2263, 4432, 241, 6353, 2341, 3060, 6270, -3029, -849, 2807, -2843, 3918, -9139, 374, -3720, -5432, -6256, -3728, -929, -2669, -5517, -8645, 65, -6878, -5401, -2159, -4765, -2179, -2787, 7285, 721, -7958, -4878, 206, 6014, -1361, 6311, -904, 1691, 9204, 4953, 1737, 4871, -455, 2770, 3858, 4437, 7477, -5261, 1703, 1222, 6017, 12142, -779, 1118, -800, -21987, 4054, 702, 8729, -8978, -1447, -5917, 10067, -2320, 7948, 7622, -3086, 11047, -4895, 7108, 1569, -3363, 3731, 3700, 4605, 807, 1044, 3752, 8872, -15144, 1583, -275, 14172, 4580, -2414, 2782, 3650, 522, -8362, -1000, 14352, 7497, 6388, 15204, 6150, -7469, -4634, 6790, 3544, -2151, 2503, -5682, -5448, 5120, 14125, -2228, -857, -1612, -1428, -6853, 2348, 9491, 1889, -787, -4058, 8109, -9530, -2545, -12897, -8695, -1546, -5663, -5234, 1640, 6129, -8422, -9229, 994, 1634, 7370, 848, 3767, 8462, 183, -5001, 4734, -5937, -9603, 10556, -2990, 4742, -4241, -3746, 5237, 3137, -3535, -5410, 8617, -12665, 384, 4428, 10429, -7488, 4605, -2942, -5069, -9673, 10345, 5911, 7666, -2527, 3194, -7624, 11444, -3664, 3431, 1916, 41, 1819, -528, -3826, 5654, 2920, 10619, -4670, -5629, -3720, 643, -5665, 3726, 5077, 4931, 10707, 7555, -3841, 8548, -6973, -2480, 1342, 6207, 3557, 1063, -664, -1686, 2613, 2709, -1696, 4512, -63, -430, -1260, -4352, -1395, 5546, 438, -1891, 3171, 3540, -6189, -6138, -2164, 2418, 7536, -762, -4533, 1111, 4174, -14218, -2361, -4078, -532, 11104, 511, -4322, 1574, -1243, -4106, -7595, 7638, 3122, 5928, -3192, 3241, -1599, -681, 3010, -2080, -11281, 2799, -3215, -4461, -8977, -862, 3131, -1679, 995, 2012, -5491, 186, -8394, -3875, -704, -6632, 1871, 8039, 10829, -3923, -4559, 1123, 2115, 1046, -2798, 1375, 3619, 6481, -974, 2602, 7288, -4002, -2244, 777, -26, -409, 3442, 1343, 8128, 120, 957, 1376, 260, -2557, 7144, 4113, -4797, -346, -2493, 4493, -3145, 7417, 323, -3758, 1522, 1664, 3317, 8552, -4038, 2506, 8549, 5193, -4402, -7040, -4864, 4195, 6552, 4883, -6393, -4799, -6003, 2060, -4751, -2535, -3884, 9901, 5991, 3336, 2071, 1216, -636, 987, -4707, -3477, 1388, 1093, 973, -3817, 2437, -5331, -1947, 119, 330, -1045, -6097, -9492, -780, -2118, -9340, -1661, -2907, -488, 855, -1645, 5139, 295, 1193, -1280, 1756, -320, 238, 2081, -3924, -2448, 4269, 5353, 3238, -5536, -716, -4315, -2981, -1927, -4258, 2570, 1464, -2791, 1922, 4880, 7761, 373, -1962, 2822, 1318, 4829, -6689, -6035, -6161, -4741, 795, -4843, 3788, 3448, 674, -7573, 13180, 3042};
static INT_T L4_F2[L4_CTEMP * L4_HF * L4_WF * 1 * 1] = {5972, -5067, 6003, 4900, -4370, 6346, 5021, 10673, 3933, 8660, 18821, 3710, 123, -7648, 898, -2933, -8538, -2462, -10798, 196, -1484, 9048, 16729, 1551, 7379, -6202, -1656, -1912, -6905, -532, -159, 16893, 1293, 5249, 1125, 2845, -1036, -3152, -2106, -8804, 6430, -9655, -6513, -2835, -5111, -4689, -1789, -3365, -2279, 39, -571, -7734, -442, -9512, 921, 356, 1030, 12503, -5010, 2116, 3082, 9945, 18, -7552, -6817, -8303, -3119, -2670, -3026, -7991, -5415, -3652, 1485, -2774, 3390, -2740, 17601, -829, -3377, -247, 2315, 9454, -4695, -4587, 6849, 4639, -9113, 4301, -3575, -3976, -5046, 22930, -7287, -2384, -14923, -5196, -3158, -3318, -2344, 569, -4585, 11445, -1894, 2260, 10984, 207, -550, -3400, 6710, -5145, -1408, 13247, -4228, -2833, 2714, -303, -312, 2781, 604, 770, 4843, 19244, -1306, 3596, -1515, 1898, 6455, 9213, 6296, 6612, -100, 4711, 747, -9056, 712, 5062, 5137, 4229, 4617, -2938, -3166, -3076, 17299, -11068, 3002, 661, 1849, 17831, -6588, -3194, -3192, 21, 381, -1351, 4360, 9139, -8665, 7421, 3405, -13063, -6552, -1828, 79, -115, 4162, 2671, -8895, -67, 14169, 8423, 1160, 592, -10451, -1531, 2385, 14179, -818, 3630, 3961, 1914, -813, -7641, 61, -3118, -11410, -5309, 115, -1441, 1984, -3462, 5516, 5954, -1586, 6763, 11897, 1618, 3924, 8504, -3753, 3804, -1618, -91, -3693, -2956, -1114, -12380, -5959, -779, -4261, 2014, -4816, -5400, -6795, -5067, -5126, -6151, 10901, 20031, 1170, -4917, -13146, -557, -1251, -4969, -4434, 3633, 2714, 3123, 17691, 521, -767, 4826, 1821, 4339, 10290, -1606, -5605, 6873, -683, -2201, 13073, 3276, -447, 2045, 2014, 27086, -7563, -19348, 12907, -1097, -10089, 7219, 1802, 7834, 9908, -10098, 7152, 1243, -11486, -102, -3496, -772, -2191, -355, 16656, -1899, 296, -1232, 39, -613, 1087, -208, 1362, 5270, 1048, -7149, 6455, -446, 22313, -1760, 1403, 1169, -86, -5200, 2331, 9183, 8227, 8848, 4484, 10129, 5219, 2707, -6760, 7534, 1434, 2243, 1074, 2935, -118, 4508, -142, -8320, 6166, 4663, 15752, 134, 5361, 10423, 681, 5891, -13130, 2073, -4346, 6712, 1656, 10550, -1166, 1351, 5395, -9604, -1470, 14630, -8207, -4624, -7409, 15649, 4032, 2852, -8113, 468, 3978, -1131, -1839, -4202, -7005, -2835, 9641, -9889, 2682, 146, -105, -372, 887, -1331, 449, -9214, 11467, -3648, -8080, -2723, -4002, 13677, -2344, -10011, 13308, -879, -10282, 7214, -2610, -3901, 1986, 12131, 4529, -905, 2876, -398, -163, 3549, -1888, -2118, -10238, -3420, -4384, -10881, -1933, 17794, 12361, 9895, -973, -87, -788, -1823, 4128, 705, -6271, 13524, -1267, -8139, 15230, 3270, -3568, -3454, 2427, 1846, 538, 1659, -1682, 218, -4121, 3818, -4952, 1251, 9863, 11257, 8892, 8301, 15491, 1404, 5212, 971, -2921, -316, 552, 313, -5169, 1679, 933, 17826, -2403, -1697, -2639, 335, 1280, 3353, -2412, 1017, 9200, 9468, 11316, -2923, -8021, 437, -8570, -89, 22007, -7705, -16141, 14449, -7288, 1968, 24418, 2135, -4088, 313, -3738, 20013, 112, -786, -453, 562, -493, 5645, -2735, 11566, 8668, -4002, -1798, -3903, -2676, -1193, -1027, 5395, -2151, -9909, 10540, -4693, 2209, 12935, -3827, 64, 1230, -945, 21913, 1286, -439, -1262, 2218, 3187, 2552, 5910, 5852, 9232, 3932, 8537, 4398, 3220, 4678, 2759, 296, 7438, 6557, -3158, 3574, 6998, 894, 8025, 13560, 6375, 3843, -3698, 4797, 893, 2903, -1207, -11172, -6098, 2239, 12871, 13497, -1722, 11745, 2772, 1721, 8285, 18546, 360, -580, -4893, -2776, 2431, -2413, 1810, 1280, -716, 243, -5254, 16182, -1752, 1968, 424, 1008, 6015, 8394, 3893, 3656, 12028, -3535, -4064, -7586, -6333, 7615, 13328, 4427, -1377, -5658, 3428, 368, -1174, 296, 10772, 8273, 4348, -619, -10352, 420, 2795, 391, -637, 2560, 3441, 2157, 2292, 16734, 2295, 1527, -3848, -1262, 5206, 2214, 9249, 1672, -25994, 3139, 4581, 2190, 8482};
static INT_T L4_F3[1 * 1 * 1 * L4_CTEMP * L4_COUT] = {-5680, -4474, -3497, -1433, -5660, 10795, -11358, -6502, -808, 5307, -5930, -4726, 7971, -3976, 1418, 329, -1668, 2235, -2361, -5032, 2938, 2850, 6591, -5167, -1634, 4897, 599, -4965, 568, 592, -5695, 455, -6772, -1433, -1051, -7674, 371, 10662, -1440, -995, -3501, 1255, -258, -6100, -2993, 5560, -6221, -3927, 7896, -1270, -1937, -3270, -1801, -3465, -6089, 4443, -1165, -2186, 3000, -4019, -15159, 2466, 996, 1240, 6086, -9959, -4692, -8303, -5322, -77, -3395, 1128, -5726, 1136, -9725, 2995, 6008, -1923, -6782, 8942, -5789, -1047, 10038, -3081, 5857, -5991, -6773, -1706, -4180, 1261, 7278, -5117, -4410, -2305, 4518, -7401, -1555, 7014, 855, -5718, 10437, 949, -6219, 1528, 261, -6718, 249, 8678, -9801, -11103, -671, 7228, -1935, -1614, 8980, -6538, -4696, 6026, 3910, 3006, -22, 4308, -1638, -6716, -9593, -2730, -5682, 652, -253, -3471, -2567, -5112, -473, 23337, -2817, -57, -4729, 6893, -3022, -1314, -3267, 3506, 894, 2720, 5616, -1706, 9844, -8092, -11451, 6467, 4697, -3977, 7624, 2481, 4669, -3704, -3857, 1999, -7912, 820, -3929, 4717, 1468, 8432, 7274, -3561, 2434, 12123, -1596, 752, -2409, -2397, -7008, -5701, -7356, -4204, 8189, 3772, 11294, -2890, -7829, -778, 8524, 6893, -8873, -7406, 2681, 14536, 927, -10891, -2885, -7573, 7499, 652, -3864, -5005, -6359, 3933, 1135, -5381, 6718, 829, 2900, 4448, 15386, -3287, -7193, 3221, 1689, -1923, -3486, -8094, 5268, 3759, -705, -5555, -10454, -121, 791, 6177, -3804, -9370, -358, -4827, -1347, 14205, 5066, 6266, -2676, -17117, 6133, 5317, -1445, 3372, 1402, -948, -2041, -4448, 1039, 513, 171, 1062, -5450, -2304, 5260, -457, -7164, 301, -7049, -5793, -3461, -1688, -3836, -6702, 4131, 3678, -2666, 9859, 1630, -2651, 616, -1967, -4696, -8876, -4571, 3518, -3505, -3403, -6477, 3011, -3116, -2067, -4990, 58, -4434, 371, -8052, 8573, -4670, -1736, 3789, -4853, -7748, -2806, 4101, 1995, -7219, -9238, -2958, -3042, 4548, -2014, 5038, -506, -2204, -4096, -3691, 5622, -2526, -2442, 5808, -978, 2633, -95, 1828, -152, -330, -5980, -3974, -4089, 2413, -4194, 2428, -948, -7457, 109, -3513, -3413, -4662, -7457, -5231, -12158, -4352, -3400, 655, 8430, -238, 14202, -4112, -1699, 9116, 1859, 5586, 6610, 5541, -3423, 7059, 6761, -2024, 4777, -8125, -5561, -3498, 3069, 492, 4641, 1838, 4122, 3821, 2472, 4312, 3302, 3425, 1082, 1165, -4818, -1861, -2497, 391, 2714, -4392, 4104, -1802, 136, -1527, 1755, -1214, 856, 2061, -2706, 4204, 3039, 7556, -1377, -8796, 1784, 2086, 1853, -4780, -4087, 2644, 4929, 2877, 534, -1727, 4906, -240, 5949, 3446, -8278, -1300, -998, 846, 4097, 3907, -1847, -1852, 358, 675, 1900, -379, 3388, 1842, 3400, 1875, -5366, 7871, -4272, 297, -927, -1572, 4520, -627, 3434, 2392, -2037, 4070, -15730, 6826, 6803, -8759, -13881, -5757, 1430, -3262, 3948, 8846, 3227, -17713, 6751, 1698, 4882, 14113, -3745, -7998, 3970, 1875, 10518, -9754, -6020, -5057, 1873, 576, 1565, 1615, 2127, -3907, -5727, -8305, -41, 5261, -2931, -4527, -3895, 5397, -7057, 6124, 10248, 1152, -5586, 3108, -500, 2277, -1533, -1572, 121, 1022, -2152, 3916, -1512, 2080, -3306, -15737, -5781, -7427, -5762, -1489, -7605, -3090, -1084, 5080, -9071, -3803, 1620, 551, 3079, 3400, 3898, -112, 7748, 3047, 5057, 6771, -4295, -11947, -16221, -6285, 1254, -3444, -2365, 8009, 3512, -206, 1619, 6053, 2530, -2609, 2786, 1310, -1140, 8221, -5371, -758, -6504, -3869, -13360, -568, 8833, -6320, 6183, -3517, -332, 1033, -1360, -3017, 7958, 3935, 1949, -3112, 155, -833, -1560, -11321, 7197, -3928, -8296, 81, -2865, 550, -602, 1987, 4206, -2792, 1837, -1505, 5609, -2237, 686, -5748, 4100, -1551, 347, -9495, 1539, -3376, 181, -3285, 7508, 3648, 3362, -1771, -309, -2502, 2020, 8516, -6744, 2953, -26, -2086, 913, -6104, -2633, 568, -1742, -304, -560, 155, 226, 4789, 2814, -2140, -2026, -1672, 2951, 8163, -4149, 3094, 3624, -1267, -12499, -1073, -8182, 4730, 1989, -3726, 1725, -3360, 3145, -5176, 2579, 1229, 1897, 8444, -2245, 144, 44, -1864, -4362, 1592, -7292, 2482, -5802, 17288, -1656, 3297, 4932, -2865, -718, 1573, -6537, 2680, -401, 4451, 8918, 13276, 429, 8388, 6935, 5373, -1803, 3258, -5920, -1527, -9284, 706, -2444, 2491, -1829, -1338, -358, -1481, 15872, 5632, 10264, -2587, -11983, 1076, -2710, 5966, -3291, 6792, -1952, -5934, -433, -1617, 10959, -4100, -7445, 7461, -1023, 5633, -761, 1480, 6330, 530, -3131, -3358, 1263, -394, 346, 6800, -4382, -5117, -4648, -4276, -5331, -4398, 2513, -9680, -873, 3967, -2958, -4724, 455, -6321, 7058, -5590, 5402, -3121, 2555, -3456, -10870, 1337, 4641, 12646, 4160, -6649, -1623, -3995, 1632, 8868, -3629, 8132, -1234, 5533, -6825, -1, -3861, 2805, 945, -3320, -3039, 4965, -5516, -5834, 3011, -12332, -6008, 2169, -1708, -2971, 3387, -1539, 3071, 13631, 5899, -4230, -745, -3092, -481, 4854, -158, 3130, -447, 334, 4966, 2610, -1850, -6574, 2476, 5250, 1973, -2662, -4029, 8231, -654, 2592, -1017, -5180, -5086, 1386, 60, 2416, -98, -508, -1965, -3228, 7384, 302, -2066, 2707, -1959, -5248, 2973, 3816, 494, -552, 3067, -345, -8472, -4098, -4470, 1693, 8065, -1103, -1967, -1566, -5367, 324, -3968, 5476, -2387, 57, -2679, -1376, -6006, -830, -2435, 3056, -6223, 3537, -2078, -2080, 1857, -831, -7216, 5898, -1070, -6839, -6569, 4894, 5932, -681, -7398, 1300, 5115, -7783, 4871, 174, -7490, 2330, -30, -3415, 8722, 1988, -2958, 1209, 10598, 147, -2072, 18151, 220, 4690, -1157, 1379, -2124, 1843, -3498, -1013, -4330, -4361, 4184, 1154, -6749, 2399, 989, 1968, 6286, -991, -2958, 6618, 1335, 479, 969, 9359, -3755, -7011, -4487, -287, -524, -3578, 150, -9235, 3321, 2103, -3737, 3191, 2028, -437, 2317, 1799, -4363, -7928, -4922, 3553, -270, -6470, 10718, -298, -4092, 3276, 8697, 7528, 1945, 9693, 1530, -5026, -9294, 193, -9143, 739, -2007, 8373, 3627, -1120, 717, -7917, -64, -2721, -2515, -5017, 4150, 357, 5455, -1790, 1254, 3188, -4292, -3714, -5178, 951, 6636, 2349, 788, 993, -766, 2969, 1118, -658, 1967, 5345, -4254, -3802, 938, -2478, -942, -662, 4857, 1388, 4002, -2046, 4229, -8344, -2204, 1711, 1791, 790, 2179, -8883, -1715, 411, -1731, 8285, 61, -2266, 9933, 2837, -5141, 2069, -2257, -10348, -735, 1435, 935, -5239, 3031, -4593, 3988, -4711, 6567, -3883, -1210, 1987, 2989, -761, -5545, 213, -1367, -6853, -8878, 5207, 8157, -7012, -4642, -12084, -2683, 4976, 1012, -6216, -1009, -4843, -3567, 141, 2524, 1365, -13489, 7993, -8582, 21030, 7535, 109, -2308, 3881, 778, 4016, -6010, -6891, 6909, -6510, -9723, -6131, -5766, -1964, -6517, -453, -3943, -4543, -6433, 5296, 3370, -709, -1605, -5115, -5080, 2981, -4323, 2387, -9088, 4229, 8121, -9611, -6038, 5888, 185, 5336, 2324, 2590, -3589, 676, -11048, 843, 1180, 6810, -1892, -8225, 7361, 1821, 3636, -3501, 2285, 1110, -5466, -8130, -933, -8115, 4433, 5751, -2526, -3569, 2904, 710, -6574, -4370, -7554, 5550, 2766, -7865, -412, 2450, 6902, -3347, -5383, -8217, 6770, -2754, 3838, -2418, -229, -1956, -204, 4829, 1463, 3316, 526, 3174, 7773, 620, -3681, 3216, 9696, -8383, -1247, -2527, -3734, 7204, -5093, -3090, -592, 7082, 1658, 4389, 4633, -4498, -2259, -4688, 6361, -70, 700, 5809, 4245, 7970, 888, 818, 4107, -2149, 3425, -6171, 169, -5497, 4143, 3942, -5253, 1991, -5295, -1600, -3304, -8463, -4536, -7290, -3589, 2416, -6101, -516, 8080, 4069, 3731, 4744, -31, -1000, -10477, -4381, 1850, -3316, 8, -1402, -13316, 1070, 1120, -393, 6197, 1846, 903, -3471, 161, 96, -3987, -887, 144, -1830, 1608, 3159, -1690, -1557, -2071, -945, 4180, -366, 7840, 3726, -6021, -2689, -2587, 3524, -733, -10725, -1445, 2900, 8042, -1656, 213, -1052, 517, -7560, 1132, 6423, 5907, 2530, 439, 1153, 5773, -3195, -3278, 5269, 4564, 1273, 5242, 2526, 294, -210, 962, -4786, -4019, -598, -1091, -1507, 4344, 1713, 2052, 717, 25, -686, -4763, -6653, -5113, 727, -2851, -4635, -6571, -8868, 1832, 5203, 488, 1988, -4488, 3380, -1844, 9176, 1464, 3534, -144, -6791, 1701, -3388, -1806, -1360, 9783, 10987, 6911, 5582, 1136, 7102, -7037, -1022, 4350, 4332, -135, -7871, 3493, -2692, 941, 7003, 377, -2582, 1578, 5410, -1902, 3574, 5463, -3521, -6835, -2628, 405, 356, 10241, -679, -247, 397, -8345, -3105, 534, 750, -2838, -937, 12025, -4228, 4226, -3827, 2940, -6886, 44, -2030, 113, 6374, 7776, -6880, 1984, 1385, -693, 3800, -9127, 7437, -3878, 1047, 4332, -11618, -4986, -3698, 7305, -9770, 5514, -4807, 9667, -5936, -1816, 4451, -6447, 3541, 1271, 4776, -1712, -1740, -506, -5598, 3617, -5928, -2143, 1194, -1728, -1425, -2711, 4605, 64, -5673, -3485, 1330, 3166, -2560, -2240, 2703, -5111, -2333, 648, 2717, 4966, 3251, 1699, 3057, -11817, -1249, -3401, -1567, 3199, -692, 1326, -2574, 1600, -1810, 269, -1185, -2088, -1349, 7618, 3972, -5009, -3632, -3915, -3778, -8180, 112, -2872, -2913, -1586, 1153, -2290, -3153, -4831, -5199, 2097, -4695, 4123, 2558, -637, 2640, 1693, -3560, 7365, 723, 2613, -5177, -6308, 3296, 1050, -7964, 1426, 308, 2963, -2809, -846, 2134, -558, 647, 8062, 2912, 2914, -1634, 357, 1411, -10526, -6020, -7061, -728, 14332, -3343, 7614, -4000, -1325, 6808, -2060, 1841, -5221, 1590, 1979, 2548, 7221, 1918, 4013, -8554, -5338, 1525, -7005, 1036, 244, 4824, -599, 8238, 2244, -3267, -1817, -2993, -519, 4559, -2221, -2514, 1399, -1427, 1833, -1202, -10137, 5432, 2278, -707, 1386, 3538, 8795, 10129, -1723, 4490, -2232, -4149, 6665, -2771, -4345, 10417, 4715, -2678, 9799, 4860, 10893, -7131, -6660, 1379, -2892, -719, 2993, -7166, -3565, -4968, 485, -3159, 9388, -91, -2345, -8579, -2899, 7816, 4816, 9157, 13399, -4238, -2954, -8408, -2500, 8778, -1691, -4980, 1405, 1355, 2325, -1889, -1980, -5624, 3583, 6582, 4578, 4542, -1234, 4669, 1637, 627, -602, -1674, 765, -2441, 8453, -9416, 4383, 2269, 3436, -4025, -11197, -4240, -2213, -10123, 10725, -3280, -4955, 2788, -2143, -2298, -7398, -5999, 3445, -8072, -5449, 8251, -5817, -1095, 5017, -6336, -873, -8907, 3606, 352, -13689, -3557, -4284, 3303, -2025, 10947, 8759, -2778, -630, -5713, 889, 4578, 9701, -5406, -6128, -4215, 4030, 6193, -2020, 11002, 188, 1665, 409, -1702, -3859, -2222, 4506, 6224, -16470, -530, 7676, 12773, 3317, -5595, 1976, 1944, -7402, -9917, -1640, -4624, 3693, 7296, 1999, -282, -7761, 1673, -1781, -5306, -10072, -4894, -1061, -2739, 3527, -9448, 4065, 2880, 2252, 2506, -3049, 8631, -5608, -6304, -4656, -9654, -9233, -3820, 9951, -7520, -1792, 354, 4225, 3583, -5722, -562, -6871, 5270, 1795, 5249, -10707, 4009, 3417, -1263, -2280, -2753, -5326, 602, -2660, 3846, 3523, -544, 331, -816, -4926, 2606, -2492, -3007, 1325, 725, -1900, 636, 350, -3374, -1033, 5747, -5666, -3615, 3805, 3495, 1758, 7208, 5365, -6295, 3014, 919, 5087, 557, 5246, -155, 4051, 3973, 2033, -343, 3883, -2548, 7959, -2065, -4687, -3441, 7507, 2467, 2011, -1982, -65, 4460, -11683, -2300, 5054, 14360, -4040, -4366, 2273, -3536, 1493, 956, 1484, -1767, -2349, -2957, -45, -2248, -2926, 3735, -14624, 3159, -8912, 4294, -7561, -9561, 7365, -9679, 3878, 291, 7869, -2883, -1013, 2470, 1671, 3690, 1737, -569, -4480, -4078, 8488, -392, -987, 434, 8841, -1680, -5946, -7154, 2498, 6551, 4730, 1204, 16817, 5503, 260, 4309, -18844, 981, 1051, -2161, -3473, 554, -2987, -2395, -2363, -178, -2338, -996, 684, -8366, 453, 6609, -6154, -4521, 3730, -5518, -9652, -2516, 5607, 1000, -2132, 6730, 1321, 8546, 5225, -7795, 3376, 1606, 3214, 6684, 9282, 2966, -2171, 5, 571, 350, 884, -252, -13, -25, 2603, -1550, -5948, -3046, -7643, 7268, -1649, -7858, 4037, -5575, -7846, -2059, -2645, 3334, 6642, -2688, 9003, 4185, 5049, 737, -12432, 10746, 3446, 7133, 671, 2873, -3573, -5245, -887, -6646, -3434, 5123, 2343, -7418, -1457, 6170, 7641, -8229, -5107, -3246, 1649, -3987, 4209, -5591, 472, -1822, -509, 5608, -2781, -6525, -5940, -1424, 7327, -1082, -7297, -8241, 6867, -947, 4215, 4485, 165, 9165, 9609, 767, 1802, -4068, -3576, -8963, 7221, 185, -2805, -3090, 2458, -3833, -2880, -4757, 858, -4944, 2171, -5374, 2086, 3622, -6695, -5968, 865, -507, 1292, -8592, 6258, -230, 3257, -246, -747, -165, 6956, 4214, 1640, 3500, 183, -8847, 2575, 1446, -7504, 2773, -1896, -5715, 4424, 1516, 12494, 3788, 7367, -3334, -1885, -7205, -917, 463, 6468, -431, 4626, -4926, 908, 2012, 9811, 1694, -3553, -1188, 3052, 15564, 136, -1627, 3461, 2389, 181, -2373, 912, 4524, 5602, -3038, 2917, -2306, -3981, 4893, 7413, -4004, 2029, 3465, -6351, 4728, 7267, -4601, -2814, -9425, 4099, 2742, -239, 5407, 3431, 781, -1105, -190, -2297, 2380, -1840, -5279, 2636, -5825, -2694, 1933, 626, -2743, -2081, 249, -2675, -7513, 1339, 741, -12738, 2206, -3350, -3357, -1423, 325, 8114, -3159, -4426, 5807, 6183, 5120, 2111, 2848, 4442, -5618, -2070, 1656, 16781, 6314, -11726, -2629, 7679, -5122, 6056, -756, -793, 3456, -491, 6153, -9050, 17562, -672, 1665, 1651, -7655, -5455, 6766, -5039, -3091, -1784, 3096, 10860, -3566, 3933, -3805, 3556, 2256, -4393, 7276, -1390, 7115, 707, 834, 6526, 3508, -584, -2912, 5902, -1242, -4090, 3984, -5423, 9141, 15292, -3473, 6442, 8080, -5351, 689, -6765, -2799, -12789, -1835, 6102, -609, -14, 3511, -9778, -8925, 5068, -2960, 53, 5041, 3328, 2974, 283, 1982, -7513, 4369, 4853, -2167, 1412, -282, -7112, -7732, 4431, -3664, 2812, 2948, -5085, 5611, 3163, -263, 2903, -5417, 5720, 594, 5465, -12007, 3451, -411, 4825, -13821, -1889, 3785, -4236, -399, -6676, 3037, -1333, 9695, -7194, 4711, 13213, 7341, 11074};
static INT_T L4_W1[L4_CTEMP] = {11649, 8266, 7175, 11556, 7321, 7090, 13587, 7559, 16464, 13497, 16198, 21758, 19422, 10488, 7974, 7421, 14117, 10495, 18944, 8327, 6600, 7838, 10724, 14664, 13613, 8170, 14198, 17154, 11597, 20266, 5373, 16886, 13440, 16555, 8120, 21297, 21558, 13605, 7209, 5391, 14252, 11329, 21274, 11192, 8543, 13306, 16235, 10333, 12108, 17245, 21477, 15495, 6627, 12175, 11233, 9861, 9921, 7723, 16406, 15800, 13030, 13756, 8256, 6626};
static INT_T L4_W2[L4_CTEMP] = {19352, 7976, 13013, 8962, 10514, 12570, 5411, 8304, 13566, 13018, 9227, 11223, 8693, 8612, 8424, 22457, 12071, 12688, 6172, 14260, 10800, 14057, 10314, 8411, 7974, 7894, 6008, 4310, 11847, 11728, 17817, 11070, 9731, 9344, 23110, 7113, 11655, 14168, 14672, 15484, 8730, 9012, 12606, 20054, 8850, 10557, 14416, 8362, 3121, 13339, 12963, 6764, 9591, 8177, 7558, 7713, 13684, 17921, 13847, 7580, 6487, 8940, 9947, 9736};
static INT_T L4_W3[L4_COUT] = {18810, 10899, 17314, 17718, 20888, 12131, 20706, 24317, 16875, 16455, 23327, 21171, 24801, 21513, 16550, 15898, 19292, 18527, 12093, 14672, 14059, 19437, 16913, 21064, 20213, 14914, 21805, 15808, 18621, 19489, 16053, 18053};
static INT_T L4_B1[L4_CTEMP] = {-18233, 21974, 13133, -1166, 20035, 2974, 6279, 29399, -5243, 1780, 18096, 9848, 468, 6013, 22670, 1724, 1389, 15819, 11731, 12320, 7469, -14544, 8345, 3749, 10114, -3265, 7297, 12036, 21345, -11945, -6759, -5806, 5007, -5744, 2447, 16273, -2349, 5363, 2976, 14039, 9698, 16191, -813, -16694, 2911, -10263, 10196, 1955, 13379, 9915, -4464, 10959, 8544, -14221, 9872, 5334, -13685, 1881, 5718, 8043, 2950, 6489, -27646, 12581};
static INT_T L4_B2[L4_CTEMP] = {-1196, -1844, -5494, -9676, 15680, 13457, -6229, 29329, -2640, 1077, 5836, -2336, -2972, -16250, -9393, -1941, -4574, 6049, -2890, -3494, 17434, -8748, 12072, 15147, -4205, -14011, -5030, -4077, 2855, -4431, -4778, -9617, -7096, -8563, -3423, 5203, -2641, 7768, 10305, 778, -7641, -2772, -3822, -2721, -4516, -8200, -1580, -4713, 1493, -5579, -2133, 599, -6071, -14991, -6928, -19481, -7995, -4416, -3421, -4959, -6581, -4673, -10479, 6787};
static INT_T L4_B3[L4_COUT] = {-19950, -23155, 6220, 7359, -19182, 26267, 11535, -18680, 29317, -4361, 18747, -26272, 22532, 4901, 9516, 1075, -17285, -7687, -8580, 6502, -22232, -13337, -3296, 20804, 11185, 7641, 19240, -6685, 4162, 6297, 9832, 17692};

static SCALE_T L4_D1 = 5;
static SCALE_T L4_D2 = 4;
static SCALE_T L4_D3 = 6;
static INTM_T L4_Limit1 = 402653184L;
static INTM_T L4_Limit2 = 402653184L;

#ifdef SHIFT
  static L_SCALE_T L4_ShRU1 = 10; //1024
  static L_SCALE_T L4_ShRB1 = 3;  //8
  static L_SCALE_T L4_ShRX1 = 14; //16384
  static L_SCALE_T L4_ShRU2 = 9;  //512
  static L_SCALE_T L4_ShRB2 = 2;  //4
  static L_SCALE_T L4_ShRX2 = 14; //16384
  static L_SCALE_T L4_ShRU3 = 8;  //256
  static L_SCALE_T L4_ShRB3 = 3;  //8
  static L_SCALE_T L4_ShRW3 = 14; //16384
  static L_SCALE_T L4_ShLU1 = 0;  //1
  static L_SCALE_T L4_ShLB1 = 0;  //1
  static L_SCALE_T L4_ShLX1 = 0;  //1
  static L_SCALE_T L4_ShLU2 = 0;  //1
  static L_SCALE_T L4_ShLB2 = 0;  //1
  static L_SCALE_T L4_ShLX2 = 0;  //1
  static L_SCALE_T L4_ShLU3 = 0;  //1
  static L_SCALE_T L4_ShLB3 = 0;  //1
  static L_SCALE_T L4_ShLW3 = 0;  //1
#else
  static L_SCALE_T L4_ShRU1 = 1024;
  static L_SCALE_T L4_ShRB1 = 8;
  static L_SCALE_T L4_ShRX1 = 16384;
  static L_SCALE_T L4_ShRU2 = 512;
  static L_SCALE_T L4_ShRB2 = 4;
  static L_SCALE_T L4_ShRX2 = 16384;
  static L_SCALE_T L4_ShRU3 = 256;
  static L_SCALE_T L4_ShRB3 = 8;
  static L_SCALE_T L4_ShRW3 = 16384;
  static L_SCALE_T L4_ShLU1 = 1;
  static L_SCALE_T L4_ShLB1 = 1;
  static L_SCALE_T L4_ShLX1 = 1;
  static L_SCALE_T L4_ShLU2 = 1;
  static L_SCALE_T L4_ShLB2 = 1;
  static L_SCALE_T L4_ShLX2 = 1;
  static L_SCALE_T L4_ShLU3 = 1;
  static L_SCALE_T L4_ShLB3 = 1;
  static L_SCALE_T L4_ShLW3 = 1;
#endif
