// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define L7_N 1
#define L7_H 30
#define L7_W 40
#define L7_CIN 32
#define L7_CTEMP 64
#define L7_HF 3
#define L7_WF 3
#define L7_COUT 32
#define L7_HOUT 30
#define L7_WOUT 40
#define L7_HPADL 1
#define L7_HPADR 1
#define L7_WPADL 1
#define L7_WPADR 1
#define L7_HSTRIDE 1
#define L7_WSTRIDE 1

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static INT_T L7_F1[1 * 1 * 1 * L7_CIN * L7_CTEMP] = {329, -13730, 3567, -5615, 8608, 6795, -8498, -6305, -8217, -3405, 358, 3898, 711, 6451, -6496, 14, -6769, 15116, -3874, -2128, -8444, 2134, 2731, 6864, -2793, -251, -4936, 7114, -620, -1443, 3853, 8999, 14883, -10409, 11106, 7263, 6420, 1838, 6281, 1470, 5561, 1490, 10128, -14472, -865, -9339, 480, 10973, 774, 756, 1259, 7569, 3219, -4396, -8456, 7753, 8001, -5672, -2940, 11866, -1447, -546, 737, 8943, 4511, -280, -3246, -10832, 10396, 8094, -20102, -16225, 101, -19041, -10174, -2331, -3971, 2341, 403, -5166, 8066, 4211, 10299, -9753, -7143, -866, -7837, -7088, 622, -15294, -8839, -11602, -10355, -3661, 3274, -4538, 6883, -14487, 1529, -6640, -3798, 6245, -10185, -4547, 637, 874, 3649, 1397, -13453, 4198, -9413, 15124, -2172, 7423, -348, 1074, -4818, -5889, 298, -3690, -1891, -2576, 877, -2472, -309, 11133, -2177, 3427, 10175, -23232, -2366, -7411, 8257, 4943, 10426, -487, -1325, 9573, -1993, 4985, 7075, -8072, 16900, 4034, 2188, 13744, 11048, -17091, 3924, 13688, -7144, 6014, -10417, 9823, 11517, -1149, -5784, 18769, -4705, 972, 1287, 3319, 12808, 15114, -3314, 3866, 604, 1677, -10588, 8580, 1, -18796, 651, 13736, 4733, -2697, -6806, -9966, -608, -8421, 16400, 15389, -3596, -6381, -599, 1566, 13690, -1772, 4721, 19385, 2184, 2199, -1094, -12590, 14895, -6173, -4965, -4269, 824, -971, -4750, 7027, -5414, -12756, 4143, 17348, 14190, 6874, 5562, 563, 732, 1556, 2016, 15258, 17502, 2880, 5404, 4160, 8550, 3855, 4866, -8684, -5702, -17472, 4740, 1363, 793, 7006, -5068, 3933, -383, 6755, -1234, 6234, 4114, -6901, 3186, 14288, 1240, 10165, -4219, 3019, 19021, -2562, -2444, -25, 1231, 3739, 4057, 6975, -3561, -10746, 1301, 1743, -2592, 4338, -8106, 3535, -6807, 5541, -6467, 6929, -1785, -9962, -291, -1486, 11288, -2818, 7745, -13159, -9681, 2618, -13982, 3091, -3875, 1209, -3440, -1363, -1037, -7364, 16294, -9316, -5272, 7260, 246, -9453, -7984, 26, 7047, -4766, 3809, 12613, -1784, 11620, -5237, -3669, -1933, 11144, 6108, 4348, -1002, -15164, 6709, -1034, 9991, 3996, -19869, 5360, 4708, -1636, 1554, 8119, 7470, 3751, -6787, 13694, -9158, -2273, -4405, -8682, -5771, 15695, 4547, -3055, -9844, 6428, 1032, 4835, 7019, -5631, 199, -5517, -2963, -2647, -5714, 428, 1361, 13177, 9549, -976, -2336, -12309, -14725, -13110, -5957, -16202, -10759, -1680, -227, -5161, -5558, 4185, -3359, -6554, -18104, 7524, 6087, -6989, -5702, -3501, 3850, 3013, 1793, 5795, 8775, 1665, -7015, -2734, 8041, 46, -6696, 1434, -3335, -12610, 10454, -5829, 6653, 1223, -2300, 4899, 3935, -11845, 4213, 9561, 17989, -5538, -11375, -1571, -6360, -2967, -2529, -5729, 12944, 11717, -3115, -12985, -4910, 6084, -8858, -513, -7557, 7789, 1559, -202, 4359, 255, -3310, -2485, -3287, -5615, 721, -1306, -5275, 14535, -2891, -960, -6860, -3735, 1408, -3598, -12280, -13414, -6602, -14923, 11455, -5260, -11489, -220, -22770, 7331, -3203, 4009, -3375, -6447, 3465, -4944, -113, -17912, 9907, -5817, 18254, -60, -6123, 921, 18944, -5355, 828, -4307, -7974, -9168, -5018, 4856, 6907, -7446, -5429, -10293, 5061, 3724, -12124, -1989, -3020, -4302, 4316, -7245, -4645, -13320, -4442, -5870, -5927, -5101, -5608, 6669, 3130, -7016, 9480, -6907, 1414, 933, -6627, -296, -5468, 4542, 2675, 6454, 2968, -1607, -12261, -1232, -2805, 3998, -11823, -6055, 5453, 1431, 2933, 2907, 1018, -9712, -9972, -4514, -2758, -260, 1106, 426, 4704, -12810, 6974, 6006, 960, 14989, 5394, 2500, 9831, 19682, 22601, 9350, 15460, 4332, -2252, 13078, 13143, -5197, 2731, 16294, 2511, -2084, -8610, -7826, 16281, 10935, -341, -2345, 18245, -1752, 16868, 1180, 14557, 3115, -3176, -4667, 6589, -3596, 8275, -629, 6033, -3729, -20356, -12489, 16196, -3476, -15681, -810, 11638, 4870, 5328, 118, -13120, -2301, -6004, 3567, 7416, -3016, -863, -17173, -5928, -12631, -3105, 7655, -6996, -6025, -4954, 14473, 5666, -1331, -722, 7864, -1736, -3128, 2502, -4239, 6859, 3137, -3017, -8368, 4003, -4055, -530, 22402, 2304, 2485, 10398, 3114, 3783, 5706, 5610, 2830, -4372, 4493, -13135, -4821, 14415, 3875, 1900, -1746, -476, 3761, 9449, -2390, -312, 7652, -13261, -3530, 998, -1462, -9203, 188, -2334, -3622, 851, -8535, -8263, 7855, 8543, -10469, -3033, -2069, 12715, -4139, 8697, 3338, -4014, 10660, -434, -830, 1145, -2010, 1253, 6628, 1714, -3177, 3844, -2199, 7921, -1432, 7739, 676, -9795, 5574, 4704, 5638, 3222, -4454, -2454, -2636, 6292, -3467, 18599, -3858, -6441, -6013, 11354, -9438, 7501, -1221, 1775, 956, 3622, -8704, -8927, 6608, -5236, -268, 3055, 4455, -16239, 3856, -5406, 1343, 1166, -118, -4868, 1657, 471, 9883, 3435, 2840, 5831, 770, 3108, -1527, 1536, 1333, 1995, 2122, -250, 346, -4489, -328, -5308, 5075, 1559, -8158, 10616, 1508, 360, -463, 5291, -206, 6645, 12062, -9787, 91, -637, -2940, 1470, 20891, 4823, 5287, -3437, -6442, 3913, -8420, 11362, -1882, 3929, 13686, 4018, 4405, -7225, -9569, -1421, -8078, 4252, 2525, 8801, 15301, -9369, -2338, -294, -7773, 1877, 9488, 12257, -12164, -928, -2073, -5419, 13545, 2701, 15111, -11330, -1291, -1596, 993, 4884, 7369, 10828, -19074, 4036, -9798, -16640, -11035, 18708, 4712, 16037, 843, 4977, 1765, 11941, -4200, -10572, -4056, -4950, -12645, 4436, 7310, -5453, -6995, -3662, 6978, 16209, -11413, 7638, 7124, 950, -10533, -2385, 11999, 6764, 4493, -1956, -3779, -6569, 2254, 11748, 3202, 15521, 3610, 5264, 13546, 10438, -19685, 6445, -1594, -282, -1590, 5021, 3082, -1665, 2939, -17419, -7436, 12681, -8186, 14416, -23501, 5959, -5466, -4085, -5550, 8421, 8037, -1839, -669, -12010, 7405, -2813, -5625, -14108, 6040, -4770, -9376, 5097, 7065, 10488, 5576, -4615, -4501, -364, 9946, 8913, -134, 939, 10360, -373, 6597, -4618, -225, -1507, 8760, -1202, -1473, -194, 986, 14079, -2438, 792, 6644, 2631, -6041, 6469, 6208, -693, 4891, 13208, -10872, -6262, 10772, 3386, 1356, -2395, -6397, -1034, 3043, -8990, -7652, -5230, 3384, -8407, 5757, 91, 2387, -8117, -6349, 5251, 1256, -1836, -5334, 7137, -4465, -185, 4671, -2320, -10175, 6564, -1678, 17930, 23368, -7851, -8752, -22223, -4263, -985, -10025, -9265, -2177, -1222, 3991, -3642, -4824, 361, -1537, -1641, -2979, 3278, 9354, 10094, -7670, 774, -1885, -265, -6895, 3567, -4078, 1796, -5169, 9730, -4376, 1467, -13590, 14509, -10644, 3213, 8225, -16126, -3809, 810, 6693, -6720, -15259, 3833, 14476, 3948, -7388, 3936, 5333, -10235, 6607, -5365, -3108, -16090, 12272, -7479, -6154, -3559, -6282, 867, 6072, -3601, -10126, -11973, 14045, -15617, 4823, 8778, 977, 6002, 2042, 1902, 7982, 9636, -1962, 10740, 3247, 1464, 8418, -6122, -7903, -10147, 1086, 12827, 15543, 5138, 6404, 1358, 10816, 1738, 3265, 9336, -31, 2536, -2596, -1154, -1424, 4788, 10724, 13431, 5223, 4008, 12432, 3330, -5684, -10131, 19199, -6282, 14268, 6149, 20573, -1337, -653, -3220, -177, 10553, 13897, -2304, 3067, 4009, 2825, 1123, -1003, -5836, -15252, -10972, 2849, 12486, -5850, -5704, -4883, 2065, -10797, -9693, 7429, 3298, -4005, 7449, 1999, -922, -9633, -6691, 2779, -3626, 22815, -1743, 13550, 7565, -21707, 2519, -15970, 17926, -3522, -10421, 3751, 6891, -16280, -2459, -12012, 9569, -2090, -3361, -6373, 8991, 3124, -3536, -3161, 4601, 3202, 7573, -6509, -7082, 700, 4586, 19357, 6802, -17912, -15867, 6, -4034, 3189, -2273, 14613, -8618, -6033, -3889, -5825, -9353, 2338, -1688, -6621, -1986, -12175, -6428, -1280, -5720, 1984, -4223, 1264, -3279, 334, 15247, -2060, -7897, 23023, -1120, 3636, 13228, -14285, -2031, -357, 3251, -14083, 5114, -2894, -9999, 9428, -6243, -7132, -3155, -2231, 5175, 8342, -3378, 5265, 5895, -7673, 4857, -3321, -3477, 835, 4106, 10573, 2766, -4671, -23077, 10057, -4256, 6234, -451, 12043, -6956, 13549, -8459, -1636, 13726, -1589, -7549, -5149, 4439, 5771, 5306, -9848, 7441, 5569, -812, -1812, 167, -3561, 2996, 8195, 9954, 3264, 3030, 7895, -11034, -14303, 4379, 16084, 6587, 4868, 1599, -19295, -6862, 10256, -1006, -7739, -9529, 7978, -1924, -2707, -8680, 2319, 13397, 17271, 2332, -803, 4063, -9203, 12873, 6440, 7412, 2416, -7089, 17188, -1398, 16962, -2702, 3715, 22564, 810, 8649, -12947, -6464, -8027, -3894, 8796, 10706, -821, -6620, 3443, -8587, 4355, 2500, -5739, -7307, -185, -11832, 4289, 10409, -10705, -4806, -2495, -3589, 2156, 2395, -10902, 6800, -9523, 397, 8304, -7848, -2420, 6193, -990, 6937, 11378, -1591, 2868, 1993, -4009, 6229, -5998, -6279, 4667, 1431, -3908, -2104, 3365, -2506, 11573, -758, -7300, 13167, 7236, 4089, -7536, 7964, 6550, 10957, 5963, 11420, 7809, -6579, -333, -743, -3076, 6744, -5536, 1023, 10735, 6600, 2162, 9901, 6386, -4252, -3042, 10483, -6569, -2838, -3848, 10882, -3393, 3737, 4845, 6762, 14269, -7971, 4404, 7926, 2004, 14643, -6578, 1600, 9349, -2182, -2688, 6770, -8178, 9216, 2044, 8887, -703, 9357, 11832, -8030, -10976, 1955, -5593, -5622, -229, 4439, 7658, -8665, 16973, -7152, -6209, -4885, -11673, -9825, -20802, -1438, -16394, -5511, -6990, -20560, 9176, 4935, 1983, -12093, -2482, 10881, -9074, 940, -16862, -1520, -5872, 1036, -5122, -2465, 10547, -6106, -1791, 11667, -10788, -3467, 515, 1190, 7353, -14479, 3133, -3358, 8514, -2619, -9443, -2137, -9226, -1630, -12929, 3218, -2642, -525, 5202, 7715, 19745, -9619, 7165, -6572, -85, 2652, -3807, 8658, 9919, -5547, -1794, -6062, 717, 7656, -7025, 3704, -9526, 17487, -2089, -7733, 6680, 4830, -1391, -7472, 5330, -4309, 11600, 4079, 1198, 5121, 8786, -5298, -4937, -9228, -187, 3033, 7162, 4893, 4069, 13842, 3529, 2970, -2070, -18892, -4597, 1107, 6016, -1718, -17613, 9184, -7334, 6813, -11295, -2908, 2276, 3786, 3668, 4701, -4401, 7646, -7622, 6950, 147, -3831, -9316, 4236, 1619, 7376, 6688, -723, 3528, 5215, 8504, 2788, 12283, 676, 14958, -325, -15448, -10730, -7674, 2195, -7214, 8315, 12869, -4888, 3144, 5329, -3237, 7, 17898, -8110, 7688, 2463, 16708, -2500, 2312, 2003, -1697, -1972, 1307, -4838, -4437, 7108, -3009, -2091, 850, -3116, 9622, -9004, -2290, 164, 4476, -4817, 5162, -10906, 2249, 10326, 4182, -245, 5435, -1426, -5295, 2808, 1131, 10969, 5229, -2026, -9844, 6021, -714, -6298, -2397, -10738, -3495, 7768, -5751, -6056, -7177, 4172, 5143, -8698, -5381, 10511, 3877, 2839, -3726, 4790, 9361, 5088, -3161, -1147, -4028, -216, -5003, 1802, 2526, -4905, -1115, 1283, 6814, 11623, -1486, -760, 2116, -994, 982, 28229, 9660, -6184, 7179, -1904, 2874, 219, -2551, 9228, -15282, 8547, 546, 18389, 1468, -1127, -3288, -11643, 1876, -1312, 5581, 5846, -2904, -147, -4705, 13920, 8971, 8089, 8936, 1595, 2969, 3689, -653, 4761, 2236, 7296, 5287, 4633, -2573, 5373, -4353, -225, 8676, -17478, -3399, 6747, 8220, -8587, 1680, -2362, -778, -4575, -8218, 6740, 4855, 2221, -8304, 6380, -4622, -861, -5530, 4340, -1835, -12295, 10508, -9310, 5530, 8567, -7149, 8919, -9001, -12506, 8026, 4130, -6330, 3948, 5378, -1578, 11958, 8408, -8898, 4520, 6030, 13072, 9801, 12145, -812, 8288, 5356, 12568, 1797, 2612, -6300, -7988, 5458, 20737, 4290, 7926, 8825, 15373, 3655, 8295, 4531, 10391, -4241, -10973, 12193, -7506, -6155, 3160, 4966, 1760, 15095, 7157, 516, -2363, 8601, -7273, -99, 5979, 318, -1551, 4896, 1667, 3318, 14641, 12644, -14106, 2237, 25721, -5603, 1095, 7157, 12429, 10275, 5346, -7458, 9130, -2380, 2923, 5049, 8727, 2503, -4748, -346, 1757, -3740, 3650, 6642, -3859, 9797, 14120, -4637, 336, 7410, 3950, -1948, 5558, -3071, -22754, -1557, 5973, -5519, 8561, 2445, -2290, -7709, 8264, 13558, 2194, -2074, 339, -5562, -5276, 7767, -6133, -5918, -3053, -5663, -15266, 490, 13827, 9176, -4466, 9315, -9382, -5268, -2803, -9220, -2958, -9742, 5430, -9165, -16756, -2532, 3531, -4614, 17231, 25849, 5295, -4040, -10056, 6449, -1832, -6766, 7183, 904, -6572, -9487, -10154, -8673, -489, -756, -6187, -9433, -240, -13620, -8145, 3418, -4473, -7292, -9208, -1664, -5273, -6355, -4448, -3134, -12964, -3475, -4940, -20926, -4846, 2191, 8435, -521, -5893, 5464, -619, 5434, 4424, 377, -713, 2611, 3078, -15349, 20162, 1738, -7435, 18727, 114, -2567, 3486, 13457, 2219, 5578, 3221, -1361, 15278, 4693, -8193, -12136, -13229, -8371, 981, 1464, -1563, 7743, 4692, 5535, 8788, -1598, -2378, 16590, 11534, 6957, 13074, 5208, 5818, -1393, 2359, 5234, -14058, 1953, 9568, 8002, -65, -3847, 14307, 14923, 6456, 17141, 5605, 12050, 12452, 5994, 656, -3571, -3753, 6301, -6041, 3981, 1415, -2657, 5857, 8825, 13878, -11832, -12021, 4098, -9748, 15516, -1573, 2486, -6718, 3090, -14410, 3653, 220, 1024, 1593, 12887, 3287, 9237, 6729, -7113, 12711, -6660, 1314, -484, -7844, -3901, 4059, -2759, 9746, 5728, -2552, 9228, -1669, -6881, 1344, -158, -6483, -7978, 3172, -15841, -5784, -15607, -369, -1526, -3098, -8032, 12133, -7388, 2072, -4414, 4801, -3585, -5143, 3554, 2876, 2979, -14154, -1882, 10628, -14487, -4057, -3026, 14289, 348, -3302, 3563, 3239, -5517, 2688, 2970, -10630, -12992, 5177, -6, -11273, -3455, 5665, -16058, -4921, -16982, 5808, -7097, 4513, 4280, 6809, 8057, 11659, -8773, -3324, 9159, -2294, 7188, 4535, -98, 5579, 9195, -17119, 9511, -5157, 185, 2790, 16893, 733, 7586, -1821, 2302, -9367, -2047, 1339, -15377, 324, 4240, 2502, -15845, -82, 12140, 2915, 10381, -3375, -10854, 1916, 8187, 18811, 1265, 5442, 6232, 15038, -8517, -351, 3800, 1208, -2649, 21210, 16908, 13965, 345, -17589, -7386, -2984, -5703, 4975, 11753, -3041, 6883, 2971, 6801, -6237, 9018, -7048, 12467, -12643, 18505, 4308, -515, -11234, 2249, -2225, -1699, -1800, 16098, 11221, -996, 6715, 18791, 3879, -1382, 5385, 11918, -157, 4218, 3894, -11036, 6918, 4121, 7678, -5331, 12337, 16029, 4732, -557, -2043, 682, 229, 7378, 1353, -12479, -17530, 9576, -6654, -7022, -2204, 17436, 8470, 3385, 2554, -10854, -2776, -1915, 5959, -1346, 3771, -7284, -5929, -6619, -7270, 13273, 7312, 3724, 9685, 5692, -5834, 59, 12708, 6632, -3528, -11762};
static INT_T L7_F2[L7_CTEMP * L7_HF * L7_WF * 1 * 1] = {3971, 6133, 12630, -2939, -1271, 1706, -5048, -5036, 429, -2963, -4097, 6869, 2291, 4516, 9157, -2763, 4213, 7480, 4991, 5717, 3118, 5540, -10331, -2453, 6117, 6676, 988, -8899, 3075, -1154, -3944, 2786, -572, -3748, -5481, -3747, -6931, -11388, -6304, -2746, 5960, -662, -3735, 13672, 938, 1509, -7773, 1180, 2805, -8734, -3472, 1035, -2532, -4684, -7707, -5823, -1089, 888, 12289, 1570, 2379, -349, 564, -122, 4727, -1336, 741, 13815, -5071, 5053, -2363, 4676, 4144, -5668, 2206, 8188, -9907, 2315, 6292, 5725, 2680, -4967, -7702, 8550, -6013, -3207, -4443, -2187, -3939, -849, -2407, -5382, -3477, -3909, -5863, 7264, -4543, -4363, 9575, 4772, -1096, 6798, 4037, -11451, 9316, -2556, 1853, 5650, 168, 413, 272, 7470, -3618, -3146, -3687, -9944, 4353, -4528, 2164, 3547, -6909, -4426, 9116, -3823, -4660, 8501, 122, 7185, 10944, 5537, 3149, -4307, 693, -6760, -1130, -1272, -2871, -4770, -4468, -7536, 14092, 4001, -1361, 672, -6262, -5165, 6076, -6097, 6201, 5170, 7380, 9011, 2567, 4101, -6613, 7218, 1256, -13344, 2874, 11065, -1164, 9950, -5071, -5738, 1285, 1575, -2202, -4507, -3742, 11128, 4891, -287, -4382, 7156, -4199, -7118, 8592, -136, -6108, 9100, -3756, -3631, 8656, 2654, -7569, -3580, -120, -2496, -9410, 527, -1807, -227, 252, 16778, 1710, 204, -2464, 2209, -817, -3786, 1738, 5775, -537, -3696, 12927, 6359, -3654, 9104, -6647, 6897, -1539, -1773, 8699, -4585, 287, 1800, -2911, -5410, -3799, -7486, -5103, -1841, -2742, -727, -2718, 171, 388, 4163, -5706, -2584, 2689, 4036, 13419, 1106, -1820, -1724, -4572, -4355, -7963, -6336, -5125, 231, -4589, 3297, -3519, -576, -3252, -8964, 1492, -1319, 9632, 328, 4701, -8862, 123, 1863, -6158, 11167, -1072, -3705, -1209, 1798, 8410, 3714, -7874, 5620, 10502, 1918, -6586, -4642, -3377, -4017, -3940, -2684, -6053, 1432, -3153, -463, -2314, -1266, -1358, 4781, -3157, -1812, 13337, 2151, -1065, 6913, 11066, -5091, -3423, 6548, -4152, -7956, 10535, 1734, -3134, -6073, 1814, 1708, -5301, 12736, 52, -4880, -1678, 1664, -2211, 3750, 3724, 294, 11423, 3659, 123, 5063, 1952, -1387, 12870, -1891, -1101, -10684, -237, 193, -237, 27, 11026, 3236, 4299, -5820, -10302, 1252, -5292, -2853, -2071, -1084, -4306, -4668, -2026, -5862, -1876, -6802, -8402, 5612, 7284, 8472, 7206, 5018, -3558, 1041, 635, -6199, -973, 8004, 10461, 1922, -4496, -6482, -904, -2653, -4090, -604, -782, -1652, 7285, -2191, -7670, -4353, -5921, 5993, -2292, -2717, -6873, -5773, -5659, 1610, 1149, -2947, -4909, -6497, 2997, 7759, 1562, 3612, 4187, 8360, -1488, -7521, -5706, 4583, -3084, 8617, 5991, 7669, 4454, -4202, -1650, 1352, 4922, 14913, -569, 331, 6196, 21, 701, -469, 626, 5997, 1443, 1479, 4177, 821, 2376, 5573, 8485, 6641, 4239, -5423, 959, 10778, -13189, -616, 891, -347, 16, 5986, -7385, 611, 2674, 835, 3870, 7841, 9563, 5722, -2863, -10356, 908, -4238, -6032, -4505, 3459, -9090, -6242, -3081, 4046, -4240, -5435, 4389, -2650, 5677, 8084, 1518, -1243, 5898, 3329, 3321, 6814, 7021, -606, -8207, -7296, 2479, 8527, 7185, 3178, -3813, -2638, 1886, -2074, -3231, -3358, -3284, 13605, -4352, -5682, -9506, 2243, 7507, 538, -264, 3885, 6918, -9327, 18013, 1450, -5441, -5218, 1585, 3, -7216, -520, 6442, 7126, 10333, 5939, 1304, 5279, 3401, -5371, 178, 12079, -13091, -4819, -1676, -1431, -2049, -1941, -4678, -683, 2842, -225, -8343, 3038, 13817, 7250, -2427, -7677, -343, 1484, 10304, -5314, 800, 763, 34, 655, -2583, -4744, 2535, -5608, -4723, -7019, -698, 8813, 846, 1554, -2145, 9765, -3695, -459, 6416, -2401, -6942, 7871, 6509, 9267, -3375, -12859, 1318, -100, 585, 4785, 7425, -2303, -5591, 1670, 1104, 6394, 10767, -2792, 1395, -514, -2812, -689, -2524, 7000, -6387, -4174, 13809, 1703, 9471, -4351, -5470, 6724, 457, -4976, 361, -7404, 385};
static INT_T L7_F3[1 * 1 * 1 * L7_CTEMP * L7_COUT] = {4656, 5982, -7489, 803, 64, 3035, 9198, -5486, -4397, -15109, -2333, -834, -6210, 5365, -5036, 1478, 8676, -2999, -1634, 2763, -7691, -6306, -2636, 13329, 8491, 4355, -12781, 7124, 1373, 6167, 6878, -10697, 8784, -13526, 10603, -2823, -4086, -6365, -7199, 3966, 6813, -2183, 2617, 7765, 2256, -5890, -8148, 5404, -3726, 12833, 1681, 3538, -6590, 7494, 8137, 1649, -929, 2499, 2568, 2653, 2208, -13804, -2405, 2153, 7596, -303, 1448, 8977, 302, 3366, 2121, 15026, 17421, 11016, 11761, 4800, -7252, -898, -6148, -6307, -2604, -5148, 7250, -3750, 3534, 183, 2441, -3178, 3724, -8460, 4429, 5261, 4059, 874, -463, -8097, 8457, 1009, -182, 6173, 2061, -381, 9866, 471, -4552, 1346, -12457, -2995, 3449, 755, -2397, 2638, 429, -7207, 6669, -12984, -8452, -504, 9474, -47, -2050, 6486, 4663, 13420, 1119, -4202, 755, -4287, 9604, -1233, 9839, 1185, -5652, -2701, 7827, -3920, -4943, 3610, 6229, 4959, -5158, -2515, 7314, -344, -10985, 206, 5989, -6351, -13059, -5149, 7149, 7528, -909, -745, -1331, -7781, 4394, -6556, 5261, -976, 4087, 4071, 2653, 3472, -19526, 1737, 6784, -1222, 4459, -305, -9019, 5071, -14927, 1137, -3027, 4246, 2247, 4990, -5424, -15301, -1115, 3745, -936, -6933, 5171, -5903, -10824, -4021, -3169, -10347, -4068, 580, -5204, -11468, 2471, 2246, 464, -5810, 5211, 845, -2667, 11461, -2202, 2495, -1159, 3607, -1613, 597, -12434, 417, 16099, 3996, 5453, 4398, -9432, 3324, 4785, 785, -1229, -8795, -2718, -998, 8453, 132, 3326, 7568, 6703, -850, 17460, -785, 1447, 5000, -8638, 10519, 1, -10468, 2805, -6108, -2686, -11177, 947, 3980, -4434, 8191, 5416, 10472, -5604, -16834, 9117, 7590, -5933, -1830, -7367, 3697, -1598, -7653, 8546, 4017, -15641, 3045, -4087, 3368, 1599, -2461, 5757, 3375, 4957, 2443, -6754, 12165, 4086, -1204, 8473, -3162, -1220, -7870, 19056, -6682, -16731, -2756, -15486, -263, -5557, -5326, 6309, -6300, -7877, 1647, 7008, 13623, 731, 7981, 8188, -1756, 6903, -2787, 5735, -9332, 1307, -3140, -2540, -9552, 11101, 4195, 5047, -14357, -950, 4986, -906, -5005, 11712, -1242, -632, 2506, 13496, 8905, -3945, 5966, 6258, -9423, -2678, 5710, -4709, 7709, 9265, 4595, -5716, 3388, -1500, -4511, 6200, -2515, -360, 5071, 5813, 837, -2694, -5523, -6094, 1321, 2294, -2139, 8084, -2059, -2149, 1425, -4265, -6238, 7178, 4406, 8324, 15108, 3626, -12211, 20136, -12311, -8824, 2021, -9653, 385, 1075, 4574, 293, 9751, -4900, 457, -8965, 8493, 2428, 7341, 2184, -4253, -7890, -9288, 7594, -5743, 4697, -2764, 4293, 2912, 8257, -5744, 1504, 17564, 116, 9784, -555, -4962, -7400, -5423, -4119, 5073, -1932, 14305, -1830, 1788, 818, 7813, -8723, 3994, 3301, 4287, 3326, 1137, -2812, 2933, 7521, 742, 1001, 11894, -5634, -1671, -3994, 2009, -7241, -9707, -2490, -11213, 5619, -1207, -14706, -9763, -4916, -4954, -1961, 2863, 2257, 6859, 2146, -5466, 10956, 8245, 3280, 7978, 4182, 1104, -14572, -8432, 5696, -6946, 1652, -1362, 2439, -5042, 9533, -3683, -4748, 6325, 11793, 11551, 2376, -9650, -4673, 9156, 1801, -1713, -8077, -12029, -283, 4466, -7087, -4471, -1877, 12258, 5139, 2857, -9106, -2434, 1232, 6160, 5532, -1664, -3237, 3765, -7755, 2170, -957, -742, -25641, 2846, -3940, 7168, -14633, 12166, 4880, 9967, 7337, -4019, 4869, -7861, 1430, 10544, 9000, -1721, -835, 1618, -5821, 4283, -386, 4593, 1300, 5779, 4505, -1882, -7629, -5749, 1743, -1305, -2821, 5492, -5718, -6060, -5318, -6557, 6732, 2422, -1703, 525, 3854, -4195, 9267, -4677, -11976, 10491, -5229, -3996, -2973, 518, 4914, 621, -7490, -10506, -9394, -4080, 4710, -533, 3667, -3496, 2577, 3476, 7473, 312, -5297, -10551, 473, 19772, -1528, 4741, 10249, 3624, 19674, -5651, 4574, 2025, 369, -15545, -797, 5294, -9881, 7263, 5080, -4969, -10158, -427, 4693, 10777, 6064, 12377, 12434, -4825, 7084, -1886, -1092, 12851, 7466, -2222, -384, 2242, -4788, 2660, 3570, 9757, 7599, -4547, -875, -319, -2273, -1336, -4381, 494, 4972, 2229, -1053, 9205, 7748, 2706, -2640, 1102, 4173, 4966, 418, 8816, -1713, -4748, -1522, 2381, 8340, 308, 5400, 21821, 2426, -4698, 6576, 2138, -3514, 1018, -5114, -5030, -349, -8833, 220, -12687, -210, 907, 3706, -404, -16368, -3498, 2925, -9322, -3719, -1334, 4828, -7862, -485, 937, 4020, -876, 1945, -7311, -280, -4117, 10160, -6066, -497, 1946, 5480, 580, 8280, 8787, 3706, -2127, 6018, -3321, -5494, 768, 968, -5013, 4951, -9619, -6695, 7551, 4248, 3430, -1682, -7164, 8665, 5537, 4317, -3457, 4034, -314, -4103, 5105, -7272, 454, 6192, 5576, -301, 6653, 1385, -3667, 9963, -14558, 413, -4656, -3045, -11920, 19899, 12223, 6568, 7366, -5364, 7940, -2069, -1397, -1709, -9100, 2009, -2481, -337, 9819, 8733, -1585, 16949, -11815, 14949, -18, -12472, -10308, -3076, -2416, -3756, 2854, 6555, 7331, -2392, 910, 3635, 5594, -5175, 3022, 3571, -2184, -16278, 3309, -1264, -2849, 1236, -1215, 5404, -958, 1682, -6939, -6619, -6300, 3284, -1983, -4287, -11232, 10248, 6286, 4515, 8692, -11270, -1974, 411, -7643, -6404, 2604, 10274, -7277, -12394, -5132, -851, 3356, -1264, -2471, -8532, -13250, -6890, -12900, -4680, 5487, -515, 5238, -7556, -1203, -3215, 14366, -5292, 5963, -11991, 5033, 8690, 9994, 3732, -10536, 1267, 3441, 446, -6771, -6631, 3772, 3320, -6928, -2462, -22264, 7972, -13868, 1370, 5012, 7431, -6829, -2953, -7283, -6589, 4487, 16227, -6161, -7225, -10104, 5955, -309, 1843, -1771, -3324, -9327, -7930, 4107, -11222, 3465, 21193, 1914, 5804, -2197, -8504, -1006, -9196, 1881, 5782, 7520, -2715, -2445, 102, -10348, 7564, 131, 3790, -9012, 13108, -2466, 4805, 7262, -16487, -934, 12684, -1250, -3328, 3915, -1325, -4034, 1858, -4603, 6636, -6572, 528, -9533, 6956, -8574, 8080, -3091, -8519, 5632, 989, 990, -1283, -8222, 927, 132, -14540, 1085, 7501, 5270, -1418, -3797, 1223, 11290, 3895, 7224, -11639, 2382, 676, -5644, 943, -6921, -121, -3514, 1794, 777, 8703, -3166, 2849, -3984, -4125, -1466, -8760, -8737, -4162, -4778, 5887, 4735, 316, 947, -1874, -4282, 3233, -12593, 76, 6729, 1515, 13601, 8289, -2002, 402, 2634, -3484, -6676, -4628, 4783, 2549, -4365, 469, -529, 5139, -5943, 814, 9541, 12392, -9721, 2306, -497, 2914, -2610, -7911, 681, -12260, -6110, 6004, -10276, 5140, -1778, -1947, 281, -3195, 2770, 2474, -1510, -4891, 387, -3927, 2026, -4191, 8156, 9810, 2579, 4617, -488, -9876, -2494, 14622, -1100, -4022, -1710, -60, -8864, -752, -8385, 520, 2936, -8756, -4923, 8812, 1465, -10528, 1760, -6409, -3509, -3982, -1162, 9061, 2339, -8616, 5070, -6319, 12395, -932, 11424, 8755, -11345, -3279, 2642, 8429, -6040, -3897, 6977, -3556, -3507, 10553, -8422, 3256, -1092, 126, -8586, 5194, -5845, 4079, -5520, -14698, -9101, -11515, -6865, 2714, -2309, 3356, -220, 2159, 4489, 1813, 5561, -3984, 5230, 3861, -2109, 9083, -3457, -13911, 1751, -15741, 10557, 3624, -3768, 1038, 7921, -9546, 11435, -275, -6638, 5181, 2082, 3035, -1787, -4602, -2194, 8029, -1199, 3519, 2836, -6486, 4338, 1579, 2451, 5330, 42, 2771, 3502, -607, 8336, 6274, 2920, 1066, 16143, -2545, 12372, 4458, 3639, -3184, 552, -11234, 16999, 914, 3917, 1725, -7116, -2610, -2776, -14516, 5887, 12797, 3704, 11013, -85, 9243, -5138, -4666, -3588, 1022, -2434, -767, -6616, -6736, 4613, -8184, -971, 1497, -7716, -2999, 6596, -4229, 2058, -6355, -7992, -6853, 3056, 1134, -6027, 4527, -1974, 5240, 6992, -19701, 2463, -3385, -3101, -7529, 70, -13803, -10577, 4744, -14948, -2350, 2246, 14412, 9058, 10723, 2459, 2114, 15916, -3833, 10734, 5726, 12228, -4317, -5499, 14484, 12706, 7610, 14906, -2527, -2405, -1766, -3119, 8523, 680, 1818, -2051, 19055, 13025, 12571, 618, -3588, -3995, 2311, -792, -7947, 10241, -1817, -4390, 2686, -1406, 16862, -1005, -5250, -8148, 1254, -12574, -2220, 2758, 571, -7038, 5868, -7986, -2549, -3155, -3534, 8919, -2814, -476, 6135, -3376, -17698, -3199, -5485, -2575, 5605, -7869, 1543, -6670, 151, 15534, -583, -8211, -4146, -8413, -3341, 8723, -10339, 4602, 8644, 2141, -6264, 2610, -625, -10140, 4078, -1567, -5274, -3823, -6154, -9621, -12311, 501, -21524, 3830, 2007, -4308, -1966, -10329, 8340, 2998, -815, -6180, -1919, 5792, 2478, 8572, 802, -897, 5541, -5334, -5499, -3806, -338, 4722, -5358, -7545, -3846, -9047, 14586, 3175, -8480, -3678, -2590, -186, -3761, -7803, -4004, 9700, -5576, -1019, -63, -2528, -987, 3232, -14267, -2261, 4792, -6255, 10979, 8579, -14906, -12600, -2511, 10020, 11423, -1944, -12487, -6777, -1507, -11355, -6008, 12493, 7027, -7906, 8272, 11436, 3431, 7968, -4822, -8485, -2964, 1349, -3255, -6907, -8290, -1810, 1074, 4017, 5882, 9529, -6666, -6857, -3736, 5574, 6545, -5, -5599, 1467, -2715, 1781, 6046, 676, -2556, 3724, 3194, 10603, -2886, -4139, 2609, 5591, 4052, 1816, 9822, -5047, -5907, -9208, 1331, -208, 12636, -12224, 790, 3980, -9933, -941, 5049, 1401, -6241, -2100, 7672, -2238, -1023, 5076, 4229, -13862, -1853, -6905, -4806, -6504, -7840, 226, 1268, -11202, 6041, 9625, -2216, -4355, 2780, 3109, 4661, -9766, -1088, 4749, -460, 1589, -12904, -4815, -5940, -213, 163, -1424, -9927, 5230, -4613, -4991, 1917, 2157, 6027, 2492, -6905, 7011, -1123, -2610, -1779, 15312, -5214, -3164, -1171, -6635, 7631, 6386, -926, -16984, -8397, 6928, 2189, 2032, -3202, -1584, 3953, -4444, -5329, -10764, 9186, -2813, 7673, 11713, 10805, -5120, 2328, -4777, -7952, -12651, 1711, -3979, -1463, -3727, 8103, -4062, -3778, 1614, -2845, 5107, 8261, 6427, -9179, 988, 16503, -6735, 1680, -7123, 9323, 4430, 11240, -2211, 4893, -11793, 384, -5999, 354, 6373, -11718, 9366, 6079, 4, 3971, 11068, -726, 21543, -2222, 15780, -751, 2718, -13631, 13824, -303, 1672, 7984, -107, 1296, -16, 6119, -15482, -5411, -4489, -16973, -3266, -12638, -617, -8485, -3847, 9800, 14005, -16733, -6692, 13525, -1352, -1002, 5264, -12094, -12500, 14885, -6181, -4448, -1441, 2441, 4751, 3528, 2704, -6489, -4306, 6785, 9270, -4542, 5144, 2564, -4026, -5623, 5788, -4486, 13423, -11868, 2548, -2764, 2481, 1608, 2325, 5916, -10719, -6964, 176, -1615, 18804, 1119, 8064, -13518, 15029, -12781, -9274, 13770, 10472, -6602, 18777, -9589, 7414, -8805, 11691, -8750, -2436, -5780, -5751, 4762, 4575, -5053, -6026, 6690, -8790, -238, -1287, 3246, -4183, -3788, -5222, -847, 7012, 6055, -6369, -12680, -194, -2135, 3897, -66, -3461, -9437, 4595, 3993, 9302, -10060, 15137, -6039, -5057, -6297, 13937, 2808, 4062, 4279, -2198, -4114, -1078, -440, 2020, -5750, 4358, 14355, 1825, -6264, -9169, 5804, -1280, -2849, -1815, 6718, 1595, 7376, 3775, -10397, -1300, 4301, -4544, 497, 4400, -3752, 9388, -1123, -384, 4433, 625, 8079, -2431, -1908, 1916, -5071, 3982, -211, -94, 4411, -295, -7288, -2736, 2187, 2377, -7166, 4586, 2938, -2664, -5333, -2514, -9294, 620, 5564, -4644, -2946, 3537, -10596, -3528, 7977, 5974, -8499, 3451, 12552, -4287, 2910, -2321, 1100, -1381, -5956, 8305, 1698, -707, 4471, 2834, -10239, -10332, 3257, -1856, -4531, 2414, -6477, -1727, 1347, 7025, -3683, 4418, -2110, -2373, -15029, -13003, -7069, -8029, 442, 5206, 7318, 9102, 399, -2172, 641, -18927, -57, -1598, -2655, -9640, 433, -8088, 7793, -14379, -3654, -597, -3605, -16466, 8921, -3479, 607, 2425, -2264, -8859, 11407, 1003, -13530, -12077, -42, 10296, 1457, -7339, -2150, -9606, -1250, -142, 10953, 4783, -3738, -887, -3508, 8444, 7289, -328, 9280, 4023, 4869, 11193, -9872, 868, -2898, 609, -593, 1503, -2486, -2630, 3624, -1137, 563, 17501, 14628, 3926, 2101, 7750, -2843, -139, 5906, 6214, 15936, 4269, -4244, 7302, -3183, 2744, 4756, 14243, -9596, -8308, 79, 379, -6177, 2985, -4165, 8133, 956, 102, -16814, 293, 3233, -8074, -4456, 8901, -3425, -3366, -2559, -3780, -12845, -2223, -1545, 5325, -66, 3925, 2300, 4339, -5166, 3059, 34, 7620, -4860, 5815, 8735, 3283, -3887, -267, -5176, -2874, -978, -3259, 8282, -271, 1707, -5294, 5280, 2738, 5991, -4080, 3864, -10882, 12009, 3530, 12979, -9387, -3458, -180, 6741, 1391, 5698, 10664, 3273, -12458, -6577, -7262, 11638, 771, 12163, 8719, 2937, 5582, -4672, -4465, 1448, -7919, 1189, -6193, 3693, 2630, -4297, -4303, 4975, 2803, 3669, 7659, -2151, 5983, -5012, -3311, -816, 5359, 5138, 500, -12332, -13492, -12174, 1688, 11094, 3438, 4595, -1670, 10814, 8945, 1183, -1050, 2506, -3684, -1204, -2859, -7285, -161, 2658, -12441, -4257, 2385, 525, -97, -3691, 5312, 3956, -4798, -2720, 8324, 5809, -1302, 2478, -12515, 12439, 2410, 3320, -10014, -564, 7861, 629, -8397, -6815, -9956, -1278, 3610, -5297, 4593, -5403, -4020, 1664, -7514, -4955, -4032, -8957, -7941, -309, 3096, -8133, -10625, 106, -5614, 101, -2569, -2381, 8004, 9723, -1668, -8506, -2905, 5126, -4801, -265, -3701, 4366, -1972, 2458, -1823, -3667, -1619, -314, 1172, 989, 3385, -1849, -3039, -1000, -1389, 2452, 4944, -2140, -6165, 5769, -1886, 3234, -1964, 559, 7616, 3570, 3477, -4223, -3375, -350, -1167, -4928, -3323, 1606, 3122, -5478, 3018, -7098, -392, -12372, -13603, 8495, 13046, 16280, 21251, 16965, -847, 11574, -3101, 10662, -12546, -7819, 2536, -2679, 7985, -13304, -16677, 11563, -18384, -11633, -3007, 275, -2191, 5540, -3208, -4427, 10087, 4650, 14673, -3363, -19554, -7555, -4264, -6630, -47, -10334, -10796, -479, 25, -980, 5385, -2269, 1184, -3322, -1091, -506, -1111, -2159, 7684, -6037, -214, -5450, -8142, 3466, 1271, -1796, -1930, 527, -8561, -1091, 7192, 1081, 7431, 5970, 399, 1437, -2249, -2670, -330, -16244, -8368, -1655, -7249, 4222, -11781, -4662, -11402, 693, -6568, -1146, -3668, 8082, -11851, -10098, -2380, 5454, 6470, 475, 12399, 8500, -4759, 1986, 7898, -4043, 14965, -1791, 115, 2833, 1420, 1229, 3893, -3099, 1240, 6664, -2409, -3304, 907, -2845, -3664, -2738, 2209, -2404, 6731, 8921, 4313, 7643, 1102, 4411, -4976, -4008, 1431, 1094, 5115, 3113, 674};
static INT_T L7_W1[L7_CTEMP] = {13650, 16183, 17950, 21636, 17984, 22418, 17890, 16251, 12213, 16748, 18255, 11058, 20170, 17043, 15003, 17021, 22871, 10397, 13118, 21988, 20927, 14109, 17116, 16552, 17851, 16137, 20024, 22070, 21191, 15964, 27370, 14949, 16991, 19428, 9569, 12522, 19779, 14699, 17385, 12102, 17743, 19791, 19999, 17161, 8598, 15791, 11884, 14802, 15361, 25657, 17211, 13915, 14704, 8823, 26120, 18506, 21362, 23429, 22782, 21716, 8122, 19198, 16598, 21112};
static INT_T L7_W2[L7_CTEMP] = {13072, 7530, 10333, 12399, 10076, 12326, 19650, 10959, 15376, 15286, 14004, 14438, 21470, 15995, 18462, 21089, 9709, 14020, 21902, 12126, 11251, 14577, 8549, 18096, 9871, 13822, 10391, 18603, 14440, 15915, 11522, 9714, 15248, 15265, 12933, 31349, 13806, 12329, 13933, 22814, 15808, 8232, 10108, 13487, 11251, 11449, 24747, 11056, 6952, 11528, 16575, 19782, 17832, 24137, 7940, 13073, 18270, 23597, 14559, 15723, 17302, 10996, 13426, 20921};
static INT_T L7_W3[L7_COUT] = {20082, 14499, 22093, 19151, 19410, 23998, 24887, 25188, 19098, 23088, 22453, 20692, 28319, 25767, 21742, 26337, 24517, 20989, 17624, 16784, 21795, 23421, 23610, 25013, 29063, 26640, 22315, 19478, 24862, 24319, 21586, 25168};
static INT_T L7_B1[L7_CTEMP] = {3606, -16748, -8430, -4218, -3979, 9502, -11867, -23028, -10155, -9826, -2104, -2023, -1660, 140, -8683, -135, -7476, 15183, 3784, 12823, -9807, -1752, -11198, -1786, 731, -15699, -5574, 4320, 2611, -16399, 393, 9185, 14626, -4318, 25327, 5071, -3262, 11674, 9152, -8381, -2863, 9802, 11664, -7319, 2383, -15246, 5594, -218, 823, -786, -1877, 19165, 12365, -16577, 4237, 12569, -19612, -2823, 12835, -7793, -6045, -268, 2005, -1251};
static INT_T L7_B2[L7_CTEMP] = {-7232, -12738, -11551, 18305, 1153, 15621, -876, -10473, -11579, 17460, 8892, -11935, 5322, -3798, -4825, 2970, -16424, -10573, 1294, -2090, 16777, -7129, -11757, -7529, 28005, -9649, 27017, 3275, 4737, -6851, 15369, -13413, -2872, -2474, -17130, -525, 4385, 20334, -8674, 1103, 10101, 25052, -7878, -8830, -11899, -14769, 4199, -17460, 32327, -5689, -5960, -7097, 1478, -5170, -15303, 3596, -9362, 490, 8970, -3074, -7004, -12049, -1933, 3918};
static INT_T L7_B3[L7_COUT] = {3202, -24789, 26833, 4786, -9023, -25788, -19562, -13291, 6065, 22602, -20926, 5323, 16979, 13572, 20343, -2935, -15814, -7522, 21870, 20810, -7901, 8903, -8871, 18645, -7003, 3166, -9920, -6101, 12524, -12270, -20803, -6135};

static SCALE_T L7_D1 = 5;
static SCALE_T L7_D2 = 4;
static SCALE_T L7_D3 = 6;
static INTM_T L7_Limit1 = 402653184L;
static INTM_T L7_Limit2 = 201326592L;

#ifdef SHIFT
  static L_SCALE_T L7_ShRU1 = 11; //2048
  static L_SCALE_T L7_ShRB1 = 4;  //16
  static L_SCALE_T L7_ShRX1 = 14; //16384
  static L_SCALE_T L7_ShRU2 = 10; //1024
  static L_SCALE_T L7_ShRB2 = 4;  //16
  static L_SCALE_T L7_ShRX2 = 13; //8192
  static L_SCALE_T L7_ShRU3 = 9;  //512
  static L_SCALE_T L7_ShRB3 = 4;  //16
  static L_SCALE_T L7_ShRW3 = 15; //32768L
  static L_SCALE_T L7_ShLU1 = 0;  //1
  static L_SCALE_T L7_ShLB1 = 0;  //1
  static L_SCALE_T L7_ShLX1 = 0;  //1
  static L_SCALE_T L7_ShLU2 = 0;  //1
  static L_SCALE_T L7_ShLB2 = 0;  //1
  static L_SCALE_T L7_ShLX2 = 0;  //1
  static L_SCALE_T L7_ShLU3 = 0;  //1
  static L_SCALE_T L7_ShLB3 = 0;  //1
  static L_SCALE_T L7_ShLW3 = 0;  //1
#else
  static L_SCALE_T L7_ShRU1 = 2048;
  static L_SCALE_T L7_ShRB1 = 16;
  static L_SCALE_T L7_ShRX1 = 16384;
  static L_SCALE_T L7_ShRU2 = 1024;
  static L_SCALE_T L7_ShRB2 = 16;
  static L_SCALE_T L7_ShRX2 = 8192;
  static L_SCALE_T L7_ShRU3 = 512;
  static L_SCALE_T L7_ShRB3 = 16;
  static L_SCALE_T L7_ShRW3 = 32768L;
  static L_SCALE_T L7_ShLU1 = 1;
  static L_SCALE_T L7_ShLB1 = 1;
  static L_SCALE_T L7_ShLX1 = 1;
  static L_SCALE_T L7_ShLU2 = 1;
  static L_SCALE_T L7_ShLB2 = 1;
  static L_SCALE_T L7_ShLX2 = 1;
  static L_SCALE_T L7_ShLU3 = 1;
  static L_SCALE_T L7_ShLB3 = 1;
  static L_SCALE_T L7_ShLW3 = 1;
#endif
