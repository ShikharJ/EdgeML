// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define L8_N 1
#define L8_H 30
#define L8_W 40
#define L8_CIN 32
#define L8_CTEMP 64
#define L8_HF 3
#define L8_WF 3
#define L8_COUT 32
#define L8_HOUT 30
#define L8_WOUT 40
#define L8_HPADL 1
#define L8_HPADR 1
#define L8_WPADL 1
#define L8_WPADR 1
#define L8_HSTRIDE 1
#define L8_WSTRIDE 1

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static const Q15_T L8_F1[1 * 1 * 1 * L8_CIN * L8_CTEMP] = {2671, -413, -938, 7553, -4986, -1065, 4960, 2828, -600, 3767, -13746, -1819, 10440, 6257, -3780, -2453, 14420, -1909, -8449, -60, -26954, 5734, 5024, -7669, -5529, 6952, 1887, -2653, 4063, -318, 1063, -7432, -2245, 1835, 3705, -4322, 2132, 7338, 683, 6688, -661, 10508, -3898, 4398, -17197, -6938, 949, 5672, 1198, 6173, -10713, -4243, 6811, -2513, 3613, -5109, 434, -2021, -1819, -3473, -928, 1960, 4363, 1497, 12699, -6415, -2838, -20851, -4198, -546, 14027, -6404, -9380, -5243, 9256, -8837, -5294, 8968, 1352, -4741, -12870, -4866, -14746, -1336, 1579, 5254, -11832, -2251, -6676, 15334, 7444, -8141, 9752, 116, 4297, -2974, -11764, -1764, 1742, 16196, 1258, 2960, -5869, 2987, -7011, -10491, -3644, -17129, 5139, -13507, -6123, -12358, -9658, 2300, 12594, -6910, 1214, 7760, -7194, -8545, -11875, 18910, -3513, -22523, 8117, -5332, 4535, -7989, -886, -6808, 458, 4854, 12957, -4433, -5525, 15446, -5395, 10621, -9132, 21380, 7905, -9685, -7257, 2459, -1930, 10026, 5261, 6718, 4319, -7818, 12652, 3720, -1615, -5624, 6337, -366, 4562, -1007, -371, -14235, 15441, 9942, 22512, -4447, 11939, -2988, 3542, -5049, 10496, 10746, 10451, 13593, 71, 6857, 12767, -6619, -2520, -6577, 2577, 12794, 3273, 1007, 2392, 2576, -3881, -3600, 9500, -1792, -10165, 7260, 11646, -15559, -5327, -7142, 5383, 6021, 16710, 6411, 8134, -1750, -1051, -5802, 1058, -3455, 102, 8134, 3755, 11298, 5795, -292, 6962, -727, 2843, -7060, -7389, 13636, 4318, 1171, 1495, 1079, 2877, -2865, -657, 24321, -5346, -6318, 8705, -2162, -11580, 12632, -2748, -16524, -2817, -4716, 10742, 1948, 7485, -1380, 2978, 1835, 5393, 5067, 1295, 41, -4011, 18642, 13961, 6629, -6874, -2830, 4973, 7349, 3574, 871, -7412, 3072, 3884, 9653, 6122, -21126, -5692, -7728, 4851, -13471, -12916, 383, 7984, -12327, -6169, -370, 2422, 5416, 9109, 1737, -3799, -4168, -6943, 808, -17556, -13727, 1907, 2323, 1154, -12006, 6039, -1493, -1406, 2181, -3474, -10076, 21012, -28831, -2811, 9135, -1323, 897, 5413, -382, -755, -12548, -9877, -5317, 10513, 5327, 4525, -4688, 2776, 1293, 5643, 13982, 6647, -6021, 750, 17993, -14537, -8689, -2821, 2834, 657, 2753, -4712, 3389, 16263, -535, -7017, -4359, -4283, 8583, -5130, -547, 14566, 88, -12098, -7872, 6770, -4536, 6045, 385, 3629, 871, 4303, -2648, -3815, -3583, -4392, -4974, -3047, 616, 3868, -2319, -5470, 6086, 189, -487, 1149, -1940, -5358, 6533, -4033, -2205, -832, -3197, -536, -2436, -2255, -6882, -9383, 11317, 6078, -363, -2737, -7798, 1543, 2722, 2965, 45, -3447, 12689, 7190, -20602, 7559, -3794, 6693, 7641, -2536, -1695, 19065, 4988, -4176, 6450, -7965, 6406, -8067, 3370, 8735, 3552, 1119, 5739, 1110, -3468, -7104, 622, 11951, -6664, 15070, -11148, -2599, -88, -3326, 3371, -2795, 6561, 3155, 2501, -10409, -9984, 3949, -6758, 9552, -2636, -418, -3265, 3637, -13023, 1193, 2511, -5378, -7965, 9054, -3805, -11087, -6925, 4096, -6874, 9906, -11929, -7969, 4845, 5388, -17005, 3221, 3072, 1758, -19708, -3241, 5588, 1342, -1058, -1947, 2023, 3181, -6675, 1031, 2059, -2512, 15553, 332, 4363, -2117, 189, -946, -4839, -548, -838, 12807, -9852, 677, 2928, 8974, -1613, -9118, 5690, -8184, -3240, 9057, 5397, -4205, 6395, -6159, -3754, -7158, -2488, -17958, 1092, 3140, -7082, -7351, -4066, 14252, 16338, -2743, -765, -3852, -5256, 3127, 2300, -4114, -3022, 8109, 4347, -123, -1687, 347, -1379, -2983, 1856, 4839, -4796, 1547, -7903, 1727, -4197, -6927, 5019, 11295, -4593, -18859, 9454, 275, 18635, 1101, -816, 8917, 4815, 1629, 4671, 503, 2297, 8830, 3717, -328, -4489, 8234, 1670, 2307, -688, -5241, 3033, -2322, -621, 4743, -2515, -18322, -13091, 1134, 3452, 32403, 28805, -52, -7714, 17103, 14986, 11353, -8837, -20095, -2592, 12977, -2810, -4831, 7605, -17682, 8784, 15661, -2983, 9694, 10761, 3148, 2996, 7864, -6733, 10949, -8646, 4021, 13280, -6183, 15142, 8445, 3463, 4053, 4572, 12741, -534, -4201, -6594, -1241, 286, -12370, 9904, 3637, 4307, -2319, 1904, 11141, 3410, 13539, -2005, 6028, -4778, -5137, -6467, -606, -7680, 7195, -960, -3014, -11087, -10776, -18605, 7234, 15077, 19815, 8521, 9344, -9980, -6194, -5948, -1142, -623, -12424, 2606, 7594, 10383, 6766, 166, 6927, -6941, 4309, 10484, -4045, -1145, -4512, -8041, -2849, 8496, -4264, 7716, 267, 9111, 4519, -3982, 6834, 6739, -11318, -1896, 3553, -8782, -3159, 1100, -5044, 8014, -17339, -2738, -8814, 11466, 3748, -908, 14647, 743, -3626, -3893, 3234, 1870, 5805, 3640, 868, -10711, 8253, 16772, 919, 472, -6649, -5222, 4119, 11464, 14453, 1504, 6748, 282, 11777, -3299, -4054, 5171, 906, 7430, 4128, 6805, -596, -2234, -3855, -12127, 8796, -4121, -6480, 41, -4204, -5168, 4229, -6452, -13543, -4131, -3118, -5902, 6834, 140, -160, -6904, -3756, 972, -3269, 4551, -5551, -6471, 2582, -3472, 6250, 6371, 13495, -10941, -14329, 296, 14412, 4229, 7749, -10654, -8856, -11875, 4174, -2261, 6179, -3956, -8488, -4079, 9333, -4114, -10648, 3270, 11426, 12186, -2998, -1296, -10249, -3601, 8836, 9122, 2981, 11501, 539, 6717, 9381, -1707, -4712, 10159, 6064, 282, -8677, -5667, 493, 3386, 1430, 187, 11215, -10966, 1928, -5483, 2144, 10703, -8201, 6770, 5113, 12856, 9267, -1624, 7406, 4712, 12102, 3590, 7258, 3997, 13111, -2070, 24331, -11154, -4218, 14, -1257, 9094, -8089, 9739, 66, -1606, -5917, -3259, 8685, 2289, 10339, -11263, 2421, -3076, 13589, 15565, 11511, -2306, -9797, 3452, 1718, 5553, 207, 6829, 1974, 15204, 2638, -10903, -3290, 46, 5656, -900, 6438, 3562, 1204, 1959, 6534, -12028, 3635, 3388, -5129, 14458, -7716, 20929, -14724, 9078, -7033, 890, -9496, -9635, 11912, 918, -266, 1253, -3233, -1918, -3071, -12990, 7587, -2682, 4748, 12621, 6187, -6062, -3095, 3813, -2269, -8795, 3319, 3636, 11850, 5071, -6423, -262, -7292, 3127, -14507, -3073, -6590, 5557, 9913, -30, -257, 16360, -4888, 5213, 16523, -1314, -6553, -17617, -3139, -679, 9918, 6723, -1075, -2829, 2235, -323, -5238, 5477, 4715, 19503, 6742, 7133, 2343, -22, 7632, 4581, -2241, -3326, -3475, 3104, -4405, 26, 1345, 14456, -3155, 12912, -7294, -7381, 9213, 3870, -5532, 5453, 2144, -14804, 11323, 1242, -6045, -8773, 9760, 1388, 4148, 8694, -7330, -3955, -4712, 4624, 5565, 2838, 4584, -12448, -1511, 249, -7347, 1273, 1733, -1612, -4763, -2530, 1475, 3875, -2240, -5439, 7148, 130, 3888, 645, -1860, 6335, 578, -3191, -6304, 3603, 3000, -7383, 4828, 4765, -546, 11055, -3206, -1165, -1617, 11319, 1168, 5933, 6730, 4561, 8083, 2619, 2257, -5243, 8117, -4732, -2692, 8324, -7373, 1862, 2400, 104, 2013, 5466, -7231, 2020, -3812, -12963, 13547, -10212, 8805, 559, -4532, 392, 1874, -1925, -11046, 4742, -7998, 14317, -2707, 7465, -11811, 6066, 1869, 12017, 6432, 1843, 8756, -5292, 4564, 18386, 1719, -3223, -1813, -13004, 5711, 13135, 6500, 4105, -8171, 1013, 6802, 16615, -7290, 331, 17411, -3946, -2417, 1279, -8034, 3581, -980, -7216, 1085, -11103, -17102, -2764, 4366, 1021, 7346, 3440, -87, -12731, -16694, 628, -461, -1381, -3370, -22010, -12714, 748, 12164, -2909, -1942, -3465, -3563, 10518, 8301, -4440, 5877, 9363, -2607, -1544, -11484, -2353, 2997, -9920, 7562, -1022, -370, 1623, 6225, -3444, -4128, -2036, 3385, -6022, 2891, 2437, -3471, 6185, 4964, 1944, -5620, -2433, -7202, 9621, -10640, 4841, 12113, -7189, -4661, 477, -8664, 4135, -1911, 251, 318, 6608, 641, -9807, -12851, -2933, 5827, 1812, -14742, 4593, -15307, 3990, -307, -8918, 178, -1862, -98, 4694, -9651, -8166, 6508, -2428, -776, -9922, -3623, 134, 9437, -3297, -734, -9282, -5342, -2912, 13656, -18873, 5429, 7096, 2204, 1456, 12322, 8869, 7136, 8419, -2741, 3163, -5390, -4921, 3435, 4272, -16629, -13249, -2892, -14428, -9642, 3823, -2331, 676, 226, -1960, -6834, -3814, -1531, 2939, 13850, -1075, 9687, 2672, 5595, -5572, 7541, 23021, 17067, 1867, 11546, -3600, -4065, 2391, 1308, 8729, -9387, -5198, 240, -3875, -6423, -5082, 10917, -5544, 2434, 4823, 15052, -400, 6444, 1830, 3616, 15551, 7238, 3182, -17428, -14124, -562, 4245, -9527, 135, -11367, -8770, -9716, 9984, 10996, 10568, -6331, -8652, 552, 12169, -9244, 5626, -6010, 9994, 445, 10878, -3623, -13133, 3482, 22627, -10115, 7161, 3217, 4785, 5829, -7038, 271, 9200, 1348, 12662, 5500, -31553, 7915, 10030, -1858, -2486, -1137, -6174, 12742, 6125, 2353, -195, -283, 2223, 1037, 6858, -2613, -19458, 4627, 12592, 7454, 1635, 2687, 12149, -4145, 6071, -8070, 14608, 19292, -28537, 2304, 3590, -6354, 11675, -12954, -1429, -2887, -17476, -6006, -698, -114, 2607, -6360, -8744, -10603, 6371, 5947, 2990, 8239, -1979, 675, 11678, -25993, 688, -1779, 1893, -6528, 7054, -7329, -2264, 3415, -2691, -8213, 2224, -3752, -2059, 4420, -4305, 1955, 10748, 10318, 4741, -2690, -11273, 11133, -203, 4877, 5172, 1032, -940, -8287, -16086, 7441, -8593, -9694, 2263, 2049, -9843, -1738, -19890, -8088, -3448, 10895, 28206, -3593, -8810, -2714, 2741, -14397, 12246, -15982, -16400, 8445, -9829, 5924, 10466, 1355, -2279, 10237, -9395, 2361, 10068, -4924, -10022, 5542, 19618, -8937, -18664, -18593, 2890, 4583, -1137, 4916, -13221, -2621, -6983, 3600, 2750, 10006, 1291, -7938, 4634, 1872, 2708, -1751, 11798, -3900, 3504, 1949, 1717, 5498, 7758, -4316, 4772, -4527, -4277, 397, 4562, 16055, -3863, -5076, 3984, 2022, 14039, -3919, 630, -1543, -24657, 2082, 3743, -7629, -9751, 3250, -8171, 12554, 1652, 7133, 2496, -559, 565, 1181, 4038, -5588, -5322, -2317, 1178, 6049, 6940, -4821, 7851, -1590, 6338, 1185, -4324, 4286, -4316, -1901, 745, 5918, 7971, 8646, -7874, 2691, -1375, -1971, 7162, -13297, -5793, 10360, -6919, -13606, 3855, -1956, 3661, -4852, 4483, -1453, -3111, -565, 16019, -1131, -3634, -499, 10057, 6958, 5501, -678, 12506, 2624, 9067, -10087, 2550, -12165, 10795, 8961, -2356, 7452, 4458, 12352, 3046, -14195, 998, -9409, -24620, -224, 711, -4361, -3231, 2267, 4199, 539, 2249, -10573, -20203, 2229, 11374, 581, 12882, -793, 3181, 1618, -7130, 529, -3593, -4644, 2353, -892, -9916, -1822, -5426, 6270, -4170, -2519, -3045, 5636, 9955, -5450, 413, -5608, 4619, 8991, -1384, 702, 1117, -4397, 2149, 9308, -5700, 3993, 18, 1554, 9647, 3614, -4457, -4597, -3739, -139, 2311, 1921, -9841, 465, -17750, -1092, 2128, -185, -4950, -220, -11398, 8918, 1923, 3291, -2686, 1530, 21766, -1774, 4561, -6924, -4327, -3099, 2069, 1553, 1129, 1570, 24, -679, -3434, 11346, -3300, -873, 10404, 1146, 1350, -5549, -2994, -8434, 975, 9721, -11376, -2129, 2807, -1145, 1299, -415, 4793, -1157, 6545, 3068, -537, 8614, 4330, 5398, -10233, 12895, -8188, 5916, -17555, -5444, 4262, 3298, -2710, 848, -11437, 1863, -2623, 9439, 3512, -2360, 7929, 71, -6612, 5670, -1075, 1648, -1853, -818, 7787, 12460, 15479, 2676, -12554, 44, 3771, 3523, -4509, 1889, 10212, 4717, -1651, 10322, -16428, -3990, -2508, 1202, -2766, -10130, -10617, -13393, 12925, -19223, 1410, 627, 7049, -14432, 2154, -3007, -1119, -7241, 4968, 4314, -3808, -1526, 1180, -1298, -3127, 7242, -2858, -5425, 535, -10389, -9093, -5389, 3792, 14127, -8133, 11337, -1846, 5294, 6415, 2943, 8184, -4401, -5428, 3609, 10922, -3836, 11046, 1420, 6382, -619, -4699, -175, 6345, -484, 11310, -3073, 751, -10517, -11998, 15728, 4932, 3477, 2805, 2227, -7142, 6881, -5795, -667, -2238, -7615, 16082, -5448, -6405, 10015, 7378, 678, -414, 12545, 690, 1395, -5547, 12527, 3019, 10100, 7668, -1565, -1131, 3476, 711, 1231, -495, -4796, -5351, -12408, 2693, 5873, 11285, 14779, -2488, -6698, -6468, 4009, -4995, 6472, -7606, -1828, -6967, 7210, -2524, 2600, 903, -976, -10174, -4050, -1242, 6605, -2215, -1130, -11282, 2529, -154, 8793, -6896, 1641, 5435, 11222, -3908, 9107, -338, -6713, 4179, 3123, -4641, 10003, 18032, -1880, 161, -7210, -187, 9066, -3495, 6684, 1195, -2823, 117, 5951, 6622, -7137, 3622, -5055, 12421, 5528, -1101, 1081, -2788, 5599, 950, 7408, 2794, -13129, -4074, -13619, -10387, 707, 1971, -8402, 12927, -365, -11148, -8141, 14026, -539, -3594, 9743, -15007, 6458, -8739, 772, 2436, 2267, 9130, -1026, -3819, -1323, 378, 15158, 2294, -122, 5111, 3119, -6461, -4388, 924, 9997, 11833, -2093, 1243, -6835, -6934, 4567, -4722, -8920, 2157, -7059, 2816, 2941, 8912, 11313, -7085, 8576, -3280, -1651, -5472, 4264, 4381, 6932, -11779, -3122, 4465, 9371, -3884, -3581, -11735, 1305, -599, 17232, 6418, 2562, -1497, 6241, 7454, 284, 1159, 930, -13403, 3422, -2710, 3616, 2620, -2296, 9881, 4847, 4782, 9944, 6761, 534, 10107, -3548, -17240, 3279, 5653, 7033, 1936, -16790, 7089, 4019, 15522, 6319, -9827, -6809, -2612, -2599, -6145, -6086, -9, -2090, -3236, -1277, 7529, 3430, -10168, 565, 778, -17075, 3705, 3464, 10486, -9024, -6665, -5895, -6376, 11, -3972, -5071, -11335, 1083, -13310, 7601, 6371, 6764, -2924, -4939, -4186, 722, -6422, 1070, -17102, -7655, -5668, -8039, -12320, -6235, -6731, 2411, -7659, -1057, -10760, -2258, 1325, -6384, -1851, 10868, 7528, 622, -16762, -7844, 461, 12316, -5596, 1905, 1475, -7051, -7586, -7579, 6879, 1372, -7061, -12203, 5908, 8588, 7423, -4144, 7791, 2226, 6361, -10095, 4799, 7738, 7974, 13264, -8384, -6523, -9388, 9631, 4695, 6668, 8553, -9728, -13219, -302, 4627, -3180, 3060, -5413, 2787, 6960, -7691, -6262, -8582, -831, -5712, 444, 5805, 17240, 12707, 4326, 606, 7493, 3420, 1632, -1795, -3575, 4766, -9137, -3573, -2, -2609, -83, 8963, -15221, -13390, 8778, -6511, 686, 3548, 2181, 6123, -10479, -15298, -439, 6628, 2304, 5392, -11306, 6682, -7075, 14745, -6116, -610, -6743, -6062, 8443, -5941, -4575, -11166, 739, 6100, -3939, -10557, -2759, 1416, -1451, 1413, -15437, 4209, -5315, -747, 3283, 18934, -38, -6916, 7413, 6230, -8813, 12628, -6196, 3986, -2213, -13966, -3307, 1238, -11779, 8578, 7271, 3311, -1280, -52, -2002, 2892, -3517, 5902, -5914, -5261, 1599, -3173, -6910, -66};
static const Q15_T L8_F2[L8_CTEMP * L8_HF * L8_WF * 1 * 1] = {-1131, -4020, 998, -4456, -8127, 4159, -5295, -5919, 1233, 4092, 6670, 6488, -5331, -3305, 5640, -5444, -1783, 2061, 2714, -2195, 3438, 7286, -1065, -2042, 3933, 11155, 8082, 8700, 12983, 4735, -4023, 4155, -113, 894, -4119, 1056, -2, 10545, -5918, 1533, 9032, -1300, -1051, 935, -13372, 170, -3852, 10446, 542, -10127, 4013, -3047, -11590, 2735, -2345, -8544, -1501, 2792, 8145, 898, 7697, -2074, 3588, -2361, -2631, -1180, -4600, 3579, 8322, -5904, 8910, 5689, -6965, -4789, -6212, -3208, 6375, -5755, -3786, -5109, -1325, -4122, -4029, -2441, -3894, 4558, -3503, 10327, -8967, -1019, 1244, -1965, 381, -116, 17174, 1924, -627, 2847, 163, -2473, -8231, -4010, 8881, 6481, 3736, -2372, -4088, -507, -291, -466, -375, -3325, -3763, -1616, 1254, 17747, -1810, 4931, -7821, 5041, 1244, 927, -798, 7500, 1567, 6559, 210, -482, 565, -1919, 11979, -611, 1031, -7805, -855, -2673, -1523, 4264, -10225, -6666, 4156, -2925, -526, 2743, 10035, -3166, 5289, 1320, 3907, -4757, 8366, -4932, 3540, 1413, -1031, -1157, 3065, -896, -1901, -4960, -6658, 8525, 11633, 8816, 6546, -1354, 1057, -30, -2703, -2893, 3269, -1434, 3172, -3120, -5121, -8404, -2506, 9773, 8176, -117, 7262, 5496, 4291, 3211, -9913, 4758, 726, 4315, -1036, 302, 2015, 2494, 427, 5585, -12951, 335, 3436, 2127, 6526, -4210, -533, 9916, -7382, -10087, 6644, 5875, 378, 646, -3469, -173, -4113, -4409, -4179, -4299, -4369, -1193, -6060, -9143, 5804, 3073, 5507, 3259, -1224, -8008, -1150, -9487, -5414, 14726, -5698, -3352, 9656, -5524, -3892, 10535, -946, 388, 1588, 3381, -7874, -1576, 511, 12452, 3501, 3796, -4435, -4287, -671, -4082, -5424, 2936, 7739, -1729, -4309, -8442, -7476, -4842, 1097, 4304, 2064, 3907, 4866, -4999, -5040, -2128, 5255, -5646, 3394, -4417, -7115, -7913, 4289, -3673, 2691, 185, 1149, 1199, 6943, 6717, 1708, -3087, -225, 324, 19700, 1436, 744, 6284, 397, 1442, -2495, 740, 4270, -3097, 459, 9525, -3763, 9655, 643, -1872, 10669, -676, -3817, -11743, 2510, 4159, 3474, 3402, -1782, -2014, -1169, -1444, 21154, -395, -1574, 582, -854, 1238, 1122, 1885, 1614, 20344, 660, 1578, 944, 144, -788, -790, -118, 9719, -4874, -299, 819, -819, 137, 11434, 4995, 11, -6702, 5976, 4033, 2701, -230, 2537, 1219, -251, 2891, 2535, -10475, 7969, 2682, -1107, 4212, -4551, -6777, 7626, -1333, -8366, 1977, 5919, -78, 9742, -6720, -5564, -234, 767, -3479, 2230, 6094, 5609, 5127, 5064, 1175, 1250, -5617, 14763, -1117, 1012, 1576, 830, 5580, 3071, -2028, 5256, 2150, 2263, 3835, 6340, -3200, 5060, 7479, 3619, 6684, 5139, 5658, 4027, 3035, 1119, 2655, 7288, 3940, 7281, 4868, 3268, 1888, -5630, -4331, -5206, -3925, 3273, -6823, -6896, 92, -5073, -2562, -4354, -6516, -7873, 1990, 12124, -6533, -1199, -2009, -1663, -2064, 8337, 752, 3499, 8318, -7835, -827, 5659, 3135, 3151, -5545, 4161, -10220, -7751, -5013, -1282, -5937, 2545, -2069, 1476, 339, 203, -2261, -1710, 522, -3395, 14036, -3683, -1758, -2916, -2621, 1191, 13060, 680, 1991, 3421, -66, 128, 1113, 159, -4290, -4847, 128, 13313, -4967, -402, -3837, 4420, -5474, -3025, -2205, -6509, -4748, -3230, -5739, 7859, 8105, 7404, 6499, 5177, -793, 6348, -5631, 290, -3388, -5489, -322, -4476, -2760, -373, 7524, 10246, -3081, 1879, 3968, -1298, 12821, 2713, -1897, -1265, -4341, 1603, 8515, -787, -4143, -598, -5235, -168, -3613, -5375, -4741, -1029, 410, -57, -1636, 17436, -95, -1752, -451, 724, 1981, 2586, -2845, 9606, 9256, -6465, 5110, 607, -9729, 11604, 5217, 1173, 11981, 12556, -4638, -1417, 804, 164, -616, 9555, -1259, 1056, -1083, -2832, 684, 4250, 6852, -1334, -5809, 3827, -4147, -7231, 4415, -5197, 8091, 4009, 2270, 2836, -356, 14834, 6502, -5, -3276, -1698, -278, 980, -4262, 4469, 4024, -9701, -6620, 7231, 1753, -1319};
static const Q15_T L8_F3[1 * 1 * 1 * L8_CTEMP * L8_COUT] = {-3452, -4961, 6998, 875, 1157, -446, 219, -578, -4697, -1285, -4672, -5024, 1066, 288, -5758, -2426, -2539, 1986, 5608, 152, 1710, -5746, 3352, -2747, 4210, 1329, -2188, -4036, 3330, -3593, -1772, -1728, 316, 426, 535, -1275, 6256, -1771, 668, -99, 1326, 4461, 3427, -640, -1787, -2181, 137, 772, 392, -2399, -3587, -2282, 5556, -4206, -1844, -1133, -546, -665, -354, -2592, -4131, 2128, -2690, 5698, 3300, 457, 1228, -2475, 1911, -1303, 1297, 5172, -3648, 3230, 987, -3573, 3193, 4869, -4460, -2246, 4184, -6267, -537, -536, 1211, -11, -2560, 849, 1698, 2634, 9067, 457, -3117, -511, -176, 5197, 3227, -4549, -3069, -3909, 3408, 7662, 497, 6159, -7735, 7948, 491, -3354, 2649, 6009, 1906, 2676, -1932, 3965, -3152, 2511, -5867, -11532, 8259, 1852, -4046, 2796, 2001, 1653, 7304, -2219, 1350, -4142, 1299, -4096, 4802, 1013, 4046, 3201, 145, 19, 1183, 776, 583, 1828, -2299, -2393, -210, -2018, 3415, 5160, 2048, 5826, 43, 3438, -2589, 984, 2540, 3902, 1526, -529, 3953, 1539, -1242, -166, 1507, 1247, -2338, 688, -50, -516, -350, -71, -2164, -1679, -2255, -1189, 3463, 1253, 2718, -3386, -311, -1429, 1773, 3255, 3155, 683, -1997, 3700, -360, 872, 137, 5561, -100, 4084, 4191, -1973, -2811, -644, -797, 3899, 4751, 2169, -1330, 2856, -1035, -934, 1588, 1408, 4927, 2344, 2054, -1536, 1308, -2216, 1301, -8050, 4414, 4294, -1036, 5872, 4672, 6046, 2465, 1874, -496, 4319, 5650, -2960, 1045, 1261, 1224, 6038, -2725, -2198, 2151, 397, -921, 2842, 1172, 2623, 5, -795, -356, -99, -1260, 6821, 1407, -9359, -6465, -3708, -4979, 215, 257, -2632, 2487, 3788, 2059, -3045, 2012, -2009, -7545, 1381, 1465, 4699, 3794, 385, 4026, 3374, -3149, 2351, 1532, -6671, 3362, 2764, -3543, 925, -5847, -722, 419, 1349, 969, 1266, -733, 1046, -776, 181, 5008, 3914, -2244, 3274, 5648, -626, 70, 6078, -1136, -2905, 2671, 587, 1911, 3915, -5913, -1706, -1082, -1376, 2252, 1111, -1368, -2258, 2518, -3124, 504, -1012, 2498, 1822, -2438, 3443, 4136, 2140, -452, 234, -157, -64, -2847, 1049, 4535, -3322, -1223, -2970, -7542, -5716, -4670, 5494, -3519, 3136, 2822, 3109, -1850, -3177, 718, 10292, -484, 740, -7322, 4423, -3314, -966, 3783, -1221, -3785, 5543, -4578, -554, 6220, 2299, 1512, 642, -3385, -7049, 1120, -3197, 2520, -4464, 501, -726, -3434, 754, -2744, 1646, -1916, -2683, -1102, 3367, -6982, 545, 359, 1782, 2172, 1556, -1573, 1730, 3741, -1663, -2393, -6807, -3263, 3650, -4389, 3029, 1665, -6820, 878, -3773, -6715, 2819, 5307, -1965, -7725, -6962, -214, 3067, 5264, -5707, 4105, -6799, 1031, -129, -561, -1625, 10586, 2773, -1943, 3456, 4452, -4809, 3085, -5756, 5369, 426, 1870, -1094, -4363, 7956, 1259, 1704, 646, 2956, 2517, -1690, -1027, 3034, 2871, -1122, 1253, 1465, -688, -506, -1426, -500, -2716, -4212, 3706, -687, -1423, -1228, -2082, -1088, 1471, 386, 1416, 1768, 2804, -103, -139, -560, -751, 2909, -2014, 4035, 278, 527, 162, -1854, 4614, 3494, -4496, 102, 278, -4862, 5172, -1405, 2442, 1699, -3670, -1693, -1904, 1416, -2196, -4297, 5201, -2393, 6076, -1607, 816, 1978, 3568, 29, -6275, 6483, 567, 2522, -3230, -2135, 3172, -4091, -3434, -1361, -2757, -2081, 3497, -2860, -2486, -2647, 2099, -4095, -1323, -3241, -1040, -1148, 1267, -3051, -1124, 1402, 4294, 391, 1540, -3148, 3499, -2065, 1161, 2842, -373, -5395, 2423, 825, 4514, 1851, 2308, -367, 1197, -2571, 124, -73, 221, 3383, -3814, -6206, 405, -2153, 1175, 2015, 1565, -7169, -4681, -3015, 4186, 1353, -306, 205, 2788, 4685, 369, 1418, 986, -830, 2103, 2776, 1173, 1677, -3827, 480, -635, -216, 1596, 1141, -149, 1247, -1803, -1509, -1955, 892, -1461, -2245, 702, 202, -353, 525, -4098, 1030, 2281, -4474, 4908, -3566, -2675, 990, -3475, -855, -7707, 2620, -1381, 1683, 3256, 2002, -3763, -805, 716, 3602, 1250, -807, -6068, -1463, -999, 2442, 6213, -5721, -1955, 5469, -3872, -4765, -1499, -3386, -7987, 4896, -3636, 5634, 1328, 3755, 2884, -1321, -2880, 1380, 742, 5525, 783, -2008, -2492, 2142, -2043, 1759, 2297, 1759, -2517, 892, -4655, -381, -1369, -3230, -1910, -35, 18, 1629, -2939, -2121, 363, 1007, -3614, 168, 1421, 1138, 4773, -1485, 1088, 1247, -905, 3476, 301, 1754, 2259, -2039, -529, 849, -3473, -1578, 400, -4051, 737, -919, -2813, 329, 908, -1325, 1550, 1476, -542, 2895, 4067, -4494, 2287, -5113, -5485, 5545, 5289, 1913, 6526, 416, 9121, 512, 2954, 6583, 3392, 6619, 69, -2752, -6063, -6272, 902, 4019, 1598, 2056, 6481, -6843, -767, -1045, 5900, 2656, -3093, -2131, -6487, 2299, 633, -664, 9208, -6228, -5043, 1824, 274, -1047, -2333, 5695, -1398, 777, -6063, 1094, -2514, -1268, -4362, -303, -903, 494, -942, -785, 513, 1616, -3511, -4662, -2754, -2360, -1020, 1539, -9721, -573, 33, 7951, -921, -522, 3759, 1813, 1357, 7964, 6603, -995, 130, -8568, 5714, 7457, -4633, -6149, 5173, -2975, 3468, 2666, 2372, -1036, -4096, 1009, -2812, -4060, 6044, -768, -4016, 3780, 995, -2243, 3065, 4460, 2243, -893, -1526, 551, -2436, 2669, 2922, -978, -1759, 764, -4128, -2609, 1063, 3592, 374, 1259, 2897, -6369, 2527, 806, -4517, -5224, 2576, 3125, -73, 1439, -536, 4005, 1469, -239, -3117, 5342, -585, 4100, 1925, -655, 633, 601, -939, 2745, -1979, -1051, 1106, 2431, -2285, 170, -1997, -1833, 1942, 2696, -864, -2346, -1625, 3264, 3735, 525, -289, 1262, 3092, 2174, 473, -6911, 694, 181, 132, 1126, 103, -2181, -7370, -638, -1070, -680, 2314, 2898, 7694, 6342, -1753, 94, -7679, -4571, 1389, 4638, 618, 3223, 2871, 7210, 1271, 5250, 2808, -3282, 226, -421, -1540, -2834, 422, 1793, 590, 643, -2542, 1455, -1020, -751, -2856, -1213, 1858, -4297, 1093, 2363, 3494, -5349, 1351, -3808, -3577, -758, 1890, 472, 2905, -1172, -1074, -3107, -551, -2380, -1570, -1283, -761, 4111, 2812, 1098, 2365, 1218, -2436, -3250, 31, 418, 3318, 52, 1184, 3397, -2294, -4610, -706, -101, -3155, 2392, 689, -5678, 2741, -1836, 1000, 754, -2123, -2082, 7853, -1184, -687, -2050, -360, -2355, 412, 899, -5446, 841, 5498, -242, 2034, 3078, -429, 3397, 857, -2702, -5164, 3507, -1634, 4209, 1437, 1708, -732, -74, -6769, -374, 196, 123, 354, -662, -1888, -1489, -4151, -49, 2636, -3041, -3223, 4246, -1155, 1389, -1079, 4208, -1361, 552, 3807, 4459, -2863, -1584, -7777, -2296, -9944, 2351, -2631, 4505, -2907, -3928, 7693, -5659, -1457, -5683, -3121, -6163, -685, -2375, 1031, -690, 1472, 2615, 815, -3149, 10777, 4916, 15480, 4708, 6229, -7242, -4168, -794, -568, -3291, -1936, -371, 2667, -1725, -1410, 9571, -5103, 8940, -2729, 343, -1576, 4042, -360, 3532, 5267, -5398, 805, -2868, 1370, -5297, 384, -5260, 555, 1050, 757, 732, -1492, -5646, -617, -5186, -3430, 1295, 2120, 181, -4402, -1473, -3090, -678, 3206, -5720, 265, -5761, 3304, -1098, 2851, -2076, 3069, -2424, 1201, -4182, 3442, 889, 2390, -7352, 2676, -3683, -1280, 2243, -1746, 2529, 868, 2730, 3562, -2996, -2832, 2758, 2644, -2472, 3751, 3839, 1849, 2405, -2349, -648, 4326, 518, -5029, -3168, 512, -2303, -3913, -2782, 183, 4029, 497, -3720, -6477, -6518, -10331, -595, 1836, 9422, -2964, -2518, -5094, 3388, 3372, 3207, -2439, 5750, -153, -8683, 3188, -3869, -3933, -5956, 1178, 3415, 399, -6819, -3971, 1666, -1458, -2115, 578, 1664, 2194, -5741, 7601, 286, 16816, 3657, 7584, -5212, -4787, -2346, -1723, -2678, -3382, -5750, -7130, -6272, 490, 13665, -6166, 6267, -1007, 6266, -597, -2714, -1887, 7912, 4123, -1401, -236, 2109, -2439, -7019, 55, -350, 1774, -1791, -4867, 2926, 2180, 4989, -185, 121, 1415, 3005, -377, 1977, -4394, -4783, -938, 442, -5661, 1835, -6333, 2403, -4786, 4898, -5493, -945, 1142, 1768, -216, 2904, -2983, 1417, 3773, 3363, -7991, -3186, -5637, -6734, 3426, 4275, -3673, -7987, -2226, -2213, -1420, -144, 1744, 3142, 1483, -7916, -876, -4603, -3974, -1682, -54, 2232, 909, -4421, -1810, -1402, -3232, 455, -4708, 359, 2103, -6058, -4671, -1083, 3036, -219, 1655, -2972, 7847, -4786, 2044, 4947, -731, 4034, 568, 2737, -2656, 2197, -4019, -1926, -409, 2577, 2614, -844, -1143, 3525, 2379, 442, -2838, 5225, -6573, 1134, 8789, 5123, 1469, -6922, -654, -1890, -1950, -4012, 4638, -165, 250, 3467, -1426, -3701, -756, 453, 1264, 137, 3686, 4223, 3016, 3574, 3353, -3720, -2867, -1780, -819, 6261, 887, -4239, -4200, 1321, 371, 1592, 1408, -1894, 803, 1375, -1378, 1651, 1388, 2624, -813, -2002, 205, -1395, 1586, -2991, 1949, -832, -1553, -4705, -6821, 1736, 1969, -24, 3135, -3900, 1800, 5815, 1094, 96, 4810, 244, -2825, 10226, -677, 6719, -2029, 2793, 5295, 2589, 2072, 3184, 2821, 2891, -2322, -10, -430, 50, 1543, -5974, -4049, 3342, 119, 3597, -2623, -519, -4821, -5861, 3140, 2160, 3193, 2610, 996, -3395, -2784, 3056, -2918, -918, -5414, -326, -3509, 3552, 1588, -1127, -1790, 2566, -498, 4098, -1163, 3085, 1645, -4208, -6294, -8444, 5531, -4192, 2102, -2886, -811, 828, -630, -2, -6224, -1665, -27, 3533, -4763, 2902, -8941, -3304, -5929, -3399, -5233, 3662, 4291, -3780, 5975, -1437, 5602, 6341, 160, -3480, -977, -8878, -3374, -15046, 6217, -4275, -2808, -3702, -1813, -4595, -5132, -1359, -5645, 2397, -269, 11424, -3394, -2751, -2310, 228, -3233, 3844, 796, 444, -579, -3342, 727, -3189, -2272, -1428, -1349, -1701, -7038, -2099, -1164, 3562, 1119, -6133, 32, 169, -1768, 3363, -3192, -5466, -208, 389, 453, -4776, -2933, 6362, -6717, 4029, 6021, -1221, -1260, -3627, 4250, -6850, -2088, -4178, -1636, -1993, 5688, -235, 1260, -1613, -7107, 2942, 4950, 2381, -323, -5075, 3241, -4158, -2567, 1240, -682, 1454, 1195, 1838, 1052, 5625, 3333, -744, -1221, 84, -179, 892, -4598, -348, -2989, 7722, 3646, -3946, -155, -714, -3550, -2208, -258, -2533, 275, 2740, -6096, 1826, 446, 4449, -1511, 266, 1507, 6322, -5510, 968, 3097, -9165, 3593, 342, 1159, 1311, -1818, -2220, 2397, -358, 5626, -4267, -2459, 2837, 1853, 3974, -1641, -774, -1633, -5319, 2184, -2891, -3081, -1084, 509, 4712, -4092, -4816, -541, 278, -3855, 5987, -477, -1420, -884, -1356, 83, -5148, -3310, 2211, -1343, -1397, 2687, 2112, -2651, -174, -2954, -991, 2034, -1256, -3183, 5708, 2688, 1872, -147, 1420, -902, -1331, -5037, -3719, 1724, -144, -2085, 217, -5981, -8763, -148, 8202, 2994, -5738, -2975, -4614, -6557, -5089, 2050, -690, 2310, 406, -6744, 1836, 349, -1966, 2560, -2441, -1006, -8220, 2068, -906, 2168, -555, 4555, 3165, 6106, -3556, -3035, 4938, -2329, 7654, -2185, 469, -782, -689, -6670, -4957, -3713, 1776, -2767, 4509, 1524, -5057, 5265, -602, 3992, 2907, 4970, 697, 225, 2930, 2031, -1371, -1208, -1958, 1734, -927, -2526, -5805, -2148, -3069, -728, 3002, -2809, -7414, -3082, -1502, 3663, -1198, -2191, -5646, -2620, 1333, 2152, -2219, 1942, -3540, 1074, 4322, -745, -375, 6192, 4154, 2255, 2952, 6169, -1111, -2372, -8188, 696, 2557, -3476, -4566, -8678, 5515, -1402, 2719, 224, 5216, 5818, 2404, 51, -1017, 1291, -5170, 4798, 4549, -5719, -2622, -3405, -1611, -3203, -10884, 1606, -1963, -1270, 867, -2556, -5984, 663, -513, -6863, 3733, 1544, -1142, -837, 20, 2380, 3835, 2204, 2679, 2770, 1510, -3112, 2147, 583, -4103, 6145, 2230, 1111, -987, -2930, -6823, 3683, 4461, -2224, -1088, -4530, 3015, -149, -310, 1564, 5912, -3165, 6749, 6482, -1895, 2791, 382, -3432, -3433, -2170, -3077, -1629, -79, 1446, -129, 1087, -418, 6751, -971, 446, -1340, -2562, -411, -2814, 2018, 1687, 3342, 1718, 4747, -794, 306, 2582, -3288, 1594, -1633, -642, -5756, -3537, -5023, -1961, -3003, 3308, -6357, 2439, 4183, -971, 922, -3679, -3426, 458, 274, -3182, 812, 4547, 1503, 5246, 12783, -1756, 2329, 1933, -5147, -93, 2467, -4883, 93, 2058, 2607, -831, -5058, 1074, -840, -3964, -73, -1982, 1199, 6534, 3822, -109, 3280, 4058, -1383, -734, -982, 3345, -3806, -1525, -3391, 135, -888, -1539, 878, -2905, 1906, -594, -3091, -1797, -3074, 5742, 4222, -4753, -28, 2746, -11979, 8068, 2111, -7007, 315, -3682, -3628, 5180, 462, 1228, -1767, -4946, -418, -1502, 1869, -5893, 836, 4992, -5530, -8429, -4716, 4455, 2853, 2384, 2171, -1372, -2103, 4273, -2920, -2370, -3145, -338, 4150, 6237, -508, -915, -697, 1487, -2104, 2478, 2536, 1363, -1977, -5674, -3214, -4417, 3117, 136, -2765, -3691, -4408, 689, -1228, 6453, 230, 313, -2084, -1713, 6252, 3284, 3108, 2896, -1294, 8041, -6442, 7098, -1962, 8980, 1171, 1231, -3220, -20, 976, 574, -1200, -5081, -5140, 1122, -4931, 282, 5316, 3035, 13678, -4075, 9379, -1498, 156, 3135, 2833, 3303, -193, 4564, -255, -1679, -4060, -1011, 2519, -3183, 790, -1249, 851, 1418, 935, -3628, -1484, -471, -2841, 3174, 1693, -1514, -1757, 1245, 411, -2876, 3208, 142, 231, -3104, 2750, -766, 1137, -455, 1182, 30, 3954, -5254, 4271, -241, -5311, 2778, 4460, -6261, -1899, -1344, 720, -6704, 4629, 10, 1393, -937, -1851, 3126, -1186, -2958, 136, 1303, 1541, 1026, -796, 1607, -261, 327, -3937, -76, -1355, 731, -2159, -2729, 919, 1314, -1200, -4030, 2582, 662, -3433, -212, -175, -168, 639, -1371, -4782, 1772, 3707, 2578, 638, 3730, 3927, 3163, -850, 4359, 597, 5035, -3924, 4276, 4224, -592, -1504, -809, -6585, 4007, 1332, -378, -4053, -2512, -1477, -7879, -3410, 1657, 3697, 4398, -4105, 1269, -2761, 317, 708, -546, -156, 2731, 974, 2153, 952, -4319, 9713, -9239, 6664, 2085, 1154, -9871, -2689, 5443, -3876, -4040, -135};
static const Q15_T L8_W1[L8_CTEMP] = {15873, 19704, 15782, 12724, 6171, 18452, 11251, 12585, 17217, 18625, 11020, 14177, 15236, 19272, 18843, 13790, 14238, 23702, 14232, 23158, 14578, 14449, 13542, 12196, 22739, 16114, 11999, 21321, 16743, 17542, 11454, 5190, 18390, 14875, 4702, 3572, 24602, 10899, 13208, 14863, 20053, 13470, 10926, 6219, 15175, 10463, 21020, 10614, 16153, 15745, 12572, 14817, 15995, 16430, 22210, 9488, 15387, 8850, 10840, 4836, 19536, 19650, 16710, 19483};
static const Q15_T L8_W2[L8_CTEMP] = {6707, 6267, 3362, 5286, 26754, 5369, 8471, 13962, 4621, 7829, 4527, 6770, 8958, 6228, 7605, 9060, 5526, 10880, 5231, 6709, 6664, 7395, 6589, 9632, 5971, 6371, 9469, 9156, 7207, 4737, 7057, 7111, 6259, 7165, 7597, 6519, 10761, 7588, 8970, 4134, 5922, 5198, 10032, 8735, 5364, 5925, 7743, 4657, 4480, 8795, 8025, 12379, 4683, 2131, 7635, 11303, 6563, 6765, 9233, 7694, 6587, 5559, 5297, 4858};
static const Q15_T L8_W3[L8_COUT] = {9095, 8927, 14459, 11239, 10263, 21452, 17462, 16847, 15514, 14581, 16125, 16281, 14370, 15869, 15743, 17812, 15428, 13128, 15146, 11300, 12039, 12693, 14006, 15803, 13568, 13626, 14464, 13766, 13758, 17587, 9183, 12878};
static const Q15_T L8_B1[L8_CTEMP] = {805, -914, -835, -583, -2236, 587, 736, -2448, 762, -30, 645, -580, 175, -728, 301, 549, 818, -294, -238, 285, -1614, 920, -734, -1792, 888, 1147, 403, -871, 1219, 787, 1069, -573, -638, 156, 2058, 2477, 115, -211, -406, 668, -328, 444, -229, -2546, -409, 157, 1077, 871, 492, -475, -1064, 576, 663, 212, 404, -1046, 238, 549, -2117, -1795, -168, 492, -404, 27};
static const Q15_T L8_B2[L8_CTEMP] = {964, -311, -1368, -783, -67, 458, -311, -465, 1673, 370, -683, -21, -305, -471, 19, 597, -783, 14, -904, 111, -807, -24, -200, 892, 337, -157, -348, 209, 381, 1457, -858, -1050, -615, -89, -257, -1503, -39, -711, -450, -55, -156, -624, -718, -1141, -897, 1382, 469, -911, 1336, -224, -490, 97, 1854, -1636, 156, -413, 919, -309, -330, -840, -654, -28, -628, 148};
static const Q15_T L8_B3[L8_COUT] = {333, 279, 313, -618, -2106, -237, -1377, -1542, 612, 495, -268, 1758, -88, -815, 1882, 1335, 782, 153, 1537, 456, -495, 55, 258, -13, 64, 802, -549, 1498, 111, -1210, 1187, 1069};

static const Q31_T L8_Limit1 = 402653184L;
static const Q31_T L8_Limit2 = 100663296L;

#ifdef SHIFT
  static const SCALE_T L8_ShRU1 = 10 + 5; //1024
  static const SCALE_T L8_ShRX1 = 14;     //16384
  static const SCALE_T L8_ShRU2 = 10 + 4; //1024
  static const SCALE_T L8_ShRX2 = 12;     //4096
  static const SCALE_T L8_ShRU3 = 8 + 6;  //256
  static const SCALE_T L8_ShRW3 = 14;     //16384
  static const SCALE_T L8_ShLU1 = 0;      //1
  static const SCALE_T L8_ShLX1 = 0;      //1
  static const SCALE_T L8_ShLU2 = 0;      //1
  static const SCALE_T L8_ShLX2 = 0;      //1
  static const SCALE_T L8_ShLU3 = 0;      //1
  static const SCALE_T L8_ShLW3 = 0;      //1
  static const SCALE_T L8_Scten1 = 0;     //1
  static const SCALE_T L8_Scten2 = 1;     //2
  static const SCALE_T L8_Scret = 0;      //1
#else
  static const SCALE_T L8_ShRU1 = 1024 * 32;
  static const SCALE_T L8_ShRX1 = 16384;
  static const SCALE_T L8_ShRU2 = 1024 * 16;
  static const SCALE_T L8_ShRX2 = 4096;
  static const SCALE_T L8_ShRU3 = 256 * 64;
  static const SCALE_T L8_ShRW3 = 16384;
  static const SCALE_T L8_ShLU1 = 1;
  static const SCALE_T L8_ShLX1 = 1;
  static const SCALE_T L8_ShLU2 = 1;
  static const SCALE_T L8_ShLX2 = 1;
  static const SCALE_T L8_ShLU3 = 1;
  static const SCALE_T L8_ShLW3 = 1;
  static const SCALE_T L8_Scten1 = 1;
  static const SCALE_T L8_Scten2 = 2;
  static const SCALE_T L8_Scret = 1;
#endif
