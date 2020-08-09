// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define L5_N 1
#define L5_H 30
#define L5_W 40
#define L5_CIN 32
#define L5_CTEMP 64
#define L5_HF 3
#define L5_WF 3
#define L5_COUT 32
#define L5_HOUT 30
#define L5_WOUT 40
#define L5_HPADL 1
#define L5_HPADR 1
#define L5_WPADL 1
#define L5_WPADR 1
#define L5_HSTRIDE 1
#define L5_WSTRIDE 1

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static Q15_T L5_F1[1 * 1 * 1 * L5_CIN * L5_CTEMP] = {4818, -1577, 4358, 6204, -199, -7369, 12543, 8249, 8318, 1715, 119, -2996, -9031, -11180, 696, 7693, -5454, -1227, 7010, 5542, 6529, -8244, 10417, 1004, -9909, 2643, -3020, -5998, 11522, 4094, 5592, 9554, 1915, 3950, 5806, -21206, -3637, -4947, 1816, -5178, -6962, -11341, 14257, 1903, -234, 4971, 9781, 20100, -8090, 8934, 8951, -11661, -5480, -3518, -11104, 8718, 3769, 5901, 3008, 17527, 3034, -7006, 2856, 13462, 6936, 2657, -25940, -2535, 7566, 16792, 6162, 12816, -1792, 11754, -16513, 6213, 13292, -7020, 3771, 5358, -8400, -7224, -18871, 7774, 2998, 150, -755, 3706, 14762, 1749, 6417, 1406, 2536, 2754, -2339, -3775, 6258, -7279, 10075, 8327, -7619, 14594, 9299, 8653, -11442, -5178, -2026, -3342, -6726, -5332, 351, 1841, -1524, -934, 10374, 2003, 2997, -6183, -9480, -14618, -19474, -3993, -2658, 574, 9544, 6235, 5656, 1290, -7739, 6584, 6251, 11315, 2973, 10519, 92, 3156, 974, 1087, 7110, -8040, 1869, 452, -11063, -7392, 1260, -5501, 10962, -6507, 5626, 6316, 3527, -4273, 6880, 10374, 9150, -10797, -902, 5528, -8446, -633, 5340, -6312, -12725, -13504, -2360, 2679, -6320, 18403, -7042, -4222, -11599, 18817, 10339, 4475, -6227, 1016, -10303, -5205, 4365, 1177, 16235, 15889, 4355, 16719, -9039, -17081, 7299, -8518, -2106, -2586, -8740, -13577, 1745, -12707, 4309, 8851, -6970, -2632, 9455, -10709, -16664, 5857, 3666, -8440, 5357, -5298, 1416, -3310, -8196, -3761, 3022, 1897, -3019, 1465, 5992, 1963, 617, -5752, 5567, -11017, -14073, 1012, -2883, 14762, 9995, 5398, 9433, 19040, -9034, -2994, -16576, -1769, 9507, 11485, -11965, 4421, -562, -5262, -8961, -3453, 128, -1706, 4014, 11317, 2854, -974, -493, -15779, -4717, 3662, 6138, -19606, -14054, 4398, 12662, 7505, 9595, 3333, 7964, 6061, -1449, 5227, 3447, 5984, 3022, 5970, 2231, 28806, 18585, -380, 4959, -7320, 17136, 5079, 5984, 15177, 8529, 15784, -7476, 15437, 2585, -13546, -10868, -11756, -1795, 4331, -2863, 10138, -2508, 11827, -8064, 1433, 3659, -1491, 8895, 7151, -1145, -7581, -3362, -1824, 9543, 16628, 9593, 5334, -3318, 10454, 395, 3269, 7226, -10608, -4098, -6217, -6730, 3124, -7628, 16656, -1808, 14167, 5656, 9131, 9686, 13047, 4978, -4643, 2302, 14726, 1220, 256, -7869, 4659, -1085, -15213, -9859, -6975, 2937, 16173, 1179, -3594, -8641, -4434, 4829, 6214, 16307, 7029, -4534, 4853, -8558, 4855, 10234, 207, 9516, 4565, -1725, -1582, -1818, 8540, 6737, 5759, -6185, -3312, -2401, -1412, -10554, -12122, 871, 97, 3200, -6367, -12480, 6711, 6603, -4107, 6235, 1901, -9187, 1134, -4097, -2881, 135, -2490, -13848, 2446, -1174, 5324, 9647, 4667, 1145, -15234, -12077, -1782, 6164, -5605, 5079, -13599, 11684, -8000, -924, -1627, 16286, 7345, -3360, -4288, -9845, -3784, -11779, -7408, 1322, 12505, 5635, 9620, 335, 2709, -7327, 8135, -5925, -10752, 1430, 10975, -4195, -8583, -8679, -4363, -13991, 12461, -115, -631, -11654, 3080, -1842, 6413, -7616, 8995, 19858, 4297, -8036, 6860, 2925, 5647, 3514, -5720, -1942, 30, -9463, 12474, -9743, 5975, 15346, -1364, -2571, -17619, -393, 5406, 1600, 6429, 1173, -3247, -13983, 7922, 4048, 6069, 11429, -1516, 9283, -1540, -9922, 3691, 1399, -6345, -19729, -3612, -8856, -3412, -7963, -2049, -2096, 926, -11415, -1643, -5021, -2870, -184, 7354, -2562, -9174, -721, -5987, -9907, 3361, 1135, -3110, -5716, -13490, 1106, -4961, 9323, 10317, -15718, -6775, -12679, -11392, -3300, 7198, -7514, 3056, -2176, 1051, -2567, 1073, 2679, 1775, -13974, -4113, 4341, -8444, 2092, -9027, 10011, 8858, 8658, 6009, 6913, 6779, 7631, 5592, -15196, 12825, 2242, 7059, 5061, 2497, 7470, 6846, 23978, 5468, -6900, 2173, 12161, -16848, 1930, 3592, 10696, -8154, 22682, 9929, 5555, -11302, -11332, -533, -2349, -15403, 1514, -10852, 8736, -1066, 7197, 6397, -1053, 3091, 9503, -2939, 8489, -3782, 5134, 8194, -12484, -3980, 886, 514, -1572, -2873, 3425, -10216, -10746, -8234, -5114, 8161, 1355, -397, 4139, -2977, -9578, 7127, -3068, 9478, -1732, 8258, 15342, -20402, 3693, -6535, -11030, 7933, 818, 15087, 8771, 2045, 9974, 4255, -7884, 24882, 4533, -3703, 20217, -7196, -7694, -973, 18353, 10629, 1015, 6978, -990, 11117, -2234, -9694, 5643, 17620, 2139, 1246, -8495, -11844, 13072, -9826, 7628, -5078, -5672, 6811, 9731, 20145, 3316, 6267, 8531, 3222, 11528, -9907, 16411, -8172, -1303, 2071, 327, -8939, 2182, -747, -6255, 2173, 7955, 5987, 948, 18921, 7211, 1741, -3298, 13620, -9209, 3695, -8585, 2419, -13487, -714, -10305, -15022, 7787, -956, -7538, -2700, -356, -5628, -219, 3048, -3633, -13027, -13308, -16002, -12821, 163, -1437, 12281, -9500, -4571, 4761, -5567, -7777, 1198, -9302, 7995, 1032, 18613, -16017, 10005, -1031, 3718, 6856, 9028, 5788, 3319, -5336, -13426, -3462, -6184, 3311, 2281, -2189, -12028, -3947, -7376, 3520, -18022, 10453, -12546, 66, -12679, 4040, -3316, -6502, -8339, 7784, 10656, 749, 3283, -1468, 6614, 6473, 9781, 109, -548, 437, -13544, -20864, -7274, 8442, -2768, -788, -2161, -2343, 10714, 7061, -1058, -11223, -5106, -5195, 11980, 1194, -8527, 2372, -7017, -21848, -3203, 7652, -439, -394, -5943, 1646, 1449, 5121, -10170, -66, 8430, 569, -1491, -3776, -3677, -4317, -10154, 827, 632, 7286, 2208, 10857, -5453, -5547, 7875, -9592, 4115, -820, 1604, -4167, -4049, -6185, -7557, -1195, -3874, 1511, -3471, 5321, 5002, 6182, 9, 11586, -7471, 18759, 7046, -16496, -10106, 1047, -2400, -6730, 9225, -12672, 14650, 8049, -1582, 524, 717, 4893, -2162, -6928, 7286, 9126, 4606, -15407, -10053, 9523, -8774, 1538, -11331, -8433, -7577, -3250, -5446, -9942, -6493, 1392, 13727, -4804, 12845, -857, 3999, -17755, 8399, -3006, 307, 4355, 11326, 17847, 4892, 13626, -19363, 3358, 11282, -12772, 12430, 9350, 1833, 4680, -1821, -3547, 13544, -5846, -2951, 17025, -395, -1365, -8694, 5480, 11113, -10331, 13036, 4585, 493, 703, -11113, 1439, 286, 6566, 15186, -4692, 2884, -2746, 466, 2556, -882, 4210, 5000, 7296, 5538, 6929, -11205, -5733, 13056, -8751, -10826, 10266, -9414, -1967, 11138, -3601, 7823, 4519, 7861, 1290, -3157, 3635, -9391, -5584, 8755, 12201, -5999, -3320, 12672, -7397, -4570, 10413, -4597, 3623, -1694, -5587, -2834, 6317, 17000, 6082, -3206, 1305, 5640, -6866, 6867, -2292, -4780, -1668, 6449, -12194, 1683, -17437, -18480, -3664, -889, -14440, -4775, 18753, 6996, -17025, -3269, 9716, 7847, 860, -1979, 686, -28888, -6482, -17786, 3875, 4714, -8251, -1526, 2287, -4649, 13137, 7238, 16609, -9507, -6274, 9495, -12457, -2691, -117, 13651, -9030, -2335, 5003, 6353, 20360, -2952, -4718, 6420, 18738, -194, 14826, 13547, -2915, 14116, 6375, 1266, 5184, 6312, 5335, -1093, 3730, 1753, -14224, 12517, 13919, -12795, -4227, 2726, 11702, -6464, 3494, 8975, 18046, -1828, 11958, 4095, 5807, 5773, 24685, 6301, 16799, 15423, 9000, 8970, -1024, 9804, -1932, 16281, 7710, -18352, 9462, 7414, -2160, 15554, 16371, -6530, -28402, 2563, 6592, 5492, 9755, 27256, 16984, -20954, 14848, 5867, 2214, -2230, 8323, 2230, -3417, -230, 7822, 7005, 4447, 31, -8943, -1002, 13015, 6868, 10801, -11246, 85, 3489, -471, 1201, -553, -4916, -5630, 1315, 6012, -13113, 7726, -17419, -2917, -8433, -412, -7534, 2809, -7180, 4703, 3865, 5882, -194, 14957, 3393, 7639, -8402, 5245, -3447, 5410, -2920, -14004, 8734, -5676, 15222, 1417, -501, -8605, -4276, -2716, 3403, -14306, -6915, -4668, -11977, -10032, 6775, -1282, 6031, 1426, -4425, 3646, -2821, -10305, -5942, 8314, 3156, -145, -10586, -8510, -12476, -18230, -7286, 12254, 1822, 5617, 331, 14495, -1320, 9900, 5418, -9419, -2266, -13849, 4811, 11703, 4076, 4121, 4610, 15880, 11597, -4013, -9171, -1894, 10465, 9944, 2343, 4922, -2799, 6339, -11359, 5084, -4555, 11658, 7256, 3490, -7435, -10014, -6826, -4345, 9754, 8848, 852, -11095, 4183, -7833, -307, -4507, -7642, 2695, -8176, 8409, 26390, -11600, 1899, 2028, 2342, -933, -105, 1009, -2651, 14112, -2681, 3133, -11045, -20790, -16405, 6758, -6473, -2150, -16591, 5059, -4313, -16187, -957, 2638, -251, -5926, -8862, -1030, -11923, -7595, -2748, 8225, 18383, -1810, -13921, -1375, -3830, 4142, 7243, 3624, 5262, -6706, 4621, 1378, 3497, -8653, -8037, -7527, -4321, 6934, -876, -5929, 10083, 16236, -13483, 9645, 4219, 5121, 6255, 7371, 17307, 2797, -2234, -14096, 8332, -9341, 4780, 15333, 6168, 13798, -4743, -12928, 7386, -7923, 21876, 21542, 6380, 19021, -930, -365, 2109, 19827, -271, -2069, -3686, 3813, 17147, -4351, 3260, 6934, -14789, 7390, 10542, 3846, 18494, 4705, 382, -7292, -3443, 15708, 18392, -13044, -10717, 3600, -7838, 1689, -3679, 5841, -9997, 4691, 685, 9160, 14376, -7468, 8788, -1504, 9292, -12757, 4944, -531, 980, 12865, 9052, -4429, 14424, -6120, 4161, 1032, 15179, 18706, 15981, 12325, 3088, 12325, 19544, 1865, -7823, 10570, 21413, 7289, -3774, 15747, 3049, 6146, -18109, -5645, -4660, 4783, 9714, -6454, -2837, 7572, -5710, -4831, 22087, -16467, 409, 8961, -6029, -3044, 1160, 2723, -1514, 1075, 9047, -16045, 1060, -13662, -6253, 9852, 827, -5180, -11161, -18748, -16995, -16401, -1890, -14287, 679, -2234, 67, -17257, 10015, -743, -20435, 0, 19422, 3, 7047, -8726, -6288, -9860, -9314, -13799, 3536, -9258, -5503, -1269, 6800, 7290, 9423, -5096, -1804, -10797, -6319, 11850, -5380, 9207, -8842, -790, 2307, 15946, -3484, -11214, 12836, -661, 6389, 1273, 890, -8632, -26157, -12757, -14830, 9657, 3473, -18567, -1391, -5115, 12049, 665, 5017, 2750, -2659, 4, 1707, 5337, 6043, -4767, -777, 8424, -11354, 8286, 657, -3598, 8540, -14276, 9827, -2077, -13869, 176, -2385, -4745, -4369, -1282, 13310, -1885, 5529, -1119, 17322, -20208, 8178, 6399, -17807, 7376, -659, -8426, 2272, -7277, 11476, 11950, 17322, 9706, 728, -8758, -5600, 10505, 452, 6052, -2663, -22194, -15918, 453, -12278, -5789, -5738, -6907, 3568, 7637, 1066, -5898, 2440, -2617, -7379, 445, -6807, 13965, 14140, -1266, -10832, 965, 11326, 9205, 7306, 9173, 7909, 10213, -11591, -7325, -16236, -2835, -10074, -2414, 15473, -12205, 5366, 9849, -5872, 1212, 1194, 2522, 4991, -3230, 9017, -14291, -12055, -1581, 5894, -3002, 13546, -10053, 1076, 6872, -2793, 2594, 6778, -7328, 21612, 10429, 1833, -8782, -6228, -6874, -8260, -4458, -1498, -6140, -370, -6821, 10451, -3435, -1377, -10412, 1745, 6353, -3347, -4591, -755, 643, 1749, 8351, 5802, -3945, 3929, -6501, 18333, -7507, 3161, -7852, 6560, 8965, 4639, 11130, 7304, 2636, -13267, 6872, -16361, 6260, 8094, -1506, 8380, -7972, -9239, -6667, 623, 583, -6763, -15161, 8013, -14272, 10430, -2036, 741, 12726, 14482, 7624, 3910, -924, 2447, -721, 5717, 344, 1522, 3283, 8463, 2280, -2400, 6077, 8028, 2717, -2221, 10867, 7770, 10261, -3712, -66, 14, 4731, -6946, 766, 4907, 1687, 6221, -7795, -10997, 7792, -11818, -6556, -7876, 16341, 4793, -15193, 15818, -2262, -92, -7061, 3945, 8645, -2731, 8885, 12443, 3498, 15771, -2037, 12291, 2352, -973, 2983, 5472, -6819, 6252, 9204, 5955, 19933, -1272, -8697, 6400, -4900, -2165, -21749, -7521, -5216, 9786, 15712, -4508, 14877, 2687, 4977, 9854, 11831, -1828, -2919, -9485, -2989, 12878, 4272, 5061, 122, 9702, 8172, 3202, -10115, -2348, 425, 4085, 7732, 9301, -8778, 4440, 8747, -2760, 3268, 7819, 6324, 7473, 20099, 11369, 4826, -4342, 14800, -121, -10440, -3004, 12160, -18553, -10576, -4452, 4747, 8294, 20100, 9215, 17434, 5927, 10837, -6444, -11138, -5089, 3950, 2836, -9127, -7481, -15045, -13888, -729, -1160, 4082, -2872, 10087, 11170, -228, 17480, -8501, 9446, 7196, -16823, -1715, 924, 2765, -2579, -15364, 3786, 9686, 5727, -3093, -21775, 733, 12453, -8186, -185, -12863, 8260, 4433, 5903, 1413, 3513, -10215, -3406, -9652, -15076, 401, -6046, 2610, 16845, 7018, -6226, -4246, 369, -4480, -462, -8468, -3175, -6010, 12277, 6081, -15729, -5712, 508, 904, 1697, 1536, -281, -5118, 9901, 17523, 2049, -3595, -6672, 2136, -11796, 15973, 1962, -11393, 8831, -71, -13938, -3868, -9798, -6058, 16674, 3817, 14579, 12372, -14453, 4322, -13193, -4478, -5660, 11560, 923, 2444, 4994, 10141, -11638, 19947, -1614, 8052, 1267, -19495, 4976, -2723, 10424, -1105, -2124, 12364, 7243, 1369, 5810, -4199, 1036, -5948, -11722, 5419, -4376, 5357, -16652, 11135, -12468, -11838, -1842, -8387, 2683, 12746, 4645, -8456, -4391, 10337, 3373, 21236, 2897, 10009, 7927, -2245, -2845, 6147, 2172, 311, -6111, 10714, -4946, 7879, -5184, 3528, -13028, 820, 9255, 5459, 5785, -3068, 634, 3612, -8256, 6301, 3661, 13161, 11738, 7384, 17379, -2407, -3945, 4001, 3404, 5218, 3421, -1352, 2522, -9870, -1682, -3243, 10817, 7781, 8358, 3421, 4106, -5597, 23468, 5748, 7256, 9956, -12915, -2692, -7154, 4955, 13012, 855, 7589, -17394, 599, 8218, -9325, 8309, -2946, 7968, -8725, 5626, 8340, -6373, 11304, -1215, 26149, -10475, 11232, 10549, -5126, 1399, 12672, 18955, -2853, -9482, -2160, -7745, 5315, 4085, -13672, 1651, -2274, 5315, -13415, 5871, 9514, -1381, 738, -3034, -8679, -4119, 1705, 5936, 2998, 4147, 14487, 1158, -6453, 10486, -2092, -3289, 113, -1311, -16115, -9639, 12055, -3611, 10507, -19780, -4181, 14238, 2379, -6432, -10358, 4123, 6969, 10608, -10019, 17826, -3527, 4067, -12015, 7600, 12846, -6729, 410, 1227, 15758, 5430, -8361, -1867, 21978, -7760, -5394, -3668, 3347, -9870, 5526, -467, 7132, -4643, 1752, 10031, 14313, -7411, 1166, -101, -14444, -5606, 261, -4014, -2702, -12313, 14773, 9093, -7058, -5534, 9030, -3837, 4909, 4787, 6216, 3081, -7458, 16533, -7120, 1941, 629, 2027, 10918, -5696, 12962, -1701, -14206, 8752, 12340, 14403, -9443, 16518, -11691, -10588, -6344, 13152, -7101, 4033, -3427, -3599, 1487, 5904, -5210, 5294, 19584, 16206, 5831, -912, -8510, 11883, 8690, 3227, -12531, -1512, 9852, 2961, 3789, 16, 9811, -1651, 13468, 6606, -7436, 12065, 4595, 7977, 5322, 8401, -115, 11225, -10061, -5231, -8472, 9127, -1295, -4982, 6234, 2479, -235, 17257, -13646, 4116, 458, -2094, -768, 8787, -1294, 4499, -7905, 12416, -2866, 12718, -17937, -5503, 1417, -562, -817, 4160, 1327, -3007, -10594};
static Q15_T L5_F2[L5_CTEMP * L5_HF * L5_WF * 1 * 1] = {-3928, -7396, -973, -6908, -1507, -5374, 7032, 10869, 9558, -608, -1532, 7581, 2461, -303, 7563, 257, 3026, 7842, 4928, -2582, 2379, 7574, 2214, 12026, -1696, -4261, 1664, -830, 397, 66, 13969, -11083, -1400, 1773, -213, -445, -4617, -4941, 1563, -5470, -8600, 5680, -3503, -5179, 11705, -5607, 389, 4675, -2282, 8921, 8175, -1355, 1597, 7732, -1624, -1483, 2620, -3383, 9240, 5579, -2463, -704, 8344, 5017, -5743, 2895, 7908, 1966, 6899, 7500, 560, 3293, -1309, -2029, -995, 7415, -7092, 3933, 3664, 1773, 10435, 9898, 8047, -3213, 5434, -6081, -4554, 316, 3690, 2642, -5826, 16142, 8960, -5205, 1644, -1984, 1733, 5659, -418, 1782, 5924, 1995, 5413, -2858, 2735, 8752, -2774, 6209, 2781, 928, -1640, 4108, 2202, 2563, -229, 9307, 8298, -2953, -1980, -836, -4190, -4803, 10806, -6753, 8980, 4064, -1643, -4328, -86, -2432, -11306, -1700, 9432, -4216, 704, -12047, 12215, -2725, -8057, -1564, 3324, -3418, 1680, 853, -3500, -4316, -3458, 3783, 16474, -1705, 437, 1850, 127, 5462, 3509, 258, 7712, -9873, -4776, 3119, -3391, -3731, 525, -9347, -2963, 3244, 12762, 1793, 542, -2632, -485, 839, -1758, 2648, 6208, -5344, 967, 3705, 7539, 1790, 6214, 2343, -1237, 8576, -8781, 2022, 10397, -2572, -1649, 6312, 7495, 6548, -1802, -3514, -1553, 4131, 714, 4441, 2170, 3692, 2743, -426, -4181, -1417, 3386, 14510, 4524, 1736, 4343, -944, 4741, -1964, 747, 12144, 4076, 6973, 635, -659, -1008, 4048, 8279, -1512, 5159, 9857, 5395, 2056, 11796, 4512, -3852, -4945, 1041, 3381, -6738, -4348, 5937, -6743, -2631, -5082, -3178, -3655, 7298, 10057, 3593, -3241, -9191, -6278, -3890, 5389, -5809, -938, 4313, -550, 10254, 3408, 9079, 8226, 1580, -5259, 1209, -64, -2611, -185, 6490, 10543, -1940, -3338, 3115, -2527, -6503, -4466, 1664, 5708, -6124, 6239, 5373, -14319, 4270, 3271, -5610, 28, 2482, -3842, -6716, 16300, -1096, -2209, 5817, -4798, -4556, -488, -3614, -3912, -5506, -8229, -1816, 12997, 3557, -6655, -4188, 9660, -4620, -5044, 977, 292, 8100, -1702, -2127, -748, -285, -1792, 8517, -1965, -923, 12259, -37, -1626, -9270, -2766, -2599, 13905, 5213, -1151, -6988, -25, -5122, -836, -9136, -2236, 12431, 5016, -2830, -4930, -7037, 1157, -4398, -6010, -4960, 2677, 1223, -7755, -5730, -4638, 6906, 23, 11758, 1241, -4729, 3477, -6484, -2992, -5388, -1111, 2484, -3109, -1362, 15363, 7631, -9515, -4496, 3993, -3826, -4199, 295, 898, 2719, 533, 3696, 13246, 5892, -1770, -8128, -2903, -9241, 516, 66, -2296, -3469, -914, -8720, 3438, 458, -3411, 13579, -6326, 3601, 3888, 212, 4484, 7196, -2630, 8850, 9440, 1047, 143, 1247, 933, 4810, 4278, -1578, -6307, -11439, -1949, -596, 8951, 3096, -1951, 2206, -2510, -2545, 3722, 477, 8143, 1299, 14968, -62, -4948, 6103, -4252, -4707, 6574, -2586, 6282, 9543, 5178, -958, 3450, 1405, 9517, 6803, -7720, -6940, 4904, -367, -3564, 2778, 7134, -2347, 8305, 1324, -8166, -217, 5540, 178, -4550, 4127, 13372, 3404, -5160, 7538, -156, 1325, -480, 4561, -4583, 2251, 7769, -2126, -5995, 8029, 137, 15691, 8712, -1355, -115, -580, 3608, 1422, 506, -8562, -750, -226, 15168, 2019, 2502, -7064, -632, -782, 1169, 11945, 2049, -4598, 4944, 3964, -2673, 2686, -656, -2154, 8119, 76, -9285, 3131, -9055, 1585, -1089, 3202, 5057, 10221, 14194, -1108, -12694, 3270, -4034, -3548, -1591, 99, -6851, 5327, -5894, -1503, 11161, -2434, -7789, 2679, -3780, -7588, -8977, 5305, -2180, -5801, 2929, -6348, -5871, 161, -3109, 2929, -1154, 14265, 2730, -7053, -11073, -4836, -2355, 193, -2828, -829, -7421, -1670, 3944, 14222, 3081, -1218, 3707, 500, 1840, -1203, 12695, -1599, 69, 5223, 398, 2119, 3412, -6874, -7728, -6494, -1709, -5247, 330, 7279, 2496, -1230, 4730, 8105, -5864, 9094, 6628, 3021, -501, 9381, -1792, 9564, -7319, -1272, 5180, -7364, -2347};
static Q15_T L5_F3[1 * 1 * 1 * L5_CTEMP * L5_COUT] = {-9883, 1393, -591, -1164, 4283, 3744, 7261, 2144, -6512, -7943, -10214, 3832, 6737, 3011, -6526, 7145, -6026, 5548, 9408, -9843, -4380, -189, 8027, 8181, 13995, 6108, -978, 3164, 3248, 4493, -5485, 10031, 1668, -1092, -8012, -12045, -3960, -1921, -16563, 6530, 1448, -5985, 9688, -4293, 1483, -3160, -494, 7005, -6534, 3766, 452, 2160, -523, -11901, -2775, 1731, 866, 27, -11059, 2226, 9228, -2410, 1724, -11681, 2897, 9534, 2986, -677, 7407, -2287, -6669, -5, -3278, -6527, -5837, 1320, 6166, -12870, -972, -104, 5726, -4526, -13107, -6209, -5946, -2838, 10680, 10484, -765, 330, 17446, 14138, 5649, 4845, 8488, -6263, 9298, 2262, 811, 7687, -4336, -2607, -6454, 6978, -5464, 7049, 11092, 11197, -14808, -1314, -6067, -5565, -2058, -1738, 2234, 9363, -8224, 6397, -8133, 3372, 984, -6765, 6003, 1525, -7105, 4125, -3061, 1200, 10227, 60, 764, -1768, -8954, -7410, 6627, -3614, -5950, 9305, 8018, -4556, 4906, 8751, -1272, 8289, 2877, 2975, -1834, -4095, 4742, 5508, -5891, -8263, 10691, -7137, -5065, 1954, 5449, 320, -6662, 1242, 1907, -3552, -1606, -2238, -6469, -2804, 2640, -10786, -15001, 15802, 9160, 5916, 608, -7033, 14071, 6788, -12801, 13029, 7840, 634, 9506, 5073, -2167, 10710, 1236, 1477, 7969, 264, 6261, -2164, 6688, -1069, -7111, 8651, 4196, 3570, 4765, -2513, 9070, -3247, -2160, -2338, -8563, -596, 6694, -5230, -5479, -2551, 15746, -5474, -1529, -714, -2004, 11435, 3049, -13122, -2207, 12588, 8789, -8756, -6593, -10325, 3857, 5621, -3969, -3693, -4427, -11635, 12367, -2534, 6001, -8239, 4684, -939, -14853, -3700, -6917, 323, -16746, -5247, -11380, 7870, 10852, -1032, -9488, -12023, 9241, 3172, 7606, -2618, -2983, -17838, 993, 652, 826, 683, -3368, -11132, 8193, -12721, -5918, -18532, 282, -5793, 4306, 1207, 5187, 7279, -5136, 7564, -6350, 1252, 3970, 3812, -4900, -7836, 1450, 3642, 11269, 3139, -3297, -5945, -1399, 8044, 11723, -7806, -136, 436, -2685, -6828, -6210, -3455, 2277, -18591, -847, -2633, 7320, -4794, -8122, 2506, -10934, 4357, -3091, -7683, -15657, -1661, 3167, 11261, -7147, -8321, -1010, 14006, 6684, -2569, -1434, 2318, -12804, 552, -10942, -3067, -4275, 134, -9036, -1166, 17418, 3682, -773, -4315, 4289, -8345, -1985, 2375, -340, -8357, -5460, -1707, 3356, 9202, -12465, 2899, 2516, -19425, 642, -156, 15415, -1016, -13039, 26014, -8652, -1144, 356, -688, 1744, 8715, 2630, 6796, -2192, 5082, 11037, -5682, 6469, 5683, -1827, -9978, -10814, 5530, 3234, 3994, 9812, -12101, -8788, 280, 5901, 11118, -505, -4031, -1078, -11674, -1115, -8758, 3838, -5693, 10589, -10976, 2268, 18206, -8262, 5986, 9948, -10525, -875, 5316, 9452, -4247, 4306, 3357, -5342, -5449, -3346, 6194, 5638, 3633, -183, 1304, 5281, 783, -4757, -7151, -8232, -4373, 6543, 4098, 4774, -610, 15124, -1256, -3269, 5263, -7161, 8387, 1732, 10613, 6325, 240, 9139, 2363, 6998, 13972, 8764, -1559, -9217, 4764, 5047, -3312, -4965, -1243, 9639, -12839, 12730, 3325, 1131, 8974, 4747, -4317, -3698, 1925, 7957, -579, -3387, -5058, 8133, 2027, -5062, -2817, 4135, 8749, -603, 54, -3649, -6193, -2459, -6517, 5947, 1007, -14217, -10150, 3753, -3824, -306, 3828, 15009, -758, 7245, 11253, 646, 7577, -2719, 4959, -732, -622, 2606, -6235, -11324, 7393, -6633, -1545, -793, 11947, -8799, 14363, 5500, -8409, -2187, -10996, 1757, 15501, 7669, 4793, 1449, 6561, -4865, -3889, -3560, -561, 5142, -2963, 2860, 6762, -2376, 5572, -8225, 6045, 9733, 22476, -3540, 8177, -7492, -4362, 11410, 472, 3681, -11453, 13322, 210, -10763, -14674, -180, 5960, 6916, 4567, 18017, 621, 3230, 3244, -7349, 4463, -17785, -419, -5788, 3476, 5106, -11365, 12283, -19375, -997, 29440, -9708, -4293, 2902, 3054, -4425, 12090, -6905, -7950, -1286, -3321, -11616, 4913, 10942, -11270, -6588, 3891, 2884, -15221, 5779, -7050, -16834, -452, -6305, -5210, 1661, -1489, 7676, 2533, -10176, -1349, -7988, -4046, 9635, 3632, 4707, 5862, 7672, -3428, -1714, 5524, -9659, -7547, 3272, -8717, 4445, -5887, -2580, 2364, 12352, -2579, 3102, 2600, -9057, 5485, -3234, 5680, 2823, -13405, -15754, -3941, 2260, 2252, 136, 3761, -4671, -3397, -7182, 6089, 7305, -4451, 14686, 13137, -6469, 6455, -5762, 12082, -7756, -371, 3890, 6091, -2175, -11989, 7759, 2077, 533, 3216, -1703, -8501, -1199, 2264, 5745, 8097, 1588, 2605, 4297, -804, 10781, 2701, 11476, 4657, 4994, -7304, 558, 2765, 6245, -10457, 10306, 2598, 6091, -19093, -1408, -12179, 5756, -4482, -3878, -2219, 1486, -4276, 3490, 8065, 3153, 13340, 646, -15738, 8674, -9534, -5245, 7428, -3958, 2063, 2340, -7110, 5105, -19616, -2084, -3230, -2697, -1942, 190, -11068, -16669, 8685, 2911, -8569, -12623, 10364, 5170, 4949, -9835, 4665, 8828, -4565, -3578, -1193, 14231, 5740, 4393, 15335, 1200, -20196, 9690, 4380, -5116, -8244, -13008, -4051, -7853, 5487, -4106, -9437, 8576, 1325, -3049, 545, -5611, -8147, 7753, 363, 315, -4402, 5506, 1376, 5174, -888, -475, -1766, -4239, -10599, -6423, -553, -3532, 5238, 426, -1631, -12297, 16385, 8849, -8594, -5390, 18500, -9920, 4868, -9819, -301, -12253, -16899, 10164, -6033, -7733, 5068, -8744, -2090, 697, -8931, -940, -5218, 16544, -12886, -4420, -9615, 4774, -3848, 18125, -13221, 7374, -5706, 6584, 30831, -1869, 8392, -6426, 946, -8316, -10716, -5862, -5551, 12654, -6449, -1323, 5920, 10722, 2127, -9884, 10763, 5680, -14033, 11617, 757, -11454, -5402, -4193, 684, 99, -7234, 6186, 8008, -2322, 739, -5536, 1115, 4563, -1208, -4022, 5271, 3364, 5230, 16222, 9353, -2655, -459, -923, 3844, 5475, 4848, 2628, 4128, -11187, 1171, 3096, 358, -705, -5336, -6559, -1935, 1961, -9334, -4281, 4937, 6811, -1716, 10371, -14540, 1001, -1175, -5033, -3114, -6835, 724, 15782, -15273, 4132, -3451, -16573, -7199, -2415, -5917, 20, -4416, -10497, 5305, 8409, 928, -5732, -4628, 2393, -1917, -1269, -6152, -7143, -13539, 2274, -18077, -4828, -11160, 3287, -4117, 6616, 11723, 3721, 11734, -14338, 993, 3809, 2973, -7099, 1760, -7626, 7487, -6121, -3185, 3969, -4005, -2703, 3775, -4325, -9583, 1318, -13032, -1347, 5524, 6899, -7632, 7517, -2053, 7704, -7714, -3250, -361, -9420, 5649, -11080, -136, 5374, -5767, 8622, 13771, 4517, -1072, -72, 12787, -2080, -321, 8706, 9977, -6377, 7681, 753, -12048, -4791, -4204, 2665, 14084, -5571, 192, -13444, 3968, 509, -1629, 1586, -1221, -8555, 1198, 1312, 11954, 8660, 1582, -5636, 1562, -6151, -2984, -1700, 6895, 8753, -8622, -4235, 6250, 3485, 8782, 12495, 1467, 8676, 4141, 8043, 6605, 2018, 18366, -2845, -4507, 7019, 610, -1368, 11359, -12664, 6370, -2746, 10656, 2340, -10700, 13151, 5323, -2061, 6085, -4738, -5203, 15164, -11354, -1345, 2375, -2564, -9799, 6765, 7585, -2651, 2853, -13117, 2465, -5093, -345, -16549, -5999, -3608, 9911, 6231, 4744, -332, 2138, -205, 1997, 4680, -5469, 4490, -11589, 11088, -7551, -4519, 2343, 1423, 4305, -11168, 7446, 1097, -1175, -12266, 5727, -1652, 902, 2010, 2937, -5511, -11779, 3976, 2727, -7124, 6612, -3433, 2043, -7013, 14044, -3562, 1413, 2547, -2748, 4341, -9200, -3707, -9650, -654, 4931, 16920, 5011, 12257, 3955, -3963, -10205, -3139, 2800, 3720, 2920, 9275, -15203, -155, 8489, -3007, 1093, 12462, 4982, 10291, 4018, -4355, 643, 173, -5335, 10911, -5217, -6238, -2041, 4442, 2694, -3203, -4804, 3069, 5232, -4735, -399, 4883, -6193, 1022, -7235, -3181, 7527, -1571, -3123, 3657, 4714, -11652, -4473, 2601, 14203, -6179, -1470, -10104, -4027, 765, -846, -94, -5961, -264, 10604, 8271, 656, -14028, -1633, 11876, 5560, -10576, -6543, -5038, -13532, -6490, 296, -5727, 1674, 1909, 2239, -4970, -12070, 3095, -2723, -1247, -11839, 3216, 6852, 2510, 2147, -3354, -2551, -3579, 3133, -2737, -493, -9047, 4, 5772, 4428, -2395, 3168, 9818, 538, 8693, -8557, -7102, -1168, -1136, -4658, 1799, -6460, 7289, 3688, -7045, 12190, 2128, 5892, 14985, -12071, 2030, 5709, 11463, 2250, -9702, 1128, -5294, 764, 15950, 41, -1589, 2223, -17984, 9028, 3109, -1604, -13024, -503, 39, -3636, -206, -13190, 13398, -9373, 5332, 8296, 2569, 3098, -2747, 7611, 17000, -1588, 8893, -2982, -396, -12771, -1728, -7146, 2630, -18862, 1820, -4627, -3443, 2671, 4262, 4368, -7936, -1895, -1977, 7463, 898, 11209, 1689, -3366, 232, 7989, 2368, 13993, -2185, -5503, 11304, -5845, 1009, -3100, -7065, 300, -5171, -3952, -2257, 455, 158, -1213, -3115, 6107, -4975, -4406, -2718, -17946, 13006, 4769, -1503, -13863, -6824, -7661, 6341, -6874, -6388, 8435, 2752, -11648, -10994, -8901, -8158, -4194, -2719, 4725, -14279, -8020, 2164, -4842, -20718, -14345, -8706, 13759, 5214, -22217, 7555, -4657, -8165, -2621, 13224, -5896, 10343, 11592, -81, 2510, -10384, 1263, -11429, 1312, -8961, -6778, -5981, 1393, -4085, -13712, 8606, -10582, 14816, -2864, 8235, 8592, 2506, 1437, -4033, -3747, -6559, 11111, 3615, -7508, 1469, -2508, -1573, 4615, 2894, 6201, 7084, 15636, -10056, -3342, 705, -9373, 3585, -4048, -380, 12132, -6176, -3377, 6128, 12897, 21856, 172, 1182, -5349, 1351, -1772, -7312, 6356, 4966, -522, 7387, 2182, 17101, 6156, -5958, -15353, -3502, 3739, 1960, -7225, 28551, -9948, 14662, -1398, 2894, 10289, -10454, -9064, 10920, 7790, 440, 4544, -3125, -3225, 757, -7572, -5361, 1284, 11262, -7007, 6395, -12665, -6654, 213, -7056, 5435, -10289, 3378, 4469, 1004, 2501, 8614, -5795, 7850, -8926, -3336, 11039, 5158, -10050, -13981, 3875, -1199, 8494, 13655, 10840, -8167, -6917, -4536, -8833, 2944, 4044, 4748, 3316, -4561, -6003, -20383, 11849, 868, -6389, -7790, -8098, -10184, -7994, 629, -12474, -11845, 5150, -7048, -4399, -5180, 3785, -17571, 923, -2034, -7922, 5484, -11863, -1942, -1983, -1233, 1486, 240, -3060, 2164, -4101, -136, 7291, 4103, -739, -2304, -7344, 4440, 11529, -7888, 11946, -72, 2268, -5057, 8141, -5112, -1957, 11628, 6441, -16281, 17151, -7191, 2131, 545, 9824, -11706, 9904, 7510, 11251, 5069, -7556, 4027, -5754, 7991, 5243, 14282, 8590, -8479, -797, 7501, -748, 4836, -2120, 10223, 62, -8227, -7188, 2445, -1213, 1463, -4868, 2785, 1199, 490, 6200, 2150, 865, -1691, -1538, -10450, 3155, 14651, -9074, -4240, 7934, -12604, -2267, -1845, 2779, -8013, 15836, -2475, 5025, -1495, -5784, -3067, -12310, -492, 5095, 6255, 2195, 1497, -3653, 7329, 16463, 1535, 5373, -581, 3256, 4976, 5409, -6921, -344, -1799, 1790, -6826, 12577, 556, 5092, 73, -17681, 10414, 9166, -9443, -3658, -6011, -4859, -7970, 560, 1550, 4406, -5807, 2384, 5884, -10713, 19823, 6319, 3052, -11747, 2649, 500, -3584, 3438, 1909, 1276, 4661, 8239, -1761, -207, 295, -102, -2722, -2089, -2736, -430, -9339, 636, 3242, -6998, -3972, -4335, 15855, 752, -4025, -9284, 6152, 3059, -16587, -6942, -7071, -1177, -925, 1281, -9004, 431, 6038, -964, 7770, -14388, 3512, 4319, -4860, 348, 342, -1233, -5628, 12821, -1473, 8050, 2946, -8501, 11199, -1287, -7757, -8237, -1598, 1818, -9483, 12239, 4701, 2404, 2593, -15165, 2040, -3713, 5244, -505, -3835, 8580, 315, -2183, -6267, 8588, 2591, 17434, -3094, 3934, 5111, -7412, -10563, 2427, -9262, -3249, 15892, -313, 3849, -5479, -2090, 3959, 3610, 6820, -14853, 7046, 3330, 16817, -15167, -745, -8924, -3, 1754, -2559, 5918, 3457, 4752, -4976, 7131, 8378, -3255, -16336, -18811, 1025, -989, 4244, 9249, -9725, 4858, 2845, -7697, -10700, 443, 4151, 14670, -3143, 9758, 10434, 3228, -62, -5609, -3657, 663, 3010, 16357, -18464, -2847, -4587, 999, 6933, -13946, -3074, -8519, -12329, 9202, -8062, -4778, 7223, -503, -2786, 9869, -2362, -1520, -928, -5573, 2070, 9509, 8644, 7509, -6720, 2382, 5815, -1066, 7929, -7637, 2705, 5831, -11291, -6840, 3954, -6081, 1441, 3281, -9221, 10712, 6484, -3301, 5882, 1724, -2302, -3137, -11567, 4000, 1643, 14792, 13044, -75, 3905, 3029, -6522, -6262, 2304, 1442, -2263, 4458, -2035, 4906, 3480, -9444, 5523, -1399, 3590, -777, -4873, -5533, -4044, 14191, -835, -2351, -2090, 3644, -1922, 1663, -437, 6459, 5209, 3294, 9123, -4577, 11925, -6193, 6144, 11545, -10951, -3673, -5623, 4745, 3367, 7012, -19867, 401, -11680, 436, 9844, 14002, -12944, 22702, 13403, 4485, -5321, -9710, 7727, 6384, -2727, -8908, 5385, -8168, 3570, 8117, 6600, -18223, -6224, 2447, -20424, -2451, 9682, -512, -9946, 10494, 4882, 2257, -5491, -13523, 1093, 731, 11086, 1887, -12622, 10909, -441, -7731, 2059, 4704, -7864, 9609, 12230, -11058, -7475, -14417, 9234, 9691, 8424, 8471, 959, -1322, -7252, 3822, -12579, 1734, 1255, -10483, 8303, -6489, -5128, 15190, -26110, 368, 771, 10821, -10223, -4306, 2430, 518, 1616, -5535, 14610, 6324, 1272, 3772, 6045, -9589, -8028, 238, 21128, -15158, 8896, -3082, -13262, -563, 10957, -11425, -2907, 433, 1831, 4434, 7710, -12871, 8899, 3122, -17664, 7446, 3445, 1816, -1240, 5811, -1684, -3024, -3415, 658, 907, -5468, -11494, 46, -5325, -11360, -14109, -14532, 17539, -7950, 2672, 10343, -469, -5229, -3031, -7390, -2162, 479, 609, 3388, 3543, -902, -7270, -2716, -8131, -16697, 14401, 8478, -6325, 12657, -3674, 5, 15082, 3328, -2109, -573, 475, -18563, -10085, -4718, -12212, -9060, -8798, 7363, -3561, 1453, 3015, -8404, 11477, -11826, 1487, 3366, -8628, 11059, -9974, -548, -6578, -2596, 26389, 2855, 3205, 71, 13331, 2383, -3637, -10707, 12050, -948, 2914, 4633, 4208, -870, -21, 7559, -7179, 2321, 3103, 9775, 6223, -2083, 855, -5535, -15468, -9592, -2647, 10585, -9841, -9953, 4029, -2554, 6190, 6079, 8629, -1216, 4249, 10605, -1634, 1548, 6907, 559, -4363, -6234, -1228, -5979, -5687, -5812, 14235, -708, -13586, -22906, -2433, -20722, -654, 1477, -119, 6784, -1264, -3696, -2010, 1667, -1752, 918, 6448, 7189, -6181, -5050, 2525, 158, 230, 9116, -1653, 11517, -3158, 8103, 3589, -5258, 10649, 2124, 1141, 628, 1157, -531, -1291, -2955, -4782, 10878, 4053, -6312, 2942, -4723, 5146, 8785, -9057, -2075, 4424, 3396, -6018, -1178, 7088, -2816, -6898, 1072, 5561, 9005, 6121, 871, -6124, -3085};
static Q15_T L5_W1[L5_CTEMP] = {9702, 12912, 10977, 9262, 16785, 15310, 14287, 10799, 11404, 11878, 13174, 10471, 14762, 7740, 16384, 9968, 6929, 10068, 14648, 16389, 11359, 8491, 9560, 10759, 10564, 12723, 9793, 11278, 13263, 19406, 9398, 10517, 7968, 13617, 8835, 15749, 8940, 14396, 14973, 8328, 13574, 9685, 13376, 12556, 11098, 10089, 12026, 9534, 14092, 10428, 12887, 13196, 9672, 18811, 13912, 9301, 13220, 13278, 16648, 11744, 14829, 15366, 15411, 17557};
static Q15_T L5_W2[L5_CTEMP] = {6535, 4076, 5826, 17534, 4241, 4300, 8868, 3975, 6186, 6584, 4059, 8255, 6653, 10438, 7593, 9052, 8349, 11083, 7153, 6802, 4529, 4663, 5750, 4702, 5196, 15842, 10284, 6164, 3624, 6019, 7224, 5691, 11183, 11384, 9765, 6534, 7896, 5661, 6460, 11204, 5579, 5865, 4853, 4790, 10192, 4230, 5639, 9046, 17783, 5156, 5918, 5244, 27020, 5069, 11413, 4638, 7609, 4220, 4490, 5585, 4482, 6749, 3538, 5683};
static Q15_T L5_W3[L5_COUT] = {14225, 10533, 12714, 11888, 13911, 16879, 10216, 15076, 10985, 14675, 16717, 13140, 15536, 15936, 12379, 12160, 14086, 13169, 10843, 11254, 9135, 12111, 11326, 13808, 18070, 12830, 15692, 12041, 13056, 14468, 12458, 16155};
static Q15_T L5_B1[L5_CTEMP] = {3946, -3613, -3121, 12222, -10293, -8248, 15407, -11759, 6089, -13326, -3570, 2499, 10709, -2357, -3189, -9048, 2811, 9094, 5108, 21149, 8745, -14774, -6916, -6124, -4371, -2464, -8418, -75, -7581, 4804, 10933, 14190, -12981, 4536, 6654, -8930, -3629, 1227, 8785, -13429, -14983, -8223, -363, 11885, -1119, 10490, 5520, 13495, -16023, 161, 7069, -5010, 2566, -7710, -11577, 953, -13206, 12663, -10797, 18849, 12236, 17300, 7320, 12038};
static Q15_T L5_B2[L5_CTEMP] = {1756, -13304, -11597, -772, 8253, -19722, -6765, -16595, -11409, -9228, -13778, -8411, -13831, -1824, 14362, 8254, -6358, 833, -926, -8962, -3650, -11921, -14518, -15429, -15161, -1204, -3433, 20192, -10511, 4032, -1827, -2874, 7034, 2852, -7577, 1780, 15188, 24081, -1150, -3694, -10636, 26386, -2179, -12347, 929, -14451, -4442, -7599, -2101, -14700, -5196, -16213, -1090, -10309, 4272, -6025, 608, 28883, 10378, -215, -5139, 18294, -27657, -711};
static Q15_T L5_B3[L5_COUT] = {-10308, -7177, 14781, 1991, -9667, 3030, -16297, 7057, 608, -8206, 7163, 1926, 16010, 11402, 15352, -4059, -3435, 10926, -12946, -14605, -3149, 759, -10723, 84, -11595, 3037, 3918, 530, -1335, 19162, -17200, 20685};

static SCALE_T L5_D1 = 5;
static SCALE_T L5_D2 = 4;
static SCALE_T L5_D3 = 6;
static Q31_T L5_Limit1 = 402653184L;
static Q31_T L5_Limit2 = 201326592L;

#ifdef SHIFT
  static L_SCALE_T L5_ShRU1 = 11; //2048
  static L_SCALE_T L5_ShRB1 = 3;  //8
  static L_SCALE_T L5_ShRX1 = 14; //16384
  static L_SCALE_T L5_ShRU2 = 9;  //512
  static L_SCALE_T L5_ShRB2 = 3;  //8
  static L_SCALE_T L5_ShRX2 = 13; //8192
  static L_SCALE_T L5_ShRU3 = 9;  //512
  static L_SCALE_T L5_ShRB3 = 3;  //8
  static L_SCALE_T L5_ShRW3 = 14; //16384
  static L_SCALE_T L5_ShLU1 = 0;  //1
  static L_SCALE_T L5_ShLB1 = 0;  //1
  static L_SCALE_T L5_ShLX1 = 0;  //1
  static L_SCALE_T L5_ShLU2 = 0;  //1
  static L_SCALE_T L5_ShLB2 = 0;  //1
  static L_SCALE_T L5_ShLX2 = 0;  //1
  static L_SCALE_T L5_ShLU3 = 0;  //1
  static L_SCALE_T L5_ShLB3 = 0;  //1
  static L_SCALE_T L5_ShLW3 = 0;  //1
#else
  static L_SCALE_T L5_ShRU1 = 2048;
  static L_SCALE_T L5_ShRB1 = 8;
  static L_SCALE_T L5_ShRX1 = 16384;
  static L_SCALE_T L5_ShRU2 = 512;
  static L_SCALE_T L5_ShRB2 = 8;
  static L_SCALE_T L5_ShRX2 = 8192;
  static L_SCALE_T L5_ShRU3 = 512;
  static L_SCALE_T L5_ShRB3 = 8;
  static L_SCALE_T L5_ShRW3 = 16384;
  static L_SCALE_T L5_ShLU1 = 1;
  static L_SCALE_T L5_ShLB1 = 1;
  static L_SCALE_T L5_ShLX1 = 1;
  static L_SCALE_T L5_ShLU2 = 1;
  static L_SCALE_T L5_ShLB2 = 1;
  static L_SCALE_T L5_ShLX2 = 1;
  static L_SCALE_T L5_ShLU3 = 1;
  static L_SCALE_T L5_ShLB3 = 1;
  static L_SCALE_T L5_ShLW3 = 1;
#endif
