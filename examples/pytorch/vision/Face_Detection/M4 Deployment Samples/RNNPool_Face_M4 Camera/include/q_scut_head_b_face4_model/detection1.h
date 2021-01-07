// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#define D1NW_HF 1
#define D1NW_WF 1
#define D1NW_CF 1
#define D1NW_COUT 1
#define D1NW_HPADL 0
#define D1NW_HPADR 0
#define D1NW_WPADL 0
#define D1NW_WPADR 0
#define D1NW_HSTRIDE 1
#define D1NW_WSTRIDE 1
#define D1NW_HDILATION 1
#define D1NW_WDILATION 1
#define D1NW_G 32

#define D1CW_HF 3
#define D1CW_WF 3
#define D1CW_CF 32
#define D1CW_COUT 4
#define D1CW_HPADL 1
#define D1CW_HPADR 1
#define D1CW_WPADL 1
#define D1CW_WPADR 1
#define D1CW_HSTRIDE 1
#define D1CW_WSTRIDE 1
#define D1CW_HDILATION 1
#define D1CW_WDILATION 1
#define D1CW_G 1

#define D1LW_HF 3
#define D1LW_WF 3
#define D1LW_CF 32
#define D1LW_COUT 4
#define D1LW_HPADL 1
#define D1LW_HPADR 1
#define D1LW_WPADL 1
#define D1LW_WPADR 1
#define D1LW_HSTRIDE 1
#define D1LW_WSTRIDE 1
#define D1LW_HDILATION 1
#define D1LW_WDILATION 1
#define D1LW_G 1

// The convention followed for filter weights is F[groups][filter_height][filter_width][in_channels][out_channels]
static const Q15_T D1NW[D1NW_G * D1NW_HF * D1NW_WF * D1NW_CF * D1NW_COUT] = {28480, 24964, 27020, 28607, 28466, 28819, 28938, 28449, 28738, 28679, 29046, 28506, 29080, 28270, 28317, 28474, 28517, 28781, 28426, 28530, 29021, 29008, 28849, 28529, 28955, 28665, 28650, 28646, 28518, 27380, 27263, 29374};
static const Q15_T D1CW[D1CW_G * D1CW_HF * D1CW_WF * D1CW_CF * D1CW_COUT] = {-1825, 1216, -1204, 1770, 1526, 1657, -774, -3520, 3192, -1125, 1762, -4576, -1464, 4047, -2120, 498, 2884, 1261, 1130, -8419, 524, 2960, 1745, -6452, -658, 6229, 5152, -8344, -340, -1763, 4643, -4018, 7988, -5925, 244, -2862, 3079, -930, 3887, -5504, 113, -1829, 3189, -672, -3321, -273, 684, 2826, -4036, 2066, -103, 2880, -2112, 3574, -3124, 1343, 324, -5333, -1160, 8073, 5288, -4166, -2917, 4463, 5768, 1122, -2328, -5572, 4514, 2783, 2752, -11402, -1458, 350, -3607, 3596, -3834, -4918, 257, 7085, 454, 5212, 512, -6096, 1684, -3298, -1789, 2648, 6148, 430, -6483, 18, 2844, 2667, -6101, 4473, -4626, -2647, -3033, 9743, -2756, 5372, -4862, 2288, -1077, -4258, -2119, 4741, -5393, 2227, 1947, 2143, -1509, 1310, -2390, 2030, 2591, 256, -757, -1600, -4472, 795, -7575, 13492, 583, -206, 3549, -2321, -2645, -809, 1057, 796, 1625, 2433, 3171, -11128, -5310, -8413, -6744, 16632, -3586, 5525, -3139, 5426, 342, -1115, 2324, -714, -555, -42, -2177, 797, -9076, 1726, 2114, 3079, -2077, -2212, 4599, -2463, 3707, -3354, 1861, -1631, 1351, -1255, 2873, -1656, -1842, -5702, 800, 9426, -8377, 1113, 2871, 7954, -6825, 4522, -4257, 6871, -4269, 2419, 7, 1359, 7810, -1003, -1659, -4512, -3617, -4679, -2778, 8750, 1995, -1761, 4191, -3681, -6987, -1373, 5803, 2334, -1045, -1761, -4350, 7275, 1950, -4501, 1719, 2391, -2500, -2566, -3869, 10203, -2125, 1175, -2682, 5666, -1872, -1480, 882, 2921, 8296, 1757, -7632, -587, -6680, -980, 3683, 3915, 1635, 4775, -1336, -6227, 4208, -1982, -1601, -1478, -626, 358, -2368, 3342, 707, -5355, -1251, 5400, -9676, 3003, 1050, 4301, -7606, 6017, -9837, 11389, 488, -4178, 1661, -484, -1574, -1312, 1664, 1242, 2712, -449, 2549, -1925, 4495, 2301, 1101, -8431, -2382, 3808, -2492, -283, -566, 1511, 885, -3026, 1633, 1637, 996, -4347, 1444, 4943, -1662, -4671, 913, -735, 1781, -771, 231, -7023, 146, 4117, 746, 449, 1872, -2397, 2542, -1298, 945, -1045, 973, 397, 699, 974, -4168, 1893, 627, -861, 396, 4115, -30, -1564, 3609, -6550, -3267, 6491, 3367, -3656, -3314, 4601, 160, 2096, 967, -4759, 897, 3832, 2729, -6305, -2224, 2043, -3297, 2072, 661, -2733, -2243, 2879, 1467, 4905, 740, -8708, -1572, -3793, -690, 4330, -1141, -1602, -2705, 3341, -251, -408, 60, -791, -2556, 405, 1540, -1056, -3744, 4420, 2245, -1616, -910, -2903, 366, 2223, -1301, 1515, 1669, -426, 1489, 1819, -1333, -187, 1690, -969, -976, -319, -5674, 396, -2528, 7671, 537, 1121, 2245, -3465, 994, 2236, -6768, 3955, 5616, -3344, 4501, -5340, 3392, -8274, -2915, 7804, 765, -812, -913, 230, 1678, -1156, 2018, -4427, 4185, 7663, 1518, -15563, 2317, 3864, -1451, -6369, 4151, -1633, 939, -1385, 4653, 2142, -2278, -3624, -2764, -4353, 6762, 1553, 1375, -3287, 718, -581, -2927, -2854, -684, 4745, -4347, 4447, 3845, -4416, -3842, 2385, -3858, 3411, 4744, 3062, -2346, -2319, 7477, -1418, -2737, -2837, 180, -1118, 2611, -1143, -1408, 1766, 3229, -5042, -6847, -536, 6494, -1825, -362, -2285, 3501, 1486, -63, 718, 978, -2211, -130, -3331, -7725, 11629, -696, 533, -2110, -736, -1283, -346, -7343, 9732, 2295, -486, -4293, 3461, 4956, 2593, -4320, -612, -1568, 636, 2100, 668, 1865, 1833, 3560, -8054, -7843, -5072, 1129, 13417, -2814, -3704, 2352, 5501, -4510, 4576, -5437, 5799, 10, 2801, -1166, 66, 6296, 2486, -6983, 1227, 2564, -1115, 5382, -6091, -1120, -9854, -2012, 12953, -1494, -1639, -3957, 5021, -1322, -3557, 5430, 287, -5541, 998, -4109, 7574, -802, 785, 3544, -3650, -7097, -592, -2439, 9016, 2771, 440, -3099, 308, 712, -5357, 1542, 1872, -3150, -7387, -2678, 12135, 3474, -1601, -176, -3341, -8519, 4095, -73, 3847, -8180, -489, -3534, 12266, -7448, 2786, 684, 6313, 986, 1301, -264, -3268, -2934, -1374, 7332, -2408, 3759, -571, 10801, -14288, -4614, 1808, 1144, 1163, -724, -3745, 4200, -374, -268, -3792, 1433, 3752, 6617, -87, -5147, -3526, -7829, -1474, 1694, 7798, 353, 2803, -5568, 2775, 8696, 1569, -56, -9634, 349, 5465, -1414, -4063, 9400, 4213, 2367, -15327, -607, -2731, 8204, -6829, -2753, -6266, 6571, 2506, -1689, -4651, 2331, 4872, -2172, 8535, -3134, -2663, 2206, -153, 1416, -4801, 123, -221, -60, 1242, 2974, -4015, -1215, 1618, 785, -4143, -1386, 5276, -3524, 1403, -1954, 3063, 3819, 1694, -1009, -2535, -1243, 2409, 1582, -3931, 1724, 121, -753, -2349, -160, 1, 2009, -361, 2536, -3190, -1402, -187, 1089, -2056, -78, 1588, 713, -2649, -157, 2370, -1055, -825, 277, 870, -22, 2867, 1953, -4643, 1328, 4275, 817, -1437, 3110, -1130, -1318, 189, 4317, -1168, -319, -1171, -1389, 545, -1352, 3943, 2730, 1029, -1118, -2742, -735, 3870, 448, -3873, -2662, -4095, 956, 7124, -877, 86, -277, 3162, -4796, -3765, -1317, 9778, -2595, 1287, -2333, 4312, -664, 1181, 1537, -553, -1525, 1041, 594, -1867, 644, 5967, -859, -7187, 3050, 1701, -2421, -842, -3611, 3052, 117, 916, -2677, 1071, -2271, 4802, 5397, -438, 1444, -4107, -5872, -601, -211, 5209, -2730, 2079, 171, -2513, 3393, -2845, 1596, -1843, 3969, -6604, 53, 2205, 2513, -2339, 1254, 496, 364, -5571, -14, 3391, -1614, 5399, 2288, -3675, -4901, 8401, -1967, -1782, -1292, 4401, 775, -3977, -438, -3702, 1221, 1719, 331, 1054, 2070, -4698, -398, -1507, -497, 3499, 518, 39, 1646, -1188, 670, -2726, 437, 2053, 2614, -1311, 175, -316, -1706, 159, 1800, -3308, 457, 69, -2145, 2926, 222, -1731, -3288, 2600, -348, 568, -2664, 2840, -1444, 2327, 2415, -5168, -962, 3323, 1942, -1011, -152, -2324, -704, 2267, -1610, 1339, -966, 475, 685, -3563, -2251, 4394, 969, -1560, -1701, 3097, -729, -3174, -1015, 4970, -444, -2505, 3715, -156, -2720, 683, -1007, 3210, 1746, 543, -416, 129, 1086, 2635, 628, -1732, 1226, -1643, 2127, -523, 134, 3271, 5244, -5835, -1144, 3325, 2606, -4054, 209, 2958, -1807, -1742, 3807, -785, -514, -2505, 5549, -4481, 1159, 150, -265, -4641, 155, 2222, -575, -2967, 1564, 4339, -3366, 662, 2414, -644, -537, 4732, 1979, -4278, 732, 4045, -830, -3589, -1270, -938, -2145, 3713, -384, 2489, 1251, -2986, -2430, -251, -3637, 4860, -2442, -1524, 7573, -2480, 2880, 1421, -2057, -1843, 3627, -1341, 229, 377, 3, 463, 516, -260, -229, -595, -3131, 3328, -41, 121, -498, 1407, -601, -641, -3220, 3049, -1609, 854, 701, -743, -3323, 1920, 1812, -174, 3244, 837, -1563, -398, -2718, -568, 407, 3454, 2826, 738, -1738, -1711, -576, -2789, -836, 1562, 1216, -23, -2181, 1753, -56, -5180, 1026, 2707, -1077, 2099, -2495, 2939, -271, 49, -2109, -1737, 2695, 281, 2726, -3470, 652, -2387, 4424, -3416, 1531, 3011, 2120, -6434, -4505, 2434, 2659, -1123, 1262, 2181, -764, -3750, 9, 346, -309, 1010, 4412, -6150, -487, 1360, 1795, -2462, -147, 1296, -1155, -113, -969, 4083, -1044, 1086, -249, -322, -3910, 5832, 2368, -4194, -905, 2303, 312, -1308, 1248, -1547, 472, 1857, 2709, -1461, 474, -1599, -1338, -960, -1469, 2556, -1620, -191, 1387, -1084, -213, -781, 969, -1436, 149, -1901, 2088, -1472, -1914, 1555, 921, -3058, 1676, -401, -3118, 635, 2126, -469, -1341, 745, 1988, 1298, -1782, 80, -2537, 4512, 2464, -4332, -2857, 4302, 1602, -4008, -1859, -3493, -897, 3868, -4026, 2033, -1359, 3870, -536, -3991, -1337, 4462, -1948, -3064, -2376, 4955, -2339, 1238, -1119, 3018, 812, -1628, 2414, -1357, -557, 2194, -810, 786, -1470, -918, -2072, 1509, 638, 1790, 1098, -4683, -2019, 1264, -614, 2565, 155, 3507, -1413, -1117, -2505, 2850, 2464, -2857, -2358, 711, -412, -2088};
static const Q15_T D1LW[D1LW_G * D1LW_HF * D1LW_WF * D1LW_CF * D1LW_COUT] = {5207, 81, -5561, -3212, -335, -4304, 5074, 4680, -3446, 3135, 7804, 7997, -379, 9407, 2529, 3228, -237, 5253, -2689, -1546, 7078, 12648, -25, 662, 1138, 2704, 6936, 6231, 7273, -284, -1712, -3007, 1454, 2965, -783, -1085, -4835, -4945, 3401, 2771, 1323, -307, 5269, 6810, 3395, 2496, 47, 1340, -3476, 761, -6012, -4157, 1307, 4057, -127, -811, 2447, 5217, -1847, -1222, -1059, -1697, 215, 1034, 3949, 8285, 1937, 3932, 9912, -2107, -2843, -1537, -3634, -2024, 1898, -415, -4538, -293, 1052, -242, -4720, -222, 7265, 9216, 2132, -3796, 292, -1816, 5240, 2303, -2392, -320, 428, -5799, 1653, 2843, -12935, -6287, 3011, 1965, -1874, -4450, -368, -319, -2343, 4273, 223, 309, -2369, 253, 76, 828, 0, -3014, -2297, 1219, 4527, -1819, -726, -142, 927, -5967, -9624, -6880, 3732, 3071, 4682, 5106, -1398, 12613, -171, 1411, -6530, 4857, 4994, 4946, 8386, 5016, 19776, 11349, 22379, 13103, 2908, 2607, 8106, 4201, 6821, 5696, -4618, 8051, 98, -5303, 24390, -14079, 5662, 8175, 19644, -20787, 2983, 3193, 14793, 9045, -2064, -385, 8639, 3728, 1336, -213, -6079, -18149, 908, 2254, -14033, 5524, 1470, 1383, -2411, -11444, -360, -7, -16430, -12182, 745, -41, -5480, 23530, 70, 2784, 5211, -19653, 5976, 6661, -5815, 10029, 1616, 5876, -17155, 2738, 146, 99, 12524, 4478, 170, -934, -5480, -3223, -237, 1122, -16594, -15273, 14398, 14766, 2552, -8835, -271, -596, 7017, -14373, 4914, 5060, 11639, -1049, 1368, 1749, 2681, 22253, 2811, -1782, 4693, -10024, 1156, 10274, -4097, -13395, 562, 3986, 18439, -11444, -976, -1677, 4946, -11648, 1190, 1781, 7069, -2261, 252, 2262, 31, -10261, -9782, -6580, -8496, 25197, 2592, -1130, 1323, -805, -763, -1117, 1116, 1067, 6230, 5564, -8199, 1391, 2882, 6062, -1002, -4806, 289, 81, 2965, 3847, 3100, 1848, -2945, 5126, -2470, -1212, -550, 886, 2006, 3808, 1335, 1415, 460, -73, 5129, 2614, 5143, 6355, -5546, 4127, -154, -335, -364, 79, 2237, 1291, 336, 1720, -1697, -1896, -613, 3088, -1367, -1758, 1077, 3195, 1737, 2549, -1272, -431, -1133, 30, 577, -1418, -2964, -2814, -2123, 278, 2875, 4228, -2009, -633, -1411, 1346, 4368, -3000, 1204, 1724, 396, -1718, -1114, 272, -5410, 1089, 445, 1195, -532, -2535, 3362, 3612, 1637, -2792, 865, 1640, 3818, 2455, -320, -739, -560, -1511, 78, 235, 3580, 2211, -356, -2685, -2157, 853, 1890, -184, 2179, -234, -2671, -2538, -4481, 2235, -5419, -3477, 1366, -211, 273, 1172, 1506, 503, -3438, -3513, 834, 2023, 1740, 2121, 2060, -6228, 2993, 4176, 2511, 8644, 10904, 12141, 3816, 7173, 3303, 6953, 11881, -218, -4897, -3899, 441, 10630, -2121, 2334, 7035, 1865, -9497, -5979, 1381, -6962, 1022, 2170, 2781, -986, -522, -456, 3927, -1998, -1234, 2418, -7699, -1758, 3257, 1768, -7856, 5426, 4533, 3972, 1805, -5949, 69, 2570, 2652, -631, -4018, -5824, -3434, -6570, -3024, -2381, 2371, -7789, -4709, -4993, 2121, 5518, -1423, 58, 9551, 41, -1606, -2307, 1763, -10395, 12, 208, -7550, -1480, 2083, -1397, -5330, 1700, -2732, -4684, 8908, -2517, -1021, 1230, -3735, -9314, -2779, -5547, -7780, 2363, -4582, -2066, -4513, -9837, -2561, -3295, -6565, 7916, 325, -1029, -13620, 9082, 10155, 5486, 1389, -109, 6785, 5919, 4955, -9410, -4233, -3651, 1009, -3023, -6174, -1623, -15470, 4475, 6599, 6437, 5014, 9010, -5478, -63, -7839, 3878, 2591, 1754, 15126, 675, 557, 3594, 503, -5738, 2333, 4986, -438, -23, 16414, 10836, -2883, -13820, -5835, -5815, -275, 7970, 1856, 4495, 20431, 5939, -6082, -7248, -16371, 11387, -162, 2869, 7556, -1607, 607, 820, 817, -11426, 9840, 11998, 14636, -11847, -4066, -3423, 12191, 6220, -1113, -1942, 12450, 4623, -2785, -2713, -4855, -12604, -8870, -6898, 4087, 5566, 2147, 496, -10559, -17442, 10278, 10157, 36, 21697, 1549, 2279, -3990, 27484, -1940, -966, -14550, -18123, 5410, 3695, 28388, 526, 5305, 3468, -3903, 4039, -5195, -6503, -4766, -14291, 470, 1726, 15303, -2372, -10116, -13481, 11338, 1698, 10924, 10814, -14079, -9453, -8049, -8069, 2933, 8600, 3705, 1599, 17467, 6665, 11562, 16440, 4746, 12144, 10498, 14002, 13563, -6272, -495, -1867, 5344, 17966, -4533, -3794, 5164, 1768, 5035, 5248, -18574, -1524, -1568, 2900, 1742, -24941, -8486, -12245, -4979, 1296, -570, -734, 4639, 4925, 5583, 6920, 774, -474, 6641, 12320, -2320, 1029, -4664, -2921, -6385, 1388, -1538, 567, -10095, 1250, -6484, -5247, 2295, 3299, 1535, 2725, -5143, -2157, -712, 153, 3443, 2383, 5187, 5173, -4291, -1632, 710, 586, -725, 1800, 732, 2651, -2645, -2811, -175, 1032, -1549, -685, 4788, 3625, 379, -2060, -1193, -1321, 6124, 3396, 216, 185, -2328, -2001, -5682, -5598, 400, -3613, 1949, 1633, 1092, 3697, -2305, -1578, -4139, -4425, -2410, -2223, 137, -1241, 2333, 2568, -3414, -4610, -3070, -1659, -2582, 970, 1462, 1861, 772, 2083, 1024, 2853, 2203, -479, -5557, -6150, -3449, -1022, -480, -789, -4396, 1032, 6829, 2350, -3355, 3480, 2286, -295, 484, -2917, 946, 1244, -566, -3308, -7984, -6250, 755, 3703, 2134, 2435, 7335, -539, -1605, -570, -71, 1175, 14, -959, -336, 5777, 4406, 5467, 356, 2443, 10100, 8522, 3864, 1109, 2361, 3391, -3368, 1324, 1324, 397, 825, -5511, -7950, -7398, -179, -2917, -2190, -2513, -338, 639, 2075, 3540, -2602, 1578, 2185, -1261, 800, 698, -4239, -2490, 118, -1293, -2596, -4237, -1114, -2157, -418, 932, -404, 5609, -67, 2396, -2161, 1154, 5337, 5906, -536, 889, 323, 513, -797, 1757, -4094, -4003, 1788, 2095, 685, 1082, -2517, 30, 1400, -167, 1553, 1161, 3088, 2801, -1721, 1555, 1906, 224, -2222, 391, -4569, -5713, -227, -2942, -1677, -2028, -1286, 3892, -2317, -2380, -292, -3176, -7109, -7813, 1775, 349, -2026, 470, -1088, -1139, -3046, -2215, 1912, -175, 669, 2529, -1825, -137, -849, -2438, -1088, 827, 4087, 5068, 3239, 6242, -4310, -1447, 4358, -4811, -933, -2694, 2279, 267, -3375, -2623, 2472, 3049, 1761, 1594, -698, 1735, 1007, 2380, 2157, 1710, 7795, 10766, -6357, 5645, 4841, -1464, 4038, 263, -768, -1615, 2417, -89, -4606, -7013, 1011, -1598, 2056, -3083, 540, 2248, -162, 3125, 48, 4278, 1990, 1953, 1052, -1001, -392, -423, -873, -1957, -2711, -1908, -2521, -1624, -2584, -2255, 1720, -1203, 239, 1650, 252, 4352, 2627, 6083, -1538, 1621, 352, -734, -997, 3143, -284, 253, 344, -727, -2741, -1250, -1763, 207, -660, -2476, 3, 114, 2153, 1910, -545, -1399, -717, -1966, 773, 5153, -1211, -3430, -428, -181, -1032, -4351, -1138, -8428, -5032, -5573, -1360, 3094, -2028, -5237, -697, -1923, -1617, 256, -542, 807, 2277, -393, -1882, -1057, -1920, 4299, 4215, 1198, 1121, 1245, -3155, 38, 6338, 3784, -3140, -1358, -5191, -6173, -1207, -2825, 706, 1317, 192, 404, 1247, 1151, -2408, -2201, -3046, -4016, -1795, 2729, 1740, 356, -929, 3789, 6079, 5067, 1336, 6436, 4532, 7366, -1306, -544, -154, 633, -4067, -3469, -5304, -4752, 7, -1202, -5700, -5300, 916, 470, 2355, 2050, 71, 1837, 850, 1279, -807, -234, -1593, -2119, 2496, 3748, 203, 1599, 1334, 4028, 1528, 1862, -905, 2924, -2634, -264, 674, 4959, 7340, 6527, 3137, -4571, 2645, 1659, -616, -324, -2394, -3014, -1380, 376, -3604, -3296, -385, 1509, 1665, 2545, 1019, -1850, -38, 673, 3244, -429, 1299, 146, -5022, 1402, 167, -2655, 1049, -88, -2155, -1025, -2186, 207, -389, -1017, -799, 336, -3537, -3117, -446, -1911, -2919, -1071, 1028, 1650, 666, 781, 1040, -2116, 3526, 1199, -142, -1340, -1163, -3179, -1518, -42, 3436, 3587, -182, -767, -3968, -1071, 1446, -4473, 0, -605, -154, 3881, -1130, -283, 2798, -1208, 2092, 1858};
static const Q15_T D1CB[D1CW_COUT] = {15886, 4132, 6713, -25561};
static const Q15_T D1LB[D1LW_COUT] = {-1319, 1403, -16876, 9564};

static const SCALE_T D1_ScaleIn = -13;
static const SCALE_T D1_ScaleOut = 8;

#ifdef SHIFT
  static const SCALE_T D1NW_Scinput = 6;  //64
  static const SCALE_T D1NW_Scoutput = 6; //64
  static const SCALE_T D1NW_Demote = 0;   //1
  static const SCALE_T D1CW_Scinput = 13; //8192
  static const SCALE_T D1CW_Scoutput = 4; //16
  static const SCALE_T D1CW_Demote = 0;   //1
  static const SCALE_T D1CB_Scten = 0;    //1
  static const SCALE_T D1CB_Scvec = 3;    //8
  static const SCALE_T D1CB_Scret = 0;    //1
  static const SCALE_T D1LW_Scinput = 13; //8192
  static const SCALE_T D1LW_Scoutput = 4; //16
  static const SCALE_T D1LW_Demote = 0;   //1
  static const SCALE_T D1LB_Scten = 0;    //1
  static const SCALE_T D1LB_Scvec = 3;    //8
  static const SCALE_T D1LB_Scret = 0;    //1
#else
  static const SCALE_T D1NW_Scinput = 64;
  static const SCALE_T D1NW_Scoutput = 64;
  static const SCALE_T D1NW_Demote = 1;
  static const SCALE_T D1CW_Scinput = 8192;
  static const SCALE_T D1CW_Scoutput = 16;
  static const SCALE_T D1CW_Demote = 1;
  static const SCALE_T D1CB_Scten = 1;
  static const SCALE_T D1CB_Scvec = 8;
  static const SCALE_T D1CB_Scret = 1;
  static const SCALE_T D1LW_Scinput = 8192;
  static const SCALE_T D1LW_Scoutput = 16;
  static const SCALE_T D1LW_Demote = 1;
  static const SCALE_T D1LB_Scten = 1;
  static const SCALE_T D1LB_Scvec = 8;
  static const SCALE_T D1LB_Scret = 1;
#endif
