"""
7GHz EVM标准配置参数
====================
基于WAIR-D数据集 Around 7GHz频段，匹配3GPP EVM标准。

数据形状推导 (按EVM 1驱4平均后):
|- 1024 AE = Nt[0] x Nt[1] x Nt[2] = 2 x 16 x 32
|- Nt_port = 2 x 16 x 8 = 256 (沿z_elem=4求平均)
|- Nr = 2 x 4 = 8
|- Nf = 52 (sampledCarriers)
|- 3D信道: (256, 8, 52)
|- 展平空域: (2048, 52)
|- 存.npy: (2, 2048, 52) 实部+虚部
"""

# 数据根目录
import os
import numpy as np

# PROJECT_ROOT 解析：params_7GHz.py 位于 test_pre/csi_pre_eval/params/ 目录下
# 因此 dirname 向上 3 层落到 test_pre/ (与 test.py 同级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data_raw')
SCENARIO_FOLDER = os.path.join(DATA_RAW_DIR, 'Dataset', 'data', 'scenario_1')
GENERATED_FOLDER = os.path.join(PROJECT_ROOT, 'data',
                                'generated_scenario_1_6_0_1000_2_16_32_2_4_1_18_52')

# 通信参数 (符合3GPP EVM标准 7GHz)
carrierFreq = '6_0'                 # 6GHz频段 (7GHz EVM配置)
BWGHz = 0.01872                     # 20MHz带宽
subcarriers = 624                   # 52RB * 12子载波
carrierSampleInterval = 12          # PRG size=4 -> 13个subband -> 52采样子载波
sampledCarriers = 52                # 采样后的子载波数

# BS天线配置 (M,N,P,Mg,Ng;Mp,Np) = (32,16,2,1,1;8,16)
Nt = [2, 16, 32]                    # x轴=2双极化, y=16, z=32 (=8端口×4AE)
spacing_t = [0.5, 0.5, 0.8]         # (dH, dV) 波长归一化间距
elements_per_port_z = 4             # 1驱4天线映射
Pattern_t = {'Power': 0}            # 全向天线, 默认0dBm
Basis_t = None                      # 占位，在数据生成器中设置

# UE天线配置 (M,N,P,Mg,Ng;Mp,Np) = (1,4,2,1,1;1,4)
Nr = [2, 4, 1]                      # x轴=2双极化, y=4, z=1
spacing_r = [0.5, 0.5, 0.5]
Basis_r = None                      # 占位，在数据生成器中设置

# EVM平均后的关键维度
NT_PORT = 256                       # = 2 * 16 * (32 // 4) = 256
NR = 8                              # = 2 * 4 = 8
NF = 52

# 数据集划分
scenario = 1
train_envs = 1000                   # 训练: Scenario1前1000个地图
test_generalization_s1 = (1000, 10000)  # Scenario1其他地图 (元组)
test_generalization_s2 = (0, 100)       # Scenario2所有地图 (元组)

# 数据集生成参数
ENVnum = 1000                       # 生成的地图数
BSlist = list(range(5))             # 5个BS
UElist = list(range(30))            # 30个UE
BSnum = len(BSlist)                 # 5
UEnum = len(UElist)                 # 30
maxPathNum = 1000                   # 路径数上限

# 存储设置
saveAsArray = True                  # 存.npy
saveAsImage = False                  # 不存图，避免占用空间

# 数据格式: 统一使用复数 (不再分实部/虚部通道)
# 存: (2048, 52) complex64 -> 读取后 reshape -> (256, 8, 52)
# 模型输入: (B, 2, 256, 8, 52) 实部+虚部 (仅在 DataLoader 中做转换)
COMPLEX_DTYPE = np.complex64         # 存储 dtype

# 原始信道维度
CHANNEL_RAW_SHAPE = (NT_PORT, NR, NF)       # (256, 8, 52) - 复数
CHANNEL_FLAT_SHAPE = (NT_PORT * NR, NF)     # (2048, 52) - 复数展平
NPY_SHAPE = (NT_PORT * NR, NF)              # (2048, 52) - 存复数，不再有通道维 2

# Scenario 2 生成路径
GENERATED_FOLDER_S2 = os.path.join(
    PROJECT_ROOT, 'data',
    f'generated_scenario_2_6_0_100_2_16_32_2_4_1_18_52'
)
