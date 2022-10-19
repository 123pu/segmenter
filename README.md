# segmenter
#### 介绍

应用mindspore深度学习框架实现segmenter分割模型复现

#### 使用说明

1. 支持硬件Ascend910-npu模型训练，该模型训练方式是应用华为ModelArts平台实现
2. 模型参数配置可参考segmenter.yaml文件

#### 代码目录结构说明

model_zoo

├── segmenter                             # segmenter模型根目录

│       ├── README.md                        # 模型说明文档

│       ├── requirements.txt                   # 依赖说明文件

│       ├── src                                # 模型定义源码目录

│       │   ├── data                          # 数据集处理目录

│       │   │  ├── database.py               # 数据集解析定义

│       │   │  ├── dataset.py                # 数据集处理定义

│       │   │  ├── dataloader.py              # 数据集加载定义

│       │   ├── loss                          # 模型损失函数目录

│       │   │  ├── loss.py                    # 损失函数定义

│       │   ├── nets                          # 模型结构目录

│       │   │  ├── vit.py                     # 模型backbone结构定义

│       │   │  ├── block.py                   # 模型transformer结构定义

│       │   │  ├── encoder.py                 # 模型解码器结构定义

│       │   │  ├── factory.py                  # 模型结构定义

│       │   │  ├── segmenter.py               # 创建网络结构定义

│       │   ├── utils                          # 模型通用工具

│       │   │  ├── learning_rates.py            # 学习率设置

│       │   │  ├── optimizer.py                # 优化器设置

│       │   ├── config.py                           # 模型参数配置文件加载

│       │   ├── segmenter.yaml                      # 模型参数配置文件

│       │   ├── eval.py                             # 模型评估脚本

│       │   ├── train_net.py                         # 训练脚本

#### 模型训练--昇腾ModelArts平台

1. 创建算法----设置启动文件train_net.py
2. 训练作业---关联创建算法，配置Ascend910环境及mindspore版本，进行模型训练

#### 

