# captcha-suite

一款基于 Python 的验证码全流程工具包，集验证码生成、深度学习模型训练与验证码识别于一体。

## 功能特点
- 生成不同长度的验证码图片（4-6个字符）
- 支持数字和大小写字母
- 包含干扰线和干扰点增加识别难度
- 基于深度学习的验证码识别模型
- 简单易用的预测接口

## 文件结构
- `config.py`: 验证码和模型配置
- `gen.py`: 验证码生成脚本
- `train.py`: PyTorch 模型训练脚本
- `predict.py`: PyTorch 验证码识别脚本
- `model.py`: 模型定义文件（包含共享的网络结构）
- `requirements.txt`: 依赖库列表
- `dataset/`: 数据集目录
  - `img/`: 生成的验证码图片保存目录
  - `model.pth`: PyTorch 模型文件
  - `history.png`: PyTorch 训练历史图表
  - `history.npy`: 训练历史数据文件

## 环境搭建

1. Python 环境

```
➜  ~ python3
Python 3.13.5 (main, Jun 11 2025, 15:36:57) [Clang 17.0.0 (clang-1700.0.13.3)] on darwin
```

2. 安装依赖库
 ```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
 ```
 > 注意：requirements.txt 包含了 PyTorch 依赖。

## 使用方法

### 1. 生成验证码
```bash
python gen.py
```
- 生成的验证码图片会保存在 `dataset/img/` 目录下
- 可以修改 `config.py` 中的参数来调整验证码的样式、长度等

### 2. 训练识别模型

```bash
python train.py
```
- 训练前确保 `dataset/img/` 目录下有足够的验证码图片
- 训练完成后，模型会保存为 `dataset/model.pth`
- 训练历史图表会保存为 `dataset/history.png`
- 训练历史数据会保存为 `dataset/history.npy`

### 3. 识别验证码

```bash
python predict.py
```
- 该脚本会识别 `dataset/img/` 目录下的验证码图片
- 可以修改代码中的路径来识别指定的验证码图片

## 注意事项
1. 训练模型需要足够的验证码样本，建议至少生成1000张图片
2. 模型训练可能需要一定时间，具体取决于计算机性能
3. 可以根据实际需求调整模型结构和训练参数以提高识别准确率

## 改进方向
1. 增加更多的验证码样式（如扭曲、旋转等）
2. 优化模型结构提高识别准确率
4. 支持更多类型的验证码（如中文验证码）