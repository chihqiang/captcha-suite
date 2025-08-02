import os
# 验证码生成与识别系统配置文件

# 验证码字符集（数字 + 大小写字母）
# 用于生成验证码文本和训练模型时的字符映射
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# 验证码长度范围（最小长度，最大长度）
# 生成验证码时会随机选择在此范围内的长度
CODE_MIN_LENGTH = 4
CODE_MAX_LENGTH = 6

# 验证码图片尺寸 (宽度, 高度)
# 生成的验证码图片将保持此尺寸
IMAGE_SIZE = (160, 60)

# 干扰线数量
# 验证码图片中添加的干扰线数量，用于增加识别难度
NOISE_LINES = 3

# 干扰点数量
# 验证码图片中添加的干扰点数量，用于增加识别难度
NOISE_DOTS = 20

# 字体大小
# 验证码中文字的字体大小
FONT_SIZE = 36

# 字体路径
# 可以使用系统字体，这里使用默认值
# 如果需要指定字体，可以设置路径如 '/System/Library/Fonts/PingFang.ttc'
FONT_PATH = None

# 验证码图片保存目录
# 生成的验证码图片将保存在此目录下
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'dataset/img')

# PyTorch模型文件保存路径
# 训练完成的模型将保存到此路径
PY_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'dataset/model.pth')

# 训练历史图表保存路径
# 训练完成后生成的准确率和损失曲线图表将保存到此路径
PY_MODEL_HISTORY = os.path.join(os.path.dirname(__file__), 'dataset/history.png')

# 训练历史数据保存路径
# 训练过程中的准确率和损失数据将保存到此路径
PY_MODEL_HISTORY_DATA = os.path.join(os.path.dirname(__file__), 'dataset/history.npy')

# 训练参数
# 训练轮数
TRAIN_EPOCHS = 150
# 训练批次大小
TRAIN_BATCH_SIZE = 16
# 训练数据加载器工作进程数
TRAIN_NUM_WORKERS = 4
# 验证数据加载器工作进程数
VAL_NUM_WORKERS = 4

# 确保输出目录存在
# 如果目录不存在，则创建该目录
os.makedirs(OUTPUT_DIR, exist_ok=True)