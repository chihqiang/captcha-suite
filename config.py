import os
# 验证码字符集（数字 + 大小写字母）
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# 验证码长度范围（最小长度，最大长度）
CODE_MIN_LENGTH = 4
CODE_MAX_LENGTH = 6

# 验证码图片尺寸 (宽度, 高度)
IMAGE_SIZE = (160, 60)
# 干扰线数量
NOISE_LINES = 3
# 干扰点数量
NOISE_DOTS = 20
# 字体大小
FONT_SIZE = 36
# 字体路径（可以使用系统字体，这里使用默认值）
FONT_PATH = None  # 如果需要指定字体，可以设置路径如 '/System/Library/Fonts/PingFang.ttc'
# 验证码图片保存目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'dataset/img')
TF_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'dataset/tf.h5')
TF_MODEL_HISTORY = os.path.join(os.path.dirname(__file__), 'dataset/tf-history.png')
PY_MODEL_FILE = os.path.join(os.path.dirname(__file__), 'dataset/model.pth')
PY_MODEL_HISTORY = os.path.join(os.path.dirname(__file__), 'dataset/history.png')
# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)