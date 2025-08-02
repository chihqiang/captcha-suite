import os
import numpy as np
import torch
from PIL import Image
from config import CHARSET, IMAGE_SIZE, OUTPUT_DIR, PY_MODEL_FILE, CODE_MAX_LENGTH
from model import CaptchaModel, device

class CaptchaPredictor:
    def __init__(self, model_path=PY_MODEL_FILE):
        self.image_width, self.image_height = IMAGE_SIZE
        # 创建字符到索引的映射和索引到字符的映射
        self.char_to_index = {char: i for i, char in enumerate(CHARSET)}
        self.index_to_char = {i: char for i, char in enumerate(CHARSET)}
        # 初始化模型
        self.model = CaptchaModel().to(device)
        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()  # 设置为评估模式
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请先训练模型或确保模型文件存在")
            self.model = None

    def preprocess_image(self, image_path):
        """预处理验证码图片"""
        image = Image.open(image_path).convert('L')  # 转为灰度图
        image = image.resize((self.image_width, self.image_height))
        image_array = np.array(image) / 255.0  # 归一化
        image_array = np.expand_dims(image_array, axis=0)  # 扩展维度 (通道)
        image_array = np.expand_dims(image_array, axis=0)  # 扩展维度 (批量大小)
        image_tensor = torch.tensor(image_array, dtype=torch.float32).to(device)
        return image_tensor

    def predict(self, image_path, true_length=None):
        """预测验证码文本
        Args:
            image_path: 图片路径
            true_length: 验证码真实长度(可选)，用于截断预测结果
        """
        if self.model is None:
            print("模型未加载，无法预测")
            return None

        # 预处理图片
        image_tensor = self.preprocess_image(image_path)

        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)

        # 解码预测结果
        captcha_text = ''
        for i in range(CODE_MAX_LENGTH):  # 模型输出固定为6个字符位置
            _, predicted_index = torch.max(output[0, i, :], 0)
            captcha_text += self.index_to_char[predicted_index.item()]

        # 如果提供了真实长度，截断预测结果
        if true_length is not None and true_length < CODE_MAX_LENGTH:
            captcha_text = captcha_text[:true_length]

        return captcha_text

    def predict_batch(self, image_dir, max_samples=10):
        """批量预测验证码"""
        if self.model is None:
            print("模型未加载，无法预测")
            return []

        results = []
        # 获取所有图片文件
        file_list = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        file_list = file_list[:max_samples]  # 限制数量

        for filename in file_list:
            image_path = os.path.join(image_dir, filename)
            # 真实标签（从文件名中提取）
            true_label = filename.split('_')[2].split('.')[0]
            true_length = len(true_label)
            # 预测，传入真实长度以截断结果
            predicted_label = self.predict(image_path, true_length)
            results.append((filename, true_label, predicted_label))
            print(f"文件: {filename}, 真实标签: {true_label}, 预测标签: {predicted_label}")

        return results

if __name__ == "__main__":
    # 创建预测器实例
    predictor = CaptchaPredictor()

    # 如果模型已加载，进行预测
    if predictor.model is not None:
        # 批量预测
        print("开始批量预测...")
        predictor.predict_batch(OUTPUT_DIR, max_samples=10)
    else:
        print("请先训练模型")