import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import random
import config
import os
from model import CaptchaModel, device

# 设置中文显示
plt.rcParams["font.family"] = []

class CaptchaDataset(Dataset):
    def __init__(self, max_samples=None):  # 默认使用所有可用样本
        self.image_width, self.image_height = config.IMAGE_SIZE
        self.charset_size = len(config.CHARSET)
        self.char_to_index = {char: i for i, char in enumerate(config.CHARSET)}
        self.index_to_char = {i: char for i, char in enumerate(config.CHARSET)}
        
        # 加载数据
        self.images = []
        self.labels = []
        
        # 获取所有验证码图片文件
        file_list = [f for f in os.listdir(config.OUTPUT_DIR) if f.endswith('.png')]
        # 限制样本数量（如果指定）
        if max_samples is not None:
            file_list = file_list[:max_samples]
        random.shuffle(file_list)
        
        for filename in file_list:
            try:
                # 从文件名中提取验证码文本
                captcha_text = filename.split('.')[0]
                # 确保验证码长度不超过最大长度
                if len(captcha_text) > config.CODE_MAX_LENGTH:
                    continue

                # 打开图片并转为灰度
                image_path = os.path.join(config.OUTPUT_DIR, filename)
                image = Image.open(image_path).convert('L')
                # 调整大小
                image = image.resize((self.image_width, self.image_height))
                
                # 数据增强
                # 随机旋转 (-5, 5) 度 (减小旋转范围)
                if random.random() > 0.5:
                    angle = random.uniform(-5, 5)
                    image = image.rotate(angle, expand=False, fillcolor=255)
                
                # 随机平移 (减小平移范围)
                if random.random() > 0.5:
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    from PIL import ImageChops
                    image = ImageChops.offset(image, dx, dy)
                    # 填充平移后的空白区域为白色
                    image = ImageOps.expand(image, border=3, fill=255)
                    image = image.resize((self.image_width, self.image_height))
                
                # 随机缩放 (减小缩放范围)
                if random.random() > 0.5:
                    scale = random.uniform(0.9, 1.1)
                    new_width = int(self.image_width * scale)
                    new_height = int(self.image_height * scale)
                    image = image.resize((new_width, new_height))
                    # 裁剪或填充到原始大小
                    if new_width > self.image_width or new_height > self.image_height:
                        left = max(0, (new_width - self.image_width) // 2)
                        top = max(0, (new_height - self.image_height) // 2)
                        right = min(new_width, left + self.image_width)
                        bottom = min(new_height, top + self.image_height)
                        image = image.crop((left, top, right, bottom))
                    else:
                        image = ImageOps.expand(image, border=((self.image_width - new_width) // 2,
                                                              (self.image_height - new_height) // 2),
                                                fill=255)
                    image = image.resize((self.image_width, self.image_height))
                
                # 随机对比度调整
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Contrast(image)
                    factor = random.uniform(0.8, 1.5)
                    image = enhancer.enhance(factor)
                
                # 随机亮度调整
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Brightness(image)
                    factor = random.uniform(0.8, 1.2)
                    image = enhancer.enhance(factor)
                
                # 随机添加噪声
                if random.random() > 0.5:
                    noise_level = random.uniform(0.01, 0.05)
                    image_array = np.array(image)
                    noise = np.random.normal(0, 255 * noise_level, image_array.shape)
                    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
                    image = Image.fromarray(noisy_image)
                
                # 转为数组并归一化
                image_array = np.array(image) / 255.0
                # 调整维度为 (1, height, width) 以匹配 PyTorch 的输入格式
                image_array = np.expand_dims(image_array, axis=0)

                # 处理标签
                label = np.zeros((config.CODE_MAX_LENGTH, self.charset_size))
                for i, char in enumerate(captcha_text):
                    if char in self.char_to_index:
                        label[i, self.char_to_index[char]] = 1
                # 对于长度不足6的验证码，其余位置填充0

                self.images.append(image_array)
                self.labels.append(label)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

class CaptchaTrainer:
    def __init__(self):
        self.model = CaptchaModel().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        self.char_to_index = {char: i for i, char in enumerate(config.CHARSET)}
        self.index_to_char = {i: char for i, char in enumerate(config.CHARSET)}
        self.best_val_acc = 0.0

    def train(self, epochs=10, batch_size=16, validation_split=0.2):
        """训练模型"""
        # 加载数据（使用所有生成的验证码）
        dataset = CaptchaDataset(max_samples=None)
        # 划分训练集和验证集
        train_size = int(len(dataset) * (1 - validation_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        # 调整工作进程数
        train_num_workers =config.TRAIN_NUM_WORKERS
        val_num_workers = config.VAL_NUM_WORKERS
        # 创建数据加载器
        print(f"🔄 创建训练数据加载器: 批次大小={batch_size}, 工作进程数={train_num_workers}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_num_workers)
        print(f"✅ 训练数据加载器创建完成")
        
        print(f"🔄 创建验证数据加载器: 批次大小={batch_size}, 工作进程数={val_num_workers}")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=val_num_workers)
        print(f"✅ 验证数据加载器创建完成")

        print(f"🚀 训练开始: 共{epochs}个epoch, 批大小{batch_size}", flush=True)
        print(f"📊 数据集信息: 训练集大小={len(train_dataset)}, 验证集大小={len(val_dataset)}", flush=True)

        # 训练历史
        history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }

        for epoch in range(epochs):
            # 当前训练轮次（从1开始）
            current_epoch = epoch + 1
            # 训练阶段
            self.model.train()
            train_correct = 0
            train_total = 0
            train_loss = 0

            total_batches = len(train_loader)
            # 计算整个训练过程的总批次数
            total_training_batches = epochs * total_batches
            for batch_idx, (images, labels) in enumerate(train_loader):
                # 打印batch进度
                # 每10个批次打印一次批次进度
                if batch_idx % 10 == 0:
                    # 计算当前已完成的全局批次数
                    completed_batches = epoch * total_batches + batch_idx
                    print(f"🔄 训练进度: 第 {current_epoch}/{epochs} 轮, 批次 {batch_idx}/{total_batches} (全局: {completed_batches}/{total_training_batches})", flush=True)
                images = images.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = self.model(images)
                # 计算损失
                loss = 0
                for i in range(config.CODE_MAX_LENGTH):
                    loss += self.criterion(outputs[:, i, :], torch.argmax(labels[:, i, :], dim=1))
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 打印每张图片的详细预测结果
                train_loss += loss.item()
                batch_correct = 0
                batch_total = 0

                # 每10个批次打印一次详细预测结果
                if batch_idx % 10 == 0:
                    print(f"📊 第 {current_epoch} 轮 批次 {batch_idx} 损失: {loss.item():.4f}")
                    # 打印前3张图片的详细预测结果
                    print_idx = 0

                for img_idx in range(labels.size(0)):
                    img_correct = True
                    predicted_code = []
                    true_code = []
                    char_details = []

                    for i in range(config.CODE_MAX_LENGTH):
                        _, predicted = torch.max(outputs[img_idx, i, :], 0)
                        true_label = torch.argmax(labels[img_idx, i, :], 0)
                        predicted_code.append(str(predicted.item()))
                        true_code.append(str(true_label.item()))

                        char_status = '✓' if predicted == true_label else '✗'
                        char_details.append(f"字符{i+1}: 预测={predicted.item()}, 真实={true_label.item()}, {char_status}")

                        if predicted != true_label:
                            img_correct = False

                    # 只统计完全正确的验证码
                    if img_correct:
                        batch_correct += 1
                    batch_total += 1

                    # 每10个批次打印前3张图片的详细结果
                    if batch_idx % 10 == 0 and print_idx < 3:
                        print(f"🔍 第 {current_epoch} 轮 批次 {batch_idx} - 样本 {img_idx} 预测详情:")
                        print(f"   预测序列: {' '.join(predicted_code)}")
                        print(f"   真实序列: {' '.join(true_code)}")
                        print(f"   结果: {'✅ 正确' if img_correct else '❌ 错误'}")
                        for detail in char_details:
                            print(f"   {detail}")
                        print_idx += 1

                train_correct += batch_correct
                train_total += batch_total

            # 计算训练准确率和平均损失
            train_acc = train_correct / train_total
            train_avg_loss = train_loss / len(train_loader)
            history['accuracy'].append(train_acc)
            history['loss'].append(train_avg_loss)

            print(f"📈 训练结果: 准确率={train_acc:.4f}, 损失={train_avg_loss:.4f}", flush=True)

            # 验证阶段
            val_correct = 0
            val_total = 0
            val_loss = 0

            self.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = self.model(images)
                    loss = 0
                    for i in range(config.CODE_MAX_LENGTH):
                        loss += self.criterion(outputs[:, i, :], torch.argmax(labels[:, i, :], dim=1))

                    val_loss += loss.item()

                    for img_idx in range(labels.size(0)):
                        img_correct = True
                        for i in range(config.CODE_MAX_LENGTH):
                            _, predicted = torch.max(outputs[img_idx, i, :], 0)
                            true_label = torch.argmax(labels[img_idx, i, :], 0)
                            if predicted != true_label:
                                img_correct = False
                                break

                        if img_correct:
                            val_correct += 1
                        val_total += 1

            val_acc = val_correct / val_total
            val_avg_loss = val_loss / len(val_loader)
            history['val_accuracy'].append(val_acc)
            history['val_loss'].append(val_avg_loss)

            print(f"🔍 验证结果: 准确率={val_acc:.4f}, 损失={val_avg_loss:.4f}", flush=True)

            # 调整学习率
            self.scheduler.step(val_acc)

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"💾 当前模型是成功的模型: (准确率达到={val_acc:.4f})")

        torch.save(self.model.state_dict(), config.PY_MODEL_FILE)
        print(f"💾 模型已保存: {config.PY_MODEL_FILE}")
        # 绘制训练历史
        self.plot_history(history)
        # 保存训练历史
        np.save(config.PY_MODEL_HISTORY_DATA, history)
        print(f"✅ 训练完成! 最佳验证准确率: {self.best_val_acc:.4f}")
        return history


    def plot_history(self, history):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))

        # 绘制准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='训练准确率')
        plt.plot(history['val_accuracy'], label='验证准确率')
        plt.title('准确率曲线')
        plt.xlabel('epoch')
        plt.ylabel('准确率')
        plt.legend()

        # 绘制损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('epoch')
        plt.ylabel('损失')
        plt.legend()

        plt.tight_layout()
        plt.savefig(config.PY_MODEL_HISTORY)
        print(f"📊 训练历史图表已保存为 {config.PY_MODEL_HISTORY}")

if __name__ == "__main__":
    trainer = CaptchaTrainer()
    trainer.train(epochs=config.TRAIN_EPOCHS, batch_size=config.TRAIN_BATCH_SIZE)