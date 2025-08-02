import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random
from config import CHARSET, IMAGE_SIZE, OUTPUT_DIR, PY_MODEL_FILE,PY_MODEL_HISTORY,CODE_MAX_LENGTH

# 设置中文显示
plt.rcParams["font.family"] = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class CaptchaDataset(Dataset):
    def __init__(self, max_samples=200):  # 数据集只有200张图片
        self.image_width, self.image_height = IMAGE_SIZE
        self.charset_size = len(CHARSET)
        self.char_to_index = {char: i for i, char in enumerate(CHARSET)}
        self.index_to_char = {i: char for i, char in enumerate(CHARSET)}
        
        # 加载数据
        self.images = []
        self.labels = []
        
        # 获取所有验证码图片文件
        file_list = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
        # 限制样本数量
        file_list = file_list[:max_samples]
        random.shuffle(file_list)
        
        for filename in file_list:
            try:
                # 从文件名中提取验证码文本
                captcha_text = filename.split('_')[2].split('.')[0]
                # 确保验证码长度不超过6
                if len(captcha_text) > CODE_MAX_LENGTH:
                    continue

                # 打开图片并转为灰度
                image_path = os.path.join(OUTPUT_DIR, filename)
                image = Image.open(image_path).convert('L')
                # 调整大小
                image = image.resize((self.image_width, self.image_height))
                
                # 数据增强
                # 随机旋转 (-5, 5) 度
                if random.random() > 0.5:
                    angle = random.uniform(-5, 5)
                    image = image.rotate(angle, expand=False)
                
                # 随机平移
                if random.random() > 0.5:
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    # 创建平移矩阵
                    from PIL import ImageChops
                    image = ImageChops.offset(image, dx, dy)
                
                # 随机缩放
                if random.random() > 0.5:
                    scale = random.uniform(0.9, 1.1)
                    new_width = int(self.image_width * scale)
                    new_height = int(self.image_height * scale)
                    image = image.resize((new_width, new_height))
                    image = image.crop((max(0, (new_width - self.image_width) // 2),
                                        max(0, (new_height - self.image_height) // 2),
                                        max(0, (new_width + self.image_width) // 2),
                                        max(0, (new_height + self.image_height) // 2)))
                    image = image.resize((self.image_width, self.image_height))
                
                # 转为数组并归一化
                image_array = np.array(image) / 255.0
                # 调整维度为 (1, height, width) 以匹配 PyTorch 的输入格式
                image_array = np.expand_dims(image_array, axis=0)

                # 处理标签
                label = np.zeros((CODE_MAX_LENGTH, self.charset_size))
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

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        self.image_width, self.image_height = IMAGE_SIZE
        self.charset_size = len(CHARSET)
        
        # 定义网络结构
        self.conv_layers = nn.Sequential(
            # 卷积层1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 卷积层2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 卷积层3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 卷积层4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 计算卷积后的输出大小
        conv_output_size = self._get_conv_output_size()
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, CODE_MAX_LENGTH * self.charset_size)
        )

    def _get_conv_output_size(self):
        """计算卷积层输出的大小"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.image_height, self.image_width)
            output = self.conv_layers(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        x = x.view(x.size(0), CODE_MAX_LENGTH, self.charset_size)  # 重塑为 (批量大小, 6, 字符集大小)
        return x

class CaptchaTrainer:
    def __init__(self):
        self.model = CaptchaModel().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.char_to_index = {char: i for i, char in enumerate(CHARSET)}
        self.index_to_char = {i: char for i, char in enumerate(CHARSET)}

    def train(self, epochs=150, batch_size=16, validation_split=0.2):  # 进一步增加epochs
        """训练模型"""
        # 加载数据
        dataset = CaptchaDataset()
        # 划分训练集和验证集
        train_size = int(len(dataset) * (1 - validation_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 训练历史
        history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_correct = 0
            train_total = 0
            train_loss = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = self.model(images)
                # 计算损失
                loss = 0
                for i in range(CODE_MAX_LENGTH):
                    loss += self.criterion(outputs[:, i, :], torch.argmax(labels[:, i, :], dim=1))
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 计算准确率
                for i in range(CODE_MAX_LENGTH):
                    _, predicted = torch.max(outputs[:, i, :], 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == torch.argmax(labels[:, i, :], dim=1)).sum().item()

                train_loss += loss.item()

            # 计算平均训练损失和准确率
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total / CODE_MAX_LENGTH  # 每个验证码有6个字符

            # 验证阶段
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # 前向传播
                    outputs = self.model(images)
                    # 计算损失
                    loss = 0
                    for i in range(CODE_MAX_LENGTH):
                        loss += self.criterion(outputs[:, i, :], torch.argmax(labels[:, i, :], dim=1))

                    # 计算准确率
                    for i in range(CODE_MAX_LENGTH):
                        _, predicted = torch.max(outputs[:, i, :], 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == torch.argmax(labels[:, i, :], dim=1)).sum().item()

                    val_loss += loss.item()

            # 计算平均验证损失和准确率
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total / CODE_MAX_LENGTH  # 每个验证码有6个字符

            # 记录历史
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # 学习率衰减
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, 学习率: {current_lr:.6f}')

        # 保存模型
        torch.save(self.model.state_dict(), PY_MODEL_FILE)
        print(f"模型已保存为 {PY_MODEL_FILE}")

        # 绘制训练历史
        self.plot_history(history)

        return history

    def plot_history(self, history):
        """绘制训练历史"""
        # 绘制准确率曲线
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='训练准确率')
        plt.plot(history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel(' epochs ')
        plt.ylabel('准确率')
        plt.legend()

        # 绘制损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel(' epochs ')
        plt.ylabel('损失')
        plt.legend()

        plt.tight_layout()
        plt.savefig(PY_MODEL_HISTORY)
        print(f"训练历史已保存为 {PY_MODEL_HISTORY}")
        plt.close()

if __name__ == "__main__":
    # 创建训练器实例
    trainer = CaptchaTrainer()
    # 打印模型结构
    print(trainer.model)
    # 开始训练
    trainer.train()