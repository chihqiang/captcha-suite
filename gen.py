import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import config

class CaptchaGenerator:
    def __init__(self):
        # 尝试加载指定字体，如果没有指定或加载失败则使用默认字体
        try:
            if config.FONT_PATH is not None:
                self.font = ImageFont.truetype(config.FONT_PATH, config.FONT_SIZE)
            else:
                raise TypeError("Font path is None")
        except (TypeError, IOError, AttributeError):
            # 使用默认字体
            self.font = ImageFont.load_default()
            print("Warning: Could not load specified font, using default.")

    def generate_text(self, length=None):
        """生成随机验证码文本"""
        if length is None:
            # 随机选择长度
            min_len, max_len = config.CODE_MIN_LENGTH, config.CODE_MAX_LENGTH
            length = random.randint(min_len, max_len)
        return ''.join(random.choice(config.CHARSET) for _ in range(length))

    def generate_image(self, text=None):
        """生成验证码图片"""
        if text is None:
            text = self.generate_text()

        width, height = config.IMAGE_SIZE
        # 创建空白图片
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # 绘制文本
        # 使用 textbbox 替代 deprecated 的 textsize
        bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = bbox[2] - bbox[0]  # right - left
        text_height = bbox[3] - bbox[1]  # bottom - top
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), text, font=self.font, fill=(0, 0, 0))

        # 绘制干扰线
        for _ in range(config.NOISE_LINES):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            draw.line(((x1, y1), (x2, y2)), fill=(random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)), width=2)

        # 绘制干扰点
        for _ in range(config.NOISE_DOTS):
            x = random.randint(0, width)
            y = random.randint(0, height)
            draw.point((x, y), fill=(random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)))

        # 添加模糊效果
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        return image, text

    def save_captcha(self, count=10):
        """保存验证码图片到指定目录"""
        for i in range(count):
            image, text = self.generate_image()
            # 保存图片，文件名包含验证码文本
            filename = f"{text}.png"
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            image.save(filepath)
            print(f"Saved captcha to {filepath}")
        return count

if __name__ == "__main__":
    generator = CaptchaGenerator()
    # 生成并保存10个验证码
    generator.save_captcha(200)