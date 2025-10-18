from PIL import Image

image1_path = r"C:\Users\caley\Desktop\srtp_vertebra\data\label_wh\15.png"
image2_path = r"C:\Users\caley\Desktop\srtp_vertebra\data\strp_yolo\sample\15.JPG"
# 加载两张图片
image1 = Image.open(image1_path).convert('RGBA')  # 第一张图片
image2 = Image.open(image2_path).convert('RGBA')  # 第二张图片

# 确保两张图片大小相同（如果不同，需要调整大小）
if image1.size != image2.size:
    image1 = image1.resize(image2.size, Image.Resampling.LANCZOS)

# 黑白反转第一张图片
image1_inverted = Image.eval(image1.convert('L'), lambda x: 255 - x).convert('RGBA')

# 创建一个新的空白图像用于处理
width, height = image1.size
result_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

# 处理像素：黑色变为蓝色，白色变为透明
pixels = image1_inverted.load()
for x in range(width):
    for y in range(height):
        r, g, b, a = pixels[x, y]
        # 假设灰度值 < 128 为黑色，>= 128 为白色
        if r < 128:  # 黑色区域变为蓝色
            pixels[x, y] = (0, 0, 255, a)  # 蓝色，不透明
        else:  # 白色区域变为透明
            pixels[x, y] = (0, 0, 0, 0)  # 完全透明

# 设置50%透明度
alpha_image = Image.new('RGBA', image1.size, (0, 0, 0, 0))
alpha_pixels = alpha_image.load()
for x in range(width):
    for y in range(height):
        r, g, b, a = pixels[x, y]
        alpha_pixels[x, y] = (r, g, b, int(a * 0.5))  # 透明度设为50%

# 叠加图片
final_image = Image.alpha_composite(image2, alpha_image)

# 保存结果
final_image.save('output2.png')

print("图片处理完成，已保存为 output2.png")