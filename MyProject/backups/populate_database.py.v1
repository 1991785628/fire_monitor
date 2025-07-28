import os
import numpy as np
from PIL import Image
from database import ImageDatabase

def get_all_image_paths(directories):
    """获取多个目录中的所有图像路径"""
    image_paths = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"警告: 目录不存在 - {directory}")
            continue
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(directory, filename))
    return image_paths

def populate_database_with_samples(db, fire_dirs, non_fire_dirs):
    """使用多个数据集的样本图像填充数据库"""
    # 获取所有图像路径
    fire_paths = get_all_image_paths(fire_dirs)
    non_fire_paths = get_all_image_paths(non_fire_dirs)
    
    # 随机打乱并限制最大样本数，确保类别平衡
    max_samples = 500  # 每个类别最大样本数
    np.random.seed(42)
    np.random.shuffle(fire_paths)
    np.random.shuffle(non_fire_paths)
    
    # 平衡类别数量
    sample_count = min(len(fire_paths), len(non_fire_paths), max_samples)
    fire_paths = fire_paths[:sample_count]
    non_fire_paths = non_fire_paths[:sample_count]
    
    # 插入火灾图像
    print("开始插入火灾图像...")
    fire_count = 0
    for file_path in fire_paths:
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                if img.mode == 'RGB':
                    channels = 3
                elif img.mode == 'RGBA':
                    channels = 4
                elif img.mode == 'L':
                    channels = 1
                else:
                    channels = 3  # 默认视为RGB
            if db.insert_image(file_path, category='fire', width=width, height=height, channels=channels):
                fire_count += 1
        except Exception as e:
            print(f"处理图像 {file_path} 时出错: {e}")
    print(f"完成插入火灾图像，共插入 {fire_count} 张图像")

    # 插入非火灾图像
    print("开始插入非火灾图像...")
    non_fire_count = 0
    for file_path in non_fire_paths:
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                if img.mode == 'RGB':
                    channels = 3
                elif img.mode == 'RGBA':
                    channels = 4
                elif img.mode == 'L':
                    channels = 1
                else:
                    channels = 3  # 默认视为RGB
            if db.insert_image(file_path, category='non_fire', width=width, height=height, channels=channels):
                non_fire_count += 1
        except Exception as e:
            print(f"处理图像 {file_path} 时出错: {e}")
    print(f"完成插入非火灾图像，共插入 {non_fire_count} 张图像")

    return fire_count, non_fire_count

if __name__ == "__main__":
    # 创建数据库连接
    db = ImageDatabase()

    # 设置图像目录路径 - 从多个数据集采样
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'fire_detection_data')
    
    # 从train、val、test三个数据集采样
    fire_dirs = [
        os.path.join(data_dir, 'train', 'fire'),
        os.path.join(data_dir, 'val', 'fire'),
        os.path.join(data_dir, 'test', 'fire')
    ]
    non_fire_dirs = [
        os.path.join(data_dir, 'train', 'non_fire'),
        os.path.join(data_dir, 'val', 'non_fire'),
        os.path.join(data_dir, 'test', 'non_fire')
    ]
    
    # 合并所有数据集的图像路径
    fire_dir = fire_dirs
    non_fire_dir = non_fire_dirs

    # 验证目录是否存在
    def validate_directories(directories, category):
        valid_dirs = []
        for dir_path in directories:
            if os.path.exists(dir_path):
                valid_dirs.append(dir_path)
            else:
                print(f"警告: {category}图像目录不存在 - {dir_path}")
        if not valid_dirs:
            print(f"错误: 未找到任何有效的{category}图像目录")
            return False
        return True

    # 验证火灾和非火灾图像目录
    if not validate_directories(fire_dirs, "火灾") or not validate_directories(non_fire_dirs, "非火灾"):
        print("请检查目录结构或修改代码中的路径")
        db.close()
        exit(1)

    # 填充数据库
    fire_count, non_fire_count = populate_database_with_samples(db, fire_dir, non_fire_dir)

    # 显示统计信息
    total = fire_count + non_fire_count
    print(f"\n数据库填充完成！")
    print(f"总插入图像: {total} 张")
    print(f"火灾图像: {fire_count} 张")
    print(f"非火灾图像: {non_fire_count} 张")

    # 关闭数据库连接
    db.close()