from modulefinder import test
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import shutil
from typing import Tuple, List, Dict, Any

def preprocess_image(image_array, target_size=(224, 224)):
    """预处理图像用于模型输入"""
    # 移除所有多余单维度
    image = tf.squeeze(image_array)
    
    # 处理不同通道数的图像
    if image.shape[-1] == 1:
        # 灰度图像转RGB
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 4:
        # RGBA转RGB
        image = tf.image.rgba_to_rgb(image)
    
    # 调整图像大小
    image = tf.image.resize(image, target_size)
    # 归一化像素值到[0, 1]
    image = image / 255.0
    # 添加批次维度
    image = tf.expand_dims(image, axis=0)
    return image


def load_multimodal_data(image_path: str, sensor_data_path: str) -> Tuple[tf.Tensor, Dict[str, Any]]:
    """加载并预处理多模态数据（图像+传感器数据）
    
    Args:
        image_path: 图像文件路径
        sensor_data_path: 传感器数据CSV文件路径
        
    Returns:
        Tuple包含预处理后的图像张量和传感器数据字典
    """
    # 加载和预处理图像
    img = load_img(image_path)
    img_array = img_to_array(img)
    processed_img = preprocess_image(img_array)
    
    # 加载传感器数据
    sensor_df = pd.read_csv(sensor_data_path)
    
    # 提取图像文件名（不含扩展名）作为键来查找对应的传感器数据
    img_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # 查找对应的传感器数据
    sensor_data = sensor_df[sensor_df['image_id'] == img_filename].to_dict('records')
    
    if not sensor_data:
        raise ValueError(f"未找到图像 {img_filename} 对应的传感器数据")
    
    # 只返回第一条匹配的数据
    return processed_img, sensor_data[0]

#数据收集路径
data_dir = r"D:/MyProject/fire_monitor/fire_detection_data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir =os.path.join(data_dir, "test")

#创建训练集，验证集和测试集目录
os.makedirs(os.path.join(train_dir, "fire"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "non_fire"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "fire"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "non_fire"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "fire"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "non_fire"), exist_ok=True)
# 创建传感器数据目录
sensor_data_dir = os.path.join(data_dir, "sensor_data")
os.makedirs(sensor_data_dir, exist_ok=True)

#数据划分比例
train_ratio = 0.5
val_ratio = 0.25
test_ratio = 0.25

#加载所有图像路径
def load_image_paths():
    # 加载原始图像路径
    fire_images = [os.path.join(data_dir, "all_fire_images", f) 
                  for f in os.listdir(os.path.join(data_dir, "all_fire_images")) 
                  if f.endswith((".jpg", ".jpeg", ".png"))]
    
    non_fire_images = [os.path.join(data_dir, "all_non_fire_images", f) 
                      for f in os.listdir(os.path.join(data_dir, "all_non_fire_images")) 
                      if f.endswith((".jpg", ".jpeg", ".png"))]
    
    print(f"原始数据: 火灾样本 {len(fire_images)} 张, 非火灾样本 {len(non_fire_images)} 张")

    #平衡数据集
    max_samples = min(len(fire_images), len(non_fire_images), 1000)

    if len(fire_images) > max_samples:
        np.random.seed(42)
        fire_images = np.random.choice(fire_images, max_samples, replace=False).tolist()

    if len(non_fire_images) > max_samples:
        np.random.seed(42)
        non_fire_images = np.random.choice(non_fire_images, max_samples, replace=False).tolist()
    
    #对于样本不足的类别，进行数据增强
    if len(fire_images) < max_samples:
        needed_samples = max_samples - len(fire_images)
        fire_images.extend(generate_augmented_images(fire_images, needed_samples, "fire"))

    if len(non_fire_images) < max_samples:
        needed_samples = max_samples - len(non_fire_images)
        non_fire_images.extend(generate_augmented_images(non_fire_images, needed_samples, "non_fire"))

    print(f"处理后数据: 火灾样本 {len(fire_images)} 张, 非火灾样本 {len(non_fire_images)} 张")
    print(f"类别比例: {len(fire_images) / len(non_fire_images):.2f}:1")

    return fire_images, non_fire_images

#生成增强图像
def generate_augmented_images(image_paths, needed_samples, class_name):
    datagen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        brightness_range = [0.8, 1.2],
        fill_mode = 'nearest'
    )

    augmented_paths = []
    temp_dir = os.path.join(data_dir, f"temp_{class_name}")
    os.makedirs(temp_dir, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        if len(augmented_paths) >= needed_samples:
            break

        img = tf.keras.preprocessing.image.load_img(img_path)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis = 0)
        
        #生成增强图像
        for j, batch in enmurate(datagen.flow(
                                img_array, 
                                batch_size = 1, 
                                save_to_dir = temp_dir,
                                save_prefix = f"aug_{i}",
                                save_format = 'jpg')):
            augmented_paths.append(os.path.join(temp_dir, f"aug_{i}_{j}.jpg"))
            if len(augmented_paths) >= needed_samples:
                break

    return augmented_paths

#划分数据集
def split_dataset(fire_images, non_fire_images):
    np.random.seed(42)
    np.random.shuffle(fire_images)
    np.random.shuffle(non_fire_images)

    # 确保两类样本在各集合中的比例一致（按较少的类别数量对齐）
    min_samples = min(len(fire_images), len(non_fire_images))
    fire_images = fire_images[:min_samples]  # 截断较多的类别，保持比例1:1
    non_fire_images = non_fire_images[:min_samples]

    #计算划分点
    fire_total = len(fire_images)
    non_fire_total = len(non_fire_images)

    fire_train_idx = int(fire_total * train_ratio)
    fire_val_idx = int(fire_total * (train_ratio + val_ratio))

    non_fire_train_idx = int(non_fire_total * train_ratio)
    non_fire_val_idx = int(non_fire_total * (train_ratio + val_ratio))

    #划分数据集
    fire_train = fire_images[:fire_train_idx]
    fire_val = fire_images[fire_train_idx:fire_val_idx]
    fire_test = fire_images[fire_val_idx:]

    non_fire_train = non_fire_images[:non_fire_train_idx]
    non_fire_val = non_fire_images[non_fire_train_idx:non_fire_val_idx]
    non_fire_test = non_fire_images[non_fire_val_idx:]

    #打印分层采样结果
    print("分层采样结果：")
    print(f"训练集：火灾图像{len(fire_train)}张，非火灾图像{len(non_fire_train)}张")
    print(f"验证集：火灾图像{len(fire_val)}张，非火灾图像{len(non_fire_val)}张")
    print(f"测试集：火灾图像{len(fire_test)}张，非火灾图像{len(non_fire_test)}张")

    return fire_train, fire_val, fire_test, non_fire_train, non_fire_val, non_fire_test

#复制图像至相应目录
def copy_images_to_dir(fire_train, fire_val, fire_test, non_fire_train, non_fire_val, non_fire_test):
    import shutil

    #清空并重新创建目录
    for dir_path in [train_dir, val_dir, test_dir]:
        for sub_dir in ["fire", "non_fire"]:
            full_path = os.path.join(dir_path, sub_dir)
            shutil.rmtree(full_path)
            os.makedirs(full_path, exist_ok=True)

    #复制火灾图像
    for i, img_path in enumerate(fire_train):
        shutil.copy(img_path, os.path.join(train_dir, "fire", f"fire_train_{i}.jpg"))
    for i, img_path in enumerate(fire_val):
        shutil.copy(img_path, os.path.join(val_dir, "fire", f"fire_val_{i}.jpg"))
    for i, img_path in enumerate(fire_test):
        shutil.copy(img_path, os.path.join(test_dir, "fire", f"fire_test_{i}.jpg"))
    
    #复制非火灾图像
    for i, img_path in enumerate(non_fire_train):
        shutil.copy(img_path, os.path.join(train_dir, "non_fire", f"non_fire_train_{i}.jpg"))
    for i, img_path in enumerate(non_fire_val):
        shutil.copy(img_path, os.path.join(val_dir, "non_fire", f"non_fire_val_{i}.jpg"))
    for i, img_path in enumerate(non_fire_test):
        shutil.copy(img_path, os.path.join(test_dir, "non_fire", f"non_fire_test_{i}.jpg"))
    
    print(f"训练集：火灾图像{len(fire_train)}张，非火灾图像{len(non_fire_train)}张")
    print(f"验证集：火灾图像{len(fire_val)}张，非火灾图像{len(non_fire_val)}张")
    print(f"测试集：火灾图像{len(fire_test)}张，非火灾图像{len(non_fire_test)}张")

#创建多模态数据生成器
def create_multimodal_generators(image_size = (224, 224), batch_size = 32, augment_training_data = False, sensor_data_path=None):
    """创建多模态数据生成器（图像+传感器数据）
    
    Args:
        image_size: 图像大小
        batch_size: 批次大小
        augment_training_data: 是否对训练数据进行增强
        sensor_data_path: 传感器数据CSV文件路径
        
    Returns:
        训练、验证、测试数据生成器
    """
    # 训练集数据增强配置
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20 if augment_training_data else 0,
        width_shift_range=0.2 if augment_training_data else 0,
        height_shift_range=0.2 if augment_training_data else 0,
        shear_range=0.2 if augment_training_data else 0,
        zoom_range=0.2 if augment_training_data else 0,
        horizontal_flip=augment_training_data,
        vertical_flip=augment_training_data,
        brightness_range=[0.8, 1.2] if augment_training_data else [1.0, 1.0]
    )
    
    # 验证集和测试集只进行 rescale
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # 创建图像数据生成器
    train_image_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_image_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )
  # 创建测试数据生成器
    test_image_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    # 修复批次数计算
    test_image_generator.samples = len(test_image_generator.filenames)
    test_image_generator.n = test_image_generator.samples
    test_image_generator._set_index_array()
    
    if sensor_data_path is None:
        # 如果没有提供传感器数据路径，返回普通图像生成器
        return train_image_generator, val_image_generator, test_image_generator
    
    # 加载传感器数据
    sensor_df = pd.read_csv(sensor_data_path)
    
    # 定义多模态数据生成器
    def multimodal_generator(image_generator, sensor_df):
        while True:
            # 获取图像批次
            images, labels = next(image_generator)
            
            # 提取图像文件名
            filenames = image_generator.filenames[image_generator.batch_index * batch_size : 
                                                (image_generator.batch_index + 1) * batch_size]
            
            # 提取图像ID
            image_ids = [os.path.splitext(os.path.basename(filename))[0] for filename in filenames]
            
            # 查找对应的传感器数据
            sensor_batch = []
            for img_id in image_ids:
                sensor_data = sensor_df[sensor_df['image_id'] == img_id].to_dict('records')
                if sensor_data:
                    # 提取需要的传感器特征
                    features = {
                        'temperature': sensor_data[0].get('temperature', 0),
                        'humidity': sensor_data[0].get('humidity', 0),
                        'smoke_level': sensor_data[0].get('smoke_level', 0),
                        'co_level': sensor_data[0].get('co_level', 0)
                    }
                    sensor_batch.append(list(features.values()))
                else:
                    # 如果没有找到传感器数据，使用默认值
                    sensor_batch.append([0, 0, 0, 0])
            
            # 转换为数组
            sensor_batch = np.array(sensor_batch, dtype=np.float32)
            
            yield [images, sensor_batch], labels
    
    # 创建多模态生成器
    train_generator = multimodal_generator(train_image_generator, sensor_df)
    val_generator = multimodal_generator(val_image_generator, sensor_df)
    test_generator = multimodal_generator(test_image_generator, sensor_df)
    
    return train_generator, val_generator, test_generator

#创建传统图像数据生成器（保持向后兼容）
def create_data_generators(image_size = (224, 224), batch_size = 32, augment_training_data = False):
    #训练集数据增强配置
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rescale = 1./255,
        rotation_range = 30,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        brightness_range = [0.7, 1.3],
        fill_mode = 'nearest',
    )

    #验证集和测试集应用与训练集相同的预处理
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rescale = 1. / 255
    )

    #创建数据生成器
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True
    )

    #创建验证集数据生成器
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = False
    )

    # 创建测试数据生成器
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size = image_size,
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = False
    )
    
    # 修复批次数计算
    test_generator.samples = len(test_generator.filenames)
    test_generator.n = test_generator.samples
    test_generator._set_index_array()

    return train_generator, val_generator, test_generator

# 在data_loader.py中添加验证
def verify_data_split():
    train_files = set(os.listdir(os.path.join(train_dir, "fire")) + 
                     os.listdir(os.path.join(train_dir, "non_fire")))
    
    val_files = set(os.listdir(os.path.join(val_dir, "fire")) + 
                   os.listdir(os.path.join(val_dir, "non_fire")))
    
    test_files = set(os.listdir(os.path.join(test_dir, "fire")) + 
                    os.listdir(os.path.join(test_dir, "non_fire")))
    
    print(f"训练集与验证集重叠: {len(train_files & val_files)}")
    print(f"训练集与测试集重叠: {len(train_files & test_files)}")
    print(f"验证集与测试集重叠: {len(val_files & test_files)}")
    
    if len(train_files & val_files) > 0 or len(train_files & test_files) > 0:
        print("警告：数据划分存在重叠！")

def check_val_data():
    val_fire = len(os.listdir(os.path.join(val_dir, "fire")))
    val_non_fire = len(os.listdir(os.path.join(val_dir, "non_fire")))
    print(f"验证集火灾样本数：{val_fire}，非火灾样本数：{val_non_fire}")
    if val_fire == 0:
        print("错误：验证集火灾样本为空！")
    if val_non_fire == 0:
        print("错误：验证集非火灾样本为空！")

#主函数
if __name__ == '__main__':
    #加载所有图像路径
    fire_images, non_fire_images = load_image_paths()

    #划分数据集
    fire_train, fire_val, fire_test, non_fire_train, non_fire_val, non_fire_test = split_dataset(fire_images, non_fire_images)

    #复制图像至相应目录
    copy_images_to_dir(fire_train, fire_val, fire_test, non_fire_train, non_fire_val, non_fire_test)

    #验证数据集划分
    verify_data_split()

    #创建数据生成器 图像大小为224*224 批次大小为32
    train_generator, val_generator, test_generator = create_data_generators(
        image_size = (224, 224), 
        batch_size = 32,
        augment_training_data = True
        )

    #打印数据生成器信息
    print(train_generator)
    print(val_generator)
    print(test_generator)

    #打印样本数量
    print("火灾样本数：", len(fire_train))
    print("非火灾样本数：", len(non_fire_train))
    print("验证集样本数：", len(fire_val))
    print("非验证集样本数：", len(non_fire_val))
    print("测试集样本数：", len(fire_test))
    print("非测试集样本数：", len(non_fire_test))

    #打印批次数量
    print("训练批次数：", len(train_generator))
    print("验证批次数：", len(val_generator))
    print("测试批次数：", len(test_generator))

    #打印验证集信息
    check_val_data()

    #打印训练集批次信息
    for batch in train_generator:
        print(batch)
        break
