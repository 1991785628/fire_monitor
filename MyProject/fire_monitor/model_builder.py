import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from typing import Tuple

def build_fire_detection_model(input_shape = (224, 224, 3), 
                               num_classes = 1, 
                               use_dropout = True, 
                               dropout_rate = 0.5, 
                               l2_reg = 0.0005,
                               learning_rate = 0.0001):

    # 使用MobileNetV2作为基础模型，加载预训练权重
    base_model = MobileNetV2(include_top = False, 
                             weights = 'imagenet',  # 使用ImageNet预训练权重
                             input_shape = input_shape,
                             alpha=1.0)  # 恢复默认模型宽度

    # 冻结所有层
    base_model.trainable = False

    # 添加注意力机制(简化版SE模块)
    inputs = tf.keras.Input(shape = input_shape)
    x = base_model(inputs, training = False)

    # 全局平均池化
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    # 简化的注意力机制
    se = layers.Dense(x.shape[-1] // 32, activation = "relu")(x)
    se = layers.Dense(x.shape[-1], activation = "sigmoid")(se)
    x = layers.Multiply()([x, se])

    # 简化全连接层结构
    if use_dropout:
        x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(32, activation = "relu",
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(num_classes, activation = "sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    # 定义Focal Loss
    def focal_loss(gamma=2., alpha=0.25):
        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
            weight = y_true * alpha + (1 - y_true) * (1 - alpha)
            # 使用TensorFlow操作替代Python条件判断
            fl = weight * tf.math.pow(1 - y_pred, gamma * y_true) * tf.math.pow(y_pred, gamma * (1 - y_true)) * ce
            return tf.reduce_mean(fl)
        return loss

    #编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    # 导入自定义F1Score指标
    from metrics import F1Score
    
    model.compile(
        optimizer = optimizer,
        loss = focal_loss(gamma=2.0, alpha=0.8),  # 进一步增加火灾类别的权重
        metrics = [
            "accuracy", 
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall'),
            tf.keras.metrics.AUC(name = 'auc'),
            F1Score(name = 'f1_score')
        ]
    )
    
    return model

def build_multimodal_fire_detection_model(image_shape: Tuple[int, int, int] = (224, 224, 3),
                                          sensor_features: int = 4,
                                          num_classes: int = 1,
                                          use_dropout: bool = True,
                                          dropout_rate: float = 0.5,
                                          l2_reg: float = 0.0005,
                                          learning_rate: float = 0.0001) -> Model:
    """构建多模态火灾检测模型（图像+传感器数据）
    
    Args:
        image_shape: 图像输入形状
        sensor_features: 传感器特征数量
        num_classes: 类别数量
        use_dropout: 是否使用dropout
        dropout_rate: dropout率
        l2_reg: L2正则化系数
        learning_rate: 学习率
        
    Returns:
        构建好的多模态模型
    """
    # 图像分支
    image_input = Input(shape=image_shape, name='image_input')
    
    # 使用MobileNetV2作为图像特征提取器
    base_model = MobileNetV2(include_top=False, 
                             weights='imagenet',
                             input_shape=image_shape,
                             alpha=1.0)
    base_model.trainable = False  # 冻结基础模型
    
    x = base_model(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # 简化的注意力机制
    se = layers.Dense(x.shape[-1] // 32, activation='relu')(x)
    se = layers.Dense(x.shape[-1], activation='sigmoid')(se)
    image_features = layers.Multiply()([x, se])
    
    # 传感器数据分支
    sensor_input = Input(shape=(sensor_features,), name='sensor_input')
    s = layers.Dense(16, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(sensor_input)
    s = layers.BatchNormalization()(s)
    if use_dropout:
        s = layers.Dropout(dropout_rate)(s)
    s = layers.Dense(8, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(s)
    sensor_features = layers.BatchNormalization()(s)
    
    # 融合两种模态特征
    combined = layers.Concatenate()([image_features, sensor_features])
    
    # 分类头
    if use_dropout:
        combined = layers.Dropout(dropout_rate)(combined)
    
    combined = layers.Dense(32, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(combined)
    combined = layers.BatchNormalization()(combined)
    
    outputs = layers.Dense(num_classes, activation='sigmoid', name='output')(combined)
    
    # 定义模型
    model = Model(inputs=[image_input, sensor_input], outputs=outputs)
    
    # 定义Focal Loss
    def focal_loss(gamma=2., alpha=0.25):
        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
            weight = y_true * alpha + (1 - y_true) * (1 - alpha)
            fl = weight * tf.math.pow(1 - y_pred, gamma * y_true) * tf.math.pow(y_pred, gamma * (1 - y_true)) * ce
            return tf.reduce_mean(fl)
        return loss
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    from metrics import F1Score
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.8),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            F1Score(name='f1_score')
        ]
    )
    
    return model

#创建模型可视化函数
def visualize_model(model, save_path = 'fire_detection_model.png'):
    tf.keras.utils.plot_model(model, to_file = save_path, show_shapes = True, show_layer_names = True, rankdir = 'TB', dpi = 96)

if __name__ == "__main__":
    model = build_fire_detection_model()
    model.summary()
    visualize_model(model)
