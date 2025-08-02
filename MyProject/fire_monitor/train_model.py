import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import History
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight
from data_loader import create_data_generators
from model_builder import build_fire_detection_model
from metrics import F1Score
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 显示所有日志
import tensorflow as tf
tf.get_logger().setLevel('DEBUG')  # 启用TensorFlow调试日志

#设置超参数
image_size = (224, 224)
batch_size = 16  # 减少批次大小以适应更复杂的模型
os.makedirs('models', exist_ok = True)

#分阶段训练参数
phase1_epochs = 20  # 冻结阶段
phase2_epochs = 40  # 微调阶段
initial_learning_rate = 0.0005
fine_tune_learning_rate = 0.00005

#创建数据生成器
train_generator, val_generator, test_generator = create_data_generators(
                                                image_size = image_size, 
                                                batch_size = batch_size,
                                                augment_training_data = True
                                                )

#验证数据生成器是否能正常产出数据
print("验证数据生成器...")
try:
    # 尝试获取一个批次的数据
    batch_images, batch_labels = next(iter(train_generator))
    print(f"数据生成器正常 - 批次形状: {batch_images.shape}, 标签形状: {batch_labels.shape}")
except Exception as e:
    print(f"数据生成器错误: {e}")
    exit(1)

#构建模型
model = build_fire_detection_model(
        input_shape = image_size + (3,), 
        num_classes = 1,
        use_dropout = True,
        dropout_rate = 0.5,
        l2_reg = 0.0005,
        learning_rate = initial_learning_rate
        )

#打印模型概要
print("模型构建完成，结构概要:")
model.summary()

# 计算训练集的类别分布
print("开始读取训练集标签...")
train_labels = train_generator.classes
classes = np.unique(train_labels)
print(f"训练标签读取完成，共 {len(train_labels)} 个样本")

#计算平衡的类别权重
class_weights_list = compute_class_weight('balanced', classes = classes, y = train_labels)
class_weights = {
    0: class_weights_list[0],
    1: class_weights_list[1]  # 使用计算出的平衡权重
}
print(f"调整后的类别权重: {class_weights}")

#定义回调函数（阶段1）
callbacks_phase1 = [
    ModelCheckpoint(filepath = 'models/best_model_phase1.keras', 
                    monitor = 'val_f1_score', 
                    save_best_only = True, 
                    mode = 'max', 
                    verbose = 1),

    EarlyStopping(monitor = 'val_f1_score', 
                  patience = 5, 
                  restore_best_weights = True, 
                  verbose = 1, 
                  mode = 'max'),
                    
    ReduceLROnPlateau(monitor = 'val_f1_score', 
                      factor = 0.5, 
                      patience = 2, 
                      mode = 'max', 
                      min_lr = 1e-7, 
                      verbose = 1) 
]

# 第一阶段训练：冻结大部分层
print("===== 开始第一阶段训练 =====")
history_phase1 = model.fit(train_generator, 
                          validation_data = val_generator, 
                          epochs = phase1_epochs, 
                          callbacks = callbacks_phase1, 
                          verbose = 2, 
                          class_weight = class_weights,
                          validation_freq = 1)

# 加载第一阶段的最佳模型
model.load_weights('models/best_model_phase1.keras')

# 第二阶段：解冻更多层进行微调
print("===== 开始第二阶段微调 =====")
# 解冻更多层
for layer in model.layers[-30:]:
    layer.trainable = True

# 降低学习率
optimizer = tf.keras.optimizers.Adam(learning_rate = fine_tune_learning_rate)
model.compile(
    optimizer = optimizer,
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = [
        "accuracy", 
        tf.keras.metrics.Precision(name = 'precision'),
        tf.keras.metrics.Recall(name = 'recall'),
        tf.keras.metrics.AUC(name = 'auc'),
        F1Score(name = 'f1_score')
    ]
)

# 定义回调函数（阶段2）
callbacks_phase2 = [
    ModelCheckpoint(filepath = 'models/best_model.keras', 
                    monitor = 'val_f1_score', 
                    save_best_only = True, 
                    mode = 'max', 
                    verbose = 1),

    EarlyStopping(monitor = 'val_f1_score', 
                  patience = 8, 
                  restore_best_weights = True, 
                  verbose = 1, 
                  mode = 'max'),
                    
    ReduceLROnPlateau(monitor = 'val_f1_score', 
                      factor = 0.3, 
                      patience = 3, 
                      mode = 'max', 
                      min_lr = 1e-8, 
                      verbose = 1) 
]

# 第二阶段训练
history_phase2 = model.fit(train_generator, 
                          validation_data = val_generator, 
                          epochs = phase2_epochs, 
                          callbacks = callbacks_phase2, 
                          verbose = 2, 
                          class_weight = class_weights,
                          initial_epoch = history_phase1.epoch[-1],
                          validation_freq = 1)

# 合并训练历史
history = History()
history.history = {k: history_phase1.history.get(k, []) + history_phase2.history.get(k, []) for k in set(history_phase1.history) | set(history_phase2.history)}

history.epoch = list(range(len(history.history['loss'])))

#保存最终模型
model.save('models/final_model.keras')

# 寻找最佳阈值以最大化F1分数
import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 1.0, 0.01)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# 获取验证集预测概率
val_pred_proba = model.predict(val_generator)
val_true = val_generator.classes

# 找到最佳阈值
best_threshold, best_f1 = find_best_threshold(val_true, val_pred_proba)
print(f"最佳阈值: {best_threshold:.2f}, 对应的F1分数: {best_f1:.4f}")

# 打印训练过程中的最佳验证指标
best_val_auc = max(history.history['val_auc'])
best_val_epoch = history.history['val_auc'].index(best_val_auc) + 1
print(f"最佳验证AUC:{best_val_auc:.4f}(第{best_val_epoch}轮)")

#绘制训练历史
import matplotlib.pyplot as plt

#配置中文字体
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history):
    plt.figure(figsize = (16, 12))

    #绘制损失曲线
    plt.subplot(3, 2, 1)
    plt.plot(history.history['loss'], label = '训练损失')
    plt.plot(history.history['val_loss'], label = '验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.legend()

    #绘制准确率曲线
    plt.subplot(3, 2, 2)
    plt.plot(history.history['accuracy'], label = '训练准确率')
    plt.plot(history.history['val_accuracy'], label = '验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.legend()

    #绘制精确率曲线
    plt.subplot(3, 2, 3)
    plt.plot(history.history['precision'], label = '训练精确率')
    plt.plot(history.history['val_precision'], label = '验证精确率')
    plt.title('训练和验证精确率')
    plt.xlabel('迭代次数')
    plt.ylabel('精确率')
    plt.legend()
    
    #绘制召回率曲线
    plt.subplot(3, 2, 4)
    plt.plot(history.history['recall'], label = '训练召回率')
    plt.plot(history.history['val_recall'], label = '验证召回率')
    plt.title('训练和验证召回率')
    plt.xlabel('迭代次数')
    plt.ylabel('召回率')
    
    #绘制AUC曲线
    plt.subplot(3, 2, 5)
    plt.plot(history.history['auc'], label = '训练AUC')
    plt.plot(history.history['val_auc'], label = '验证AUC')
    plt.title('训练和验证AUC')
    plt.xlabel('迭代次数')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)
