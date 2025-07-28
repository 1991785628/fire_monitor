import os
import io
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from database import ImageDatabase
from data_loader import preprocess_image
from metrics import F1Score

class ImageProcessor:
    def __init__(self, model_path, db_name='image_database.db'):
        self.model = self.load_detection_model(model_path)
        self.db = ImageDatabase(db_name)
        self.image_size = (224, 224)  # 与模型训练时使用的尺寸保持一致

    def load_detection_model(self, model_path):
        """加载火灾检测模型，包含自定义指标"""
        try:
            from metrics import F1Score
            model = load_model(model_path, custom_objects={'F1Score': F1Score})
            print(f"成功加载模型: {model_path}")
            return model
        except Exception as e:
            print(f"加载模型错误: {e}")
            raise

    def predict_image(self, image_blob):
        """预测图像是否包含火灾"""
        try:
            # 使用PIL直接打开图像字节流
            img = Image.open(io.BytesIO(image_blob))
            # 转换为数组
            image_array = np.array(img)
            # 预处理图像
            processed_img = preprocess_image(image_array)
            # 预测
            prediction = self.model.predict(processed_img)
            confidence = float(prediction[0][0])
            # 模型输出表示非火灾概率，使用训练确定的最佳阈值提高分类准确性
            # 与model_evaluation.py中的评估逻辑保持一致
            result = 'fire' if confidence < 0.4946 else 'non_fire'
            return result, confidence
        except Exception as e:
            print(f"预测错误: {e}")
            return None, None

    def process_unprocessed_images(self, limit=None):
        """处理数据库中所有未处理的图像"""
        unprocessed_images = self.db.get_unprocessed_images(limit)
        print(f"找到 {len(unprocessed_images)} 张未处理图像")

        processed_count = 0
        for img_data in unprocessed_images:
            image_id, filename, image_blob, timestamp = img_data
            print(f"处理图像: {filename}")

            result, confidence = self.predict_image(image_blob)
            if result is not None:
                # 更新数据库中的预测结果
                if self.db.update_image_prediction(image_id, result, confidence):
                    processed_count += 1
                    print(f"预测结果: {result} (置信度: {confidence:.4f})")

        print(f"处理完成，共处理 {processed_count} 张图像")
        return processed_count

    def close(self):
        """关闭数据库连接"""
        self.db.close()

if __name__ == "__main__":
    # 设置模型路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'models', 'best_model.keras')

    # 创建处理器实例
    processor = ImageProcessor(model_path)

    try:
        # 处理所有未处理的图像，不设置限制
        processor.process_unprocessed_images(limit=None)
    finally:
        # 确保关闭连接
        processor.close()

    # 生成风险地图和详细报告并保存到数据库
    print("正在生成风险地图和详细报告...")
    import subprocess
    # 使用正确的相对路径调用generate_reports.py
    subprocess.run(["python", os.path.join(base_dir, "generate_reports.py")])
    print("风险地图和详细报告生成完成")