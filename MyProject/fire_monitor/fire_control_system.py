import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from .data_loader import load_multimodal_data
from .model_evaluation import evaluate_model
from .train_model import evaluate_fire_risk
import time
import json
from .metrics import F1Score

class FireControlSystem:
    """火灾防控决策系统
    整合火灾检测、风险评估和蔓延预测功能
    """
    def __init__(self, model_path='models/final_model.keras'):
        # 加载模型
        self.model = self._load_model(model_path)
        self.image_size = (224, 224)
        self.results_dir = 'fire_control_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _load_model(self, model_path):
        """加载训练好的模型
        Args:
            model_path: 模型路径
        Returns:
            加载的模型
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            model = load_model(model_path, custom_objects={'F1Score': F1Score})
            print(f"成功加载模型: {model_path}")
            return model
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            raise
        
    def detect_fire(self, image_path, sensor_data_path=None):
        """检测火灾
        Args:
            image_path: 图像路径
            sensor_data_path: 传感器数据路径
        Returns:
            检测结果
        """
        print(f"开始检测图像: {image_path}")
        start_time = time.time()
        
        # 加载和预处理数据
        if sensor_data_path:
            image, sensor_data = load_multimodal_data(image_path, sensor_data_path)
            # 多模态模型预测
            inputs = [np.expand_dims(image, axis=0), 
                     np.expand_dims(np.array(list(sensor_data.values())), axis=0)]
        else:
            # 单模态模型预测
            from .data_loader import preprocess_image
            import tensorflow as tf
            # 加载图像并转换为数组
            img = tf.keras.utils.load_img(image_path, target_size=self.image_size)
            image_array = tf.keras.utils.img_to_array(img)
            # 预处理图像
            image = preprocess_image(image_array, self.image_size)
            inputs = image
        
        # 预测
        prediction = self.model.predict(inputs, verbose=0)
        # 处理不同模型的输出格式
        try:
            if isinstance(prediction, list):
                # 处理多输出模型
                fire_probability = float(prediction[0][0])
            elif hasattr(prediction, 'shape'):
                # 处理张量输出
                if len(prediction.shape) > 1 and prediction.shape[0] > 0:
                    fire_probability = float(prediction[0][0])
                elif prediction.shape[0] > 0:
                    fire_probability = float(prediction[0])
                else:
                    raise ValueError("预测结果为空")
            else:
                # 处理其他可能的格式
                fire_probability = float(prediction)
            is_fire = fire_probability < 0.5
        except (IndexError, TypeError, ValueError) as e:
            print(f"处理预测结果时出错: {str(e)}")
            print(f"预测结果格式: {type(prediction)}, 内容: {prediction}")
            # 设置默认值以避免程序崩溃
            fire_probability = 0.0
            is_fire = False
        
        end_time = time.time()
        print(f"检测完成，耗时 {end_time - start_time:.2f} 秒")
        
        # 构建结果
        result = {
            'image_path': image_path,
            'fire_detected': is_fire,
            'fire_probability': fire_probability,
            'detection_time': end_time - start_time,
            'timestamp': time.time()
        }
        
        return result
        
    def analyze_fire_risk(self, detection_results, environmental_data):
        """分析火灾风险
        Args:
            detection_results: 检测结果
            environmental_data: 环境数据
        Returns:
            风险分析结果
        """
        print("开始火灾风险分析...")
        
        # 格式化检测结果以适应风险评估函数
        formatted_results = []
        for i, result in enumerate(detection_results):
            if result['fire_detected']:
                # 假设位置信息可以从图像路径或其他来源获取
                # 这里简化处理，使用索引作为位置
                formatted_results.append({
                    'fire_probability': result['fire_probability'],
                    'position': {'x': (i+1)*100, 'y': (i+1)*100}
                })
        
        # 调用风险评估函数
        risk_result = evaluate_fire_risk(formatted_results, environmental_data)
        
        # 整合结果
        analysis_result = {
            'detection_summary': {
                'total_images': len(detection_results),
                'fire_images': sum(1 for r in detection_results if r['fire_detected']),
                'average_probability': float(np.mean([r['fire_probability'] for r in detection_results]))
            },
            'risk_assessment': risk_result,
            'timestamp': time.time()
        }
        
        # 保存结果
        result_path = os.path.join(self.results_dir, f'analysis_result_{int(time.time())}.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=4)
        
        print(f"风险分析完成，结果已保存到: {result_path}")
        return analysis_result
        
    def generate_report(self, analysis_result, report_path=None):
        """生成综合报告
        Args:
            analysis_result: 分析结果
            report_path: 报告保存路径
        Returns:
            报告路径
        """
        if not report_path:
            report_path = os.path.join(self.results_dir, f'fire_control_report_{int(time.time())}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("==================== 火灾防控决策报告 =====================\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")
            
            # 检测摘要
            f.write("1. 检测摘要\n")
            summary = analysis_result['detection_summary']
            f.write(f"   总图像数: {summary['total_images']}\n")
            f.write(f"   火灾图像数: {summary['fire_images']}\n")
            f.write(f"   平均火灾概率: {summary['average_probability']:.4f}\n\n")
            
            # 风险评估
            f.write("2. 风险评估\n")
            risk = analysis_result['risk_assessment']
            f.write(f"   风险等级: {risk['risk_level']}\n")
            max_risk_value = risk.get('max_risk_value')
            if max_risk_value is not None:
                f.write(f"   最大风险值: {max_risk_value:.4f}\n")
            risk_map_path = risk.get('risk_map_path')
            if risk_map_path:
                f.write(f"   风险地图: {risk_map_path}\n")
            
            f.write("\n==================== 报告结束 ====================")
        
        print(f"综合报告已生成: {report_path}")
        return report_path

if __name__ == '__main__':
    # 示例用法
    try:
        print("开始初始化系统...")
        # 初始化系统
        system = FireControlSystem()
        print("系统初始化完成")
        
        # 连接数据库
        from database import ImageDatabase
        db = ImageDatabase()
        print("成功连接到图像数据库")
        
        # 从数据库获取未处理的图像 (限制10张)
        unprocessed_images = db.get_unprocessed_images(limit=10)
        print(f"从数据库获取到 {len(unprocessed_images)} 张未处理图像")
        
        # 检测火灾
        detection_results = []
        if unprocessed_images:
            for img_id, filename, img_data, timestamp in unprocessed_images:
                print(f"正在检测图像: {filename}")
                # 由于我们需要文件路径，这里假设图像保存在fire_detection_data/test目录下
                # 在实际应用中，可能需要从数据库中提取图像数据并保存为临时文件
                img_path = f"fire_detection_data/test/{filename}"
                result = system.detect_fire(img_path)
                detection_results.append(result)
                print(f"图像 {filename}: 火灾概率 = {result['fire_probability']:.4f}, {'检测到火灾' if result['fire_detected'] else '未检测到火灾'}")
                
                # 更新数据库中的预测结果
                prediction = 'fire' if result['fire_detected'] else 'non_fire'
                db.update_image_prediction(img_id, prediction, result['fire_probability'])
        else:
            print("警告: 没有找到未处理的图像，使用测试图像进行演示")
            # 使用默认测试图像
            image_paths = ['fire_detection_data/test/fire/fire_test_1.jpg', 
                          'fire_detection_data/test/fire/fire_test_2.jpg',
                          'fire_detection_data/test/non_fire/non_fire_test_1.jpg']
            for img_path in image_paths:
                print(f"正在检测图像: {img_path}")
                result = system.detect_fire(img_path)
                detection_results.append(result)
                print(f"图像 {img_path}: 火灾概率 = {result['fire_probability']:.4f}, {'检测到火灾' if result['fire_detected'] else '未检测到火灾'}")
        
        # 关闭数据库连接
        db.close()
        
        # 分析风险
        print("开始分析风险...")
        environmental_data = {
            'wind_speed': 6.5,
            'wind_direction': 90,  # 东风
            'humidity': 25,
            'temperature': 28
        }
        
        analysis_result = system.analyze_fire_risk(detection_results, environmental_data)
        print("风险分析完成")
        
        # 生成报告
        print("开始生成报告...")
        report_path = system.generate_report(analysis_result)
        print(f"报告已生成: {report_path}")
        
    except Exception as e:
        import traceback
        print(f"系统运行出错: {str(e)}")
        print("错误堆栈:")
        print(traceback.format_exc())