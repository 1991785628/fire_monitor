import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import time
import sqlite3
import os
import random
import json

# 配置 matplotlib 中文显示并抑制字体警告
import matplotlib
import warnings
# 抑制 matplotlib 字体管理器的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class KalmanFilter:
    """卡尔曼滤波器实现，用于优化传感器数据"""
    def __init__(self, initial_state, initial_error, process_noise, measurement_noise):
        self.state = initial_state
        self.error = initial_error
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.gain = 0

    def predict(self):
        """预测下一个状态"""
        # 状态预测（假设为常数模型）
        self.error += self.process_noise

    def update(self, measurement):
        """使用测量值更新状态"""
        # 计算卡尔曼增益
        self.gain = self.error / (self.error + self.measurement_noise)
        # 更新状态
        self.state += self.gain * (measurement - self.state)
        # 更新误差协方差
        self.error = (1 - self.gain) * self.error
        return self.state


class SensorSimulator:
    """传感器数据模拟器，生成与火灾风险相关的传感器数据并应用卡尔曼滤波"""
    def __init__(self, seed, fire_model):
        self.seed = seed
        self.fire_model = fire_model
        np.random.seed(seed)
        # 初始化卡尔曼滤波器
        self.temp_filter = KalmanFilter(25, 1, 0.1, 0.5)
        self.humidity_filter = KalmanFilter(50, 1, 0.1, 0.5)
        self.smoke_filter = KalmanFilter(0, 1, 0.1, 0.5)
        # 创建传感器数据目录
        self.data_dir = 'sensor_data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        # 存储同一张图片的所有传感器数据
        self.all_sensor_data = {'image_id': None, 'steps': []}

    def generate_sensor_data(self, step):
        """生成模拟的传感器数据"""
        # 获取当前火灾模型状态
        fire_grid = self.fire_model.fire_grid
        risk_grid = self.fire_model.risk_grid

        # 计算火灾区域比例
        fire_area = np.sum(fire_grid == 1) / (fire_grid.shape[0] * fire_grid.shape[1])

        # 基于火灾状态生成温度数据（火灾越大温度越高）
        base_temp = 25 + fire_area * 100
        temp_noise = np.random.normal(0, 2)
        raw_temp = base_temp + temp_noise
        # 应用卡尔曼滤波
        self.temp_filter.predict()
        filtered_temp = self.temp_filter.update(raw_temp)

        # 生成湿度数据（火灾越大湿度越低）
        base_humidity = 50 - fire_area * 40
        humidity_noise = np.random.normal(0, 3)
        raw_humidity = base_humidity + humidity_noise
        # 应用卡尔曼滤波
        self.humidity_filter.predict()
        filtered_humidity = self.humidity_filter.update(raw_humidity)

        # 生成烟雾浓度数据（火灾越大烟雾越浓）
        base_smoke = fire_area * 1000
        smoke_noise = np.random.normal(0, 50)
        raw_smoke = base_smoke + smoke_noise
        # 应用卡尔曼滤波
        self.smoke_filter.predict()
        filtered_smoke = self.smoke_filter.update(raw_smoke)

        # 确保数值在合理范围内
        filtered_temp = max(20, min(100, filtered_temp))
        filtered_humidity = max(5, min(95, filtered_humidity))
        filtered_smoke = max(0, filtered_smoke)

        return {
            'step': step,
            'timestamp': time.time(),
            'temperature': round(filtered_temp, 2),
            'humidity': round(filtered_humidity, 2),
            'smoke': round(filtered_smoke, 2),
            'fire_area': round(fire_area, 4)
        }

    def add_sensor_data(self, data, image_id):
        """添加传感器数据到内存"""
        self.all_sensor_data['image_id'] = image_id
        self.all_sensor_data['steps'].append(data)

    def save_all_sensor_data(self, image_id):
        """保存所有传感器数据到单个文件"""
        filename = os.path.join(self.data_dir, f'sensor_data_img_{image_id}.json')
        with open(filename, 'w') as f:
            json.dump(self.all_sensor_data, f, indent=4)
        print(f'所有传感器数据已保存到 {filename}')


class FireSpreadModel:
    """基于物理模型的火灾蔓延预测系统
    实现森林/建筑火灾的蔓延路径和风险评估
    """
    def __init__(self, db_path=None):
        # 初始化数据库连接
        self.db_path = db_path if db_path else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'image_database.db')
        self.conn = None
        self.cursor = None
        self._connect_db()
        # 初始化模型参数
        self.grid_size = (100, 100)  # 模拟网格大小
        self.cell_size = 5  # 每个网格单元的实际大小(米) - 调整为更精细的5米
        self.wind_speed = 5.0  # 风速(m/s)
        self.wind_direction = 0  # 风向(度，0表示正北)
        self.humidity = 30  # 湿度(%)
        self.temperature = 25  # 环境温度(摄氏度)
        self.fuel_load = 0.5  # 燃料负载(kg/m²)
        self.ignition_threshold = 0.4  # 点燃阈值 - 降低阈值使火灾更容易蔓延
        self.spread_rate_factor = 0.2  # 蔓延速率因子 - 增加速率使火灾更容易蔓延
        
        # 初始化模拟网格
        self.fire_grid = np.zeros(self.grid_size)  # 0:未燃烧, 1:燃烧中, 2:已燃尽
        self.risk_grid = np.zeros(self.grid_size)  # 风险等级(0-1)
        
    def _connect_db(self):
        """连接到图像数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.text_factory = str
            self.cursor = self.conn.cursor()
            print(f"成功连接到数据库: {self.db_path}")
        except Exception as e:
            print(f"数据库连接失败: {e}")
            self.conn = None
            self.cursor = None

    def close_db(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("数据库连接已关闭")

    def get_high_risk_fire_images(self, count=5):
        """从数据库中获取高风险火灾图像
        Args:
            count: 要获取的图像数量
        Returns:
            图像ID列表
        """
        if not self.cursor:
            print("数据库未连接，无法获取图像")
            return []

        # 获取高风险火灾图像 (prediction为'fire'且confidence较低的视为高风险)
        self.cursor.execute("SELECT id FROM images WHERE prediction = 'fire' AND confidence < 0.3 ORDER BY RANDOM() LIMIT ?", (count,))
        image_ids = [row[0] for row in self.cursor.fetchall()]
        print(f"从数据库中获取了 {len(image_ids)} 张高风险火灾图像")
        return image_ids

    def set_environment_params(self, params: Dict):
        """设置环境参数
        Args:
            params: 包含环境参数的字典
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
    def set_initial_fire(self, positions: List[Tuple[int, int]]):
        """设置初始起火点
        Args:
            positions: 起火点坐标列表 [(x1, y1), (x2, y2), ...]
        """
        for x, y in positions:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.fire_grid[x, y] = 1
        
    def calculate_spread_direction(self) -> np.ndarray:
        """计算蔓延方向概率
        Returns:
            方向概率矩阵
        """
        # 基于风向计算蔓延方向权重
        wind_rad = np.radians(self.wind_direction)
        direction_weights = np.zeros((3, 3))  # 3x3邻域权重
        
        # 中心为当前单元格
        # 计算8个方向的权重
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # 跳过中心
                
                # 计算方向与风向的夹角
                cell_angle = np.arctan2(dy, dx)
                angle_diff = abs(cell_angle - wind_rad)
                
                # 风向相同的方向权重更高
                weight = np.cos(angle_diff) * 0.5 + 0.5
                direction_weights[dx+1, dy+1] = weight
        
        # 归一化权重
        direction_weights /= direction_weights.sum()
        return direction_weights
        
    def update_fire_spread(self) -> None:
        """更新火灾蔓延状态
        """
        new_fire_grid = self.fire_grid.copy()
        direction_weights = self.calculate_spread_direction()
        
        # 计算风险因子
        risk_factor = self.fuel_load * (1 - self.humidity / 100) * np.exp(0.1 * (self.temperature - 25)) * (1 + 0.1 * self.wind_speed)
        
        # 遍历网格
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.fire_grid[i, j] == 1:  # 燃烧中的单元格
                    # 向周围蔓延
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                                if self.fire_grid[ni, nj] == 0:  # 未燃烧
                                    # 计算蔓延概率
                                    spread_prob = self.spread_rate_factor * risk_factor * direction_weights[dx+1, dy+1]
                                    
                                    if np.random.random() < spread_prob:
                                        new_fire_grid[ni, nj] = 1
                    
                    # 燃烧一段时间后变为燃尽状态
                    if np.random.random() < 0.1:
                        new_fire_grid[i, j] = 2
        
        self.fire_grid = new_fire_grid
        
        # 更新风险网格
        self._update_risk_grid()
        
    def _update_risk_grid(self) -> None:
        """更新风险评估网格
        """
        # 重置风险网格
        self.risk_grid = np.zeros(self.grid_size)
        
        # 基于火灾距离和蔓延方向计算风险
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.fire_grid[i, j] == 1:
                    # 对周围单元格产生风险影响
                    for dx in range(-10, 11):
                        for dy in range(-10, 11):
                            ni, nj = i + dx, j + dy
                            if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                                distance = np.sqrt(dx**2 + dy**2)
                                if distance > 0:
                                    # 距离越近风险越高
                                    risk = 1.0 / (distance * 0.5)
                                    # 考虑风向影响
                                    wind_effect = np.cos(np.radians(self.wind_direction) - np.arctan2(dy, dx)) * 0.5 + 0.5
                                    self.risk_grid[ni, nj] = max(self.risk_grid[ni, nj], 
                                                                 risk * wind_effect * 0.3)
        
        # 限制风险值在0-1之间
        self.risk_grid = np.clip(self.risk_grid, 0, 1)
        
    def run_simulation(self, steps: int) -> List[np.ndarray]:
        """运行火灾蔓延模拟
        Args:
            steps: 模拟步数
        Returns:
            每一步的火灾状态列表
        """
        history = [self.fire_grid.copy()]
        
        for _ in range(steps):
            self.update_fire_spread()
            history.append(self.fire_grid.copy())
            
        return history
        
    def visualize_fire_spread(self, history: List[np.ndarray], image_id: int, interval: float = 0.5) -> None:
        """可视化火灾蔓延过程
        Args:
            history: 火灾状态历史
            image_id: 图像ID
            interval: 帧间隔(秒)
        """
        plt.figure(figsize=(10, 10))
        
        for i, fire_state in enumerate(history):
            plt.clf()
            plt.imshow(fire_state, cmap='hot', interpolation='nearest')
            plt.title(f'图像 ID: {image_id} - 火灾蔓延模拟 - 步骤 {i}')
            plt.colorbar(label='燃烧状态 (0:未燃烧, 1:燃烧中, 2:已燃尽)')
            plt.draw()
            plt.pause(interval)
        
        plt.close()
        
    def visualize_risk(self, image_id: int) -> None:
        """可视化风险评估
        Args:
            image_id: 图像ID
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.risk_grid, cmap='viridis', interpolation='nearest')
        plt.title(f'图像 ID: {image_id} - 火灾风险评估')
        plt.colorbar(label='风险等级 (0-1)')
        # 创建风险评估文件夹
        output_dir = 'risk_evaluations'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, f'risk_evaluation_{image_id}.png')
        plt.savefig(filename)
        plt.close()
        print(f'风险评估图已保存为 {filename}')

if __name__ == '__main__':
    # 从数据库获取高风险火灾图像
    model = FireSpreadModel()
    high_risk_images = []
    
    try:
        high_risk_images = model.get_high_risk_fire_images(5)
        
        if high_risk_images:
            print(f"将按顺序处理 {len(high_risk_images)} 张高风险火灾图像")
        else:
            print("未找到高风险火灾图像，程序退出")
    finally:
        model.close_db()
    
    if not high_risk_images:
        exit()
    
    # 按顺序处理每张图像
    for img_id in high_risk_images:
        print(f'\n开始处理图像 ID: {img_id}')
        
        # 创建新的模型实例
        img_model = FireSpreadModel()
        img_model.close_db()  # 已获取图像ID，关闭数据库连接
        
        # 设置环境参数
        img_model.set_environment_params({
            'wind_speed': 8.0,
            'wind_direction': 45,  # 东北风
            'humidity': 20,
            'temperature': 30
        })
        
        # 使用图像ID生成模拟起火点
        np.random.seed(img_id)
        x = np.random.randint(20, 80)
        y = np.random.randint(20, 80)
        img_model.set_initial_fire([(x, y)])
        
        # 创建传感器模拟器
        sensor_sim = SensorSimulator(img_id, img_model)

        # 运行模拟并收集传感器数据
        print('开始火灾蔓延模拟并收集传感器数据...')
        start_time = time.time()
        history = []
        history.append(img_model.fire_grid.copy())

        for step in range(100):
            img_model.update_fire_spread()
            history.append(img_model.fire_grid.copy())

            # 每10步生成一次传感器数据并添加到内存
            if step % 10 == 0:
                sensor_data = sensor_sim.generate_sensor_data(step)
                sensor_sim.add_sensor_data(sensor_data, img_id)
                print(f'已生成第 {step} 步的传感器数据')

        # 模拟完成后保存所有传感器数据
        sensor_sim.save_all_sensor_data(img_id)

        end_time = time.time()
        print(f'模拟完成，耗时 {end_time - start_time:.2f} 秒')
        
        # 可视化结果
        img_model.visualize_fire_spread(history, img_id)
        img_model.visualize_risk(img_id)
        print(f'图像 ID: {img_id} 处理完成')