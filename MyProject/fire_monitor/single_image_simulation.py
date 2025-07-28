import numpy as np
import matplotlib.pyplot as plt
import time
import sqlite3
import os
import random
import json
from datetime import datetime

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class KalmanFilter:
    """卡尔曼滤波器类，用于优化传感器数据"""
    def __init__(self, process_noise, measurement_noise, estimate_error):
        self.process_noise = process_noise  # 过程噪声
        self.measurement_noise = measurement_noise  # 测量噪声
        self.estimate_error = estimate_error  # 估计误差
        self.priori_estimate = 0  # 先验估计
        self.priori_error = 0  # 先验误差
        self.gain = 0  # 卡尔曼增益
        self.updated_estimate = 0  # 更新后的估计
        self.updated_error = 0  # 更新后的误差
        self.first_run = True  # 首次运行标志

    def predict(self):
        """预测步骤"""
        if self.first_run:
            return
        # 先验估计 = 上一次的更新估计
        self.priori_estimate = self.updated_estimate
        # 先验误差 = 上一次的更新误差 + 过程噪声
        self.priori_error = self.updated_error + self.process_noise

    def update(self, measurement):
        """更新步骤"""
        if self.first_run:
            # 首次运行，初始化估计值
            self.updated_estimate = measurement
            self.updated_error = self.estimate_error
            self.first_run = False
            return self.updated_estimate

        # 计算卡尔曼增益
        self.gain = self.priori_error / (self.priori_error + self.measurement_noise)
        # 更新估计
        self.updated_estimate = self.priori_estimate + self.gain * (measurement - self.priori_estimate)
        # 更新误差
        self.updated_error = (1 - self.gain) * self.priori_error
        return self.updated_estimate

class SensorSimulator:
    """传感器模拟器类，用于生成模拟的传感器数据"""
    def __init__(self, image_id):
        self.image_id = image_id
        self.temperature_filter = KalmanFilter(0.1, 0.5, 1.0)
        self.humidity_filter = KalmanFilter(0.1, 0.5, 1.0)
        self.smoke_filter = KalmanFilter(0.1, 0.5, 1.0)
        self.sensor_data_dir = "d:/MyProject/sensor_data"
        # 确保传感器数据目录存在
        os.makedirs(self.sensor_data_dir, exist_ok=True)
        # 存储同一张图片的所有传感器数据
        self.all_sensor_data = {'image_id': self.image_id, 'steps': []}

    def generate_sensor_data(self, step, fire_area_ratio):
        """根据火灾面积比例生成传感器数据"""
        # 基础值
        base_temperature = 25.0  # 基础温度(℃)
        base_humidity = 50.0     # 基础湿度(%).
        base_smoke = 0.1         # 基础烟雾浓度

        # 根据火灾面积比例调整
        temp_variation = fire_area_ratio * 100  # 火灾面积越大，温度越高
        humidity_variation = -fire_area_ratio * 30  # 火灾面积越大，湿度越低
        smoke_variation = fire_area_ratio * 0.9  # 火灾面积越大，烟雾浓度越高

        # 添加随机噪声
        temperature = base_temperature + temp_variation + random.uniform(-2, 2)
        humidity = base_humidity + humidity_variation + random.uniform(-5, 5)
        smoke = base_smoke + smoke_variation + random.uniform(-0.05, 0.05)

        # 应用卡尔曼滤波
        self.temperature_filter.predict()
        filtered_temp = self.temperature_filter.update(temperature)

        self.humidity_filter.predict()
        filtered_humidity = self.humidity_filter.update(humidity)

        self.smoke_filter.predict()
        filtered_smoke = self.smoke_filter.update(smoke)

        # 确保值在合理范围内
        filtered_temp = max(0, min(100, filtered_temp))
        filtered_humidity = max(0, min(100, filtered_humidity))
        filtered_smoke = max(0, min(1, filtered_smoke))

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "step": step,
            "timestamp": timestamp,
            "temperature": round(filtered_temp, 2),
            "humidity": round(filtered_humidity, 2),
            "smoke": round(filtered_smoke, 4),
            "fire_area_ratio": round(fire_area_ratio, 4)
        }

    def add_sensor_data(self, data):
        """添加传感器数据到内存"""
        self.all_sensor_data['steps'].append(data)

    def save_all_sensor_data(self):
        """保存所有传感器数据到单个文件"""
        filename = os.path.join(self.sensor_data_dir, f'sensor_data_img_{self.image_id}.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_sensor_data, f, ensure_ascii=False, indent=4)
        print(f"所有传感器数据已保存到: {filename}")

class FireSpreadModel:
    """火灾蔓延模型类，用于模拟火灾蔓延过程"""
    def __init__(self, image_id):
        self.image_id = image_id
        self.grid_size = 100  # 网格大小
        self.cell_size = 5  # 单元格大小(米)
        self.wind_direction = random.uniform(0, 2 * np.pi)  # 风向(弧度)
        self.wind_speed = random.uniform(0, 10)  # 风速(米/秒)
        self.humidity = random.uniform(20, 80)  # 湿度(%)
        self.temperature = random.uniform(15, 35)  # 温度(℃)
        self.ignition_probability = 0.02  # 初始着火概率
        self.simulation_steps = 100  # 模拟步数
        self.fire_grid = np.zeros((self.grid_size, self.grid_size))  # 火灾网格
        self.fire_history = []  # 火灾历史记录
        self.risk_evaluation_dir = "d:/MyProject/risk_evaluations"
        os.makedirs(self.risk_evaluation_dir, exist_ok=True)
        self.sensor_simulator = SensorSimulator(image_id)

        # 从数据库获取图像数据来初始化火灾网格
        self.initialize_fire_grid()

    def initialize_fire_grid(self):
        """从数据库初始化火灾网格"""
        try:
            conn = sqlite3.connect('d:/MyProject/image_database.db')
            cursor = conn.cursor()

            # 获取指定ID的图像数据
            cursor.execute("SELECT image_data FROM images WHERE id = ?", (self.image_id,))
            result = cursor.fetchone()

            if result is None:
                print(f"未找到ID为 {self.image_id} 的图像数据，使用随机初始化")
                # 随机初始化一个着火点
                x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
                self.fire_grid[x, y] = 1.0
            else:
                # 这里简化处理，实际应用中应该解析图像数据
                print(f"已加载ID为 {self.image_id} 的图像数据")
                # 随机初始化一些着火点
                num_ignition_points = random.randint(1, 5)
                for _ in range(num_ignition_points):
                    x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
                    self.fire_grid[x, y] = 1.0

            conn.close()
        except Exception as e:
            print(f"数据库错误: {e}")
            # 随机初始化一个着火点
            x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            self.fire_grid[x, y] = 1.0

        # 保存初始状态
        self.fire_history.append(np.copy(self.fire_grid))

    def calculate_spread_probability(self, i, j):
        """计算火灾蔓延概率"""
        if self.fire_grid[i, j] > 0:
            return 0.0  # 已经着火

        # 检查周围8个方向的着火点
        spread_prob = 0.0
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                if self.fire_grid[ni, nj] > 0:
                    # 基础蔓延概率
                    base_prob = 0.3

                    # 考虑风向影响
                    wind_effect = 1.0
                    if (di, dj) == self.get_wind_direction_vector():
                        wind_effect = 1.0 + self.wind_speed * 0.1
                    elif (di, dj) == self.get_opposite_wind_direction_vector():
                        wind_effect = max(0.1, 1.0 - self.wind_speed * 0.1)

                    # 考虑湿度影响
                    humidity_effect = max(0.1, 1.0 - self.humidity * 0.01)

                    # 考虑温度影响
                    temperature_effect = 1.0 + (self.temperature - 25) * 0.02

                    spread_prob += base_prob * wind_effect * humidity_effect * temperature_effect

        # 归一化概率
        spread_prob = min(1.0, spread_prob)
        return spread_prob

    def get_wind_direction_vector(self):
        """将风向转换为方向向量"""
        # 将弧度转换为8个方向之一
        angle_deg = np.degrees(self.wind_direction)
        if 22.5 <= angle_deg < 67.5:
            return (-1, 1)  # 东北
        elif 67.5 <= angle_deg < 112.5:
            return (-1, 0)  # 北
        elif 112.5 <= angle_deg < 157.5:
            return (-1, -1)  # 西北
        elif 157.5 <= angle_deg < 202.5:
            return (0, -1)  # 西
        elif 202.5 <= angle_deg < 247.5:
            return (1, -1)  # 西南
        elif 247.5 <= angle_deg < 292.5:
            return (1, 0)  # 南
        elif 292.5 <= angle_deg < 337.5:
            return (1, 1)  # 东南
        else:
            return (0, 1)  # 东

    def get_opposite_wind_direction_vector(self):
        """获取与风向相反的方向向量"""
        dir_vector = self.get_wind_direction_vector()
        return (-dir_vector[0], -dir_vector[1])

    def simulate_step(self):
        """模拟一步火灾蔓延"""
        new_fire_grid = np.copy(self.fire_grid)
        spread_probabilities = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                spread_prob = self.calculate_spread_probability(i, j)
                spread_probabilities[i, j] = spread_prob
                if random.random() < spread_prob:
                    new_fire_grid[i, j] = 1.0

        self.fire_grid = new_fire_grid
        self.fire_history.append(np.copy(self.fire_grid))
        return spread_probabilities

    def run_simulation(self):
        """运行完整的火灾蔓延模拟"""
        print(f"开始对图片ID: {self.image_id} 进行火灾蔓延模拟...")
        start_time = time.time()

        for step in range(self.simulation_steps):
            spread_probabilities = self.simulate_step()
            # 每10步生成并保存传感器数据
            if step % 10 == 0:
                fire_area_ratio = np.sum(self.fire_grid) / (self.grid_size * self.grid_size)
                sensor_data = self.sensor_simulator.generate_sensor_data(fire_area_ratio)
                self.sensor_simulator.save_sensor_data(step, sensor_data)
                print(f"模拟步数: {step}, 火灾面积比例: {fire_area_ratio:.4f}, 传感器数据: {sensor_data}")

        end_time = time.time()
        print(f"模拟完成，耗时: {end_time - start_time:.2f} 秒")

        # 生成风险评估图像
        self.generate_risk_evaluation()

    def generate_risk_evaluation(self):
        """生成风险评估图像"""
        plt.figure(figsize=(10, 8))
        # 使用最后一步的火灾网格作为风险图
        risk_map = self.fire_history[-1]
        im = plt.imshow(risk_map, cmap='jet', interpolation='bilinear')
        plt.title(f'火灾风险热分布图 - 图像ID: {self.image_id}')
        plt.colorbar(im, label='风险等级')
        plt.axis('off')

        # 保存图像
        filename = f'risk_evaluation_{self.image_id}.png'
        filepath = os.path.join(self.risk_evaluation_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"风险评估图像已保存到: {filepath}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python single_image_simulation.py <image_id>")
        sys.exit(1)

    try:
        image_id = int(sys.argv[1])
        print(f"处理图像ID: {image_id}")
        fire_model = FireSpreadModel(image_id)
        num_steps = fire_model.simulation_steps

        print(f"开始对图片ID: {image_id} 进行火灾蔓延模拟...")

        # 初始化传感器模拟器
        sensor_sim = SensorSimulator(image_id)

        # 运行模拟
        start_time = time.time()
        for step in range(num_steps):
            fire_model.simulate_step()

            # 计算火灾区域比例
            fire_area_ratio = np.sum(fire_model.fire_grid) / (fire_model.grid_size * fire_model.grid_size)

            # 每10步生成传感器数据并添加到内存
            if step % 10 == 0:
                sensor_data = sensor_sim.generate_sensor_data(step, fire_area_ratio)
                sensor_sim.add_sensor_data(sensor_data)
                print(f"模拟步数: {step}, 火灾面积比例: {fire_area_ratio:.4f}, 传感器数据: {sensor_data}")

                # 显示进度
                if step % 20 == 0:
                    print(f'模拟进度: {step}/{num_steps} 步')

            # 模拟完成后保存所有传感器数据
            sensor_sim.save_all_sensor_data()

            # 生成风险评估图像
            fire_model.generate_risk_evaluation()

            end_time = time.time()
            print(f"模拟完成，耗时: {end_time - start_time:.2f} 秒")
    except ValueError:
        print("错误: 请提供有效的图像ID")
        sys.exit(1)
    except Exception as e:
        print(f"模拟过程中出错: {e}")
        sys.exit(1)