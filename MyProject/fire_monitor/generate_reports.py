# -*- coding: utf-8 -*- 
import sqlite3
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from io import BytesIO
import time

# 配置 matplotlib 中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class ReportGenerator:
    def __init__(self):
        # 获取当前文件目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 连接图像数据库（使用绝对路径）
        image_db_path = os.path.join(base_dir, '..', 'image_database.db')
        self.image_conn = sqlite3.connect(image_db_path)
        self.image_conn.text_factory = str
        self.image_cursor = self.image_conn.cursor()

        # 连接或创建报告数据库（使用绝对路径）
        report_db_path = os.path.join(base_dir, '..', 'report_database.db')
        self.report_conn = sqlite3.connect(report_db_path)
        self.report_conn.text_factory = str
        self.report_cursor = self.report_conn.cursor()

        # 确保报告表存在
        self._create_report_table()

    def _create_report_table(self):
        """创建报告表（如果不存在）"""
        self.report_cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            risk_map BLOB,
            report_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.report_conn.commit()

    def get_unreported_images(self):
        """获取尚未生成报告的图像"""
        # 首先获取所有已处理的图像ID
        self.image_cursor.execute("SELECT id, filename, prediction, confidence FROM images WHERE prediction IS NOT NULL")
        all_images = self.image_cursor.fetchall()
        print(f"从图像数据库中找到 {len(all_images)} 张已处理的图像")

        # 获取已生成报告的图像ID
        self.report_cursor.execute("SELECT DISTINCT image_id FROM reports")
        reported_image_ids = {row[0] for row in self.report_cursor.fetchall()}
        print(f"从报告数据库中找到 {len(reported_image_ids)} 张已生成报告的图像")

        # 返回尚未生成报告的图像
        unreported = [img for img in all_images if img[0] not in reported_image_ids]
        print(f"找到 {len(unreported)} 张未生成报告的图像")
        return unreported

    def generate_risk_map(self, image_id, prediction, confidence):
        """生成风险地图 - 热量分布图"""
        # 设置随机种子，确保相同图像ID生成相同的风险模式
        np.random.seed(image_id)
        size = 100
        x, y = np.meshgrid(np.arange(size), np.arange(size))

        if prediction == 'fire':
            # 火灾图像 - 生成更自然的火灾形状，而非完美圆形
            # 主热点
            center = (50, 50)
            distance_main = np.sqrt((x - center[0])**2 + (y - center[1])** 2)
            risk_main = np.exp(-distance_main / 15) * (1 - confidence)

            # 添加次级热点增加真实感
            num_secondary = np.random.randint(1, 4)
            risk_secondary = np.zeros((size, size))
            for _ in range(num_secondary):
                sec_center = (center[0] + np.random.randint(-20, 21),
                             center[1] + np.random.randint(-20, 21))
                distance_sec = np.sqrt((x - sec_center[0])**2 + (y - sec_center[1])** 2)
                risk_secondary += np.exp(-distance_sec / 25) * (0.3 - 0.6 * confidence)

            # 结合主热点和次级热点
            risk_data = risk_main + risk_secondary
            # 归一化到0-1范围
            risk_data = np.clip(risk_data, 0, 1)
        else:
            # 非火灾图像 - 生成更均匀的低风险分布
            base_risk = 0.1 * confidence  # 基础风险值
            # 添加轻微的随机波动
            risk_fluctuations = np.random.normal(0, 0.05, (size, size))
            risk_data = base_risk + risk_fluctuations
            # 确保风险值在合理范围内
            risk_data = np.clip(risk_data, 0, 0.3)  # 非火灾风险不超过0.3

        # 创建热量分布图
        plt.figure(figsize=(10, 10))
        # 使用更适合火灾的颜色映射，从蓝色(低)到红色(高)
        plt.imshow(risk_data, cmap='coolwarm', interpolation='bilinear')
        plt.colorbar(label='风险等级 (0-1)')
        plt.title(f'火灾风险热量分布图 - 图像ID: {image_id}')
        plt.axis('off')  # 关闭坐标轴

        # 转换为字节流
        buffer = BytesIO()
        plt.savefig(buffer, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close()

        return buffer.getvalue()

    def generate_report(self, image_id, filename, prediction, confidence):
        """生成分析报告"""
        # 生成风险等级
        if prediction == 'fire':
            risk_level = '高风险' if confidence < 0.1 else '中风险'
            recommendation = '立即采取灭火措施，并疏散人员。'
        else:
            risk_level = '低风险'
            recommendation = '无需特殊处理，继续监控。'

        # 构建报告文本
        report = f"火灾分析报告\n"
        report += f"====================\n"
        report += f"图像ID: {image_id}\n"
        report += f"文件名: {filename}\n"
        report += f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"预测结果: {prediction}\n"
        report += f"置信度: {confidence:.4f}\n"
        report += f"风险等级: {risk_level}\n"
        report += f"建议措施: {recommendation}\n"
        report += f"====================\n"
        report += "报告生成完毕。"

        return report

    def save_report(self, image_id, risk_map, report_text):
        """保存报告到数据库"""
        self.report_cursor.execute(
            "INSERT INTO reports (image_id, risk_map, report_text) VALUES (?, ?, ?)",
            (image_id, risk_map, report_text)
        )
        self.report_conn.commit()

    def process_unreported_images(self):
        """处理所有未生成报告的图像"""
        unreported_images = self.get_unreported_images()
        print(f"找到 {len(unreported_images)} 张未生成报告的图像")

        for image in unreported_images:
            image_id, filename, prediction, confidence = image
            print(f"正在处理图像 {image_id}: {filename}")

            # 生成风险地图
            risk_map = self.generate_risk_map(image_id, prediction, confidence)

            # 生成报告
            report_text = self.generate_report(image_id, filename, prediction, confidence)

            # 保存报告
            self.save_report(image_id, risk_map, report_text)
            print(f"图像 {image_id} 的报告已生成")

        return len(unreported_images)

    def close(self):
        """关闭数据库连接"""
        self.image_conn.close()
        self.report_conn.close()

    def regenerate_report(self, image_id):
        """重新生成特定图像ID的报告"""
        # 检查图像是否存在
        self.image_cursor.execute("SELECT filename, prediction, confidence FROM images WHERE id = ?", (image_id,))
        image = self.image_cursor.fetchone()
        if not image:
            print(f"图像ID {image_id} 不存在")
            return False

        filename, prediction, confidence = image
        print(f"正在重新生成图像 {image_id}: {filename} 的报告")

        # 生成风险地图
        risk_map = self.generate_risk_map(image_id, prediction, confidence)

        # 生成报告
        report_text = self.generate_report(image_id, filename, prediction, confidence)

        # 删除旧报告
        self.report_cursor.execute("DELETE FROM reports WHERE image_id = ?", (image_id,))

        # 保存新报告
        self.save_report(image_id, risk_map, report_text)
        print(f"图像 {image_id} 的报告已重新生成")
        return True

    def regenerate_all_reports(self):
        """重新生成所有已处理图像的报告"""
        # 获取所有已处理的图像ID
        self.image_cursor.execute("SELECT id FROM images WHERE prediction IS NOT NULL")
        image_ids = [row[0] for row in self.image_cursor.fetchall()]
        print(f"找到 {len(image_ids)} 张已处理的图像，准备重新生成报告")

        # 对每个图像ID重新生成报告
        success_count = 0
        for image_id in image_ids:
            if self.regenerate_report(image_id):
                success_count += 1

        print(f"共重新生成 {success_count} 份报告")
        return success_count

if __name__ == "__main__":
    generator = ReportGenerator()
    try:
        # 处理未报告的图像
        count = generator.process_unreported_images()
        print(f"共生成 {count} 份新报告")

        # 重新生成所有已处理图像的报告
        generator.regenerate_all_reports()
    finally:
        generator.close()