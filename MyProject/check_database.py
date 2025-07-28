import sqlite3
import os
import sys
import time
from fire_monitor.fire_spread_prediction import FireSpreadModel
from fire_monitor.fire_control_system import FireControlSystem

def check_database():
    try:
        # 连接数据库
        conn = sqlite3.connect('image_database.db')
        cursor = conn.cursor()
        print("成功连接到数据库")

        # 查询最新处理的记录
        print("\n最新处理的图像记录:\n")
        cursor.execute('''
            SELECT id, filename, prediction, confidence, processed, timestamp 
            FROM images 
            WHERE processed = 1
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        processed_results = cursor.fetchall()

        if not processed_results:
            print("没有找到已处理的记录")
        else:
            # 打印已处理结果表头
            print(f"{'ID':<5} {'文件名':<20} {'预测':<10} {'置信度':<10} {'处理状态':<10} {'时间戳'}")
            print("-" * 80)
            
            # 打印已处理结果行
            for row in processed_results:
                try:
                    image_id, filename, prediction, confidence, processed, timestamp = row
                    
                    # 调试信息
                    print(f"调试: 已处理行数据={row}")
                    
                    # 处理各个字段
                    image_id_str = str(image_id) if image_id is not None else "N/A"
                    filename_str = filename if filename is not None else "N/A"
                    prediction_str = prediction if prediction is not None else "N/A"
                    confidence_str = f"{confidence:.4f}" if isinstance(confidence, (int, float)) else "N/A"
                    processed_str = "是" if processed else "否"
                    timestamp_str = timestamp if timestamp is not None else "N/A"
                    
                    print(f"{image_id_str:<5} {filename_str:<20} {prediction_str:<10} {confidence_str:<10} {processed_str:<10} {timestamp_str}")
                except Exception as e:
                    print(f"处理已处理行时出错: {e}, 行数据={row}")

        # 查询未处理的记录
        print("\n未处理的图像记录:\n")
        cursor.execute('''
            SELECT id, filename, prediction, confidence, processed, timestamp 
            FROM images 
            WHERE processed = 0
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        unprocessed_results = cursor.fetchall()

        if not unprocessed_results:
            print("没有找到未处理的记录")
        else:
            # 打印未处理结果表头
            print(f"{'ID':<5} {'文件名':<20} {'预测':<10} {'置信度':<10} {'处理状态':<10} {'时间戳'}")
            print("-" * 80)
            
            # 打印未处理结果行
            for row in unprocessed_results:
                try:
                    image_id, filename, prediction, confidence, processed, timestamp = row
                    
                    # 调试信息
                    print(f"调试: 行数据={row}")
                    
                    # 处理各个字段
                    image_id_str = str(image_id) if image_id is not None else "N/A"
                    filename_str = filename if filename is not None else "N/A"
                    prediction_str = prediction if prediction is not None else "N/A"
                    confidence_str = f"{confidence:.4f}" if isinstance(confidence, (int, float)) else "N/A"
                    processed_str = "是" if processed else "否"
                    timestamp_str = timestamp if timestamp is not None else "N/A"
                    
                    print(f"{image_id_str:<5} {filename_str:<20} {prediction_str:<10} {confidence_str:<10} {processed_str:<10} {timestamp_str}")
                except Exception as e:
                    print(f"处理行时出错: {e}, 行数据={row}")

    except Exception as e:
        print(f"查询错误: {e}")
    finally:
        # 关闭数据库连接
        if 'conn' in locals():
            conn.close()
            print("数据库连接已关闭")

def check_and_add_columns():    
    try:
        conn = sqlite3.connect('image_database.db')
        cursor = conn.cursor()
        
        # 检查analysis_report列是否存在
        cursor.execute("PRAGMA table_info(images)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'analysis_report' not in columns:
            print("添加analysis_report列到images表...")
            cursor.execute("ALTER TABLE images ADD COLUMN analysis_report TEXT")
            conn.commit()
            print("添加列成功")
        
        conn.close()
    except Exception as e:
        print(f"检查/添加列时出错: {e}")

def process_fire_analysis():
    try:
        # 检查并添加必要的列
        check_and_add_columns()
        conn = sqlite3.connect('image_database.db')
        cursor = conn.cursor()
        print("\n开始火灾分析处理...")
        
        # 初始化火灾蔓延模型和防控系统
        fire_spread_model = FireSpreadModel()
        fire_control_system = FireControlSystem()
        
        # 获取需要分析的图像（已预测为火灾且未分析的图像）
        try:
            cursor.execute('''
                SELECT id, filename, image_data
                FROM images 
                WHERE prediction = 'fire' AND processed = 1
                LIMIT 10
            ''')
            images_to_analyze = cursor.fetchall()
        except sqlite3.OperationalError as e:
            # 处理可能不存在的analyzed列
            print(f"查询错误: {e}，尝试不使用analyzed列")
            cursor.execute('''
                SELECT id, filename, image_data
                FROM images 
                WHERE prediction = 'fire' AND processed = 1
                LIMIT 10
            ''')
            images_to_analyze = cursor.fetchall()
        
        if not images_to_analyze:
            print("没有需要分析的火灾图像")
            return
        
        print(f"找到 {len(images_to_analyze)} 张需要分析的火灾图像")
        
        for image_id, filename, image_data in images_to_analyze:
            print(f"\n分析图像: {filename} (ID: {image_id})")
            
            # 保存图像临时文件用于分析
            temp_image_path = f"temp_{image_id}.jpg"
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
            
            try:
                # 检测火灾
                detection_result = fire_control_system.detect_fire(temp_image_path)
                print(f"火灾检测结果: {'是' if detection_result['fire_detected'] else '否'}, 概率: {detection_result['fire_probability']:.4f}")
                
                if detection_result['fire_detected']:
                    # 设置初始起火点进行模拟
                    fire_spread_model.set_initial_fire([(50, 50)])
                    
                    # 运行火灾蔓延模拟
                    print("运行火灾蔓延模拟...")
                    history = fire_spread_model.run_simulation(30)
                    
                    # 生成风险图
                    fire_spread_model.visualize_risk()
                    
                    # 准备分析结果
                    analysis_result = {
                        'detection_summary': {
                            'total_images': 1,
                            'fire_images': 1,
                            'average_probability': detection_result['fire_probability']
                        },
                        'risk_assessment': {
                            'risk_level': '高',
                            'max_risk_value': 0.9,
                            'risk_map_path': 'risk_evaluation.png'
                        },
                        'timestamp': time.time()
                    }
                    
                    # 生成报告
                    report_path = fire_control_system.generate_report(analysis_result)
                    print(f"生成火灾控制报告: {report_path}")
                    
                    # 更新数据库
                    try:
                        cursor.execute('''
                            UPDATE images
                            SET analysis_report = ?
                            WHERE id = ?
                        ''', (report_path, image_id))
                        conn.commit()
                        print(f"数据库已更新，图像ID: {image_id}")
                    except sqlite3.OperationalError as e:
                        print(f"更新数据库时出错: {e}，可能缺少analysis_report列")
                else:
                    print("该图像未检测到火灾，跳过分析")
                
            except Exception as e:
                print(f"分析图像时出错: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
    except Exception as e:
        print(f"火灾分析处理错误: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("数据库连接已关闭")

def check_table_structure():
    try:
        conn = sqlite3.connect('image_database.db')
        cursor = conn.cursor()
        
        # 查询表结构
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='images';")
        result = cursor.fetchone()
        if result:
            print("\n表结构:\n")
            print(result[0])
        else:
            print("未找到images表")
    except Exception as e:
        print(f"查询表结构错误: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_table_structure()
    check_database()
    process_fire_analysis()