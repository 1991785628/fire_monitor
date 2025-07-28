import sqlite3
import os

db_path = 'image_database.db'

# 检查数据库文件是否存在
if not os.path.exists(db_path):
    print(f"错误: 数据库文件 '{db_path}' 不存在")
    exit(1)

try:
    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 清空images表
    print("正在清空images表...")
    cursor.execute('DELETE FROM images')
    conn.commit()
    print("images表已成功清空")
    
    # 关闭连接
    conn.close()
    print("数据库连接已关闭")

except sqlite3.Error as e:
    print(f"数据库错误: {e}")
    exit(1)

except Exception as e:
    print(f"发生错误: {e}")
    exit(1)