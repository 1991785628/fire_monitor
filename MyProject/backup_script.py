import os
import shutil
import re
from datetime import datetime

# 配置参数
SOURCE_DIR = r'd:\MyProject\fire_monitor'
BACKUP_DIR = r'd:\MyProject\backups'
MAX_BACKUPS = 3
FILE_PATTERNS = ['.py', '.h5', '.keras']  # 需要备份的文件类型

print(f"源目录: {SOURCE_DIR}")
print(f"备份目录: {BACKUP_DIR}")
print(f"是否存在源目录: {os.path.exists(SOURCE_DIR)}")

def create_backup_dir():
    """创建备份目录如果不存在"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"创建备份目录: {BACKUP_DIR}")
    else:
        print(f"备份目录已存在: {BACKUP_DIR}")

def get_backup_versions(file_name):
    """获取文件的所有备份版本"""
    # 修改正则表达式，使其更灵活地匹配备份文件
    backup_pattern = re.compile(f"^{re.escape(file_name)}\.v(\d+)")
    versions = []
    
    if not os.path.exists(BACKUP_DIR):
        print(f"备份目录不存在: {BACKUP_DIR}")
        return versions
    
    print(f"扫描备份目录: {BACKUP_DIR} 中的文件 {file_name}")
    all_items = os.listdir(BACKUP_DIR)
    print(f"备份目录中的所有项目: {all_items}")
    
    for item in all_items:
        match = backup_pattern.match(item)
        if match:
            version = int(match.group(1))
            versions.append(version)
            print(f"找到匹配的备份: {item}, 版本号: {version}")
        else:
            print(f"不匹配的项目: {item}")
    
    sorted_versions = sorted(versions)
    print(f"排序后的版本号: {sorted_versions}")
    return sorted_versions

def backup_file(file_path):
    """备份单个文件"""
    file_name = os.path.basename(file_path)
    print(f"处理文件: {file_path}")
    
    # 检查文件类型是否需要备份
    if not any(file_name.endswith(pattern) for pattern in FILE_PATTERNS):
        print(f"跳过文件 {file_name}，不符合备份类型")
        return
    
    # 获取现有备份版本
    versions = get_backup_versions(file_name)
    print(f"文件 {file_name} 的现有备份版本: {versions}")
    
    # 确定新版本号
    if versions:
        new_version = max(versions) + 1
    else:
        new_version = 1
    print(f"新版本号: {new_version}")
    
    # 构建备份文件名
    backup_file_name = f"{file_name}.v{new_version}"
    backup_file_path = os.path.join(BACKUP_DIR, backup_file_name)
    print(f"备份路径: {backup_file_path}")
    
    try:
        # 复制文件进行备份
        shutil.copy2(file_path, backup_file_path)
        print(f"已成功备份: {file_path} -> {backup_file_path}")
    except Exception as e:
        print(f"备份失败: {str(e)}")
        return
    
    # 检查并删除旧备份
    if len(versions) >= MAX_BACKUPS:
        # 按版本号排序，删除最旧的
        versions.sort()
        old_versions = versions[:-MAX_BACKUPS+1]  # 保留最近MAX_BACKUPS-1个，因为我们刚添加了一个新的
        
        for old_version in old_versions:
            old_backup_name = f"{file_name}.v{old_version}"
            old_backup_path = os.path.join(BACKUP_DIR, old_backup_name)
            
            if os.path.exists(old_backup_path):
                try:
                    os.remove(old_backup_path)
                    print(f"已删除旧备份: {old_backup_path}")
                except Exception as e:
                    print(f"删除旧备份失败: {str(e)}")

def backup_all_files():
    """主函数"""
    create_backup_dir()
    
    # 只处理源目录下的指定类型文件，不递归子目录
    print(f"扫描源目录: {SOURCE_DIR}")
    for file in os.listdir(SOURCE_DIR):
        file_path = os.path.join(SOURCE_DIR, file)
        # 只处理文件，不处理目录
        if os.path.isfile(file_path):
            print(f"处理文件: {file_path}")
            
            # 检查文件类型是否符合备份要求
            if any(file.endswith(ext) for ext in FILE_PATTERNS):
                backup_file(file_path)
            else:
                print(f"跳过文件 {file}，不符合备份类型")
    
    print("备份完成!")

if __name__ == "__main__":
    backup_all_files()