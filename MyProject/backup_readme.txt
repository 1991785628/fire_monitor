# 代码备份脚本说明

## 功能介绍
该脚本用于自动备份指定目录下的代码文件，并保持最近3个版本的备份，旧版本将自动覆盖。

## 配置参数
- `SOURCE_DIR`: 源目录路径，默认为 'd:\MyProject\fire_monitor'
- `BACKUP_DIR`: 备份目录路径，默认为 'd:\MyProject\backups'
- `MAX_BACKUPS`: 最大备份数量，默认为 3
- `FILE_PATTERNS`: 需要备份的文件类型，默认为 ['.py', '.h5', '.keras']

## 使用方法
1. 直接运行 `backup_script.py` 文件
2. 或双击运行 `run_backup.bat` 批处理文件

## 备份规则
- 备份文件命名格式: 原文件名.v版本号 (例如: model_builder.py.v1)
- 每次备份时版本号自动递增
- 当备份数量超过 `MAX_BACKUPS` 时，自动删除最旧的备份

## 注意事项
- 确保源目录和备份目录路径正确
- 脚本会跳过 `__pycache__` 目录
- 只有符合 `FILE_PATTERNS` 的文件才会被备份