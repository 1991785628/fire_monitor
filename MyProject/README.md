# 火灾监控系统 (Fire Monitor System)

## 项目概述
火灾监控系统是一个基于深度学习的火灾检测与风险评估平台。该系统能够通过图像识别技术检测火灾，并结合环境传感器数据进行风险评估和蔓延预测，为火灾防控提供决策支持。

## 主要功能
- 基于深度学习的火灾图像检测
- 多模态数据融合（图像+传感器数据）
- 火灾风险等级评估
- 火灾蔓延趋势预测
- 自动生成分析报告
- 图像数据管理与存储

## 技术栈
- 编程语言: Python 3.9+, C++ 17
- 深度学习框架: TensorFlow 2.x
- 数据库: SQLite
- 图像处理: PIL/Pillow
- 数据可视化: Matplotlib
- 界面框架: Qt 6.9.1 (Community Edition)
- 开发工具: Qt Creator 17.0 (Community Edition)
- 其他库: NumPy, Pandas, Scikit-learn

## 项目结构
```
MyProject/
├── fire_monitor/              # 主应用目录
│   ├── __pycache__/           # 编译后的Python文件
│   ├── data_loader.py         # 数据加载和预处理
│   ├── database.py            # 数据库交互
│   ├── fire_control_system.py # 系统主控类
│   ├── fire_detection_data/   # 火灾检测数据集
│   │   ├── all_fire_images/   # 所有火灾图像
│   │   ├── all_non_fire_images/ # 所有非火灾图像
│   │   ├── sensor_data/       # 传感器数据
│   │   ├── test/              # 测试集
│   │   ├── train/             # 训练集
│   │   └── val/               # 验证集
│   ├── fire_spread_prediction.py # 火灾蔓延预测
│   ├── generate_reports.py    # 报告生成
│   ├── image_database.db      # 图像数据库
│   ├── metrics.py             # 自定义评估指标
│   ├── model_builder.py       # 模型构建
│   ├── model_evaluation.py    # 模型评估
│   ├── models/                # 训练好的模型
│   ├── populate_database.py   # 数据库填充
│   ├── process_database_images.py # 数据库图像处理
│   ├── report_database.db     # 报告数据库
│   ├── single_image_simulation.py # 单图像模拟
│   ├── train_model.py         # 模型训练
│   └── view_results.py        # 结果查看
├── cpp_interface/             # C++接口（Qt框架）
│   ├── build/                 # 构建目录
│   ├── fire_monitor_interface.pro # Qt项目文件
│   ├── fire_monitor_interface.pro.user # 用户特定项目设置
│   ├── image_importer.pro     # 图像导入器项目文件
│   ├── main.cpp               # 主程序入口
│   ├── mainwindow.cpp         # 主窗口实现
│   ├── mainwindow.h           # 主窗口头文件
│   ├── mainwindow.ui          # 主窗口UI设计
│   └── resources.qrc          # 资源文件
├── models/                    # 模型备份
├── risk_evaluations/          # 风险评估结果
├── sensor_data/               # 传感器数据备份
├── backup_readme.txt          # 备份说明
├── backup_script.py           # 备份脚本
├── backups/                   # 代码备份
├── check_database.py          # 数据库检查
├── clear_database.py          # 清除数据库
├── image_database.db          # 图像数据库备份
├── image_database_backup.db   # 图像数据库备份
├── report_database.db         # 报告数据库备份
├── run_backup.bat             # 运行备份批处理
└── run_detection_pipeline.bat # 运行检测流程批处理
```

## 安装指南

### 前提条件
- Python 3.9或更高版本
- pip 21.0或更高版本
- Qt 6.9.1 (Community Edition)
- Qt Creator 17.0 (Community Edition)

### 安装步骤
1. 克隆或下载项目到本地
   ```
   git clone <repository-url>
   cd MyProject
   ```

2. 创建并激活虚拟环境
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. 安装依赖包
   ```
   pip install -r requirements.txt
   ```
   > 注意: 如果没有requirements.txt文件，请手动安装以下包：
   > tensorflow, numpy, pandas, scikit-learn, pillow, matplotlib, imbalanced-learn

## 使用方法

### 1. 准备数据
- 将火灾图像放入 `fire_monitor/fire_detection_data/all_fire_images/`
- 将非火灾图像放入 `fire_monitor/fire_detection_data/all_non_fire_images/`
- 将传感器数据CSV文件放入 `fire_monitor/fire_detection_data/sensor_data/`

### 2. 数据预处理与划分
```python
cd fire_monitor
python data_loader.py
```

### 3. 训练模型
```python
python train_model.py
```

### 4. 运行火灾检测系统
```python
python fire_control_system.py
```

### 5. 查看结果
- 检测结果和报告将保存在 `fire_control_results/` 目录下
- 风险评估图像将保存在 `risk_evaluations/` 目录下

## 模块说明

### data_loader.py
负责数据加载、预处理和划分。支持图像增强、数据集平衡和多模态数据加载。

### model_builder.py
定义了火灾检测模型的结构，包括单模态图像模型和多模态融合模型。使用MobileNetV2作为基础模型，并添加了注意力机制。

### train_model.py
实现模型训练流程，支持分阶段训练（冻结特征提取层和微调），并包含火灾风险评估功能。

### fire_control_system.py
系统主控类，整合火灾检测、风险评估和报告生成功能，提供统一的API接口。

### database.py
负责图像数据的存储和管理，支持图像插入、查询和更新操作。

## 贡献指南
1.  Fork 本项目
2.  创建特性分支 (`git checkout -b feature/fooBar`)
3.  提交更改 (`git commit -am 'Add some fooBar'`)
4.  推送到分支 (`git push origin feature/fooBar`)
5.  创建新的Pull Request

## 许可证
本项目采用MIT许可证 - 详情请见LICENSE文件

## 联系方式
如有问题或建议，请联系: [camellia@fmzxyd.onmicrosoft.com]

## Qt界面使用指南

### 1. 打开项目
1. 启动Qt Creator 17.0
2. 选择"文件" -> "打开文件或项目"
3. 导航到项目目录下的`cpp_interface`文件夹
4. 选择`fire_monitor_interface.pro`文件并打开

### 2. 编译项目
1. 选择合适的构建套件（已配置Qt 6.9.1）
2. 点击左下角的"构建"按钮（锤子图标）
3. 等待编译完成

### 3. 运行界面程序
1. 点击绿色的"运行"按钮
2. 程序将启动火灾监控系统的图形界面
3. 通过界面可以导入图像、运行检测和查看结果