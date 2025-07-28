# 火灾监测系统 - C++ 交互界面

## 环境搭建

1. 安装 Qt 6.9.1 (Community Edition) 和 Qt Creator 17.0 (Community Edition)
   - 下载地址: https://www.qt.io/download
   - 安装时确保勾选 "Qt 6.9.1" 和 "MinGW 11.2.0 64-bit" 组件

2. 安装 SQLite 开发库
   - Windows: 可通过 Qt 安装程序一并安装
   - Linux: `sudo apt-get install libsqlite3-dev`
   - macOS: `brew install sqlite3`

3. 确保已安装 Python 3.9 或更高版本
   - 并安装必要的 Python 依赖: `pip install tensorflow pillow numpy pandas scikit-learn matplotlib`

## 编译运行

1. 启动 Qt Creator 17.0
2. 选择 "文件" -> "打开文件或项目"
3. 导航到项目目录下的 `cpp_interface` 文件夹
4. 选择 `fire_monitor_interface.pro` 文件并打开
5. 选择合适的构建套件（已配置 Qt 6.9.1 和 MinGW 64-bit）
6. 点击左下角的"构建"按钮（锤子图标）编译项目
7. 点击绿色的"运行"按钮启动应用程序

## 功能说明

1. **处理未识别图像** 按钮: 调用 Python 脚本处理数据库中未识别的图像
2. **刷新结果** 按钮: 从数据库刷新显示最新的识别结果
3. **结果表格**: 显示最近 20 条识别记录，包括 ID、文件名、预测结果、置信度和处理时间
4. **图像预览**: 显示最新识别图像的预览
5. **状态栏**: 显示当前操作状态和系统消息

## 项目结构
```
cpp_interface/
├── build/                 # 构建目录
├── fire_monitor_interface.pro # Qt项目文件
├── fire_monitor_interface.pro.user # 用户特定项目设置
├── image_importer.pro     # 图像导入器项目文件
├── main.cpp               # 主程序入口
├── mainwindow.cpp         # 主窗口实现
├── mainwindow.h           # 主窗口头文件
├── mainwindow.ui          # 主窗口UI设计
└── resources.qrc          # 资源文件
```

## 故障排除

1. **数据库连接失败**
   - 检查 `image_database.db` 文件是否存在于项目根目录
   - 确保 Qt 已正确安装 SQLite 驱动

2. **图像处理失败**
   - 检查 Python 是否正确安装
   - 确保 `process_database_images.py` 脚本存在于 `fire_monitor` 目录
   - 检查 Python 依赖是否已安装

3. **编译错误**
   - 确保 Qt 版本为 6.9.1
   - 检查构建套件配置是否正确
   - 确认所有源文件已正确加载
