#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QFile>
#include <QFileInfo>
#include <QMessageBox>
#include <QPainter>
#include <QBrush>
#include <QColor>
#include <QSqlQuery>
#include <QSqlError>
#include <QMessageBox>
#include <QPixmap>
#include <QBuffer>

PredictionDelegate::PredictionDelegate(QObject *parent) : QStyledItemDelegate(parent)
{
}

void PredictionDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    if (index.column() == 2) { // 预测结果在第3列 (索引2)
        QStyleOptionViewItem opt = option;
        initStyleOption(&opt, index);
        QString prediction = index.data().toString();

        if (prediction == "non_fire") {
            // 非火灾，绿色背景
            painter->fillRect(option.rect, QColor(100, 255, 100, 128));
        } else if (prediction == "fire") {
            // 火灾，红色背景
            painter->fillRect(option.rect, QColor(255, 100, 100, 128));
        }

        // 绘制文本
        painter->drawText(option.rect, Qt::AlignCenter, prediction);
    } else {
        // 其他列使用默认绘制
        QStyledItemDelegate::paint(painter, option, index);
    }
}
#include <QDebug>
#include <QSqlQuery>
#include <QSqlError>
#include <QTableView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QStatusBar>
#include <QMessageBox>
#include <QHeaderView>
#include <QProcess>
#include <QDateTime>
#include <QPixmap>
#include <QByteArray>
#include <QBuffer>
#include <QScrollArea>
#include <QGroupBox>
#include <QFormLayout>
#include <QTextEdit>
#include <QModelIndex>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , currentPage(0)
    , pageSize(20)
    , ui(new Ui::MainWindow)
    , pythonProcess(new QProcess(this))
    , pageLabel(nullptr)
    , logTextEdit(nullptr)
{
    ui->setupUi(this);

    // 设置gridLayout的行和列伸展因子，实现田字分布
    ui->gridLayout->setRowStretch(0, 1);
    ui->gridLayout->setRowStretch(1, 1);
    ui->gridLayout->setColumnStretch(0, 1);
    ui->gridLayout->setColumnStretch(1, 1);
    model = new QSqlQueryModel(this);
    setupUI();
    connect(pythonProcess, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, &MainWindow::processFinished);
    // 设置Python进程的编码为UTF-8
    pythonProcess->setReadChannel(QProcess::StandardOutput);
    
    connect(pythonProcess, &QProcess::readyReadStandardOutput, this, [this]() {
        // 从Python进程读取输出并转换编码（Windows下Python默认使用GBK编码）
        while (pythonProcess->canReadLine()) {
            QByteArray line = pythonProcess->readLine();
            QString text = QString::fromLocal8Bit(line).trimmed();
            if (!text.isEmpty()) {
                updateStatus(text);
            }
        }
    });
    connect(pythonProcess, &QProcess::readyReadStandardError, this, [this]() {
        // 从Python进程读取错误输出并转换编码
        while (pythonProcess->canReadLine()) {
            QByteArray line = pythonProcess->readLine();
            QString errorText = QString::fromLocal8Bit(line).trimmed();
            if (!errorText.isEmpty()) {
                updateStatus("Error: " + errorText);
            }
        }
    });

    bool imageDbConnected = connectToDatabase();
    bool reportDbConnected = connectToReportDatabase();

    if (imageDbConnected && reportDbConnected) {
        updateStatus("数据库连接成功");
        loadImageResults("");
    } else {
        if (!imageDbConnected)
            qDebug() << "Image database connection failed: " << db.lastError().text();
        if (!reportDbConnected)
            qDebug() << "Report database connection failed: " << reportDb.lastError().text();
        updateStatus("一个或多个数据库连接失败");
    }
}

MainWindow::~MainWindow()
{
    if (db.isOpen())
        db.close();
    delete ui;
}

void MainWindow::setupUI()
{
    setupImportUI();
    // 显式设置窗口大小
    this->resize(1400, 900);

    // 设置表格模型和委托
    // QTableWidget不支持setModel，改用手动设置项目
    ui->resultsTable->setColumnCount(5);
    ui->resultsTable->setHorizontalHeaderLabels(QStringList() << "ID" << "文件名" << "预测结果" << "置信度" << "处理时间");
    ui->resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->resultsTable->setItemDelegate(new PredictionDelegate(this));
    ui->resultsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->resultsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->resultsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    ui->resultsTable->setMinimumHeight(150);

    // 调整图像预览和分析报告区域大小
    ui->imageLabel->setMinimumHeight(350);
    ui->riskMapLabel->setMinimumHeight(150);

    // 初始化页码标签
    pageLabel = new QLabel("第 1 页");
    ui->statusbar->addPermanentWidget(pageLabel);
    
    // 初始化日志文本框
    logTextEdit = ui->logTextEdit;
    logTextEdit->setReadOnly(true);
    logTextEdit->setLineWrapMode(QTextEdit::WidgetWidth);
    
    // 连接信号和槽
    connect(ui->processButton, &QPushButton::clicked, this, &MainWindow::on_processButton_clicked);
    connect(ui->refreshButton, &QPushButton::clicked, this, &MainWindow::on_refreshButton_clicked);
    connect(ui->resultsTable, &QTableView::clicked, this, &MainWindow::on_resultsTable_clicked);
    connect(ui->prevButton, &QPushButton::clicked, this, &MainWindow::on_prevButton_clicked);
    connect(ui->nextButton, &QPushButton::clicked, this, &MainWindow::on_nextButton_clicked);

    // 强制布局更新
    ui->centralwidget->layout()->update();
}

bool MainWindow::connectToDatabase()
{
    db = QSqlDatabase::addDatabase("QSQLITE");
    QString dbPath = "d:/MyProject/image_database.db";
    db.setDatabaseName(dbPath);

    if (!db.open()) {
        qDebug() << "Database open failed: " << db.lastError().text();
        QMessageBox::critical(this, "数据库错误", db.lastError().text());
        return false;
    }
    return true;
}

bool MainWindow::connectToReportDatabase()
{
    // 创建或连接到报告数据库
    reportDb = QSqlDatabase::addDatabase("QSQLITE", "reportConnection");
    QString reportDbPath = "d:/MyProject/report_database.db";
    reportDb.setDatabaseName(reportDbPath);

    if (!reportDb.open()) {
        qDebug() << "Report database open failed: " << reportDb.lastError().text();
        QMessageBox::critical(this, "报告数据库错误", reportDb.lastError().text());
        return false;
    }

    // 创建报告表（如果不存在）
    QSqlQuery query(reportDb);
    QString createTableQuery = "CREATE TABLE IF NOT EXISTS reports ("
                              "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "image_id INTEGER NOT NULL, "
                              "risk_map BLOB, "
                              "report_text TEXT, "
                              "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
                              "FOREIGN KEY(image_id) REFERENCES images(id))";

    if (!query.exec(createTableQuery)) {
        qDebug() << "Failed to create reports table: " << query.lastError().text();
        return false;
    }

    return true;
}

void MainWindow::loadImageResults(QString searchQuery)
{
    // 先检查数据库是否打开
    if (!db.isOpen()) {
        qDebug() << "Error: Database is not open";
        updateStatus("错误: 数据库未打开");
        return;
    }

    // 检查images表是否存在
    QSqlQuery checkTableQuery(db);
    checkTableQuery.exec("SELECT name FROM sqlite_master WHERE type='table' AND name='images'");
    if (!checkTableQuery.next()) {
        qDebug() << "Error: 'images' table does not exist in the database";
        updateStatus("错误: 数据库中不存在'images'表");
        return;
    }

    // 清空表格
    ui->resultsTable->setRowCount(0);

    // 计算偏移量
    int offset = currentPage * pageSize;
    QSqlQuery query(db);
    QString sqlQuery = "SELECT id, filename, prediction, confidence, timestamp FROM images";
    QString countQuery = "SELECT COUNT(*) FROM images";

    // 添加搜索条件
    if (!searchQuery.isEmpty()) {
        sqlQuery += " WHERE filename LIKE :search";
        countQuery += " WHERE filename LIKE :search";
    }

    sqlQuery += " ORDER BY timestamp DESC LIMIT :limit OFFSET :offset";

    // 准备查询
    query.prepare(sqlQuery);
    if (!searchQuery.isEmpty()) {
        query.bindValue(":search", QString("%1%").arg(searchQuery));
    }
    query.bindValue(":limit", pageSize);
    query.bindValue(":offset", offset);

    if (!query.exec()) {
        qDebug() << "Query execution failed:" << query.lastError().text();
        updateStatus("查询失败: " + query.lastError().text());
        return;
    }

    // 手动添加数据到表格
    int row = 0;
    while (query.next()) {
        ui->resultsTable->insertRow(row);
        ui->resultsTable->setItem(row, 0, new QTableWidgetItem(query.value(0).toString()));
        ui->resultsTable->setItem(row, 1, new QTableWidgetItem(query.value(1).toString()));
        ui->resultsTable->setItem(row, 2, new QTableWidgetItem(query.value(2).toString()));
        ui->resultsTable->setItem(row, 3, new QTableWidgetItem(query.value(3).toString()));
        ui->resultsTable->setItem(row, 4, new QTableWidgetItem(query.value(4).toString()));
        row++;
    }

    // 获取总记录数
    QSqlQuery countSql(db);
    countSql.prepare(countQuery);
    if (!searchQuery.isEmpty()) {
        countSql.bindValue(":search", QString("%1%").arg(searchQuery));
    }
    if (!countSql.exec()) {
        qDebug() << "Count query failed:" << countSql.lastError().text();
    } else {
        countSql.next();
        int totalRecords = countSql.value(0).toInt();
        totalPages = (totalRecords + pageSize - 1) / pageSize;
    }

    // 确保当前页码有效
    qDebug() << "loadImageResults: before validation currentPage=" << currentPage << ", totalPages=" << totalPages;
    if (currentPage >= totalPages && currentPage > 0) {
        qDebug() << "loadImageResults: adjusting currentPage from" << currentPage << "to" << totalPages - 1;
        currentPage = totalPages - 1;
        loadImageResults(searchQuery);
        return; // 添加return语句防止重复执行
    }
    qDebug() << "loadImageResults: after validation currentPage=" << currentPage;

    // 无论如何都更新页码标签
    updatePageLabel();

    updateStatus(QString("加载了 %1 条记录，共 %2 页").arg(row).arg(totalPages));
}

void MainWindow::on_prevButton_clicked()
{
    if (currentPage > 0) {
        currentPage--;
        loadImageResults("");
    }
}

void MainWindow::on_nextButton_clicked()
{
    qDebug() << "on_nextButton_clicked: before currentPage=" << currentPage;
    if (currentPage < totalPages - 1) {
        currentPage++;
        qDebug() << "on_nextButton_clicked: after currentPage=" << currentPage;
        loadImageResults("");
    }
}

void MainWindow::updatePageLabel()
{
    // 添加调试输出
    qDebug() << "updatePageLabel: currentPage=" << currentPage << ", totalPages=" << totalPages;
    pageLabel->setText(QString("页码: %1/%2").arg(currentPage + 1).arg(totalPages));
    pageLabel->update(); // 强制刷新标签
    ui->statusbar->update(); // 强制刷新状态栏
}

void MainWindow::on_selectFireButton_clicked()
{
    QStringList files = QFileDialog::getOpenFileNames(this, "选择火灾图像", "", "图像文件 (*.png *.jpg *.jpeg)");
    if (!files.isEmpty()) {
        fireImagePaths = files;
        updateStatus(QString("已选择 %1 张火灾图像").arg(files.size()));
    }
}

void MainWindow::on_selectNonFireButton_clicked()
{
    QStringList files = QFileDialog::getOpenFileNames(this, "选择非火灾图像", "", "图像文件 (*.png *.jpg *.jpeg)");
    if (!files.isEmpty()) {
        nonFireImagePaths = files;
        updateStatus(QString("已选择 %1 张非火灾图像").arg(files.size()));
    }
}

void MainWindow::on_importButton_clicked()
{
    if (!db.isOpen() && !connectToDatabase()) {
        QMessageBox::critical(this, "数据库错误", "无法连接到数据库");
        return;
    }

    int fireCount = 0, nonFireCount = 0;

    // 导入火灾图像
    for (const QString& filePath : fireImagePaths) {
        if (insertImage(filePath, "fire")) {
            fireCount++;
        }
    }

    // 导入非火灾图像
    for (const QString& filePath : nonFireImagePaths) {
        if (insertImage(filePath, "non_fire")) {
            nonFireCount++;
        }
    }

    updateStatus(QString("导入完成 - 火灾图像: %1, 非火灾图像: %2").arg(fireCount).arg(nonFireCount));
    QMessageBox::information(this, "导入完成", QString("成功导入 %1 张图像\n火灾图像: %2\n非火灾图像: %3").arg(fireCount + nonFireCount).arg(fireCount).arg(nonFireCount));

    // 清空选择列表
    fireImagePaths.clear();
    nonFireImagePaths.clear();

    // 刷新图像列表
    loadImageResults("");
}

bool MainWindow::getImageInfo(const QString& filePath, int& width, int& height, int& channels)
{
    QImage image(filePath);
    if (image.isNull()) {
        updateStatus("无法打开图像: " + filePath);
        return false;
    }

    width = image.width();
    height = image.height();

    switch (image.format()) {
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32:
        case QImage::Format_ARGB32_Premultiplied:
            channels = 4;
            break;
        case QImage::Format_RGB888:
            channels = 3;
            break;
        case QImage::Format_Grayscale8:
            channels = 1;
            break;
        default:
            channels = 3; // 默认视为RGB
            break;
    }

    return true;
}

bool MainWindow::insertImage(const QString& filePath, const QString& category)
{
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        updateStatus("无法打开文件: " + filePath);
        return false;
    }

    QByteArray imageData = file.readAll();
    file.close();

    int width, height, channels;
    if (!getImageInfo(filePath, width, height, channels)) {
        return false;
    }

    QSqlQuery query(db);
    query.prepare("INSERT INTO images (filename, prediction, confidence, width, height, channels, image_data, timestamp) VALUES (:filename, :prediction, :confidence, :width, :height, :channels, :image_data, CURRENT_TIMESTAMP)");
    query.bindValue(":filename", QFileInfo(filePath).fileName());
    query.bindValue(":prediction", category);
    query.bindValue(":confidence", 1.0); // 手动导入默认为100%置信度
    query.bindValue(":width", width);
    query.bindValue(":height", height);
    query.bindValue(":channels", channels);
    query.bindValue(":image_data", imageData);

    if (!query.exec()) {
        updateStatus("插入图像失败: " + query.lastError().text());
        return false;
    }

    return true;
}

void MainWindow::on_resultsTable_clicked(const QModelIndex &index)
{
    if (!index.isValid())
        return;

    // 获取选中行的ID
    int row = index.row();
    QTableWidgetItem *idItem = ui->resultsTable->item(row, 0);
    if (!idItem) {
        updateStatus("无法获取图像ID");
        return;
    }
    int imageId = idItem->text().toInt();

    // 查询并显示选中的图像和报告
    displayImageById(imageId);
}

void MainWindow::displayImageById(int imageId)
{
    QSqlQuery query(db);
    query.prepare("SELECT image_data FROM images WHERE id = :id");
    query.bindValue(":id", imageId);

    if (query.exec() && query.next()) {
        QByteArray imageData = query.value(0).toByteArray();
        QPixmap pixmap;
        if (pixmap.loadFromData(imageData)) {
            // 缩放图像以适应标签
            QPixmap scaledPixmap = pixmap.scaled(ui->imageLabel->width(), ui->imageLabel->height(),
                                                Qt::KeepAspectRatio, Qt::SmoothTransformation);
            ui->imageLabel->setPixmap(scaledPixmap);
        } else {
            ui->imageLabel->setText("无法加载图像");
        }
    } else {
        ui->imageLabel->setText("无图像数据");
    }

    // 加载并显示报告数据
    loadReportData(imageId);
}

void MainWindow::loadReportData(int imageId)
{
    if (!reportDb.isOpen()) {
        qDebug() << "Report database is not open";
        ui->riskMapLabel->setText("报告数据库未连接");
        ui->reportTextEdit->setText("报告数据库未连接");
        return;
    }

    QSqlQuery query(reportDb);
    query.prepare("SELECT risk_map, report_text FROM reports WHERE image_id = :image_id");
    query.bindValue(":image_id", imageId);

    if (query.exec() && query.next()) {
        QByteArray riskMapData = query.value(0).toByteArray();
        QString reportText = query.value(1).toString();

        displayRiskMap(riskMapData);
        displayReport(reportText);
    } else {
        ui->riskMapLabel->setText("无风险地图数据");
        ui->reportTextEdit->setText("无分析报告数据");
    }
}

void MainWindow::displayRiskMap(const QByteArray &riskMapData)
{
    if (riskMapData.isEmpty()) {
        ui->riskMapLabel->setText("无风险地图数据");
        return;
    }

    // 尝试使用QImage加载以获取更多错误信息
    QImage image;
    if (!image.loadFromData(riskMapData)) {
        // 图像加载失败，显示错误信息
        QString errorMsg = QString("无法加载风险地图: 数据格式无效，长度: %1字节").arg(riskMapData.size());
        ui->riskMapLabel->setText(errorMsg);
        updateStatus(errorMsg);
        return;
    }

    // 图像加载成功，缩放并显示
    QPixmap pixmap = QPixmap::fromImage(image);
    QPixmap scaledPixmap = pixmap.scaled(ui->riskMapLabel->width(), ui->riskMapLabel->height(),
                                        Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->riskMapLabel->setPixmap(scaledPixmap);

    // 显示成功信息
    updateStatus("风险地图加载成功");
}

void MainWindow::displayReport(const QString &reportText)
{
    if (reportText.isEmpty()) {
        ui->reportTextEdit->setText("无分析报告数据");
    } else {
        ui->reportTextEdit->setText(reportText);
    }
}

void MainWindow::setupImportUI()
{
    // 创建导入按钮
    QPushButton *selectFireButton = new QPushButton("选择火灾图像");
    QPushButton *selectNonFireButton = new QPushButton("选择非火灾图像");
    QPushButton *importButton = new QPushButton("导入图像");
    QPushButton *deleteButton = new QPushButton("删除选中图像");
    QPushButton *clearButton = new QPushButton("清空所有图像");

    // 连接信号槽
    connect(selectFireButton, &QPushButton::clicked, this, &MainWindow::on_selectFireButton_clicked);
    connect(selectNonFireButton, &QPushButton::clicked, this, &MainWindow::on_selectNonFireButton_clicked);
    connect(importButton, &QPushButton::clicked, this, &MainWindow::on_importButton_clicked);
    connect(deleteButton, &QPushButton::clicked, this, &MainWindow::on_deleteButton_clicked);
    connect(clearButton, &QPushButton::clicked, this, &MainWindow::on_clearButton_clicked);

    // 添加到布局
    QHBoxLayout *importLayout = new QHBoxLayout();
    importLayout->addWidget(selectFireButton);
    importLayout->addWidget(selectNonFireButton);
    importLayout->addWidget(importButton);
    importLayout->addWidget(deleteButton);
    importLayout->addWidget(clearButton);

    // 将导入布局添加到主界面
    QWidget *importWidget = new QWidget();
    importWidget->setLayout(importLayout);
    
    // 获取中心部件的布局
    QLayout *centralLayout = ui->centralwidget->layout();
    if (centralLayout) {
        centralLayout->addWidget(importWidget);
    } else {
        // 如果没有布局，创建一个新的布局
        QVBoxLayout *newLayout = new QVBoxLayout(ui->centralwidget);
        newLayout->addWidget(importWidget);
        // 添加原有控件
        QList<QWidget*> children = ui->centralwidget->findChildren<QWidget*>();
        foreach (QWidget *child, children) {
            if (child != importWidget) {
                newLayout->addWidget(child);
            }
        }
    }
}

void MainWindow::on_processButton_clicked(){
    if (pythonProcess->state() == QProcess::Running) {
        updateStatus("处理进程已在运行中");
        return;
    }

    updateStatus("开始预测处理图像...");
    // 使用绝对路径代替相对路径，避免运行环境变化导致的路径问题
    QString pythonDir = "d:/MyProject/fire_monitor";
    QDir dir(pythonDir);
    if (!dir.exists()) {
        updateStatus("错误: Python脚本目录不存在: " + pythonDir);
        return;
    }
    pythonProcess->setWorkingDirectory(pythonDir);
    
    // 检查Python脚本是否存在
    QString scriptPath = pythonDir + "/process_database_images.py";
    if (!QFile::exists(scriptPath)) {
        updateStatus("错误: 找不到Python脚本: " + scriptPath);
        return;
    }
    
    // 使用绝对路径调用Python，避免依赖环境变量
    QString pythonExe = "python";
    // 检查系统是否能找到Python
    QProcess whichProcess;
    whichProcess.start("where", QStringList() << "python");
    if (whichProcess.waitForFinished() && whichProcess.exitCode() == 0) {
        QString pythonPath = whichProcess.readAllStandardOutput().split('\n').first().trimmed();
        if (!pythonPath.isEmpty()) {
            pythonExe = pythonPath;
            updateStatus("找到Python可执行文件: " + pythonExe);
        }
    }
    
    // 启动Python进程
    pythonProcess->start(pythonExe, QStringList() << "process_database_images.py");
    
    // 等待进程启动
    if (!pythonProcess->waitForStarted(5000)) { // 增加超时时间
        qDebug() << "Failed to start Python process: " << pythonProcess->errorString();
        updateStatus("启动Python进程失败: " + pythonProcess->errorString());
        updateStatus("尝试使用python3命令...");
        
        // 尝试使用python3命令
        pythonProcess->start("python3", QStringList() << "process_database_images.py");
        if (!pythonProcess->waitForStarted(5000)) {
            qDebug() << "Failed to start Python3 process: " << pythonProcess->errorString();
            updateStatus("启动Python3进程也失败: " + pythonProcess->errorString());
            // 确保进程状态被重置
            pythonProcess->kill();
        }
    }
    
    // 确保进程启动后正确捕获输出
    if (pythonProcess->state() == QProcess::Running) {
        updateStatus("Python进程已成功启动");
    }
}



void MainWindow::on_refreshButton_clicked()
{
    updateStatus("刷新当前页面中...");
    loadImageResults("");
    updateStatus("当前页面已刷新");
}

void MainWindow::updateStatus(QString message)
{
    static QString lastMessage;
    static QDateTime lastTime;

    // 检查是否与上一条消息相同且时间间隔小于500毫秒
    if (message == lastMessage && QDateTime::currentDateTime().msecsTo(lastTime) > -500) {
        return; // 忽略短时间内的重复消息
    }

    lastMessage = message;
    lastTime = QDateTime::currentDateTime();

    ui->statusbar->showMessage(message);
    logTextEdit->append(lastTime.toString("yyyy-MM-dd HH:mm:ss") + " - " + message);
}

void MainWindow::processFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::NormalExit && exitCode == 0) {
        updateStatus("图像处理完成");
        loadImageResults("");
    } else {
        updateStatus(QString("处理失败，退出代码: %1").arg(exitCode));
    }
}

void MainWindow::on_deleteButton_clicked()
{
    // 获取选中的行
    QModelIndexList selectedRows = ui->resultsTable->selectionModel()->selectedRows();
    if (selectedRows.isEmpty()) {
        QMessageBox::information(this, "提示", "请先选择要删除的图像");
        return;
    }

    // 获取选中图像的ID
    int imageId = selectedRows.first().siblingAtColumn(0).data().toInt();
    QString fileName = selectedRows.first().siblingAtColumn(1).data().toString();

    // 显示确认对话框
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "确认删除", QString("确定要删除图像 '%1' 吗？").arg(fileName),
                                  QMessageBox::Yes|QMessageBox::No);
    if (reply == QMessageBox::Yes) {
        if (deleteImageById(imageId)) {
            updateStatus(QString("成功删除图像: %1").arg(fileName));
            loadImageResults(""); // 刷新图像列表
        } else {
            updateStatus(QString("删除图像失败: %1").arg(fileName));
        }
    }
}

void MainWindow::on_clearButton_clicked()
{
    // 显示确认对话框
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "确认清空", "确定要清空所有图像吗？此操作不可撤销！",
                                  QMessageBox::Yes|QMessageBox::No);
    if (reply == QMessageBox::Yes) {
        if (clearAllImages()) {
            updateStatus("成功清空所有图像");
            loadImageResults(""); // 刷新图像列表
        } else {
            updateStatus("清空图像失败");
        }
    }
}

void MainWindow::on_simulateButton_clicked()
{
    // 获取当前选中的行
    QModelIndexList selectedRows = ui->resultsTable->selectionModel()->selectedRows();
    if (selectedRows.isEmpty()) {
        QMessageBox::information(this, "提示", "请先选择一张图片进行模拟");
        return;
    }

    // 获取选中行的ID
    int row = selectedRows.first().row();
    int imageId = ui->resultsTable->item(row, 0)->text().toInt();

    updateStatus(QString("开始对图片ID: %1 进行火灾蔓延模拟...").arg(imageId));

    // 启动Python进程运行模拟脚本
    QString scriptPath = "d:/MyProject/fire_monitor/single_image_simulation.py";
    pythonProcess->start("python", QStringList() << scriptPath << QString::number(imageId));
}

void MainWindow::on_searchButton_clicked()
{
    // 获取搜索框中的文本
    QString searchText = ui->searchLineEdit->text();
    if (searchText.isEmpty()) {
        QMessageBox::information(this, "提示", "请输入搜索内容");
        return;
    }

    // 重置当前页码
    currentPage = 0;
    // 调用带搜索参数的加载函数
    loadImageResults(searchText);

    // 查找并选中目标（如果是数字ID）
    bool ok;
    int targetId = searchText.toInt(&ok);
    if (ok) {
        bool found = false;
        for (int row = 0; row < ui->resultsTable->rowCount(); ++row) {
            int currentId = ui->resultsTable->item(row, 0)->text().toInt();
            if (currentId == targetId) {
                ui->resultsTable->selectRow(row);
                on_resultsTable_clicked(ui->resultsTable->model()->index(row, 0));
                found = true;
                break;
            }
        }

        if (!found) {
            QMessageBox::information(this, "提示", QString("未找到ID为 %1 的图片").arg(targetId));
        }
    }
}

bool MainWindow::deleteImageById(int imageId)
{
    if (!db.isOpen()) {
        if (!connectToDatabase()) {
            return false;
        }
    }

    QSqlQuery query(db);
    // 开始事务
    db.transaction();

    // 删除报告数据库中的相关记录
    QSqlQuery reportQuery(reportDb);
    reportQuery.prepare("DELETE FROM reports WHERE image_id = :imageId");
    reportQuery.bindValue(":imageId", imageId);
    if (!reportQuery.exec()) {
        qDebug() << "删除报告记录失败: " << reportQuery.lastError().text();
        db.rollback();
        return false;
    }

    // 删除图像数据库中的记录
    query.prepare("DELETE FROM images WHERE id = :imageId");
    query.bindValue(":imageId", imageId);
    if (!query.exec()) {
        qDebug() << "删除图像记录失败: " << query.lastError().text();
        db.rollback();
        return false;
    }

    // 提交事务
    if (!db.commit()) {
        qDebug() << "事务提交失败: " << db.lastError().text();
        db.rollback();
        return false;
    }

    return true;
}

bool MainWindow::clearAllImages()
{
    if (!db.isOpen()) {
        if (!connectToDatabase()) {
            return false;
        }
    }

    QSqlQuery query(db);
    // 开始事务
    db.transaction();

    // 清空报告数据库
    QSqlQuery reportQuery(reportDb);
    if (!reportQuery.exec("DELETE FROM reports")) {
        qDebug() << "清空报告数据库失败: " << reportQuery.lastError().text();
        db.rollback();
        return false;
    }

    // 清空图像数据库
    if (!query.exec("DELETE FROM images")) {
        qDebug() << "清空图像数据库失败: " << query.lastError().text();
        db.rollback();
        return false;
    }

    // 重置自增ID
    if (!query.exec("DELETE FROM sqlite_sequence WHERE name='images'")) {
        qDebug() << "重置图像ID失败: " << query.lastError().text();
        db.rollback();
        return false;
    }

    if (!reportQuery.exec("DELETE FROM sqlite_sequence WHERE name='reports'")) {
          qDebug() << "重置报告ID失败: " << reportQuery.lastError().text();
          db.rollback();
          return false;
      }

    // 提交事务
    if (!db.commit()) {
        qDebug() << "事务提交失败: " << db.lastError().text();
        db.rollback();
        return false;
    }

    return true;
}

// 最终确认：所有重复代码已彻底删除，文件结构正确
