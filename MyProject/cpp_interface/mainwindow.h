#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSqlDatabase>
#include <QSqlQueryModel>
#include <QTimer>
#include <QProcess>
#include <QLabel>
#include <QTextEdit>
#include <QModelIndex>
#include <QStyledItemDelegate>

class PredictionDelegate : public QStyledItemDelegate
{
    Q_OBJECT
public:
    explicit PredictionDelegate(QObject *parent = nullptr);
    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override;
};


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    int currentPage = 0;
    int pageSize = 20;
    int totalPages = 0;

private slots:
    void on_processButton_clicked();
    void on_refreshButton_clicked();
    void updateStatus(QString message);
    void processFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void on_prevButton_clicked();
    void on_nextButton_clicked();
    void updatePageLabel();
    void loadImageResults(QString searchQuery = QString());
    void on_resultsTable_clicked(const QModelIndex &index);
    void on_selectFireButton_clicked();
    void on_selectNonFireButton_clicked();
    void on_importButton_clicked();
    void on_deleteButton_clicked();
    void on_clearButton_clicked();
    void on_simulateButton_clicked();
    void on_searchButton_clicked();

private:
    Ui::MainWindow *ui;
    QSqlDatabase db;
    QSqlDatabase reportDb;
    QSqlQueryModel *model;
    QProcess *pythonProcess;
    QLabel *pageLabel;
    QTextEdit *logTextEdit;
    bool connectToDatabase();
    bool connectToReportDatabase();
    void setupUI();
    void setupImportUI();
    void displayImageById(int imageId);
    void loadReportData(int imageId);
    void displayRiskMap(const QByteArray &riskMapData);
    void displayReport(const QString &reportText);
    bool insertImage(const QString& filePath, const QString& category);
    bool getImageInfo(const QString& filePath, int& width, int& height, int& channels);
    bool deleteImageById(int imageId);
    bool clearAllImages();

private:
    QStringList fireImagePaths;
    QStringList nonFireImagePaths;
};

#endif // MAINWINDOW_H