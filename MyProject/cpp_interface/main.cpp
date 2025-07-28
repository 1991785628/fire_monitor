#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle("火灾监测系统");
    w.resize(1024, 768);
    w.show();
    return a.exec();
}