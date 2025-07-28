QT += core gui sql widgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = fire_monitor_interface
TEMPLATE = app

SOURCES += main.cpp \
           mainwindow.cpp

HEADERS += mainwindow.h

FORMS += mainwindow.ui

RESOURCES += resources.qrc

# 数据库支持
INCLUDEPATH += ../
DEPENDPATH += ../

# 编译器设置
QMAKE_CXXFLAGS += -std=c++17

# Qt SQL模块已包含SQLite支持，无需额外链接