/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.9.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QGroupBox *previewGroup;
    QVBoxLayout *verticalLayout;
    QLabel *imageLabel;
    QGroupBox *controlGroup;
    QVBoxLayout *verticalLayout_2;
    QHBoxLayout *horizontalLayout;
    QPushButton *processButton;
    QPushButton *refreshButton;
    QPushButton *simulateButton;
    QHBoxLayout *searchLayout;
    QLineEdit *searchLineEdit;
    QPushButton *searchButton;
    QTableWidget *resultsTable;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *prevButton;
    QPushButton *nextButton;
    QGroupBox *analysisImageGroup;
    QVBoxLayout *verticalLayout_3;
    QLabel *riskMapLabel;
    QGroupBox *reportAndLogGroup;
    QVBoxLayout *verticalLayout_6;
    QTextEdit *reportTextEdit;
    QTextEdit *logTextEdit;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(1400, 900);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName("gridLayout");
        previewGroup = new QGroupBox(centralwidget);
        previewGroup->setObjectName("previewGroup");
        verticalLayout = new QVBoxLayout(previewGroup);
        verticalLayout->setObjectName("verticalLayout");
        imageLabel = new QLabel(previewGroup);
        imageLabel->setObjectName("imageLabel");
        imageLabel->setAlignment(Qt::AlignmentFlag::AlignCenter);

        verticalLayout->addWidget(imageLabel);


        gridLayout->addWidget(previewGroup, 0, 0, 1, 1);

        controlGroup = new QGroupBox(centralwidget);
        controlGroup->setObjectName("controlGroup");
        verticalLayout_2 = new QVBoxLayout(controlGroup);
        verticalLayout_2->setObjectName("verticalLayout_2");
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName("horizontalLayout");
        processButton = new QPushButton(controlGroup);
        processButton->setObjectName("processButton");

        horizontalLayout->addWidget(processButton);

        refreshButton = new QPushButton(controlGroup);
        refreshButton->setObjectName("refreshButton");

        horizontalLayout->addWidget(refreshButton);

        simulateButton = new QPushButton(controlGroup);
        simulateButton->setObjectName("simulateButton");

        horizontalLayout->addWidget(simulateButton);


        verticalLayout_2->addLayout(horizontalLayout);

        searchLayout = new QHBoxLayout();
        searchLayout->setObjectName("searchLayout");
        searchLineEdit = new QLineEdit(controlGroup);
        searchLineEdit->setObjectName("searchLineEdit");

        searchLayout->addWidget(searchLineEdit);

        searchButton = new QPushButton(controlGroup);
        searchButton->setObjectName("searchButton");

        searchLayout->addWidget(searchButton);


        verticalLayout_2->addLayout(searchLayout);

        resultsTable = new QTableWidget(controlGroup);
        if (resultsTable->columnCount() < 5)
            resultsTable->setColumnCount(5);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        resultsTable->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        resultsTable->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        resultsTable->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        resultsTable->setHorizontalHeaderItem(3, __qtablewidgetitem3);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        resultsTable->setHorizontalHeaderItem(4, __qtablewidgetitem4);
        resultsTable->setObjectName("resultsTable");

        verticalLayout_2->addWidget(resultsTable);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName("horizontalLayout_2");
        prevButton = new QPushButton(controlGroup);
        prevButton->setObjectName("prevButton");

        horizontalLayout_2->addWidget(prevButton);

        nextButton = new QPushButton(controlGroup);
        nextButton->setObjectName("nextButton");

        horizontalLayout_2->addWidget(nextButton);


        verticalLayout_2->addLayout(horizontalLayout_2);


        gridLayout->addWidget(controlGroup, 0, 1, 1, 1);

        analysisImageGroup = new QGroupBox(centralwidget);
        analysisImageGroup->setObjectName("analysisImageGroup");
        verticalLayout_3 = new QVBoxLayout(analysisImageGroup);
        verticalLayout_3->setObjectName("verticalLayout_3");
        riskMapLabel = new QLabel(analysisImageGroup);
        riskMapLabel->setObjectName("riskMapLabel");
        riskMapLabel->setAlignment(Qt::AlignmentFlag::AlignCenter);

        verticalLayout_3->addWidget(riskMapLabel);


        gridLayout->addWidget(analysisImageGroup, 1, 0, 1, 1);

        reportAndLogGroup = new QGroupBox(centralwidget);
        reportAndLogGroup->setObjectName("reportAndLogGroup");
        verticalLayout_6 = new QVBoxLayout(reportAndLogGroup);
        verticalLayout_6->setObjectName("verticalLayout_6");
        reportTextEdit = new QTextEdit(reportAndLogGroup);
        reportTextEdit->setObjectName("reportTextEdit");
        reportTextEdit->setReadOnly(true);

        verticalLayout_6->addWidget(reportTextEdit);

        logTextEdit = new QTextEdit(reportAndLogGroup);
        logTextEdit->setObjectName("logTextEdit");
        logTextEdit->setReadOnly(true);

        verticalLayout_6->addWidget(logTextEdit);


        gridLayout->addWidget(reportAndLogGroup, 1, 1, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "\347\201\253\347\201\276\347\233\221\346\265\213\347\263\273\347\273\237", nullptr));
        previewGroup->setTitle(QCoreApplication::translate("MainWindow", "\345\233\276\345\203\217\351\242\204\350\247\210", nullptr));
        imageLabel->setText(QCoreApplication::translate("MainWindow", "\346\227\240\345\233\276\345\203\217\346\225\260\346\215\256", nullptr));
        controlGroup->setTitle(QCoreApplication::translate("MainWindow", "\345\244\204\347\220\206\350\257\206\345\210\253\345\233\276\345\203\217", nullptr));
        processButton->setText(QCoreApplication::translate("MainWindow", "\345\244\204\347\220\206\345\233\276\345\203\217", nullptr));
        refreshButton->setText(QCoreApplication::translate("MainWindow", "\345\210\267\346\226\260\347\273\223\346\236\234", nullptr));
        simulateButton->setText(QCoreApplication::translate("MainWindow", "\347\201\253\347\201\276\350\224\223\345\273\266\346\250\241\346\213\237", nullptr));
        searchLineEdit->setPlaceholderText(QCoreApplication::translate("MainWindow", "\350\276\223\345\205\245\345\233\276\347\211\207ID\346\220\234\347\264\242", nullptr));
        searchButton->setText(QCoreApplication::translate("MainWindow", "\346\220\234\347\264\242", nullptr));
        QTableWidgetItem *___qtablewidgetitem = resultsTable->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QCoreApplication::translate("MainWindow", "ID", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = resultsTable->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QCoreApplication::translate("MainWindow", "\346\226\207\344\273\266\345\220\215", nullptr));
        QTableWidgetItem *___qtablewidgetitem2 = resultsTable->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QCoreApplication::translate("MainWindow", "\351\242\204\346\265\213\347\273\223\346\236\234", nullptr));
        QTableWidgetItem *___qtablewidgetitem3 = resultsTable->horizontalHeaderItem(3);
        ___qtablewidgetitem3->setText(QCoreApplication::translate("MainWindow", "\347\275\256\344\277\241\345\272\246", nullptr));
        QTableWidgetItem *___qtablewidgetitem4 = resultsTable->horizontalHeaderItem(4);
        ___qtablewidgetitem4->setText(QCoreApplication::translate("MainWindow", "\345\244\204\347\220\206\346\227\266\351\227\264", nullptr));
        prevButton->setText(QCoreApplication::translate("MainWindow", "\344\270\212\344\270\200\351\241\265", nullptr));
        nextButton->setText(QCoreApplication::translate("MainWindow", "\344\270\213\344\270\200\351\241\265", nullptr));
        analysisImageGroup->setTitle(QCoreApplication::translate("MainWindow", "\347\201\253\347\201\276\345\210\206\346\236\220\345\233\276\345\203\217", nullptr));
        riskMapLabel->setText(QCoreApplication::translate("MainWindow", "\346\227\240\351\243\216\351\231\251\345\234\260\345\233\276\346\225\260\346\215\256", nullptr));
        reportAndLogGroup->setTitle(QCoreApplication::translate("MainWindow", "\345\210\206\346\236\220\346\212\245\345\221\212\344\270\216\346\227\245\345\277\227", nullptr));
        reportTextEdit->setPlaceholderText(QCoreApplication::translate("MainWindow", "\346\227\240\345\210\206\346\236\220\346\212\245\345\221\212\346\225\260\346\215\256", nullptr));
        logTextEdit->setPlaceholderText(QCoreApplication::translate("MainWindow", "\346\227\240\346\227\245\345\277\227\346\225\260\346\215\256", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
