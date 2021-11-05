/********************************************************************************
** Form generated from reading UI file 'camerastatistics.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CAMERASTATISTICS_H
#define UI_CAMERASTATISTICS_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CameraStatistics
{
public:
    QVBoxLayout *verticalLayout;
    QFormLayout *formLayout;
    QLabel *label;
    QLineEdit *leFramesTotal;
    QLabel *label_2;
    QLineEdit *leFramesDropped;
    QLabel *label_3;
    QLineEdit *leFramesIncomplete;
    QLabel *label_4;
    QLineEdit *leCurrFrameID;
    QLabel *label_5;
    QLabel *label_6;
    QLabel *label_7;
    QLineEdit *leCurrTimestamp;
    QLineEdit *leCurrFPS;
    QLineEdit *leCurrThoughput;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *CameraStatistics)
    {
        if (CameraStatistics->objectName().isEmpty())
            CameraStatistics->setObjectName(QStringLiteral("CameraStatistics"));
        CameraStatistics->resize(302, 315);
        verticalLayout = new QVBoxLayout(CameraStatistics);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        formLayout = new QFormLayout();
        formLayout->setObjectName(QStringLiteral("formLayout"));
        label = new QLabel(CameraStatistics);
        label->setObjectName(QStringLiteral("label"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label);

        leFramesTotal = new QLineEdit(CameraStatistics);
        leFramesTotal->setObjectName(QStringLiteral("leFramesTotal"));
        leFramesTotal->setReadOnly(true);

        formLayout->setWidget(0, QFormLayout::FieldRole, leFramesTotal);

        label_2 = new QLabel(CameraStatistics);
        label_2->setObjectName(QStringLiteral("label_2"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_2);

        leFramesDropped = new QLineEdit(CameraStatistics);
        leFramesDropped->setObjectName(QStringLiteral("leFramesDropped"));
        leFramesDropped->setReadOnly(true);

        formLayout->setWidget(1, QFormLayout::FieldRole, leFramesDropped);

        label_3 = new QLabel(CameraStatistics);
        label_3->setObjectName(QStringLiteral("label_3"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_3);

        leFramesIncomplete = new QLineEdit(CameraStatistics);
        leFramesIncomplete->setObjectName(QStringLiteral("leFramesIncomplete"));
        leFramesIncomplete->setReadOnly(true);

        formLayout->setWidget(2, QFormLayout::FieldRole, leFramesIncomplete);

        label_4 = new QLabel(CameraStatistics);
        label_4->setObjectName(QStringLiteral("label_4"));

        formLayout->setWidget(3, QFormLayout::LabelRole, label_4);

        leCurrFrameID = new QLineEdit(CameraStatistics);
        leCurrFrameID->setObjectName(QStringLiteral("leCurrFrameID"));
        leCurrFrameID->setReadOnly(true);

        formLayout->setWidget(3, QFormLayout::FieldRole, leCurrFrameID);

        label_5 = new QLabel(CameraStatistics);
        label_5->setObjectName(QStringLiteral("label_5"));

        formLayout->setWidget(4, QFormLayout::LabelRole, label_5);

        label_6 = new QLabel(CameraStatistics);
        label_6->setObjectName(QStringLiteral("label_6"));

        formLayout->setWidget(5, QFormLayout::LabelRole, label_6);

        label_7 = new QLabel(CameraStatistics);
        label_7->setObjectName(QStringLiteral("label_7"));

        formLayout->setWidget(6, QFormLayout::LabelRole, label_7);

        leCurrTimestamp = new QLineEdit(CameraStatistics);
        leCurrTimestamp->setObjectName(QStringLiteral("leCurrTimestamp"));
        leCurrTimestamp->setReadOnly(true);

        formLayout->setWidget(4, QFormLayout::FieldRole, leCurrTimestamp);

        leCurrFPS = new QLineEdit(CameraStatistics);
        leCurrFPS->setObjectName(QStringLiteral("leCurrFPS"));
        leCurrFPS->setReadOnly(true);

        formLayout->setWidget(5, QFormLayout::FieldRole, leCurrFPS);

        leCurrThoughput = new QLineEdit(CameraStatistics);
        leCurrThoughput->setObjectName(QStringLiteral("leCurrThoughput"));
        leCurrThoughput->setReadOnly(true);

        formLayout->setWidget(6, QFormLayout::FieldRole, leCurrThoughput);


        verticalLayout->addLayout(formLayout);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(CameraStatistics);

        QMetaObject::connectSlotsByName(CameraStatistics);
    } // setupUi

    void retranslateUi(QWidget *CameraStatistics)
    {
        CameraStatistics->setWindowTitle(QApplication::translate("CameraStatistics", "Form", Q_NULLPTR));
        label->setText(QApplication::translate("CameraStatistics", "Frames total", Q_NULLPTR));
        label_2->setText(QApplication::translate("CameraStatistics", "Frames dropped", Q_NULLPTR));
        label_3->setText(QApplication::translate("CameraStatistics", "Frames incomplete", Q_NULLPTR));
        label_4->setText(QApplication::translate("CameraStatistics", "Current Frame ID", Q_NULLPTR));
        label_5->setText(QApplication::translate("CameraStatistics", "Current Timestamp", Q_NULLPTR));
        label_6->setText(QApplication::translate("CameraStatistics", "FPS", Q_NULLPTR));
        label_7->setText(QApplication::translate("CameraStatistics", "Throughput [Mb/s]", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class CameraStatistics: public Ui_CameraStatistics {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CAMERASTATISTICS_H
