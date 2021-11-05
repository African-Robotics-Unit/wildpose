/********************************************************************************
** Form generated from reading UI file 'CameraSetupWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CAMERASETUPWIDGET_H
#define UI_CAMERASETUPWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CameraSetupWidget
{
public:
    QVBoxLayout *verticalLayout;
    QFormLayout *formLayout;
    QLabel *label;
    QDoubleSpinBox *spnFrameRate;
    QLabel *label_2;
    QSpinBox *spnExposureTime;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *CameraSetupWidget)
    {
        if (CameraSetupWidget->objectName().isEmpty())
            CameraSetupWidget->setObjectName(QStringLiteral("CameraSetupWidget"));
        CameraSetupWidget->resize(283, 99);
        verticalLayout = new QVBoxLayout(CameraSetupWidget);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        formLayout = new QFormLayout();
        formLayout->setObjectName(QStringLiteral("formLayout"));
        formLayout->setContentsMargins(-1, 6, 0, -1);
        label = new QLabel(CameraSetupWidget);
        label->setObjectName(QStringLiteral("label"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);

        formLayout->setWidget(0, QFormLayout::LabelRole, label);

        spnFrameRate = new QDoubleSpinBox(CameraSetupWidget);
        spnFrameRate->setObjectName(QStringLiteral("spnFrameRate"));
        spnFrameRate->setEnabled(false);
        spnFrameRate->setMinimum(1);

        formLayout->setWidget(0, QFormLayout::FieldRole, spnFrameRate);

        label_2 = new QLabel(CameraSetupWidget);
        label_2->setObjectName(QStringLiteral("label_2"));
        sizePolicy.setHeightForWidth(label_2->sizePolicy().hasHeightForWidth());
        label_2->setSizePolicy(sizePolicy);

        formLayout->setWidget(1, QFormLayout::LabelRole, label_2);

        spnExposureTime = new QSpinBox(CameraSetupWidget);
        spnExposureTime->setObjectName(QStringLiteral("spnExposureTime"));
        spnExposureTime->setEnabled(false);

        formLayout->setWidget(1, QFormLayout::FieldRole, spnExposureTime);


        verticalLayout->addLayout(formLayout);

        verticalSpacer = new QSpacerItem(20, 65, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(CameraSetupWidget);

        QMetaObject::connectSlotsByName(CameraSetupWidget);
    } // setupUi

    void retranslateUi(QWidget *CameraSetupWidget)
    {
        CameraSetupWidget->setWindowTitle(QApplication::translate("CameraSetupWidget", "Dialog", Q_NULLPTR));
        label->setText(QApplication::translate("CameraSetupWidget", "Frame rate", Q_NULLPTR));
        label_2->setText(QApplication::translate("CameraSetupWidget", "Exposure time (mcs)", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class CameraSetupWidget: public Ui_CameraSetupWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CAMERASETUPWIDGET_H
