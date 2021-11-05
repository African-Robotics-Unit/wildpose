/********************************************************************************
** Form generated from reading UI file 'DenoiseController.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DENOISECONTROLLER_H
#define UI_DENOISECONTROLLER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DenoiseController
{
public:
    QGridLayout *gridLayout;
    QTabWidget *tabWidget;
    QWidget *tabCommon;
    QGridLayout *gridLayout_3;
    QLabel *label_3;
    QComboBox *cboThresholdType;
    QComboBox *cboWaveletType;
    QDoubleSpinBox *spnMaxColor;
    QDoubleSpinBox *spnMaxY;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_6;
    QLabel *label_5;
    QSpinBox *spnDecompLevel;
    QSpacerItem *verticalSpacer;
    QWidget *tabY;
    QGridLayout *gridLayout_2;
    QHBoxLayout *horizontalLayout;
    QSlider *sldY0;
    QSlider *sldY1;
    QSlider *sldY2;
    QSlider *sldY3;
    QSlider *sldY4;
    QSlider *sldYThreshold;
    QToolButton *btnResetY;
    QLabel *lblCurIntensity;
    QLabel *label_7;
    QWidget *tabCb;
    QGridLayout *gridLayout_5;
    QSlider *sldCbThreshold;
    QToolButton *btnResetCb;
    QLabel *lblCurCb;
    QLabel *label_8;
    QHBoxLayout *horizontalLayout_2;
    QSlider *sldCb0;
    QSlider *sldCb1;
    QSlider *sldCb2;
    QSlider *sldCb3;
    QSlider *sldCb4;
    QWidget *tabCr;
    QGridLayout *gridLayout_4;
    QLabel *label_9;
    QSlider *sldCrThreshold;
    QLabel *lblCurCr;
    QToolButton *btnResetCr;
    QHBoxLayout *horizontalLayout_3;
    QSlider *sldCr0;
    QSlider *sldCr1;
    QSlider *sldCr2;
    QSlider *sldCr3;
    QSlider *sldCr4;
    QCheckBox *chkSyncColor;
    QCheckBox *chkUseDenoise;

    void setupUi(QWidget *DenoiseController)
    {
        if (DenoiseController->objectName().isEmpty())
            DenoiseController->setObjectName(QStringLiteral("DenoiseController"));
        DenoiseController->resize(384, 295);
        gridLayout = new QGridLayout(DenoiseController);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(6, 6, 6, 6);
        tabWidget = new QTabWidget(DenoiseController);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setEnabled(true);
        tabWidget->setStyleSheet(QStringLiteral(""));
        tabCommon = new QWidget();
        tabCommon->setObjectName(QStringLiteral("tabCommon"));
        gridLayout_3 = new QGridLayout(tabCommon);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        label_3 = new QLabel(tabCommon);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_3->addWidget(label_3, 5, 0, 1, 1);

        cboThresholdType = new QComboBox(tabCommon);
        cboThresholdType->setObjectName(QStringLiteral("cboThresholdType"));

        gridLayout_3->addWidget(cboThresholdType, 3, 2, 1, 2);

        cboWaveletType = new QComboBox(tabCommon);
        cboWaveletType->setObjectName(QStringLiteral("cboWaveletType"));

        gridLayout_3->addWidget(cboWaveletType, 4, 2, 1, 2);

        spnMaxColor = new QDoubleSpinBox(tabCommon);
        spnMaxColor->setObjectName(QStringLiteral("spnMaxColor"));
        spnMaxColor->setMaximum(65535);
        spnMaxColor->setValue(1000);

        gridLayout_3->addWidget(spnMaxColor, 7, 2, 1, 2);

        spnMaxY = new QDoubleSpinBox(tabCommon);
        spnMaxY->setObjectName(QStringLiteral("spnMaxY"));
        spnMaxY->setMaximum(65535);
        spnMaxY->setValue(100);

        gridLayout_3->addWidget(spnMaxY, 6, 2, 1, 2);

        label = new QLabel(tabCommon);
        label->setObjectName(QStringLiteral("label"));

        gridLayout_3->addWidget(label, 3, 0, 1, 1);

        label_2 = new QLabel(tabCommon);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout_3->addWidget(label_2, 4, 0, 1, 1);

        label_6 = new QLabel(tabCommon);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout_3->addWidget(label_6, 7, 0, 1, 1);

        label_5 = new QLabel(tabCommon);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_3->addWidget(label_5, 6, 0, 1, 1);

        spnDecompLevel = new QSpinBox(tabCommon);
        spnDecompLevel->setObjectName(QStringLiteral("spnDecompLevel"));
        spnDecompLevel->setMinimum(1);
        spnDecompLevel->setMaximum(11);
        spnDecompLevel->setValue(5);

        gridLayout_3->addWidget(spnDecompLevel, 5, 2, 1, 2);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_3->addItem(verticalSpacer, 8, 2, 1, 1);

        tabWidget->addTab(tabCommon, QString());
        tabY = new QWidget();
        tabY->setObjectName(QStringLiteral("tabY"));
        gridLayout_2 = new QGridLayout(tabY);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        sldY0 = new QSlider(tabY);
        sldY0->setObjectName(QStringLiteral("sldY0"));
        sldY0->setMinimum(-200);
        sldY0->setMaximum(200);
        sldY0->setValue(0);
        sldY0->setOrientation(Qt::Vertical);
        sldY0->setTickPosition(QSlider::TicksAbove);
        sldY0->setTickInterval(40);

        horizontalLayout->addWidget(sldY0);

        sldY1 = new QSlider(tabY);
        sldY1->setObjectName(QStringLiteral("sldY1"));
        sldY1->setMinimum(-200);
        sldY1->setMaximum(200);
        sldY1->setValue(0);
        sldY1->setOrientation(Qt::Vertical);
        sldY1->setTickPosition(QSlider::TicksAbove);
        sldY1->setTickInterval(40);

        horizontalLayout->addWidget(sldY1);

        sldY2 = new QSlider(tabY);
        sldY2->setObjectName(QStringLiteral("sldY2"));
        sldY2->setMinimum(-200);
        sldY2->setMaximum(200);
        sldY2->setValue(0);
        sldY2->setOrientation(Qt::Vertical);
        sldY2->setTickPosition(QSlider::TicksAbove);
        sldY2->setTickInterval(40);

        horizontalLayout->addWidget(sldY2);

        sldY3 = new QSlider(tabY);
        sldY3->setObjectName(QStringLiteral("sldY3"));
        sldY3->setMinimum(-200);
        sldY3->setMaximum(200);
        sldY3->setValue(0);
        sldY3->setOrientation(Qt::Vertical);
        sldY3->setTickPosition(QSlider::TicksAbove);
        sldY3->setTickInterval(40);

        horizontalLayout->addWidget(sldY3);

        sldY4 = new QSlider(tabY);
        sldY4->setObjectName(QStringLiteral("sldY4"));
        sldY4->setMinimum(-200);
        sldY4->setMaximum(200);
        sldY4->setValue(0);
        sldY4->setOrientation(Qt::Vertical);
        sldY4->setTickPosition(QSlider::TicksAbove);
        sldY4->setTickInterval(40);

        horizontalLayout->addWidget(sldY4);


        gridLayout_2->addLayout(horizontalLayout, 1, 0, 1, 4);

        sldYThreshold = new QSlider(tabY);
        sldYThreshold->setObjectName(QStringLiteral("sldYThreshold"));
        sldYThreshold->setMaximum(10000);
        sldYThreshold->setSingleStep(10);
        sldYThreshold->setPageStep(100);
        sldYThreshold->setValue(5000);
        sldYThreshold->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(sldYThreshold, 0, 1, 1, 1);

        btnResetY = new QToolButton(tabY);
        btnResetY->setObjectName(QStringLiteral("btnResetY"));
        btnResetY->setMinimumSize(QSize(23, 22));
        btnResetY->setMaximumSize(QSize(23, 22));
        QIcon icon;
        icon.addFile(QStringLiteral(":/res/reset.svg"), QSize(), QIcon::Normal, QIcon::Off);
        btnResetY->setIcon(icon);

        gridLayout_2->addWidget(btnResetY, 0, 3, 1, 1);

        lblCurIntensity = new QLabel(tabY);
        lblCurIntensity->setObjectName(QStringLiteral("lblCurIntensity"));
        lblCurIntensity->setMinimumSize(QSize(35, 0));
        lblCurIntensity->setFrameShape(QFrame::StyledPanel);

        gridLayout_2->addWidget(lblCurIntensity, 0, 2, 1, 1);

        label_7 = new QLabel(tabY);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout_2->addWidget(label_7, 0, 0, 1, 1);

        tabWidget->addTab(tabY, QString());
        tabCb = new QWidget();
        tabCb->setObjectName(QStringLiteral("tabCb"));
        gridLayout_5 = new QGridLayout(tabCb);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        sldCbThreshold = new QSlider(tabCb);
        sldCbThreshold->setObjectName(QStringLiteral("sldCbThreshold"));
        sldCbThreshold->setMaximum(100000);
        sldCbThreshold->setSingleStep(10);
        sldCbThreshold->setPageStep(100);
        sldCbThreshold->setValue(25000);
        sldCbThreshold->setOrientation(Qt::Horizontal);

        gridLayout_5->addWidget(sldCbThreshold, 0, 1, 1, 1);

        btnResetCb = new QToolButton(tabCb);
        btnResetCb->setObjectName(QStringLiteral("btnResetCb"));
        btnResetCb->setMinimumSize(QSize(23, 22));
        btnResetCb->setMaximumSize(QSize(23, 22));
        btnResetCb->setIcon(icon);

        gridLayout_5->addWidget(btnResetCb, 0, 3, 1, 1);

        lblCurCb = new QLabel(tabCb);
        lblCurCb->setObjectName(QStringLiteral("lblCurCb"));
        lblCurCb->setMinimumSize(QSize(35, 0));
        lblCurCb->setFrameShape(QFrame::StyledPanel);

        gridLayout_5->addWidget(lblCurCb, 0, 2, 1, 1);

        label_8 = new QLabel(tabCb);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout_5->addWidget(label_8, 0, 0, 1, 1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        sldCb0 = new QSlider(tabCb);
        sldCb0->setObjectName(QStringLiteral("sldCb0"));
        sldCb0->setMinimum(-200);
        sldCb0->setMaximum(200);
        sldCb0->setValue(0);
        sldCb0->setOrientation(Qt::Vertical);
        sldCb0->setTickPosition(QSlider::TicksAbove);
        sldCb0->setTickInterval(40);

        horizontalLayout_2->addWidget(sldCb0);

        sldCb1 = new QSlider(tabCb);
        sldCb1->setObjectName(QStringLiteral("sldCb1"));
        sldCb1->setMinimum(-200);
        sldCb1->setMaximum(200);
        sldCb1->setValue(0);
        sldCb1->setOrientation(Qt::Vertical);
        sldCb1->setTickPosition(QSlider::TicksAbove);
        sldCb1->setTickInterval(40);

        horizontalLayout_2->addWidget(sldCb1);

        sldCb2 = new QSlider(tabCb);
        sldCb2->setObjectName(QStringLiteral("sldCb2"));
        sldCb2->setMinimum(-200);
        sldCb2->setMaximum(200);
        sldCb2->setValue(0);
        sldCb2->setOrientation(Qt::Vertical);
        sldCb2->setTickPosition(QSlider::TicksAbove);
        sldCb2->setTickInterval(40);

        horizontalLayout_2->addWidget(sldCb2);

        sldCb3 = new QSlider(tabCb);
        sldCb3->setObjectName(QStringLiteral("sldCb3"));
        sldCb3->setMinimum(-200);
        sldCb3->setMaximum(200);
        sldCb3->setValue(0);
        sldCb3->setOrientation(Qt::Vertical);
        sldCb3->setTickPosition(QSlider::TicksAbove);
        sldCb3->setTickInterval(40);

        horizontalLayout_2->addWidget(sldCb3);

        sldCb4 = new QSlider(tabCb);
        sldCb4->setObjectName(QStringLiteral("sldCb4"));
        sldCb4->setMinimum(-200);
        sldCb4->setMaximum(200);
        sldCb4->setValue(0);
        sldCb4->setOrientation(Qt::Vertical);
        sldCb4->setTickPosition(QSlider::TicksAbove);
        sldCb4->setTickInterval(40);

        horizontalLayout_2->addWidget(sldCb4);


        gridLayout_5->addLayout(horizontalLayout_2, 4, 0, 1, 4);

        tabWidget->addTab(tabCb, QString());
        tabCr = new QWidget();
        tabCr->setObjectName(QStringLiteral("tabCr"));
        gridLayout_4 = new QGridLayout(tabCr);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        label_9 = new QLabel(tabCr);
        label_9->setObjectName(QStringLiteral("label_9"));

        gridLayout_4->addWidget(label_9, 0, 0, 1, 1);

        sldCrThreshold = new QSlider(tabCr);
        sldCrThreshold->setObjectName(QStringLiteral("sldCrThreshold"));
        sldCrThreshold->setMaximum(100000);
        sldCrThreshold->setSingleStep(10);
        sldCrThreshold->setPageStep(100);
        sldCrThreshold->setValue(25000);
        sldCrThreshold->setOrientation(Qt::Horizontal);

        gridLayout_4->addWidget(sldCrThreshold, 0, 1, 1, 1);

        lblCurCr = new QLabel(tabCr);
        lblCurCr->setObjectName(QStringLiteral("lblCurCr"));
        lblCurCr->setMinimumSize(QSize(35, 0));
        lblCurCr->setFrameShape(QFrame::StyledPanel);

        gridLayout_4->addWidget(lblCurCr, 0, 2, 1, 1);

        btnResetCr = new QToolButton(tabCr);
        btnResetCr->setObjectName(QStringLiteral("btnResetCr"));
        btnResetCr->setMinimumSize(QSize(23, 22));
        btnResetCr->setMaximumSize(QSize(23, 22));
        btnResetCr->setIcon(icon);

        gridLayout_4->addWidget(btnResetCr, 0, 3, 1, 1);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        sldCr0 = new QSlider(tabCr);
        sldCr0->setObjectName(QStringLiteral("sldCr0"));
        sldCr0->setMinimum(-200);
        sldCr0->setMaximum(200);
        sldCr0->setValue(0);
        sldCr0->setOrientation(Qt::Vertical);
        sldCr0->setTickPosition(QSlider::TicksAbove);
        sldCr0->setTickInterval(40);

        horizontalLayout_3->addWidget(sldCr0);

        sldCr1 = new QSlider(tabCr);
        sldCr1->setObjectName(QStringLiteral("sldCr1"));
        sldCr1->setMinimum(-200);
        sldCr1->setMaximum(200);
        sldCr1->setValue(0);
        sldCr1->setOrientation(Qt::Vertical);
        sldCr1->setTickPosition(QSlider::TicksAbove);
        sldCr1->setTickInterval(40);

        horizontalLayout_3->addWidget(sldCr1);

        sldCr2 = new QSlider(tabCr);
        sldCr2->setObjectName(QStringLiteral("sldCr2"));
        sldCr2->setMinimum(-200);
        sldCr2->setMaximum(200);
        sldCr2->setValue(0);
        sldCr2->setOrientation(Qt::Vertical);
        sldCr2->setTickPosition(QSlider::TicksAbove);
        sldCr2->setTickInterval(40);

        horizontalLayout_3->addWidget(sldCr2);

        sldCr3 = new QSlider(tabCr);
        sldCr3->setObjectName(QStringLiteral("sldCr3"));
        sldCr3->setMinimum(-200);
        sldCr3->setMaximum(200);
        sldCr3->setValue(0);
        sldCr3->setOrientation(Qt::Vertical);
        sldCr3->setTickPosition(QSlider::TicksAbove);
        sldCr3->setTickInterval(40);

        horizontalLayout_3->addWidget(sldCr3);

        sldCr4 = new QSlider(tabCr);
        sldCr4->setObjectName(QStringLiteral("sldCr4"));
        sldCr4->setMinimum(-200);
        sldCr4->setMaximum(200);
        sldCr4->setValue(0);
        sldCr4->setOrientation(Qt::Vertical);
        sldCr4->setTickPosition(QSlider::TicksAbove);
        sldCr4->setTickInterval(40);

        horizontalLayout_3->addWidget(sldCr4);


        gridLayout_4->addLayout(horizontalLayout_3, 4, 0, 1, 4);

        tabWidget->addTab(tabCr, QString());

        gridLayout->addWidget(tabWidget, 1, 0, 1, 3);

        chkSyncColor = new QCheckBox(DenoiseController);
        chkSyncColor->setObjectName(QStringLiteral("chkSyncColor"));
        chkSyncColor->setChecked(true);

        gridLayout->addWidget(chkSyncColor, 0, 2, 1, 1);

        chkUseDenoise = new QCheckBox(DenoiseController);
        chkUseDenoise->setObjectName(QStringLiteral("chkUseDenoise"));

        gridLayout->addWidget(chkUseDenoise, 0, 0, 1, 2);


        retranslateUi(DenoiseController);

        tabWidget->setCurrentIndex(1);
        cboThresholdType->setCurrentIndex(1);
        cboWaveletType->setCurrentIndex(-1);


        QMetaObject::connectSlotsByName(DenoiseController);
    } // setupUi

    void retranslateUi(QWidget *DenoiseController)
    {
        DenoiseController->setWindowTitle(QApplication::translate("DenoiseController", "Form", Q_NULLPTR));
        label_3->setText(QApplication::translate("DenoiseController", "Decomp level", Q_NULLPTR));
        cboThresholdType->clear();
        cboThresholdType->insertItems(0, QStringList()
         << QApplication::translate("DenoiseController", "Hard", Q_NULLPTR)
         << QApplication::translate("DenoiseController", "Soft", Q_NULLPTR)
         << QApplication::translate("DenoiseController", "Garrote", Q_NULLPTR)
        );
        label->setText(QApplication::translate("DenoiseController", "Threshold type", Q_NULLPTR));
        label_2->setText(QApplication::translate("DenoiseController", "Wavelet type", Q_NULLPTR));
        label_6->setText(QApplication::translate("DenoiseController", "Max color thr.", Q_NULLPTR));
        label_5->setText(QApplication::translate("DenoiseController", "Max Y thr.", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tabCommon), QApplication::translate("DenoiseController", "Common", Q_NULLPTR));
        btnResetY->setText(QString());
        lblCurIntensity->setText(QApplication::translate("DenoiseController", "50.00", Q_NULLPTR));
        label_7->setText(QApplication::translate("DenoiseController", "Y thresh.", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tabY), QApplication::translate("DenoiseController", "Y", Q_NULLPTR));
        btnResetCb->setText(QString());
        lblCurCb->setText(QApplication::translate("DenoiseController", "250.00", Q_NULLPTR));
        label_8->setText(QApplication::translate("DenoiseController", "Cb thresh.", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tabCb), QApplication::translate("DenoiseController", "Cb", Q_NULLPTR));
        label_9->setText(QApplication::translate("DenoiseController", "Cr thresh.", Q_NULLPTR));
        lblCurCr->setText(QApplication::translate("DenoiseController", "250.00", Q_NULLPTR));
        btnResetCr->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tabCr), QApplication::translate("DenoiseController", "Cr", Q_NULLPTR));
        chkSyncColor->setText(QApplication::translate("DenoiseController", "Sync Cb and Cr data", Q_NULLPTR));
        chkUseDenoise->setText(QApplication::translate("DenoiseController", "RGB denoise", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DenoiseController: public Ui_DenoiseController {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DENOISECONTROLLER_H
