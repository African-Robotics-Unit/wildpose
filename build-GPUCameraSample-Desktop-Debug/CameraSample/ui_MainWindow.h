/********************************************************************************
** Form generated from reading UI file 'MainWindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QToolButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "CameraSetupWidget.h"
#include "DenoiseController.h"
#include "Widgets/GtGWidget.h"
#include "camerastatistics.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpenCamera;
    QAction *actionPlay;
    QAction *actionStop;
    QAction *actionRecord;
    QAction *actionExit;
    QAction *actionOptions;
    QAction *actionWB_picker;
    QAction *actionOpenBayerPGM;
    QAction *actionOpenGrayPGM;
    QAction *actionShowImage;
    QWidget *centralWidget;
    QVBoxLayout *MediaViewerLayout;
    QMenuBar *menuBar;
    QMenu *menuCamera;
    QMenu *menuTools;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QGridLayout *gridLayout_2;
    QToolButton *btnGetGrayFile;
    QLabel *label_39;
    QLineEdit *txtFPNFileName;
    QToolButton *btnGetFPNFile;
    QLineEdit *txtFlatFieldFile;
    QLabel *label_13;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_32;
    QComboBox *cboBayerPattern;
    QLabel *label_2;
    QComboBox *cboBayerType;
    QDockWidget *processWidget;
    QWidget *dockWidgetContents_3;
    QGridLayout *gridLayout;
    QGroupBox *groupBox_8;
    QGridLayout *gridLayout_24;
    QCheckBox *chkZoomFit;
    QLabel *lblZoom;
    QToolButton *btnResetZoom;
    QSlider *sldZoom;
    QLabel *label_47;
    QComboBox *cboCUDADevice;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QComboBox *cboGamma;
    QHBoxLayout *horizontalLayout_3;
    QCheckBox *chkBPC;
    QCheckBox *chkSAM;
    QDockWidget *colorCorrectionWidget;
    QWidget *dockWidgetContents_5;
    QGridLayout *gridLayout_5;
    QLabel *lblRed;
    QSlider *sldRed;
    QToolButton *btnResetRed;
    QLabel *lblGreen;
    QSlider *sldGreen;
    QToolButton *btnResetGreen;
    QLabel *lblBlue;
    QSlider *sldBlue;
    QToolButton *btnResetBlue;
    QDockWidget *exposureWidget;
    QWidget *dockWidgetContents_6;
    QGridLayout *gridLayout_7;
    QLabel *lblEV;
    QSlider *sldEV;
    QToolButton *btnResetEV;
    QDockWidget *denoiseWidget;
    QWidget *dockWidgetContents_7;
    QVBoxLayout *verticalLayout_3;
    DenoiseController *denoiseCtlr;
    QDockWidget *benchMarksWidget;
    QWidget *dockWidgetContents_8;
    QVBoxLayout *verticalLayout_2;
    QPlainTextEdit *lblInfo;
    QDockWidget *recordingWidget;
    QWidget *dockWidgetContents_2;
    QGridLayout *gridLayout_3;
    QLabel *label_3;
    QLabel *label_4;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_6;
    QLineEdit *txtOutPath;
    QToolButton *btnGetOutPath;
    QLabel *label_7;
    QSpinBox *spnJpegQty;
    QLabel *label_17;
    QLineEdit *txtFilePrefix;
    QComboBox *cboSamplingFmt;
    QSpacerItem *verticalSpacer_2;
    QComboBox *cboOutFormat;
    QLabel *label_5;
    QComboBox *cbBitrate;
    QDockWidget *gtgDockWidget;
    QWidget *dockWidgetContents_4;
    QHBoxLayout *gtgLayout;
    GtGWidget *gtgWidget;
    QDockWidget *cameraSettingsWidget;
    QWidget *dockWidgetContents_9;
    QVBoxLayout *verticalLayout;
    CameraSetupWidget *cameraController;
    QDockWidget *cameraStatWidget;
    QWidget *dockWidgetContents_10;
    QVBoxLayout *verticalLayout1;
    CameraStatistics *cameraStatistics;
    QDockWidget *dw_RtspServer;
    QWidget *dockWidgetContents_91;
    QVBoxLayout *verticalLayout2;
    QHBoxLayout *horizontalLayout_7;
    QLabel *label_9;
    QComboBox *cboFormatEnc;
    QStackedWidget *swRtspOptions;
    QWidget *page;
    QVBoxLayout *verticalLayout_4;
    QHBoxLayout *horizontalLayout_9;
    QLabel *label_12;
    QComboBox *cboSamplingFmtRtsp;
    QLabel *label_11;
    QSpinBox *spnJpegQtyRtsp;
    QWidget *page_2;
    QVBoxLayout *verticalLayout_5;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label_10;
    QComboBox *cbBitrateRtsp;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_8;
    QLineEdit *txtRtspServer;
    QLabel *lblStatusRtspServer;
    QHBoxLayout *horizontalLayout_6;
    QPushButton *btnStartRtspServer;
    QPushButton *btnStopRtspServer;
    QSpacerItem *verticalSpacer;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1571, 1354);
        MainWindow->setDockNestingEnabled(true);
        actionOpenCamera = new QAction(MainWindow);
        actionOpenCamera->setObjectName(QStringLiteral("actionOpenCamera"));
        QIcon icon;
        icon.addFile(QStringLiteral(":/res/camera.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpenCamera->setIcon(icon);
        actionPlay = new QAction(MainWindow);
        actionPlay->setObjectName(QStringLiteral("actionPlay"));
        actionPlay->setCheckable(true);
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/res/play_fwd.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionPlay->setIcon(icon1);
        actionStop = new QAction(MainWindow);
        actionStop->setObjectName(QStringLiteral("actionStop"));
        QIcon icon2;
        icon2.addFile(QStringLiteral(":/res/stop.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionStop->setIcon(icon2);
        actionRecord = new QAction(MainWindow);
        actionRecord->setObjectName(QStringLiteral("actionRecord"));
        actionRecord->setCheckable(true);
        QIcon icon3;
        icon3.addFile(QStringLiteral(":/res/record.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionRecord->setIcon(icon3);
        actionExit = new QAction(MainWindow);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        QIcon icon4;
        icon4.addFile(QStringLiteral(":/res/exit.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionExit->setIcon(icon4);
        actionOptions = new QAction(MainWindow);
        actionOptions->setObjectName(QStringLiteral("actionOptions"));
        QIcon icon5;
        icon5.addFile(QStringLiteral(":/res/gear.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionOptions->setIcon(icon5);
        actionWB_picker = new QAction(MainWindow);
        actionWB_picker->setObjectName(QStringLiteral("actionWB_picker"));
        actionWB_picker->setCheckable(true);
        QIcon icon6;
        icon6.addFile(QStringLiteral(":/res/WBPicker.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionWB_picker->setIcon(icon6);
        actionOpenBayerPGM = new QAction(MainWindow);
        actionOpenBayerPGM->setObjectName(QStringLiteral("actionOpenBayerPGM"));
        QIcon icon7;
        icon7.addFile(QStringLiteral(":/res/image.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpenBayerPGM->setIcon(icon7);
        actionOpenGrayPGM = new QAction(MainWindow);
        actionOpenGrayPGM->setObjectName(QStringLiteral("actionOpenGrayPGM"));
        QIcon icon8;
        icon8.addFile(QStringLiteral(":/res/imageGray.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpenGrayPGM->setIcon(icon8);
        actionShowImage = new QAction(MainWindow);
        actionShowImage->setObjectName(QStringLiteral("actionShowImage"));
        actionShowImage->setCheckable(true);
        actionShowImage->setChecked(true);
        QIcon icon9;
        icon9.addFile(QStringLiteral(":/res/PicturesPlace.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionShowImage->setIcon(icon9);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        MediaViewerLayout = new QVBoxLayout(centralWidget);
        MediaViewerLayout->setSpacing(6);
        MediaViewerLayout->setContentsMargins(11, 11, 11, 11);
        MediaViewerLayout->setObjectName(QStringLiteral("MediaViewerLayout"));
        MediaViewerLayout->setContentsMargins(0, 0, 0, 0);
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1571, 22));
        menuCamera = new QMenu(menuBar);
        menuCamera->setObjectName(QStringLiteral("menuCamera"));
        menuTools = new QMenu(menuBar);
        menuTools->setObjectName(QStringLiteral("menuTools"));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);
        dockWidget = new QDockWidget(MainWindow);
        dockWidget->setObjectName(QStringLiteral("dockWidget"));
        dockWidget->setMinimumSize(QSize(367, 178));
        dockWidget->setMaximumSize(QSize(524287, 180));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QStringLiteral("dockWidgetContents"));
        gridLayout_2 = new QGridLayout(dockWidgetContents);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(6, 6, 6, 6);
        btnGetGrayFile = new QToolButton(dockWidgetContents);
        btnGetGrayFile->setObjectName(QStringLiteral("btnGetGrayFile"));

        gridLayout_2->addWidget(btnGetGrayFile, 5, 3, 1, 1);

        label_39 = new QLabel(dockWidgetContents);
        label_39->setObjectName(QStringLiteral("label_39"));
        label_39->setMinimumSize(QSize(120, 0));

        gridLayout_2->addWidget(label_39, 2, 0, 1, 2);

        txtFPNFileName = new QLineEdit(dockWidgetContents);
        txtFPNFileName->setObjectName(QStringLiteral("txtFPNFileName"));

        gridLayout_2->addWidget(txtFPNFileName, 3, 0, 1, 3);

        btnGetFPNFile = new QToolButton(dockWidgetContents);
        btnGetFPNFile->setObjectName(QStringLiteral("btnGetFPNFile"));

        gridLayout_2->addWidget(btnGetFPNFile, 3, 3, 1, 1);

        txtFlatFieldFile = new QLineEdit(dockWidgetContents);
        txtFlatFieldFile->setObjectName(QStringLiteral("txtFlatFieldFile"));

        gridLayout_2->addWidget(txtFlatFieldFile, 5, 0, 1, 3);

        label_13 = new QLabel(dockWidgetContents);
        label_13->setObjectName(QStringLiteral("label_13"));
        label_13->setMinimumSize(QSize(120, 0));
        label_13->setMaximumSize(QSize(120, 16777215));

        gridLayout_2->addWidget(label_13, 4, 0, 1, 2);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        label_32 = new QLabel(dockWidgetContents);
        label_32->setObjectName(QStringLiteral("label_32"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label_32->sizePolicy().hasHeightForWidth());
        label_32->setSizePolicy(sizePolicy);
        label_32->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout_2->addWidget(label_32);

        cboBayerPattern = new QComboBox(dockWidgetContents);
        cboBayerPattern->setObjectName(QStringLiteral("cboBayerPattern"));

        horizontalLayout_2->addWidget(cboBayerPattern);

        label_2 = new QLabel(dockWidgetContents);
        label_2->setObjectName(QStringLiteral("label_2"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label_2->sizePolicy().hasHeightForWidth());
        label_2->setSizePolicy(sizePolicy1);
        label_2->setMaximumSize(QSize(60, 16777215));

        horizontalLayout_2->addWidget(label_2);

        cboBayerType = new QComboBox(dockWidgetContents);
        cboBayerType->setObjectName(QStringLiteral("cboBayerType"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(cboBayerType->sizePolicy().hasHeightForWidth());
        cboBayerType->setSizePolicy(sizePolicy2);
        cboBayerType->setStyleSheet(QLatin1String("QComboBox  QAbstractItemView {\n"
"	selection-background-color: rgb(136, 136, 136);\n"
"	background-color: rgb(64, 64, 64);\n"
"}\n"
"\n"
"QComboBox:editable {\n"
"	selection-background-color: rgb(136, 136, 136);\n"
"}"));

        horizontalLayout_2->addWidget(cboBayerType);


        gridLayout_2->addLayout(horizontalLayout_2, 0, 0, 1, 4);

        dockWidget->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(2), dockWidget);
        processWidget = new QDockWidget(MainWindow);
        processWidget->setObjectName(QStringLiteral("processWidget"));
        processWidget->setMinimumSize(QSize(213, 224));
        processWidget->setMaximumSize(QSize(524287, 224));
        dockWidgetContents_3 = new QWidget();
        dockWidgetContents_3->setObjectName(QStringLiteral("dockWidgetContents_3"));
        gridLayout = new QGridLayout(dockWidgetContents_3);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(6, 6, 6, 6);
        groupBox_8 = new QGroupBox(dockWidgetContents_3);
        groupBox_8->setObjectName(QStringLiteral("groupBox_8"));
        gridLayout_24 = new QGridLayout(groupBox_8);
        gridLayout_24->setSpacing(6);
        gridLayout_24->setContentsMargins(11, 11, 11, 11);
        gridLayout_24->setObjectName(QStringLiteral("gridLayout_24"));
        gridLayout_24->setContentsMargins(6, 6, 6, 6);
        chkZoomFit = new QCheckBox(groupBox_8);
        chkZoomFit->setObjectName(QStringLiteral("chkZoomFit"));
        chkZoomFit->setCheckable(true);
        chkZoomFit->setChecked(true);

        gridLayout_24->addWidget(chkZoomFit, 3, 0, 1, 2);

        lblZoom = new QLabel(groupBox_8);
        lblZoom->setObjectName(QStringLiteral("lblZoom"));
        lblZoom->setMaximumSize(QSize(30, 16777215));
        lblZoom->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout_24->addWidget(lblZoom, 0, 0, 1, 1);

        btnResetZoom = new QToolButton(groupBox_8);
        btnResetZoom->setObjectName(QStringLiteral("btnResetZoom"));
        btnResetZoom->setEnabled(true);
        btnResetZoom->setMinimumSize(QSize(22, 22));
        QIcon icon10;
        icon10.addFile(QStringLiteral(":/res/reset.svg"), QSize(), QIcon::Normal, QIcon::Off);
        btnResetZoom->setIcon(icon10);
        btnResetZoom->setAutoRepeat(true);

        gridLayout_24->addWidget(btnResetZoom, 0, 2, 1, 1);

        sldZoom = new QSlider(groupBox_8);
        sldZoom->setObjectName(QStringLiteral("sldZoom"));
        sldZoom->setEnabled(true);
        sldZoom->setMinimum(10);
        sldZoom->setMaximum(1000);
        sldZoom->setValue(99);
        sldZoom->setOrientation(Qt::Horizontal);

        gridLayout_24->addWidget(sldZoom, 0, 1, 1, 1);


        gridLayout->addWidget(groupBox_8, 4, 0, 1, 2);

        label_47 = new QLabel(dockWidgetContents_3);
        label_47->setObjectName(QStringLiteral("label_47"));

        gridLayout->addWidget(label_47, 2, 0, 1, 1);

        cboCUDADevice = new QComboBox(dockWidgetContents_3);
        cboCUDADevice->setObjectName(QStringLiteral("cboCUDADevice"));
        cboCUDADevice->setStyleSheet(QLatin1String("QComboBox  QAbstractItemView {\n"
"	selection-background-color: rgb(136, 136, 136);\n"
"	background-color: rgb(64, 64, 64);\n"
"}\n"
"\n"
"QComboBox:editable {\n"
"	selection-background-color: rgb(136, 136, 136);\n"
"}"));

        gridLayout->addWidget(cboCUDADevice, 3, 0, 1, 2);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(dockWidgetContents_3);
        label->setObjectName(QStringLiteral("label"));
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);

        horizontalLayout->addWidget(label);

        cboGamma = new QComboBox(dockWidgetContents_3);
        cboGamma->setObjectName(QStringLiteral("cboGamma"));

        horizontalLayout->addWidget(cboGamma);


        gridLayout->addLayout(horizontalLayout, 8, 0, 1, 2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        chkBPC = new QCheckBox(dockWidgetContents_3);
        chkBPC->setObjectName(QStringLiteral("chkBPC"));
        chkBPC->setChecked(true);

        horizontalLayout_3->addWidget(chkBPC);

        chkSAM = new QCheckBox(dockWidgetContents_3);
        chkSAM->setObjectName(QStringLiteral("chkSAM"));
        chkSAM->setChecked(true);

        horizontalLayout_3->addWidget(chkSAM);


        gridLayout->addLayout(horizontalLayout_3, 5, 0, 1, 2);

        processWidget->setWidget(dockWidgetContents_3);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(2), processWidget);
        colorCorrectionWidget = new QDockWidget(MainWindow);
        colorCorrectionWidget->setObjectName(QStringLiteral("colorCorrectionWidget"));
        colorCorrectionWidget->setMinimumSize(QSize(97, 122));
        colorCorrectionWidget->setMaximumSize(QSize(524287, 122));
        dockWidgetContents_5 = new QWidget();
        dockWidgetContents_5->setObjectName(QStringLiteral("dockWidgetContents_5"));
        gridLayout_5 = new QGridLayout(dockWidgetContents_5);
        gridLayout_5->setSpacing(6);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        gridLayout_5->setContentsMargins(6, 6, 6, 6);
        lblRed = new QLabel(dockWidgetContents_5);
        lblRed->setObjectName(QStringLiteral("lblRed"));
        lblRed->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_5->addWidget(lblRed, 0, 0, 1, 1);

        sldRed = new QSlider(dockWidgetContents_5);
        sldRed->setObjectName(QStringLiteral("sldRed"));
        sldRed->setMinimumSize(QSize(0, 16));
        sldRed->setStyleSheet(QLatin1String("QSlider::groove:horizontal {\n"
"     border: 1px solid rgb(64, 0, 0);\n"
"	 border-style: groove;\n"
"     height: 3px;\n"
"     background: rgb(255, 192, 192);\n"
"	 border-radius: 2px;\n"
"\n"
" }\n"
"QSlider::sub-page:horizontal {\n"
"    background: rgb(160, 64, 64);\n"
"	border-radius: 2px;\n"
"	height: 3px;\n"
"}\n"
"\n"
"\n"
"QSlider::handle:horizontal {\n"
"	background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fd0000, stop:1 #9d0000);\n"
"	\n"
"	border: 1px solid #5c5c5c;\n"
"	width: 11px;\n"
"	 \n"
"	margin-top: -5px;\n"
"	margin-bottom: -5px;\n"
"    border-radius: 2px;\n"
" }"));
        sldRed->setMaximum(199);
        sldRed->setSingleStep(1);
        sldRed->setPageStep(10);
        sldRed->setValue(99);
        sldRed->setSliderPosition(99);
        sldRed->setOrientation(Qt::Horizontal);

        gridLayout_5->addWidget(sldRed, 0, 1, 1, 1);

        btnResetRed = new QToolButton(dockWidgetContents_5);
        btnResetRed->setObjectName(QStringLiteral("btnResetRed"));
        btnResetRed->setMinimumSize(QSize(23, 22));
        btnResetRed->setMaximumSize(QSize(23, 22));
        btnResetRed->setIcon(icon10);
        btnResetRed->setAutoRaise(true);

        gridLayout_5->addWidget(btnResetRed, 0, 2, 1, 1);

        lblGreen = new QLabel(dockWidgetContents_5);
        lblGreen->setObjectName(QStringLiteral("lblGreen"));
        lblGreen->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_5->addWidget(lblGreen, 1, 0, 1, 1);

        sldGreen = new QSlider(dockWidgetContents_5);
        sldGreen->setObjectName(QStringLiteral("sldGreen"));
        sldGreen->setStyleSheet(QLatin1String("\n"
"QSlider::groove:horizontal {\n"
"     border: 1px solid rgb(0, 64, 0);\n"
"	 border-style: groove;\n"
"     height: 3px;\n"
"     background: rgb(192, 255, 192);\n"
"	 border-radius: 2px;\n"
"}\n"
"\n"
" QSlider::sub-page:horizontal {\n"
"    background: rgb(64, 160, 64);\n"
"	border-radius: 2px;\n"
"	height: 3px;\n"
"} \n"
"\n"
"QSlider::handle:horizontal {\n"
"	background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00fd00, stop:1 #009d00);\n"
"    border: 1px solid #5c5c5c;\n"
"    width: 11px;\n"
"	 \n"
"	margin-top: -5px;\n"
"	margin-bottom: -5px;\n"
"    border-radius: 2px;\n"
" }"));
        sldGreen->setMaximum(199);
        sldGreen->setSingleStep(1);
        sldGreen->setPageStep(10);
        sldGreen->setValue(99);
        sldGreen->setSliderPosition(99);
        sldGreen->setOrientation(Qt::Horizontal);

        gridLayout_5->addWidget(sldGreen, 1, 1, 1, 1);

        btnResetGreen = new QToolButton(dockWidgetContents_5);
        btnResetGreen->setObjectName(QStringLiteral("btnResetGreen"));
        btnResetGreen->setMinimumSize(QSize(23, 22));
        btnResetGreen->setMaximumSize(QSize(23, 22));
        btnResetGreen->setIcon(icon10);
        btnResetGreen->setAutoRaise(true);

        gridLayout_5->addWidget(btnResetGreen, 1, 2, 1, 1);

        lblBlue = new QLabel(dockWidgetContents_5);
        lblBlue->setObjectName(QStringLiteral("lblBlue"));
        lblBlue->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_5->addWidget(lblBlue, 2, 0, 1, 1);

        sldBlue = new QSlider(dockWidgetContents_5);
        sldBlue->setObjectName(QStringLiteral("sldBlue"));
        sldBlue->setStyleSheet(QLatin1String("QSlider::groove:horizontal {\n"
"    border: 1px solid rgb(0, 0, 64);\n"
"	border-style: groove;\n"
"    height: 3px;\n"
"	border-radius: 2px;\n"
"	background: rgb(192, 192, 255);\n"
"}\n"
"QSlider::sub-page:horizontal {\n"
"    background: rgb(64, 64, 160);\n"
"	border-radius: 2px;\n"
"	height: 3px;\n"
"}\n"
"QSlider::handle:horizontal {\n"
"	background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0000fd, stop:1 #00009d);\n"
"\n"
"    border: 1px solid #5c5c5c;\n"
"	width: 11px;\n"
"	 \n"
"	margin-top: -5px;\n"
"	margin-bottom: -5px;\n"
"    border-radius: 2px;	\n"
"}"));
        sldBlue->setMaximum(199);
        sldBlue->setSingleStep(1);
        sldBlue->setPageStep(10);
        sldBlue->setValue(99);
        sldBlue->setSliderPosition(99);
        sldBlue->setOrientation(Qt::Horizontal);

        gridLayout_5->addWidget(sldBlue, 2, 1, 1, 1);

        btnResetBlue = new QToolButton(dockWidgetContents_5);
        btnResetBlue->setObjectName(QStringLiteral("btnResetBlue"));
        btnResetBlue->setMinimumSize(QSize(23, 22));
        btnResetBlue->setMaximumSize(QSize(23, 22));
        btnResetBlue->setIcon(icon10);
        btnResetBlue->setAutoRaise(true);

        gridLayout_5->addWidget(btnResetBlue, 2, 2, 1, 1);

        colorCorrectionWidget->setWidget(dockWidgetContents_5);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(2), colorCorrectionWidget);
        exposureWidget = new QDockWidget(MainWindow);
        exposureWidget->setObjectName(QStringLiteral("exposureWidget"));
        exposureWidget->setMinimumSize(QSize(114, 60));
        exposureWidget->setMaximumSize(QSize(524287, 60));
        dockWidgetContents_6 = new QWidget();
        dockWidgetContents_6->setObjectName(QStringLiteral("dockWidgetContents_6"));
        gridLayout_7 = new QGridLayout(dockWidgetContents_6);
        gridLayout_7->setSpacing(6);
        gridLayout_7->setContentsMargins(11, 11, 11, 11);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        gridLayout_7->setContentsMargins(6, 6, 6, 6);
        lblEV = new QLabel(dockWidgetContents_6);
        lblEV->setObjectName(QStringLiteral("lblEV"));
        lblEV->setAlignment(Qt::AlignCenter);

        gridLayout_7->addWidget(lblEV, 0, 0, 1, 1);

        sldEV = new QSlider(dockWidgetContents_6);
        sldEV->setObjectName(QStringLiteral("sldEV"));
        sldEV->setMinimum(-500);
        sldEV->setMaximum(500);
        sldEV->setSingleStep(1);
        sldEV->setPageStep(50);
        sldEV->setOrientation(Qt::Horizontal);
        sldEV->setTickPosition(QSlider::TicksBelow);
        sldEV->setTickInterval(50);

        gridLayout_7->addWidget(sldEV, 0, 1, 1, 1);

        btnResetEV = new QToolButton(dockWidgetContents_6);
        btnResetEV->setObjectName(QStringLiteral("btnResetEV"));
        btnResetEV->setMinimumSize(QSize(23, 22));
        btnResetEV->setMaximumSize(QSize(23, 22));
        btnResetEV->setIcon(icon10);
        btnResetEV->setAutoRaise(true);

        gridLayout_7->addWidget(btnResetEV, 0, 2, 1, 1);

        exposureWidget->setWidget(dockWidgetContents_6);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(2), exposureWidget);
        denoiseWidget = new QDockWidget(MainWindow);
        denoiseWidget->setObjectName(QStringLiteral("denoiseWidget"));
        dockWidgetContents_7 = new QWidget();
        dockWidgetContents_7->setObjectName(QStringLiteral("dockWidgetContents_7"));
        verticalLayout_3 = new QVBoxLayout(dockWidgetContents_7);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        denoiseCtlr = new DenoiseController(dockWidgetContents_7);
        denoiseCtlr->setObjectName(QStringLiteral("denoiseCtlr"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(denoiseCtlr->sizePolicy().hasHeightForWidth());
        denoiseCtlr->setSizePolicy(sizePolicy3);
        denoiseCtlr->setMaximumSize(QSize(16777215, 16777215));

        verticalLayout_3->addWidget(denoiseCtlr);

        denoiseWidget->setWidget(dockWidgetContents_7);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(2), denoiseWidget);
        benchMarksWidget = new QDockWidget(MainWindow);
        benchMarksWidget->setObjectName(QStringLiteral("benchMarksWidget"));
        sizePolicy3.setHeightForWidth(benchMarksWidget->sizePolicy().hasHeightForWidth());
        benchMarksWidget->setSizePolicy(sizePolicy3);
        dockWidgetContents_8 = new QWidget();
        dockWidgetContents_8->setObjectName(QStringLiteral("dockWidgetContents_8"));
        verticalLayout_2 = new QVBoxLayout(dockWidgetContents_8);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        lblInfo = new QPlainTextEdit(dockWidgetContents_8);
        lblInfo->setObjectName(QStringLiteral("lblInfo"));
        lblInfo->setFrameShape(QFrame::NoFrame);
        lblInfo->setReadOnly(true);

        verticalLayout_2->addWidget(lblInfo);

        benchMarksWidget->setWidget(dockWidgetContents_8);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), benchMarksWidget);
        recordingWidget = new QDockWidget(MainWindow);
        recordingWidget->setObjectName(QStringLiteral("recordingWidget"));
        recordingWidget->setMaximumSize(QSize(524287, 250));
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QStringLiteral("dockWidgetContents_2"));
        gridLayout_3 = new QGridLayout(dockWidgetContents_2);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setContentsMargins(6, 6, 6, 6);
        label_3 = new QLabel(dockWidgetContents_2);
        label_3->setObjectName(QStringLiteral("label_3"));
        sizePolicy.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy);

        gridLayout_3->addWidget(label_3, 0, 0, 1, 1);

        label_4 = new QLabel(dockWidgetContents_2);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_3->addWidget(label_4, 1, 0, 1, 1);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        label_6 = new QLabel(dockWidgetContents_2);
        label_6->setObjectName(QStringLiteral("label_6"));

        horizontalLayout_4->addWidget(label_6);

        txtOutPath = new QLineEdit(dockWidgetContents_2);
        txtOutPath->setObjectName(QStringLiteral("txtOutPath"));

        horizontalLayout_4->addWidget(txtOutPath);

        btnGetOutPath = new QToolButton(dockWidgetContents_2);
        btnGetOutPath->setObjectName(QStringLiteral("btnGetOutPath"));

        horizontalLayout_4->addWidget(btnGetOutPath);


        gridLayout_3->addLayout(horizontalLayout_4, 4, 0, 1, 4);

        label_7 = new QLabel(dockWidgetContents_2);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout_3->addWidget(label_7, 3, 0, 1, 1);

        spnJpegQty = new QSpinBox(dockWidgetContents_2);
        spnJpegQty->setObjectName(QStringLiteral("spnJpegQty"));
        spnJpegQty->setMinimum(50);
        spnJpegQty->setMaximum(100);
        spnJpegQty->setValue(90);

        gridLayout_3->addWidget(spnJpegQty, 1, 3, 1, 1);

        label_17 = new QLabel(dockWidgetContents_2);
        label_17->setObjectName(QStringLiteral("label_17"));
        sizePolicy.setHeightForWidth(label_17->sizePolicy().hasHeightForWidth());
        label_17->setSizePolicy(sizePolicy);

        gridLayout_3->addWidget(label_17, 2, 0, 1, 1);

        txtFilePrefix = new QLineEdit(dockWidgetContents_2);
        txtFilePrefix->setObjectName(QStringLiteral("txtFilePrefix"));

        gridLayout_3->addWidget(txtFilePrefix, 3, 1, 1, 3);

        cboSamplingFmt = new QComboBox(dockWidgetContents_2);
        cboSamplingFmt->setObjectName(QStringLiteral("cboSamplingFmt"));

        gridLayout_3->addWidget(cboSamplingFmt, 1, 1, 1, 1);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_3->addItem(verticalSpacer_2, 5, 0, 1, 1);

        cboOutFormat = new QComboBox(dockWidgetContents_2);
        cboOutFormat->setObjectName(QStringLiteral("cboOutFormat"));

        gridLayout_3->addWidget(cboOutFormat, 0, 1, 1, 3);

        label_5 = new QLabel(dockWidgetContents_2);
        label_5->setObjectName(QStringLiteral("label_5"));
        sizePolicy.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy);

        gridLayout_3->addWidget(label_5, 1, 2, 1, 1);

        cbBitrate = new QComboBox(dockWidgetContents_2);
        cbBitrate->setObjectName(QStringLiteral("cbBitrate"));
        cbBitrate->setEditable(true);

        gridLayout_3->addWidget(cbBitrate, 2, 1, 1, 3);

        recordingWidget->setWidget(dockWidgetContents_2);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), recordingWidget);
        gtgDockWidget = new QDockWidget(MainWindow);
        gtgDockWidget->setObjectName(QStringLiteral("gtgDockWidget"));
        dockWidgetContents_4 = new QWidget();
        dockWidgetContents_4->setObjectName(QStringLiteral("dockWidgetContents_4"));
        gtgLayout = new QHBoxLayout(dockWidgetContents_4);
        gtgLayout->setSpacing(6);
        gtgLayout->setContentsMargins(11, 11, 11, 11);
        gtgLayout->setObjectName(QStringLiteral("gtgLayout"));
        gtgWidget = new GtGWidget(dockWidgetContents_4);
        gtgWidget->setObjectName(QStringLiteral("gtgWidget"));

        gtgLayout->addWidget(gtgWidget);

        gtgDockWidget->setWidget(dockWidgetContents_4);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), gtgDockWidget);
        cameraSettingsWidget = new QDockWidget(MainWindow);
        cameraSettingsWidget->setObjectName(QStringLiteral("cameraSettingsWidget"));
        dockWidgetContents_9 = new QWidget();
        dockWidgetContents_9->setObjectName(QStringLiteral("dockWidgetContents_9"));
        verticalLayout = new QVBoxLayout(dockWidgetContents_9);
        verticalLayout->setSpacing(0);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        cameraController = new CameraSetupWidget(dockWidgetContents_9);
        cameraController->setObjectName(QStringLiteral("cameraController"));

        verticalLayout->addWidget(cameraController);

        cameraSettingsWidget->setWidget(dockWidgetContents_9);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), cameraSettingsWidget);
        cameraStatWidget = new QDockWidget(MainWindow);
        cameraStatWidget->setObjectName(QStringLiteral("cameraStatWidget"));
        dockWidgetContents_10 = new QWidget();
        dockWidgetContents_10->setObjectName(QStringLiteral("dockWidgetContents_10"));
        verticalLayout1 = new QVBoxLayout(dockWidgetContents_10);
        verticalLayout1->setSpacing(0);
        verticalLayout1->setContentsMargins(11, 11, 11, 11);
        verticalLayout1->setObjectName(QStringLiteral("verticalLayout1"));
        verticalLayout1->setContentsMargins(0, 0, 0, 0);
        cameraStatistics = new CameraStatistics(dockWidgetContents_10);
        cameraStatistics->setObjectName(QStringLiteral("cameraStatistics"));

        verticalLayout1->addWidget(cameraStatistics);

        cameraStatWidget->setWidget(dockWidgetContents_10);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), cameraStatWidget);
        dw_RtspServer = new QDockWidget(MainWindow);
        dw_RtspServer->setObjectName(QStringLiteral("dw_RtspServer"));
        dockWidgetContents_91 = new QWidget();
        dockWidgetContents_91->setObjectName(QStringLiteral("dockWidgetContents_91"));
        verticalLayout2 = new QVBoxLayout(dockWidgetContents_91);
        verticalLayout2->setSpacing(6);
        verticalLayout2->setContentsMargins(11, 11, 11, 11);
        verticalLayout2->setObjectName(QStringLiteral("verticalLayout2"));
        horizontalLayout_7 = new QHBoxLayout();
        horizontalLayout_7->setSpacing(6);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        label_9 = new QLabel(dockWidgetContents_91);
        label_9->setObjectName(QStringLiteral("label_9"));
        sizePolicy.setHeightForWidth(label_9->sizePolicy().hasHeightForWidth());
        label_9->setSizePolicy(sizePolicy);

        horizontalLayout_7->addWidget(label_9);

        cboFormatEnc = new QComboBox(dockWidgetContents_91);
        cboFormatEnc->setObjectName(QStringLiteral("cboFormatEnc"));

        horizontalLayout_7->addWidget(cboFormatEnc);


        verticalLayout2->addLayout(horizontalLayout_7);

        swRtspOptions = new QStackedWidget(dockWidgetContents_91);
        swRtspOptions->setObjectName(QStringLiteral("swRtspOptions"));
        QSizePolicy sizePolicy4(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(swRtspOptions->sizePolicy().hasHeightForWidth());
        swRtspOptions->setSizePolicy(sizePolicy4);
        page = new QWidget();
        page->setObjectName(QStringLiteral("page"));
        QSizePolicy sizePolicy5(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(page->sizePolicy().hasHeightForWidth());
        page->setSizePolicy(sizePolicy5);
        verticalLayout_4 = new QVBoxLayout(page);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_9 = new QHBoxLayout();
        horizontalLayout_9->setSpacing(6);
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        horizontalLayout_9->setContentsMargins(-1, 0, -1, -1);
        label_12 = new QLabel(page);
        label_12->setObjectName(QStringLiteral("label_12"));
        sizePolicy.setHeightForWidth(label_12->sizePolicy().hasHeightForWidth());
        label_12->setSizePolicy(sizePolicy);

        horizontalLayout_9->addWidget(label_12);

        cboSamplingFmtRtsp = new QComboBox(page);
        cboSamplingFmtRtsp->setObjectName(QStringLiteral("cboSamplingFmtRtsp"));

        horizontalLayout_9->addWidget(cboSamplingFmtRtsp);

        label_11 = new QLabel(page);
        label_11->setObjectName(QStringLiteral("label_11"));
        sizePolicy.setHeightForWidth(label_11->sizePolicy().hasHeightForWidth());
        label_11->setSizePolicy(sizePolicy);

        horizontalLayout_9->addWidget(label_11);

        spnJpegQtyRtsp = new QSpinBox(page);
        spnJpegQtyRtsp->setObjectName(QStringLiteral("spnJpegQtyRtsp"));
        spnJpegQtyRtsp->setMinimum(50);
        spnJpegQtyRtsp->setMaximum(100);
        spnJpegQtyRtsp->setValue(90);

        horizontalLayout_9->addWidget(spnJpegQtyRtsp);


        verticalLayout_4->addLayout(horizontalLayout_9);

        swRtspOptions->addWidget(page);
        page_2 = new QWidget();
        page_2->setObjectName(QStringLiteral("page_2"));
        verticalLayout_5 = new QVBoxLayout(page_2);
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setContentsMargins(11, 11, 11, 11);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_8 = new QHBoxLayout();
        horizontalLayout_8->setSpacing(6);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        label_10 = new QLabel(page_2);
        label_10->setObjectName(QStringLiteral("label_10"));
        sizePolicy.setHeightForWidth(label_10->sizePolicy().hasHeightForWidth());
        label_10->setSizePolicy(sizePolicy);

        horizontalLayout_8->addWidget(label_10);

        cbBitrateRtsp = new QComboBox(page_2);
        cbBitrateRtsp->setObjectName(QStringLiteral("cbBitrateRtsp"));
        cbBitrateRtsp->setEditable(true);

        horizontalLayout_8->addWidget(cbBitrateRtsp);


        verticalLayout_5->addLayout(horizontalLayout_8);

        swRtspOptions->addWidget(page_2);

        verticalLayout2->addWidget(swRtspOptions);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        label_8 = new QLabel(dockWidgetContents_91);
        label_8->setObjectName(QStringLiteral("label_8"));

        horizontalLayout_5->addWidget(label_8);

        txtRtspServer = new QLineEdit(dockWidgetContents_91);
        txtRtspServer->setObjectName(QStringLiteral("txtRtspServer"));

        horizontalLayout_5->addWidget(txtRtspServer);

        lblStatusRtspServer = new QLabel(dockWidgetContents_91);
        lblStatusRtspServer->setObjectName(QStringLiteral("lblStatusRtspServer"));
        lblStatusRtspServer->setMinimumSize(QSize(20, 20));
        lblStatusRtspServer->setMaximumSize(QSize(20, 20));
        lblStatusRtspServer->setStyleSheet(QLatin1String("background: qradialgradient(spread:pad, cx:0.5, cy:0.5, radius:0.5, fx:0.295, fy:0.272727, stop:0 rgba(255, 255, 255, 255), stop:1 rgba(80, 80, 80, 255));\n"
"border-radius: 9px;"));

        horizontalLayout_5->addWidget(lblStatusRtspServer);


        verticalLayout2->addLayout(horizontalLayout_5);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        btnStartRtspServer = new QPushButton(dockWidgetContents_91);
        btnStartRtspServer->setObjectName(QStringLiteral("btnStartRtspServer"));

        horizontalLayout_6->addWidget(btnStartRtspServer);

        btnStopRtspServer = new QPushButton(dockWidgetContents_91);
        btnStopRtspServer->setObjectName(QStringLiteral("btnStopRtspServer"));

        horizontalLayout_6->addWidget(btnStopRtspServer);


        verticalLayout2->addLayout(horizontalLayout_6);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout2->addItem(verticalSpacer);

        dw_RtspServer->setWidget(dockWidgetContents_91);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dw_RtspServer);

        menuBar->addAction(menuCamera->menuAction());
        menuBar->addAction(menuTools->menuAction());
        menuCamera->addAction(actionOpenBayerPGM);
        menuCamera->addSeparator();
        menuCamera->addAction(actionOpenGrayPGM);
        menuCamera->addAction(actionPlay);
        menuCamera->addAction(actionRecord);
        menuCamera->addSeparator();
        menuCamera->addAction(actionExit);
        menuTools->addAction(actionWB_picker);
        mainToolBar->addAction(actionOpenBayerPGM);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionPlay);
        mainToolBar->addAction(actionRecord);
        mainToolBar->addAction(actionWB_picker);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionShowImage);
        mainToolBar->addSeparator();
        mainToolBar->addAction(actionExit);
        mainToolBar->addSeparator();

        retranslateUi(MainWindow);

        cboBayerPattern->setCurrentIndex(-1);
        cbBitrate->setCurrentIndex(2);
        swRtspOptions->setCurrentIndex(1);
        cbBitrateRtsp->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "GPU Camera Sample", Q_NULLPTR));
        actionOpenCamera->setText(QApplication::translate("MainWindow", "Open camera", Q_NULLPTR));
        actionPlay->setText(QApplication::translate("MainWindow", "Play", Q_NULLPTR));
        actionStop->setText(QApplication::translate("MainWindow", "Stop", Q_NULLPTR));
        actionRecord->setText(QApplication::translate("MainWindow", "Record", Q_NULLPTR));
        actionExit->setText(QApplication::translate("MainWindow", "Exit", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        actionExit->setToolTip(QApplication::translate("MainWindow", "Exit program", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        actionOptions->setText(QApplication::translate("MainWindow", "Options", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        actionOptions->setToolTip(QApplication::translate("MainWindow", "Program options", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        actionWB_picker->setText(QApplication::translate("MainWindow", "WB picker", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        actionWB_picker->setToolTip(QApplication::translate("MainWindow", "Pick WB from gray point", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        actionOpenBayerPGM->setText(QApplication::translate("MainWindow", "Open Bayer PGM", Q_NULLPTR));
        actionOpenGrayPGM->setText(QApplication::translate("MainWindow", "Open Gray PGM", Q_NULLPTR));
        actionShowImage->setText(QApplication::translate("MainWindow", "Show Image", Q_NULLPTR));
        menuCamera->setTitle(QApplication::translate("MainWindow", "Camera", Q_NULLPTR));
        menuTools->setTitle(QApplication::translate("MainWindow", "Tools", Q_NULLPTR));
        dockWidget->setWindowTitle(QApplication::translate("MainWindow", "Camera settings", Q_NULLPTR));
        btnGetGrayFile->setText(QApplication::translate("MainWindow", "...", Q_NULLPTR));
        label_39->setText(QApplication::translate("MainWindow", "Dark frame file", Q_NULLPTR));
        txtFPNFileName->setText(QString());
        btnGetFPNFile->setText(QApplication::translate("MainWindow", "...", Q_NULLPTR));
        txtFlatFieldFile->setText(QString());
        label_13->setText(QApplication::translate("MainWindow", "Flat field file", Q_NULLPTR));
        label_32->setText(QApplication::translate("MainWindow", "Bayer pattern", Q_NULLPTR));
        label_2->setText(QApplication::translate("MainWindow", "Algorithm", Q_NULLPTR));
        processWidget->setWindowTitle(QApplication::translate("MainWindow", "Processing options", Q_NULLPTR));
        groupBox_8->setTitle(QApplication::translate("MainWindow", "Zoom", Q_NULLPTR));
        chkZoomFit->setText(QApplication::translate("MainWindow", "Fit window", Q_NULLPTR));
        lblZoom->setText(QApplication::translate("MainWindow", "100%", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        btnResetZoom->setToolTip(QApplication::translate("MainWindow", "Reset zoom to 100%", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        btnResetZoom->setText(QApplication::translate("MainWindow", "+", Q_NULLPTR));
        label_47->setText(QApplication::translate("MainWindow", "CUDA Device", Q_NULLPTR));
        label->setText(QApplication::translate("MainWindow", "Gamma", Q_NULLPTR));
        chkBPC->setText(QApplication::translate("MainWindow", "BPC", Q_NULLPTR));
        chkSAM->setText(QApplication::translate("MainWindow", "FPN/FFC", Q_NULLPTR));
        colorCorrectionWidget->setWindowTitle(QApplication::translate("MainWindow", "White balance", Q_NULLPTR));
        lblRed->setText(QApplication::translate("MainWindow", "1.00", Q_NULLPTR));
        btnResetRed->setText(QString());
        lblGreen->setText(QApplication::translate("MainWindow", "1.00", Q_NULLPTR));
        btnResetGreen->setText(QString());
        lblBlue->setText(QApplication::translate("MainWindow", "1.00", Q_NULLPTR));
        btnResetBlue->setText(QApplication::translate("MainWindow", "...", Q_NULLPTR));
        exposureWidget->setWindowTitle(QApplication::translate("MainWindow", "Exposure correction", Q_NULLPTR));
        lblEV->setText(QApplication::translate("MainWindow", "0.0 eV", Q_NULLPTR));
        btnResetEV->setText(QApplication::translate("MainWindow", "...", Q_NULLPTR));
        denoiseWidget->setWindowTitle(QApplication::translate("MainWindow", "Denoise", Q_NULLPTR));
        benchMarksWidget->setWindowTitle(QApplication::translate("MainWindow", "Benchmarks", Q_NULLPTR));
        recordingWidget->setWindowTitle(QApplication::translate("MainWindow", "Recording", Q_NULLPTR));
        label_3->setText(QApplication::translate("MainWindow", "File format", Q_NULLPTR));
        label_4->setText(QApplication::translate("MainWindow", "Sampling", Q_NULLPTR));
        label_6->setText(QApplication::translate("MainWindow", "Path", Q_NULLPTR));
        btnGetOutPath->setText(QApplication::translate("MainWindow", "...", Q_NULLPTR));
        label_7->setText(QApplication::translate("MainWindow", "File prefix", Q_NULLPTR));
        label_17->setText(QApplication::translate("MainWindow", "Bitrate", Q_NULLPTR));
        label_5->setText(QApplication::translate("MainWindow", "Quality", Q_NULLPTR));
        cbBitrate->clear();
        cbBitrate->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "100 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "50 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "20 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "10 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "5 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "1 MB", Q_NULLPTR)
        );
        gtgDockWidget->setWindowTitle(QApplication::translate("MainWindow", "Glass to Glass test", Q_NULLPTR));
        cameraSettingsWidget->setWindowTitle(QApplication::translate("MainWindow", "Camera settings", Q_NULLPTR));
        cameraStatWidget->setWindowTitle(QApplication::translate("MainWindow", "Camera Statistics", Q_NULLPTR));
        dw_RtspServer->setWindowTitle(QApplication::translate("MainWindow", "RTSP Server", Q_NULLPTR));
        label_9->setText(QApplication::translate("MainWindow", "Format", Q_NULLPTR));
        cboFormatEnc->clear();
        cboFormatEnc->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "JPEG", Q_NULLPTR)
         << QApplication::translate("MainWindow", "H264", Q_NULLPTR)
         << QApplication::translate("MainWindow", "H265", Q_NULLPTR)
        );
        label_12->setText(QApplication::translate("MainWindow", "Sampling", Q_NULLPTR));
        label_11->setText(QApplication::translate("MainWindow", "Quality", Q_NULLPTR));
        label_10->setText(QApplication::translate("MainWindow", "Bitrate", Q_NULLPTR));
        cbBitrateRtsp->clear();
        cbBitrateRtsp->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "100 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "50 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "20 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "10 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "5 MB", Q_NULLPTR)
         << QApplication::translate("MainWindow", "1 MB", Q_NULLPTR)
        );
        label_8->setText(QApplication::translate("MainWindow", "Server", Q_NULLPTR));
        txtRtspServer->setText(QApplication::translate("MainWindow", "rtsp://127.0.0.1:1234/live.sdp", Q_NULLPTR));
        lblStatusRtspServer->setText(QString());
        btnStartRtspServer->setText(QApplication::translate("MainWindow", "Start", Q_NULLPTR));
        btnStopRtspServer->setText(QApplication::translate("MainWindow", "Stop", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
