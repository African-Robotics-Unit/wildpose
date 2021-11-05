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
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "Widgets/GtGWidget.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen_RTSP_server;
    QAction *actionClose_RTSP_server;
    QAction *actionOpen_RTSP_client;
    QAction *actionPlay;
    QAction *actionExit;
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QVBoxLayout *MediaViewerLayout;
    QMenuBar *menubar;
    QMenu *menuFile;
    QStatusBar *statusbar;
    QToolBar *toolBar;
    QDockWidget *dwRtspConfig;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *gbTransportProtocol;
    QVBoxLayout *verticalLayout_6;
    QRadioButton *rbRtp;
    QRadioButton *rbCtp;
    QGroupBox *gbDecodersH264;
    QVBoxLayout *verticalLayout_5;
    QRadioButton *rbCuvid;
    QRadioButton *rbOtherAvailable;
    QRadioButton *rbCuvidNV;
    QGroupBox *gbMJpegParameters;
    QVBoxLayout *verticalLayout_4;
    QRadioButton *rbJpegTurbo;
    QRadioButton *rbFastvideoJpeg;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *le_rtsp_address;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *pb_openRtsp;
    QPushButton *pb_stopRtsp;
    QGridLayout *gridLayout;
    QLabel *label_3;
    QLabel *lb_count_frames;
    QLabel *label_2;
    QLabel *lb_fps;
    QLabel *label_4;
    QLabel *lb_bitrate;
    QLabel *lb_durations;
    QSpacerItem *verticalSpacer;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents_2;
    QVBoxLayout *verticalLayout_3;
    GtGWidget *gtgWidget;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(890, 600);
        actionOpen_RTSP_server = new QAction(MainWindow);
        actionOpen_RTSP_server->setObjectName(QStringLiteral("actionOpen_RTSP_server"));
        actionClose_RTSP_server = new QAction(MainWindow);
        actionClose_RTSP_server->setObjectName(QStringLiteral("actionClose_RTSP_server"));
        actionOpen_RTSP_client = new QAction(MainWindow);
        actionOpen_RTSP_client->setObjectName(QStringLiteral("actionOpen_RTSP_client"));
        actionPlay = new QAction(MainWindow);
        actionPlay->setObjectName(QStringLiteral("actionPlay"));
        actionPlay->setCheckable(true);
        actionPlay->setChecked(true);
        QIcon icon;
        icon.addFile(QStringLiteral(":/res/play_fwd.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionPlay->setIcon(icon);
        actionExit = new QAction(MainWindow);
        actionExit->setObjectName(QStringLiteral("actionExit"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        MediaViewerLayout = new QVBoxLayout();
        MediaViewerLayout->setObjectName(QStringLiteral("MediaViewerLayout"));

        verticalLayout->addLayout(MediaViewerLayout);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 890, 21));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);
        toolBar = new QToolBar(MainWindow);
        toolBar->setObjectName(QStringLiteral("toolBar"));
        toolBar->setMinimumSize(QSize(0, 0));
        toolBar->setIconSize(QSize(24, 24));
        toolBar->setToolButtonStyle(Qt::ToolButtonIconOnly);
        MainWindow->addToolBar(Qt::TopToolBarArea, toolBar);
        dwRtspConfig = new QDockWidget(MainWindow);
        dwRtspConfig->setObjectName(QStringLiteral("dwRtspConfig"));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QStringLiteral("dockWidgetContents"));
        verticalLayout_2 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        gbTransportProtocol = new QGroupBox(dockWidgetContents);
        gbTransportProtocol->setObjectName(QStringLiteral("gbTransportProtocol"));
        verticalLayout_6 = new QVBoxLayout(gbTransportProtocol);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        rbRtp = new QRadioButton(gbTransportProtocol);
        rbRtp->setObjectName(QStringLiteral("rbRtp"));
        rbRtp->setChecked(true);

        verticalLayout_6->addWidget(rbRtp);

        rbCtp = new QRadioButton(gbTransportProtocol);
        rbCtp->setObjectName(QStringLiteral("rbCtp"));

        verticalLayout_6->addWidget(rbCtp);


        verticalLayout_2->addWidget(gbTransportProtocol);

        gbDecodersH264 = new QGroupBox(dockWidgetContents);
        gbDecodersH264->setObjectName(QStringLiteral("gbDecodersH264"));
        verticalLayout_5 = new QVBoxLayout(gbDecodersH264);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        rbCuvid = new QRadioButton(gbDecodersH264);
        rbCuvid->setObjectName(QStringLiteral("rbCuvid"));
        rbCuvid->setChecked(true);

        verticalLayout_5->addWidget(rbCuvid);

        rbOtherAvailable = new QRadioButton(gbDecodersH264);
        rbOtherAvailable->setObjectName(QStringLiteral("rbOtherAvailable"));

        verticalLayout_5->addWidget(rbOtherAvailable);

        rbCuvidNV = new QRadioButton(gbDecodersH264);
        rbCuvidNV->setObjectName(QStringLiteral("rbCuvidNV"));

        verticalLayout_5->addWidget(rbCuvidNV);


        verticalLayout_2->addWidget(gbDecodersH264);

        gbMJpegParameters = new QGroupBox(dockWidgetContents);
        gbMJpegParameters->setObjectName(QStringLiteral("gbMJpegParameters"));
        verticalLayout_4 = new QVBoxLayout(gbMJpegParameters);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        rbJpegTurbo = new QRadioButton(gbMJpegParameters);
        rbJpegTurbo->setObjectName(QStringLiteral("rbJpegTurbo"));
        rbJpegTurbo->setChecked(true);

        verticalLayout_4->addWidget(rbJpegTurbo);

        rbFastvideoJpeg = new QRadioButton(gbMJpegParameters);
        rbFastvideoJpeg->setObjectName(QStringLiteral("rbFastvideoJpeg"));

        verticalLayout_4->addWidget(rbFastvideoJpeg);


        verticalLayout_2->addWidget(gbMJpegParameters);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        label = new QLabel(dockWidgetContents);
        label->setObjectName(QStringLiteral("label"));

        horizontalLayout->addWidget(label);

        le_rtsp_address = new QLineEdit(dockWidgetContents);
        le_rtsp_address->setObjectName(QStringLiteral("le_rtsp_address"));

        horizontalLayout->addWidget(le_rtsp_address);


        verticalLayout_2->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(-1, 0, -1, -1);
        pb_openRtsp = new QPushButton(dockWidgetContents);
        pb_openRtsp->setObjectName(QStringLiteral("pb_openRtsp"));

        horizontalLayout_2->addWidget(pb_openRtsp);

        pb_stopRtsp = new QPushButton(dockWidgetContents);
        pb_stopRtsp->setObjectName(QStringLiteral("pb_stopRtsp"));

        horizontalLayout_2->addWidget(pb_stopRtsp);


        verticalLayout_2->addLayout(horizontalLayout_2);

        gridLayout = new QGridLayout();
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(-1, 0, -1, -1);
        label_3 = new QLabel(dockWidgetContents);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 1, 0, 1, 1);

        lb_count_frames = new QLabel(dockWidgetContents);
        lb_count_frames->setObjectName(QStringLiteral("lb_count_frames"));
        lb_count_frames->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(lb_count_frames, 0, 1, 1, 1);

        label_2 = new QLabel(dockWidgetContents);
        label_2->setObjectName(QStringLiteral("label_2"));

        gridLayout->addWidget(label_2, 0, 0, 1, 1);

        lb_fps = new QLabel(dockWidgetContents);
        lb_fps->setObjectName(QStringLiteral("lb_fps"));
        lb_fps->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(lb_fps, 1, 1, 1, 1);

        label_4 = new QLabel(dockWidgetContents);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout->addWidget(label_4, 2, 0, 1, 1);

        lb_bitrate = new QLabel(dockWidgetContents);
        lb_bitrate->setObjectName(QStringLiteral("lb_bitrate"));
        lb_bitrate->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout->addWidget(lb_bitrate, 2, 1, 1, 1);


        verticalLayout_2->addLayout(gridLayout);

        lb_durations = new QLabel(dockWidgetContents);
        lb_durations->setObjectName(QStringLiteral("lb_durations"));

        verticalLayout_2->addWidget(lb_durations);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);

        dwRtspConfig->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dwRtspConfig);
        dockWidget = new QDockWidget(MainWindow);
        dockWidget->setObjectName(QStringLiteral("dockWidget"));
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QStringLiteral("dockWidgetContents_2"));
        verticalLayout_3 = new QVBoxLayout(dockWidgetContents_2);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        gtgWidget = new GtGWidget(dockWidgetContents_2);
        gtgWidget->setObjectName(QStringLiteral("gtgWidget"));
        gtgWidget->setMinimumSize(QSize(0, 50));

        verticalLayout_3->addWidget(gtgWidget);

        dockWidget->setWidget(dockWidgetContents_2);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);

        menubar->addAction(menuFile->menuAction());
        menuFile->addAction(actionExit);
        toolBar->addAction(actionPlay);

        retranslateUi(MainWindow);
        QObject::connect(actionExit, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "RTSPPlayer", Q_NULLPTR));
        actionOpen_RTSP_server->setText(QApplication::translate("MainWindow", "Open RTSP server", Q_NULLPTR));
        actionClose_RTSP_server->setText(QApplication::translate("MainWindow", "Close RTSP server", Q_NULLPTR));
        actionOpen_RTSP_client->setText(QApplication::translate("MainWindow", "Open RTSP client", Q_NULLPTR));
        actionPlay->setText(QApplication::translate("MainWindow", "\342\226\272", Q_NULLPTR));
        actionExit->setText(QApplication::translate("MainWindow", "Exit", Q_NULLPTR));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", Q_NULLPTR));
        toolBar->setWindowTitle(QApplication::translate("MainWindow", "toolBar", Q_NULLPTR));
        dwRtspConfig->setWindowTitle(QApplication::translate("MainWindow", "RTSP configuration", Q_NULLPTR));
        gbTransportProtocol->setTitle(QApplication::translate("MainWindow", "Transport protocol", Q_NULLPTR));
        rbRtp->setText(QApplication::translate("MainWindow", "RTP (UDP)", Q_NULLPTR));
        rbCtp->setText(QApplication::translate("MainWindow", "CTP (UDP)", Q_NULLPTR));
        gbDecodersH264->setTitle(QApplication::translate("MainWindow", "H.264 parameters", Q_NULLPTR));
        rbCuvid->setText(QApplication::translate("MainWindow", "cuvid decoder (FFMPEG)", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        rbOtherAvailable->setToolTip(QApplication::translate("MainWindow", "H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        rbOtherAvailable->setText(QApplication::translate("MainWindow", "H.264 on CPU (FFMPEG)", Q_NULLPTR));
        rbCuvidNV->setText(QApplication::translate("MainWindow", "cuvid decoder (NVCodec)", Q_NULLPTR));
        gbMJpegParameters->setTitle(QApplication::translate("MainWindow", "MJPEG parameters", Q_NULLPTR));
        rbJpegTurbo->setText(QApplication::translate("MainWindow", "JpegTurbo decoder", Q_NULLPTR));
        rbFastvideoJpeg->setText(QApplication::translate("MainWindow", "FastVideo decoder", Q_NULLPTR));
        label->setText(QApplication::translate("MainWindow", "Address", Q_NULLPTR));
        le_rtsp_address->setText(QApplication::translate("MainWindow", "rtsp://127.0.0.1:1234/live.sdp", Q_NULLPTR));
        pb_openRtsp->setText(QApplication::translate("MainWindow", "Open", Q_NULLPTR));
        pb_stopRtsp->setText(QApplication::translate("MainWindow", "Stop", Q_NULLPTR));
        label_3->setText(QApplication::translate("MainWindow", "FPS:", Q_NULLPTR));
        lb_count_frames->setText(QApplication::translate("MainWindow", "0", Q_NULLPTR));
        label_2->setText(QApplication::translate("MainWindow", "Frames:", Q_NULLPTR));
        lb_fps->setText(QApplication::translate("MainWindow", "0", Q_NULLPTR));
        label_4->setText(QApplication::translate("MainWindow", "Bitrate:", Q_NULLPTR));
        lb_bitrate->setText(QApplication::translate("MainWindow", "0", Q_NULLPTR));
        lb_durations->setText(QString());
        dockWidget->setWindowTitle(QApplication::translate("MainWindow", "Glass to Glass test", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
