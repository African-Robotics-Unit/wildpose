/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.5)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/CameraSample/MainWindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainWindow_t {
    QByteArrayData data[68];
    char stringdata0[1272];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MainWindow"
QT_MOC_LITERAL(1, 11, 13), // "onZoomChanged"
QT_MOC_LITERAL(2, 25, 0), // ""
QT_MOC_LITERAL(3, 26, 4), // "zoom"
QT_MOC_LITERAL(4, 31, 21), // "on_chkZoomFit_toggled"
QT_MOC_LITERAL(5, 53, 7), // "checked"
QT_MOC_LITERAL(6, 61, 23), // "on_sldZoom_valueChanged"
QT_MOC_LITERAL(7, 85, 5), // "value"
QT_MOC_LITERAL(8, 91, 23), // "on_btnResetZoom_clicked"
QT_MOC_LITERAL(9, 115, 38), // "on_cboBayerPattern_currentInd..."
QT_MOC_LITERAL(10, 154, 5), // "index"
QT_MOC_LITERAL(11, 160, 35), // "on_cboBayerType_currentIndexC..."
QT_MOC_LITERAL(12, 196, 21), // "on_sldEV_valueChanged"
QT_MOC_LITERAL(13, 218, 21), // "on_btnResetEV_clicked"
QT_MOC_LITERAL(14, 240, 21), // "onDenoiseStateChanged"
QT_MOC_LITERAL(15, 262, 2), // "on"
QT_MOC_LITERAL(16, 265, 22), // "onThresholdTypeChanged"
QT_MOC_LITERAL(17, 288, 7), // "newType"
QT_MOC_LITERAL(18, 296, 20), // "onWaveletTypeChanged"
QT_MOC_LITERAL(19, 317, 19), // "onYThresholdChanged"
QT_MOC_LITERAL(20, 337, 12), // "newThreshold"
QT_MOC_LITERAL(21, 350, 20), // "onCbThresholdChanged"
QT_MOC_LITERAL(22, 371, 18), // "onShrinkageChanged"
QT_MOC_LITERAL(23, 390, 11), // "newSrinkage"
QT_MOC_LITERAL(24, 402, 22), // "onDenoiseParamsChanged"
QT_MOC_LITERAL(25, 425, 22), // "on_sldRed_valueChanged"
QT_MOC_LITERAL(26, 448, 24), // "on_sldGreen_valueChanged"
QT_MOC_LITERAL(27, 473, 23), // "on_sldBlue_valueChanged"
QT_MOC_LITERAL(28, 497, 22), // "on_btnResetRed_clicked"
QT_MOC_LITERAL(29, 520, 24), // "on_btnResetGreen_clicked"
QT_MOC_LITERAL(30, 545, 23), // "on_btnResetBlue_clicked"
QT_MOC_LITERAL(31, 569, 31), // "on_cboGamma_currentIndexChanged"
QT_MOC_LITERAL(32, 601, 29), // "on_actionOpenCamera_triggered"
QT_MOC_LITERAL(33, 631, 23), // "on_actionRecord_toggled"
QT_MOC_LITERAL(34, 655, 4), // "arg1"
QT_MOC_LITERAL(35, 660, 23), // "on_actionExit_triggered"
QT_MOC_LITERAL(36, 684, 26), // "on_actionWB_picker_toggled"
QT_MOC_LITERAL(37, 711, 21), // "on_actionPlay_toggled"
QT_MOC_LITERAL(38, 733, 17), // "getErrDescription"
QT_MOC_LITERAL(39, 751, 12), // "fastStatus_t"
QT_MOC_LITERAL(40, 764, 4), // "code"
QT_MOC_LITERAL(41, 769, 10), // "onGPUError"
QT_MOC_LITERAL(42, 780, 13), // "onGPUFinished"
QT_MOC_LITERAL(43, 794, 17), // "on_chkBPC_toggled"
QT_MOC_LITERAL(44, 812, 16), // "onNewWBFromPoint"
QT_MOC_LITERAL(45, 829, 2), // "pt"
QT_MOC_LITERAL(46, 832, 10), // "openCamera"
QT_MOC_LITERAL(47, 843, 8), // "uint32_t"
QT_MOC_LITERAL(48, 852, 5), // "devID"
QT_MOC_LITERAL(49, 858, 11), // "openPGMFile"
QT_MOC_LITERAL(50, 870, 7), // "isBayer"
QT_MOC_LITERAL(51, 878, 13), // "initNewCamera"
QT_MOC_LITERAL(52, 892, 14), // "GPUCameraBase*"
QT_MOC_LITERAL(53, 907, 3), // "cmr"
QT_MOC_LITERAL(54, 911, 20), // "onCameraStateChanged"
QT_MOC_LITERAL(55, 932, 29), // "GPUCameraBase::cmrCameraState"
QT_MOC_LITERAL(56, 962, 8), // "newState"
QT_MOC_LITERAL(57, 971, 31), // "on_actionOpenBayerPGM_triggered"
QT_MOC_LITERAL(58, 1003, 30), // "on_actionOpenGrayPGM_triggered"
QT_MOC_LITERAL(59, 1034, 24), // "on_btnGetOutPath_clicked"
QT_MOC_LITERAL(60, 1059, 24), // "on_btnGetFPNFile_clicked"
QT_MOC_LITERAL(61, 1084, 25), // "on_btnGetGrayFile_clicked"
QT_MOC_LITERAL(62, 1110, 17), // "on_chkSAM_toggled"
QT_MOC_LITERAL(63, 1128, 29), // "on_btnStartRtspServer_clicked"
QT_MOC_LITERAL(64, 1158, 28), // "on_btnStopRtspServer_clicked"
QT_MOC_LITERAL(65, 1187, 19), // "onTimeoutStatusRtsp"
QT_MOC_LITERAL(66, 1207, 35), // "on_cboFormatEnc_currentIndexC..."
QT_MOC_LITERAL(67, 1243, 28) // "on_actionShowImage_triggered"

    },
    "MainWindow\0onZoomChanged\0\0zoom\0"
    "on_chkZoomFit_toggled\0checked\0"
    "on_sldZoom_valueChanged\0value\0"
    "on_btnResetZoom_clicked\0"
    "on_cboBayerPattern_currentIndexChanged\0"
    "index\0on_cboBayerType_currentIndexChanged\0"
    "on_sldEV_valueChanged\0on_btnResetEV_clicked\0"
    "onDenoiseStateChanged\0on\0"
    "onThresholdTypeChanged\0newType\0"
    "onWaveletTypeChanged\0onYThresholdChanged\0"
    "newThreshold\0onCbThresholdChanged\0"
    "onShrinkageChanged\0newSrinkage\0"
    "onDenoiseParamsChanged\0on_sldRed_valueChanged\0"
    "on_sldGreen_valueChanged\0"
    "on_sldBlue_valueChanged\0on_btnResetRed_clicked\0"
    "on_btnResetGreen_clicked\0"
    "on_btnResetBlue_clicked\0"
    "on_cboGamma_currentIndexChanged\0"
    "on_actionOpenCamera_triggered\0"
    "on_actionRecord_toggled\0arg1\0"
    "on_actionExit_triggered\0"
    "on_actionWB_picker_toggled\0"
    "on_actionPlay_toggled\0getErrDescription\0"
    "fastStatus_t\0code\0onGPUError\0onGPUFinished\0"
    "on_chkBPC_toggled\0onNewWBFromPoint\0"
    "pt\0openCamera\0uint32_t\0devID\0openPGMFile\0"
    "isBayer\0initNewCamera\0GPUCameraBase*\0"
    "cmr\0onCameraStateChanged\0"
    "GPUCameraBase::cmrCameraState\0newState\0"
    "on_actionOpenBayerPGM_triggered\0"
    "on_actionOpenGrayPGM_triggered\0"
    "on_btnGetOutPath_clicked\0"
    "on_btnGetFPNFile_clicked\0"
    "on_btnGetGrayFile_clicked\0on_chkSAM_toggled\0"
    "on_btnStartRtspServer_clicked\0"
    "on_btnStopRtspServer_clicked\0"
    "onTimeoutStatusRtsp\0"
    "on_cboFormatEnc_currentIndexChanged\0"
    "on_actionShowImage_triggered"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      48,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,  254,    2, 0x08 /* Private */,
       4,    1,  257,    2, 0x08 /* Private */,
       6,    1,  260,    2, 0x08 /* Private */,
       8,    0,  263,    2, 0x08 /* Private */,
       9,    1,  264,    2, 0x08 /* Private */,
      11,    1,  267,    2, 0x08 /* Private */,
      12,    1,  270,    2, 0x08 /* Private */,
      13,    0,  273,    2, 0x08 /* Private */,
      14,    1,  274,    2, 0x08 /* Private */,
      16,    1,  277,    2, 0x08 /* Private */,
      18,    1,  280,    2, 0x08 /* Private */,
      19,    1,  283,    2, 0x08 /* Private */,
      21,    1,  286,    2, 0x08 /* Private */,
      22,    1,  289,    2, 0x08 /* Private */,
      24,    0,  292,    2, 0x08 /* Private */,
      25,    1,  293,    2, 0x08 /* Private */,
      26,    1,  296,    2, 0x08 /* Private */,
      27,    1,  299,    2, 0x08 /* Private */,
      28,    0,  302,    2, 0x08 /* Private */,
      29,    0,  303,    2, 0x08 /* Private */,
      30,    0,  304,    2, 0x08 /* Private */,
      31,    1,  305,    2, 0x08 /* Private */,
      32,    0,  308,    2, 0x08 /* Private */,
      33,    1,  309,    2, 0x08 /* Private */,
      35,    0,  312,    2, 0x08 /* Private */,
      36,    1,  313,    2, 0x08 /* Private */,
      37,    1,  316,    2, 0x08 /* Private */,
      38,    1,  319,    2, 0x08 /* Private */,
      41,    0,  322,    2, 0x08 /* Private */,
      42,    0,  323,    2, 0x08 /* Private */,
      43,    1,  324,    2, 0x08 /* Private */,
      44,    1,  327,    2, 0x08 /* Private */,
      46,    1,  330,    2, 0x08 /* Private */,
      49,    1,  333,    2, 0x08 /* Private */,
      49,    0,  336,    2, 0x28 /* Private | MethodCloned */,
      51,    2,  337,    2, 0x08 /* Private */,
      54,    1,  342,    2, 0x08 /* Private */,
      57,    0,  345,    2, 0x08 /* Private */,
      58,    0,  346,    2, 0x08 /* Private */,
      59,    0,  347,    2, 0x08 /* Private */,
      60,    0,  348,    2, 0x08 /* Private */,
      61,    0,  349,    2, 0x08 /* Private */,
      62,    1,  350,    2, 0x08 /* Private */,
      63,    0,  353,    2, 0x08 /* Private */,
      64,    0,  354,    2, 0x08 /* Private */,
      65,    0,  355,    2, 0x08 /* Private */,
      66,    1,  356,    2, 0x08 /* Private */,
      67,    1,  359,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::QReal,    3,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   15,
    QMetaType::Void, QMetaType::Int,   17,
    QMetaType::Void, QMetaType::Int,   17,
    QMetaType::Void, QMetaType::Float,   20,
    QMetaType::Void, QMetaType::Float,   20,
    QMetaType::Void, QMetaType::Int,   23,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   34,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   34,
    QMetaType::Void, QMetaType::Bool,   34,
    QMetaType::QString, 0x80000000 | 39,   40,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::QPoint,   45,
    QMetaType::Void, 0x80000000 | 47,   48,
    QMetaType::Void, QMetaType::Bool,   50,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 52, 0x80000000 | 47,   53,   48,
    QMetaType::Void, 0x80000000 | 55,   56,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Bool,    5,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MainWindow *_t = static_cast<MainWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->onZoomChanged((*reinterpret_cast< qreal(*)>(_a[1]))); break;
        case 1: _t->on_chkZoomFit_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: _t->on_sldZoom_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->on_btnResetZoom_clicked(); break;
        case 4: _t->on_cboBayerPattern_currentIndexChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->on_cboBayerType_currentIndexChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->on_sldEV_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->on_btnResetEV_clicked(); break;
        case 8: _t->onDenoiseStateChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->onThresholdTypeChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->onWaveletTypeChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: _t->onYThresholdChanged((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 12: _t->onCbThresholdChanged((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 13: _t->onShrinkageChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 14: _t->onDenoiseParamsChanged(); break;
        case 15: _t->on_sldRed_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 16: _t->on_sldGreen_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 17: _t->on_sldBlue_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 18: _t->on_btnResetRed_clicked(); break;
        case 19: _t->on_btnResetGreen_clicked(); break;
        case 20: _t->on_btnResetBlue_clicked(); break;
        case 21: _t->on_cboGamma_currentIndexChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 22: _t->on_actionOpenCamera_triggered(); break;
        case 23: _t->on_actionRecord_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 24: _t->on_actionExit_triggered(); break;
        case 25: _t->on_actionWB_picker_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 26: _t->on_actionPlay_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 27: { QString _r = _t->getErrDescription((*reinterpret_cast< fastStatus_t(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QString*>(_a[0]) = std::move(_r); }  break;
        case 28: _t->onGPUError(); break;
        case 29: _t->onGPUFinished(); break;
        case 30: _t->on_chkBPC_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 31: _t->onNewWBFromPoint((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 32: _t->openCamera((*reinterpret_cast< uint32_t(*)>(_a[1]))); break;
        case 33: _t->openPGMFile((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 34: _t->openPGMFile(); break;
        case 35: _t->initNewCamera((*reinterpret_cast< GPUCameraBase*(*)>(_a[1])),(*reinterpret_cast< uint32_t(*)>(_a[2]))); break;
        case 36: _t->onCameraStateChanged((*reinterpret_cast< GPUCameraBase::cmrCameraState(*)>(_a[1]))); break;
        case 37: _t->on_actionOpenBayerPGM_triggered(); break;
        case 38: _t->on_actionOpenGrayPGM_triggered(); break;
        case 39: _t->on_btnGetOutPath_clicked(); break;
        case 40: _t->on_btnGetFPNFile_clicked(); break;
        case 41: _t->on_btnGetGrayFile_clicked(); break;
        case 42: _t->on_chkSAM_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 43: _t->on_btnStartRtspServer_clicked(); break;
        case 44: _t->on_btnStopRtspServer_clicked(); break;
        case 45: _t->onTimeoutStatusRtsp(); break;
        case 46: _t->on_cboFormatEnc_currentIndexChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 47: _t->on_actionShowImage_triggered((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 35:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< GPUCameraBase* >(); break;
            }
            break;
        }
    }
}

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow.data,
      qt_meta_data_MainWindow,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 48)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 48;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 48)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 48;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
