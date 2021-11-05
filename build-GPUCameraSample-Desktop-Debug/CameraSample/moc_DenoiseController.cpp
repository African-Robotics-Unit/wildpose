/****************************************************************************
** Meta object code from reading C++ file 'DenoiseController.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.5)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/CameraSample/Widgets/DenoiseController.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'DenoiseController.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_DenoiseController_t {
    QByteArrayData data[29];
    char stringdata0[534];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_DenoiseController_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_DenoiseController_t qt_meta_stringdata_DenoiseController = {
    {
QT_MOC_LITERAL(0, 0, 17), // "DenoiseController"
QT_MOC_LITERAL(1, 18, 12), // "stateChanged"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 2), // "on"
QT_MOC_LITERAL(4, 35, 20), // "thresholdTypeChanged"
QT_MOC_LITERAL(5, 56, 7), // "newType"
QT_MOC_LITERAL(6, 64, 18), // "waveletTypeChanged"
QT_MOC_LITERAL(7, 83, 16), // "shrinkageChanged"
QT_MOC_LITERAL(8, 100, 12), // "newShrinkage"
QT_MOC_LITERAL(9, 113, 17), // "yThresholdChanged"
QT_MOC_LITERAL(10, 131, 12), // "newThreshold"
QT_MOC_LITERAL(11, 144, 18), // "cbThresholdChanged"
QT_MOC_LITERAL(12, 163, 13), // "paramsChanged"
QT_MOC_LITERAL(13, 177, 39), // "on_cboThresholdType_currentIn..."
QT_MOC_LITERAL(14, 217, 5), // "index"
QT_MOC_LITERAL(15, 223, 37), // "on_cboWaveletType_currentInde..."
QT_MOC_LITERAL(16, 261, 30), // "on_sldCbThreshold_valueChanged"
QT_MOC_LITERAL(17, 292, 5), // "value"
QT_MOC_LITERAL(18, 298, 29), // "on_sldYThreshold_valueChanged"
QT_MOC_LITERAL(19, 328, 23), // "on_spnMaxY_valueChanged"
QT_MOC_LITERAL(20, 352, 4), // "arg1"
QT_MOC_LITERAL(21, 357, 27), // "on_spnMaxColor_valueChanged"
QT_MOC_LITERAL(22, 385, 20), // "onSliderValueChanged"
QT_MOC_LITERAL(23, 406, 30), // "on_sldCrThreshold_valueChanged"
QT_MOC_LITERAL(24, 437, 20), // "on_btnResetY_clicked"
QT_MOC_LITERAL(25, 458, 21), // "on_btnResetCb_clicked"
QT_MOC_LITERAL(26, 480, 21), // "on_btnResetCr_clicked"
QT_MOC_LITERAL(27, 502, 23), // "on_chkSyncColor_toggled"
QT_MOC_LITERAL(28, 526, 7) // "checked"

    },
    "DenoiseController\0stateChanged\0\0on\0"
    "thresholdTypeChanged\0newType\0"
    "waveletTypeChanged\0shrinkageChanged\0"
    "newShrinkage\0yThresholdChanged\0"
    "newThreshold\0cbThresholdChanged\0"
    "paramsChanged\0on_cboThresholdType_currentIndexChanged\0"
    "index\0on_cboWaveletType_currentIndexChanged\0"
    "on_sldCbThreshold_valueChanged\0value\0"
    "on_sldYThreshold_valueChanged\0"
    "on_spnMaxY_valueChanged\0arg1\0"
    "on_spnMaxColor_valueChanged\0"
    "onSliderValueChanged\0"
    "on_sldCrThreshold_valueChanged\0"
    "on_btnResetY_clicked\0on_btnResetCb_clicked\0"
    "on_btnResetCr_clicked\0on_chkSyncColor_toggled\0"
    "checked"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_DenoiseController[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      19,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       7,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,  109,    2, 0x06 /* Public */,
       4,    1,  112,    2, 0x06 /* Public */,
       6,    1,  115,    2, 0x06 /* Public */,
       7,    1,  118,    2, 0x06 /* Public */,
       9,    1,  121,    2, 0x06 /* Public */,
      11,    1,  124,    2, 0x06 /* Public */,
      12,    0,  127,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      13,    1,  128,    2, 0x08 /* Private */,
      15,    1,  131,    2, 0x08 /* Private */,
      16,    1,  134,    2, 0x08 /* Private */,
      18,    1,  137,    2, 0x08 /* Private */,
      19,    1,  140,    2, 0x08 /* Private */,
      21,    1,  143,    2, 0x08 /* Private */,
      22,    1,  146,    2, 0x08 /* Private */,
      23,    1,  149,    2, 0x08 /* Private */,
      24,    0,  152,    2, 0x08 /* Private */,
      25,    0,  153,    2, 0x08 /* Private */,
      26,    0,  154,    2, 0x08 /* Private */,
      27,    1,  155,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, QMetaType::Float,   10,
    QMetaType::Void, QMetaType::Float,   10,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,   14,
    QMetaType::Void, QMetaType::Int,   14,
    QMetaType::Void, QMetaType::Int,   17,
    QMetaType::Void, QMetaType::Int,   17,
    QMetaType::Void, QMetaType::Double,   20,
    QMetaType::Void, QMetaType::Double,   20,
    QMetaType::Void, QMetaType::Int,   17,
    QMetaType::Void, QMetaType::Int,   17,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   28,

       0        // eod
};

void DenoiseController::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        DenoiseController *_t = static_cast<DenoiseController *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->stateChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->thresholdTypeChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->waveletTypeChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->shrinkageChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->yThresholdChanged((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 5: _t->cbThresholdChanged((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 6: _t->paramsChanged(); break;
        case 7: _t->on_cboThresholdType_currentIndexChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->on_cboWaveletType_currentIndexChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->on_sldCbThreshold_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->on_sldYThreshold_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: _t->on_spnMaxY_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 12: _t->on_spnMaxColor_valueChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 13: _t->onSliderValueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 14: _t->on_sldCrThreshold_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 15: _t->on_btnResetY_clicked(); break;
        case 16: _t->on_btnResetCb_clicked(); break;
        case 17: _t->on_btnResetCr_clicked(); break;
        case 18: _t->on_chkSyncColor_toggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            typedef void (DenoiseController::*_t)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DenoiseController::stateChanged)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (DenoiseController::*_t)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DenoiseController::thresholdTypeChanged)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (DenoiseController::*_t)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DenoiseController::waveletTypeChanged)) {
                *result = 2;
                return;
            }
        }
        {
            typedef void (DenoiseController::*_t)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DenoiseController::shrinkageChanged)) {
                *result = 3;
                return;
            }
        }
        {
            typedef void (DenoiseController::*_t)(float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DenoiseController::yThresholdChanged)) {
                *result = 4;
                return;
            }
        }
        {
            typedef void (DenoiseController::*_t)(float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DenoiseController::cbThresholdChanged)) {
                *result = 5;
                return;
            }
        }
        {
            typedef void (DenoiseController::*_t)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&DenoiseController::paramsChanged)) {
                *result = 6;
                return;
            }
        }
    }
}

const QMetaObject DenoiseController::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_DenoiseController.data,
      qt_meta_data_DenoiseController,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *DenoiseController::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *DenoiseController::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_DenoiseController.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int DenoiseController::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 19)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 19;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 19)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 19;
    }
    return _id;
}

// SIGNAL 0
void DenoiseController::stateChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void DenoiseController::thresholdTypeChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void DenoiseController::waveletTypeChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void DenoiseController::shrinkageChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void DenoiseController::yThresholdChanged(float _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void DenoiseController::cbThresholdChanged(float _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void DenoiseController::paramsChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
