/****************************************************************************
** Meta object code from reading C++ file 'AsyncFileWriter.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.5)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../src/CameraSample/AsyncFileWriter.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'AsyncFileWriter.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.5. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_AsyncWriter_t {
    QByteArrayData data[4];
    char stringdata0[30];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_AsyncWriter_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_AsyncWriter_t qt_meta_stringdata_AsyncWriter = {
    {
QT_MOC_LITERAL(0, 0, 11), // "AsyncWriter"
QT_MOC_LITERAL(1, 12, 8), // "progress"
QT_MOC_LITERAL(2, 21, 0), // ""
QT_MOC_LITERAL(3, 22, 7) // "percent"

    },
    "AsyncWriter\0progress\0\0percent"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_AsyncWriter[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,

       0        // eod
};

void AsyncWriter::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        AsyncWriter *_t = static_cast<AsyncWriter *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->progress((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            typedef void (AsyncWriter::*_t)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&AsyncWriter::progress)) {
                *result = 0;
                return;
            }
        }
    }
}

const QMetaObject AsyncWriter::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_AsyncWriter.data,
      qt_meta_data_AsyncWriter,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *AsyncWriter::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *AsyncWriter::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_AsyncWriter.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int AsyncWriter::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void AsyncWriter::progress(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
struct qt_meta_stringdata_AsyncFileWriter_t {
    QByteArrayData data[1];
    char stringdata0[16];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_AsyncFileWriter_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_AsyncFileWriter_t qt_meta_stringdata_AsyncFileWriter = {
    {
QT_MOC_LITERAL(0, 0, 15) // "AsyncFileWriter"

    },
    "AsyncFileWriter"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_AsyncFileWriter[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void AsyncFileWriter::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObject AsyncFileWriter::staticMetaObject = {
    { &AsyncWriter::staticMetaObject, qt_meta_stringdata_AsyncFileWriter.data,
      qt_meta_data_AsyncFileWriter,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *AsyncFileWriter::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *AsyncFileWriter::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_AsyncFileWriter.stringdata0))
        return static_cast<void*>(this);
    return AsyncWriter::qt_metacast(_clname);
}

int AsyncFileWriter::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AsyncWriter::qt_metacall(_c, _id, _a);
    return _id;
}
struct qt_meta_stringdata_AsyncMJPEGWriter_t {
    QByteArrayData data[1];
    char stringdata0[17];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_AsyncMJPEGWriter_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_AsyncMJPEGWriter_t qt_meta_stringdata_AsyncMJPEGWriter = {
    {
QT_MOC_LITERAL(0, 0, 16) // "AsyncMJPEGWriter"

    },
    "AsyncMJPEGWriter"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_AsyncMJPEGWriter[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

void AsyncMJPEGWriter::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObject AsyncMJPEGWriter::staticMetaObject = {
    { &AsyncWriter::staticMetaObject, qt_meta_stringdata_AsyncMJPEGWriter.data,
      qt_meta_data_AsyncMJPEGWriter,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *AsyncMJPEGWriter::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *AsyncMJPEGWriter::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_AsyncMJPEGWriter.stringdata0))
        return static_cast<void*>(this);
    return AsyncWriter::qt_metacast(_clname);
}

int AsyncMJPEGWriter::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = AsyncWriter::qt_metacall(_c, _id, _a);
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
