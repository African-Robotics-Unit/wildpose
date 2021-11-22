#ifndef TEST_ENCODER_H
#define TEST_ENCODER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>

#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <JetsonGPIO.h>
#include <memory.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <chrono>
#include <mutex>

using namespace std;

class test_encoder : public QObject
{
    Q_OBJECT
public:
    explicit test_encoder(QObject *parent = nullptr);
    ~test_encoder();
    void start();
    int encoder_count(const int& value, int encoder_index);
    void stop();
    void wake();
    void encoder_update_thread(int encoder_index);
    void open_csv();
    void close_file();
    void write_encoders(unsigned counter);

signals:

public slots:

private:
    int counter0;
    int counter1;
    int stop_flag;
    bool eWake;
    std::mutex m;
    std::ofstream encoderlog1;
    QThread encoder_thread;
    QMutex eMutex;
    QWaitCondition eWait;
};

#endif // TEST_ENCODER_H