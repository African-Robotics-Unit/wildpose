#include "test_encoder.h"

#include <algorithm>
#include <string.h>
#include <stdio.h>
#include <JetsonGPIO.h>
#include <memory.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <chrono>
#ifdef WIN32
#include <xiApi.h>       // Windows
#else
#include <m3api/xiApi.h> // Linux, OSX
#include <unistd.h>      // usleep
#endif

#include <QElapsedTimer>
#include <QDateTime>
#include <QDebug>
#include <QPoint>

using namespace std;
using namespace std::chrono;
using namespace GPIO;

test_encoder::test_encoder(QObject *parent) : QObject(parent)
{
    this -> counter0 = 0;
    this -> counter1 = 0;
    this -> stop_flag = 0;

    //moveToThread(&encoder_thread);
    //encoder_thread.start();
}

test_encoder::~test_encoder(){
    stop();
    encoder_thread.quit();
    encoder_thread.wait(3000);
}

void test_encoder::wake(){
    eWake = true;
    eWait.wakeAll();
}

// Tester function for updating global variable in separate thread
int test_encoder::encoder_count(const int& value = 0, int encoder_index = 0)
{
    std::lock_guard<std::mutex> lock(m);
    //eMutex.lock();
    //eWait.wait(&eMutex);
    //eMutex.unlock();
    if (encoder_index == 0){
        if (value == 0)
        {
            return this -> counter0;
        }
        else
        {
            this -> counter0 = value;
            return 0;
        }   
    } else {
        if (value == 0)
        {
            return this -> counter1;
        }
        else
        {
            this -> counter1 = value;
            return 0;
        }   
    }
}

// Threaded function for updating encoder count variable based on pin values
void test_encoder::encoder_update_thread(int encoder_index = 0){
    int encode_a;
    int encode_b;
    int val_a;
    int val_b;
    int last_reading;

    if (encoder_index == 0){
        encode_a = 15;
        encode_b = 13;
        GPIO::setmode(GPIO::BOARD);
	    GPIO::setup(13, GPIO::INTO);
        GPIO::setup(15, GPIO::INTO);
    } else {
        encode_a = 29;
        encode_b = 31;
        GPIO::setmode(GPIO::BOARD);
	    GPIO::setup(29, GPIO::INTO);
        GPIO::setup(31, GPIO::INTO);
    }
    
    last_reading = GPIO::input(encode_b);
    eWake = false;

    while (stop_flag == 0){
        // get a sample
        //if(!eWake)
        //{
        //    eMutex.lock();
        //    eWait.wait(&eMutex);
        //    eMutex.unlock();
        //}
        eWake = false;
        val_a = GPIO::input(encode_a);
        val_b = GPIO::input(encode_b);

        if (encoder_index == 0){
            
            int var0 = this -> encoder_count(0,0);

            if ((val_b != last_reading) && val_b==1){
                if (val_a != val_b) {
                    var0 --;
                    //cout << "Decrement 0" << counter0 << endl;
                    this -> encoder_count(var0, 0);
                } else {
                    // Encoder is rotating CW so increment
                    //cout << "Increment 0" << counter0 << endl;
                    var0 ++;
                    this -> encoder_count(var0, 0);
                }

            }
        } else {
            int var1 = this -> encoder_count(0,1);

            if ((val_b != last_reading) && val_b==1){
                if (val_a != val_b) {
                    var1 --;
                    this -> encoder_count(var1, 1);
                } else {
                    // Encoder is rotating CW so increment
                    cout << "Increment 1" << counter1 << endl;
                    var1 ++;
                    this -> encoder_count(var1, 1);
                }

            }
        }
        
        last_reading = val_b;
    }

    return;

}

void test_encoder::stop(){
    cout << "Stopped threads!" << endl;
    eWait.wakeAll();
    this -> stop_flag = 1;
}

void test_encoder::start(){
    cout << "Running threads!" << endl;
    //this -> encoder_update_thread();
    std::thread t1(&test_encoder::encoder_update_thread, this, 0);
    std::thread t2(&test_encoder::encoder_update_thread, this, 1);
    t1.detach();
    t2.detach();
    //QTimer::singleShot(0, this, [this](){encoder_update_thread();});
    

}

void test_encoder::open_csv(){

    encoderlog1.open("encodetest.csv");
    encoderlog1 << "Count,encoder_a,encoder_b\n";
}

void test_encoder::write_encoders(unsigned counter){
    std::string count_string = to_string(counter);
    std::string enc_a = to_string(this -> encoder_count(0,0));
    std::string enc_b = to_string(this -> encoder_count(0,1));
    cout << count_string << " enc1 " << enc_a << " enc2 " << enc_b << endl;
    encoderlog1 << count_string+","+enc_a+","+enc_b+"\n";
}

void test_encoder::close_file(){
    encoderlog1.close();
    cout << "Closed file csv" << endl;
}
