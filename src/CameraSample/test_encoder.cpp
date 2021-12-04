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
using namespace GPIO;

test_encoder::test_encoder(QObject *parent) : QObject(parent)
{
    this -> counter0 = 0;
    this -> counter1 = 0;
    this -> counter2 = 0;
    this -> counter3 = 0;
    this -> stop_flag = 0;

    //moveToThread(&encoder_thread);
    //encoder_thread.start();
}

test_encoder::~test_encoder(){
    stop();
    //encoder_thread.quit();
    //encoder_thread.wait(3000);
}

void test_encoder::wake(){
    //eWake = true;
    //eWait.wakeAll();
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
    } else if(encoder_index == 1)
    {
        if (value == 0)
        {
            return this -> counter1;
        }
        else
        {
            this -> counter1 = value;
            return 0;
        }   
    } else if(encoder_index == 2){
        if (value == 0)
        {
            return this -> counter2;
        }
        else
        {
            this -> counter2 = value;
            return 0;
        }   
    } else{
        if (value == 0)
        {
            return this -> counter3;
        }
        else
        {
            this -> counter3 = value;
            return 0;
        }   
    }
}

// Threaded function for updating encoder count variable based on pin values
void test_encoder::encoder_update_thread(int encoder_index = 0){
    int encode_a = 13;
    int encode_b = 15;
    int val_a;
    int val_b;
    int last_reading;

    //GPIO::cleanup();
    //GPIO::setmode(GPIO::BOARD);

    if (encoder_index == 0){
        //uncomment when encoders are fixed
        encode_a = 19;
        encode_b = 21;
	    GPIO::setup(encode_a, GPIO::INTO);
        GPIO::setup(encode_b, GPIO::INTO);
    } else if(encoder_index == 1)
    {
        ///uncomment when encoders are fixed
        encode_a = 13;
        encode_b = 15;
	    GPIO::setup(encode_a, GPIO::INTO);
        GPIO::setup(encode_b, GPIO::INTO);
    }else if(encoder_index == 2)
    {
        encode_a = 12;
        encode_b = 16;
	    GPIO::setup(encode_a, GPIO::INTO);
        GPIO::setup(encode_b, GPIO::INTO);
    }else
    {
        encode_a = 24;
        encode_b = 26;
	    GPIO::setup(encode_a, GPIO::INTO);
        GPIO::setup(encode_b, GPIO::INTO);
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
        } else if(encoder_index == 1){
            int var1 = this -> encoder_count(0,1);

            if ((val_b != last_reading) && val_b==1){
                if (val_a != val_b) {
                    var1 --;
                    this -> encoder_count(var1, 1);
                } else {
                    // Encoder is rotating CW so increment
                    //cout << "Increment 1" << counter1 << endl;
                    var1 ++;
                    this -> encoder_count(var1, 1);
                }

            }
        } else if(encoder_index == 2){
            int var2 = this -> encoder_count(0,2);

            if ((val_b != last_reading) && val_b==1){
                if (val_a != val_b) {
                    var2 --;
                    this -> encoder_count(var2, 2);
                } else {
                    // Encoder is rotating CW so increment
                    //cout << "Increment 1" << counter1 << endl;
                    var2 ++;
                    this -> encoder_count(var2, 2);
                }

            }
        } else if(encoder_index == 3){
            int var3 = this -> encoder_count(0,3);

            if ((val_b != last_reading) && val_b==1){
                if (val_a != val_b) {
                    var3 --;
                    this -> encoder_count(var3, 3);
                } else {
                    // Encoder is rotating CW so increment
                    //cout << "Increment 1" << counter1 << endl;
                    var3 ++;
                    this -> encoder_count(var3, 3);
                }

            }
        }
        
        last_reading = val_b;
    }

    return;

}
void test_encoder::loopy()
{
	int right_out = 37;
	int left_out = 33;
	int up_out = 31;
	int down_out = 29;

	int right_in = 40;
	int left_in = 38;
	int up_in = 36;
	int down_in = 32;

	GPIO::setup(right_in, GPIO::INTO);
	GPIO::setup(left_in, GPIO::INTO);
	GPIO::setup(up_in, GPIO::INTO);
	GPIO::setup(down_in, GPIO::INTO);
	
	GPIO::setup(right_out, GPIO::OUTOF, GPIO::LOW);	
	GPIO::setup(left_out, GPIO::OUTOF, GPIO::LOW);
	GPIO::setup(up_out, GPIO::OUTOF, GPIO::LOW);
	GPIO::setup(down_out, GPIO::OUTOF, GPIO::LOW);

	cout << "Running joystick control!" << endl;

	while (!stop_flag)
	{
		int is_right = GPIO::input(right_in);
		int is_left = GPIO::input(left_in);
		int is_up = GPIO::input(up_in);
		int is_down = GPIO::input(down_in);
		if(is_right){
			GPIO::output(right_out, GPIO::HIGH);
			
		}else{
			
			GPIO::output(right_out, GPIO::LOW);
		}	
		
		if(is_left){
			
			GPIO::output(left_out, GPIO::HIGH);
		}else{
			
			GPIO::output(left_out, GPIO::LOW);
		}	
		
		if(is_up){
		
			GPIO::output(up_out, GPIO::HIGH);
		}else{
			
			GPIO::output(up_out, GPIO::LOW);
		}	
		
		if(is_down){
		
			GPIO::output(down_out, GPIO::HIGH);
		}else{
			
			GPIO::output(down_out, GPIO::LOW);
		}
	}

    return;
}

void test_encoder::stop(){
    cout << "Stopping threads!" << endl;
    //eWait.wakeAll();
    this -> stop_flag = 1;
    //wait for threads to end
    usleep(1000);
    GPIO::cleanup();
    cout << "Stoppedthreads!" << endl;
}

void test_encoder::start(){
    cout << "Running threads!" << endl;
    //this -> encoder_update_thread();
    //GPIO::cleanup();
    GPIO::setmode(GPIO::BOARD);
    std::thread t1(&test_encoder::encoder_update_thread, this, 0);
    std::thread t2(&test_encoder::encoder_update_thread, this, 1);
    std::thread t3(&test_encoder::encoder_update_thread, this, 2);
    std::thread t4(&test_encoder::encoder_update_thread, this, 3);
    std::thread j1(&test_encoder::loopy, this);
    t1.detach();
    t2.detach();
    t3.detach();
    t4.detach();
    j1.detach();
    //QTimer::singleShot(0, this, [this](){encoder_update_thread();});
    

}

void test_encoder::open_csv(){

    encoderlog1.open("encodetest.csv");
    encoderlog1 << "Count,encoder_a,encoder_b,encoder_c,encoder_d\n";
}

void test_encoder::write_encoders(unsigned counter){
    std::string count_string = to_string(counter);

    std::string enc_a = to_string(this -> encoder_count(0,0));
    std::string enc_b = to_string(this -> encoder_count(0,1));
    std::string enc_c = to_string(this -> encoder_count(0,2));
    std::string enc_d = to_string(this -> encoder_count(0,3));

    cout << count_string << " enc1 " << enc_a << " enc2 " << enc_b << endl;
    cout << count_string << " enc3 " << enc_c << " enc4 " << enc_d << endl;
    encoderlog1 << count_string+","+enc_a+","+enc_b+","+enc_c+","+enc_d+"\n";
}

void test_encoder::close_file(){
    encoderlog1.close();
    cout << "Closed file csv" << endl;
}
