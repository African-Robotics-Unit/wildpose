#include <algorithm>
#include <string.h>
#include <cmdline.h>
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

#include "xiAPI_tiff.h"

using namespace std;

std::mutex m;
int counter0 = 0;
int counter1 = 0;
int stop = 0;

// Tester function for updating global variable in separate thread
int encoder_count(const int& value = 0, const int encoder_index = 0)
{
    std::lock_guard<std::mutex> lock(m);
    if (encoder_index == 0){
        if (value == 0)
        {
            return counter0;
        }
        else
        {
            counter0 = value;
            return 0;
        }   
    } else if (encoder_index == 1){
        if (value == 0)
        {
            return counter1;
        }
        else
        {
            counter1 = value;
            return 0;
        }   
    }
}

// Threaded function for updating encoder count variable based on pin values
void encoder_update_thread(const int encoder_index = 0){
    int encode_a;
    int encode_b;
    int val_a;
    int val_b;
    int last_reading;

    if (encoder_index == 0){
        encode_a = 15;
        encode_b = 13;
    } else {
        encode_a = 29;
        encode_b = 31;
    }
    
    GPIO::cleanup();
	GPIO::setmode(GPIO::BOARD);
	GPIO::setup(encode_a, GPIO::INTO);
    GPIO::setup(encode_b, GPIO::INTO);

    last_reading = GPIO::input(encode_b);

    steady_clock::time_point last_time = steady_clock::now();

    while (stop == 0){
        // get a sample
        val_a = GPIO::input(encode_a);
        val_b = GPIO::input(encode_b);

        if (encoder_index == 0){
            
            int counter0 = encoder_count();

            if ((val_b != last_reading) && val_b==1){
                if (val_a != val_b) {
                    counter0 --;
                    encoder_count(counter0);
                } else {
                    // Encoder is rotating CW so increment
                    counter0 ++;
                    encoder_count(counter0);
                }

            }
        } else {
            int counter1 = encoder_count(encoder_index=1);

            if ((val_b != last_reading) && val_b==1){
                if (val_a != val_b) {
                    counter1 --;
                    encoder_count(counter1, encoder_index=1);
                } else {
                    // Encoder is rotating CW so increment
                    counter1 ++;
                    encoder_count(counter1, encoder_index=1);
                }

            }
        }
        
        last_reading = val_b;
    }

}

void encoder_save(const int encoder_index = 0){
    std::ofstream myfile;
    if (encoder_index == 0){
        myfile.open(std::format("encoder0.csv"));
        myfile << "Counts:\n";
        while(stop==0){
            int counter = encoder_count(encoder_index=0);
            std::string count_string = to_string(counter);
            myfile << count_string+"\n";
            sleep(2);
        }
    } else {
        myfile.open(std::format("encoder1.csv"));
        myfile << "Counts:\n";
        while(stop==0){
            int counter = encoder_count(encoder_index=1);
            std::string count_string = to_string(counter);
            myfile << count_string+"\n";
            sleep(2);
        }
    }
    
    myfile.close();
    cout << "Saving complete!" << endl;
}

void stop_threads(){
    stop = 1;
}