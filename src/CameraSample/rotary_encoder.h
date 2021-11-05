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

class rotary_encoder
{

    public:
    
        int counter0;
        int counter1;
        int stop;
        std::mutex m;
        std::ofstream encoderlog1;

        rotary_encoder();
        void run_threads();
        int encoder_count(const int& value, int encoder_index);
        void stop_threads();
        void encoder_update_thread(int encoder_index);
        void open_csv();
        void close_file();
        void write_encoders(unsigned counter);
};