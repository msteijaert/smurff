/*
 * Copyright (c) 2014-2016, imec
 * All rights reserved.
 */

#include <chrono>

#ifdef PROFILING

#include <mutex>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>

#include "utils.h"
#include "counters.h"

Counter::Counter(std::string name)
    : name(name), diff(0), count(1), total_counter(false)
{
    start = tick();
}

Counter::Counter() 
    : name(std::string()), diff(0), count(0), total_counter(true)
{} 

Counter::~Counter() {
    static std::mutex mtx;

    if(total_counter) return;
    stop = tick();
    diff = stop - start;

    mtx.lock();
    perf_data[name] += *this;
    mtx.unlock();
}

void Counter::operator+=(const Counter &other) {
    if (name.empty()) name = other.name;
    diff += other.diff;
    count += other.count;
}

std::string Counter::as_string(const Counter &total) {
    std::ostringstream os;
    int percent = round(100.0 * diff / (total.diff + 0.000001));
    os << ">> " << name << ":\t" << std::fixed << std::setw(11)
       << std::setprecision(4) << diff << "\t(" << percent << "%) in\t" << count << "\n";
    return os.str();
}

TotalsCounter perf_data;

TotalsCounter::TotalsCounter(int p) : procid(p) {}

void TotalsCounter::print() {
    if (data.empty()) return;
    char hostname[1024];
    gethostname(hostname, 1024);
    std::cout << "\nTotals on " << hostname << " (" << procid << "):\n";
    for(auto &t : data)
       std::cout << t.second.as_string(data["main"]);
}

#endif // PROFILING

double tick() 
{
   return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}