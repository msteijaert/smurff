#pragma once

#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <array>
#include <map>

#include <Eigen/Sparse>
#include <Eigen/Dense>



#include <SmurffCpp/Utils/Error.h>



#ifdef NDEBUG
#define SHOW(m)
#else
#define SHOW(m) std::cout << #m << ":\n" << m << std::endl;
#endif

inline double tick() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

inline void split_work_mpi(int num_latent, int num_nodes, int* work) {
   double avg_work = num_latent / (double) num_nodes;
   int work_unit;
   if (2 <= avg_work) work_unit = 2;
   else work_unit = 1;

   int min_work  = work_unit * (int)floor(avg_work / work_unit);
   int work_left = num_latent;

   for (int i = 0; i < num_nodes; i++) {
      work[i]    = min_work;
      work_left -= min_work;
   }
   int i = 0;
   while (work_left > 0) {
      int take = std::min(work_left, work_unit);
      work[i]   += take;
      work_left -= take;
      i = (i + 1) % num_nodes;
   }
}