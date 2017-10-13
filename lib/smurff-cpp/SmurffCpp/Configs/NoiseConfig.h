#pragma once

#include <string>

namespace smurff
{
   class NoiseConfig
   {
   public:
      std::string name  = "noiseless";
      int cccc;
      
      // for fixed gaussian noise
      double precision  = 5.0;
   
      // for adaptive gausssian noise
      double sn_init    = 1.0;
      double sn_max     = 10.0;

   public:
      NoiseConfig(std::string n = "noiseless") : name(n)
      {
         static int c = 0;
         c++;
         cccc = c;
      }

      //NoiseConfig(double p) : name("fixed"), precision(p) {}
      //NoiseConfig(double i, double m) : name("adaptive"), sn_init(i), sn_max(m) {}

   public:
      bool validate(bool = true) const;
   };
}