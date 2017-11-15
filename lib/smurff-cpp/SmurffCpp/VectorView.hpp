#pragma once

#include <stdint.h>

#include <memory>
#include <vector>

namespace smurff {

// this class creates vector from another vector and removes 1 element (selected with m_removed)
// this is basically needed to select all V matrices from the model and exclude U matrix
template<class T>
class VectorView 
{
private:
   std::shared_ptr<std::vector<std::shared_ptr<T> > > m_vec;
   std::uint32_t m_removed;

protected:
   VectorView(){}

public:
   VectorView(const std::vector<std::shared_ptr<T> >& vec, uint32_t removed) 
      : m_vec(new std::vector<std::shared_ptr<T> >()),
        m_removed(-1)
   {
      for(size_t i = 0; i < vec.size(); i++) 
      {
         if(removed == i)
            continue;

         m_vec->push_back(vec[i]);
      }

      m_removed = removed;
   }

   VectorView(std::shared_ptr<std::vector<std::shared_ptr<T> > > vec, uint32_t removed)
      : m_vec(vec),
        m_removed(-1)
   {

   }

public:
   std::shared_ptr<const T> get(uint32_t i) const 
   {
      return m_vec->operator[](i);
   }

   std::shared_ptr<T> get(uint32_t i) 
   {
      return m_vec->operator[](i);
   }

   size_t size() const
   {
      return m_vec->size();
   }

   std::uint32_t removed() const
   {
      return m_removed;
   }
};

}