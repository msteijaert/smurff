#pragma once

#include <algorithm>
#include <memory>
#include <cstdint>

#include <SmurffCpp/Model.h>
#include <SmurffCpp/Utils/PVec.hpp>

//http://en.cppreference.com/w/cpp/iterator/iterator

namespace smurff {

template<class T>
class VMatrixIterator : public std::iterator<
	std::input_iterator_tag,   // iterator_category
	std::shared_ptr<T>,        // value_type
	std::uint32_t,             // difference_type
	std::shared_ptr<T>,        // pointer - NOTE: it is logically correct to use std::shared_ptr<T>* here but for convenience we ommit *
	std::shared_ptr<T>         // reference
>
{
private:
	std::shared_ptr<Model> m_model;
	std::uint32_t m_mode;
	std::uint32_t m_num;

public:
	//begin constructor
   VMatrixIterator(std::shared_ptr<Model> model, std::uint32_t mode, std::uint32_t num)
      : m_model(model), m_mode(mode), m_num(num == mode ? num + 1 : num)
   {
   }

	//end constructor
   VMatrixIterator(std::uint32_t num)
      : m_model(std::shared_ptr<Model>()), m_mode(-1), m_num(num)
   {
   }

public:
	//copy constructors
   VMatrixIterator(const VMatrixIterator<T>& other)
      : m_model(other.m_model),
        m_mode(other.m_mode),
        m_num(other.m_num)
   {
   }

public:
   VMatrixIterator<T>& operator++()
   {
      m_num++;
      if (m_num == m_mode)
         m_num++;
      return *this;
   }

   VMatrixIterator<T> operator++(int)
   {
      VMatrixIterator<T> retval = *this;
      ++(*this);
      return retval;
   }

   bool operator==(VMatrixIterator<T> other) const
   {
      return m_num == other.m_num;
   }

   bool operator!=(VMatrixIterator<T> other) const
   {
      return !(*this == other);
   }

   std::shared_ptr<T> operator*()
   {
      return m_model->U(m_num);
   }

   std::shared_ptr<T> operator->()
   {
      return this->operator*();
   }
};

}