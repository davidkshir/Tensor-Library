#pragma once

#include <stdexcept>
#include <utility>
namespace tensor {

   template<typename T>
   Storage<T>::Storage(std::size_t size)
            :storage_(size)
   {}

   template<typename T>
   Storage<T>::Storage(const std::vector<T>& values)
            :storage_(values)
   {}

   template<typename T>
   Storage<T>::Storage(std::vector<T>&& values)
            :storage_(std::move(values))
   {}

   // Having a non-constant and constant indexing will be useful for views
   template<typename T>
   T& Storage<T>::operator[](std::size_t index) {
      return storage_[index];
   }

   template<typename T>
   const T& Storage<T>::operator[](std::size_t index) const {
      return storage_[index];
   }

   template<typename T>
   std::size_t Storage<T>::size() const{
      return storage_.size();
   }
}
