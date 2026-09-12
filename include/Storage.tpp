#pragma once

#include <stdexcept>
namespace tensor {

   template<typename T>
   Storage<T>::Storage(std::size_t size)
            :storage(size)
   {
      if (size == 0) {
         throw std::invalid_argument("Storage size must be greater than zero.");
      }
   }

   template<typename T>
   Storage<T>::Storage(const std::vector<T>& values)
            :storage(values)
   {
      if (storage.empty()) {
         throw std::invalid_argument("Storage size must be greater than zero.");
      }
   }

   template<typename T>
   Storage<T>::Storage(std::vector<T>&& values)
            :storage(std::move(values))
   {
      if (storage.empty()) {
         throw std::invalid_argument("Storage size must be greater than zero.");
      }
   }

   // Having a non-constant and constant indexing will be useful for views
   template<typename T>
   T& Storage<T>::operator[](std::size_t index) {
      return storage[index];
   }

   template<typename T>
   const T& Storage<T>::operator[](std::size_t index) const {
      return storage[index];
   }

   template<typename T>
   std::size_t Storage<T>::size() const{
      return storage.size();
   }
}
