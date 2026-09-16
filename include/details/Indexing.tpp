#pragma once
#include "Storage.hpp"
#include "TensorLayout.hpp"

namespace tensor::detail {
    template<typename T>
    T& at(Storage<T>& storage, const TensorLayout& layout, const Indices& indices) {
        return storage[layout.getStorageIndex(indices)];
    }
    template<typename T>
    const T& at(const Storage<T>& storage, const TensorLayout& layout, const Indices& indices){
        return storage[layout.getStorageIndex(indices)];
    }

}