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

    template<typename T>
    void validateView(const std::shared_ptr<Storage<T>>&  storage, const TensorLayout& layout) {
        if (storage == nullptr) {
            throw std::invalid_argument("Storage pointer cannot be null.");
        }
        if (layout.numElements() == 0) {
            if (layout.getOffset() > storage->size()) {
                throw std::out_of_range("Offset is out of storage range.");
            }
        }
        else {
            Indices maxIndex;

            for (std::size_t axis : layout.getShape()) {
                maxIndex.push_back(axis - 1);
            }
            if (layout.getStorageIndex(maxIndex) >= storage->size()) {
                throw std::invalid_argument("Tensor view exceeds max storage.");
            }
        }
    }

}