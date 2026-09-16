#pragma once
#include<memory>
#include<utility>
#include<stdexcept>

#include "Indexing.hpp"

namespace tensor {

    template<typename T>
    TensorView<T>::TensorView(std::shared_ptr<Storage<T>> storage, Shape shape, Strides strides, std::size_t offset)
        :layout_(std::move(shape), std::move(strides), offset),
        storage_(std::move(storage))
    {
        if (storage_ == nullptr) {
            throw std::invalid_argument("Storage pointer cannot be null.");
        }
        if (layout_.numElements() == 0) {
            if (offset > storage_->size()) {
                throw std::out_of_range("Offset is out of storage range.");
            }
        }
        else {
            Indices maxIndex;

            for (std::size_t axis : layout_.getShape()) {
                maxIndex.push_back(axis - 1);
            }
            if (layout_.getStorageIndex(maxIndex) >= storage_->size()) {
                throw std::invalid_argument("Tensor view exceeds max storage.");
            }
        }
    }

    template<typename T>
    const Shape& TensorView<T>::getShape() const {
        return layout_.getShape();
    }

    template<typename T>
    const Strides& TensorView<T>::getStrides() const {
        return layout_.getStrides();
    }

    template<typename T>
    std::size_t TensorView<T>::getDim() const{
        return layout_.getDim();
    }

    template<typename T>
    std::size_t TensorView<T>::numElements() const{
        return layout_.numElements();
    }

    template<typename T>
    bool TensorView<T>::isContiguous() const{
        return layout_.isContiguous();
    }

    template<typename T>
    const T& TensorView<T>::at(const Indices& indices) const {
        const Storage<T>& storage = *storage_;
        return detail::at(storage, layout_, indices);
    }


}
