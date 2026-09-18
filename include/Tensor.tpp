#pragma once

#include <utility>
#include <stdexcept>
#include "details/TensorHelpers.tpp"
#include "TensorView.hpp"
#include "MutableTensorView.hpp"


namespace tensor {
    template<typename T>
    Tensor<T>::Tensor(Shape shape)
        :layout_(std::move(shape)),
        storage_(std::make_shared<Storage<T>>(layout_.numElements()))
    {}

    template<typename T>
    Tensor<T>::Tensor(Shape shape, const std::vector<T>& values)
        :layout_(std::move(shape)),
        storage_(std::make_shared<Storage<T>>(values))
    {
        if (storage_->size() != layout_.numElements()) {
            throw std::invalid_argument("Number of elements must match number of elements in data.");
        }
    }

    template<typename T>
    Tensor<T>::Tensor(Shape shape, std::vector<T>&& values)
        :layout_(std::move(shape)),
        storage_(std::make_shared<Storage<T>>(std::move(values)))
    {
        if (storage_->size() != layout_.numElements()) {
            throw std::invalid_argument("Number of elements must match number of elements in data.");
        }
    }

    template<typename T>
    const Shape& Tensor<T>::getShape() const {
        return layout_.getShape();
    }

    template<typename T>
    const Strides& Tensor<T>::getStrides() const {
        return layout_.getStrides();
    }


    template<typename T>
    std::size_t Tensor<T>::getDim() const{
        return layout_.getDim();
    }

    template<typename T>
    std::size_t Tensor<T>::numElements() const{
        return layout_.numElements();
    }

    template<typename T>
    bool Tensor<T>::isContiguous() const{
        return layout_.isContiguous();
    }

    template<typename T>
    T& Tensor<T>::at(const Indices& indices) {
        return detail::at(*storage_, layout_, indices);
    }

    template<typename T>
    const T& Tensor<T>::at(const Indices& indices) const {
        const Storage<T>& storage = *storage_; // ensures storage reference is not mutable
        return detail::at(storage, layout_, indices);
    }

    template<typename T>
    TensorView<T> Tensor<T>::slice(const Slices& slices) const {
        TensorLayout newLayout = layout_.slice(slices);
        return TensorView<T>(storage_, std::move(newLayout));
    }

    template<typename T>
    MutableTensorView<T> Tensor<T>::mutableSlice(const Slices& slices) {
        TensorLayout newLayout = layout_.slice(slices);
        return MutableTensorView<T>(storage_, std::move(newLayout));
    }
}

