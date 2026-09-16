#pragma once

#include<memory>
#include<utility>
#include<stdexcept>
#include "TensorHelpers.hpp"

namespace tensor {

    template<typename T>
    MutableTensorView<T>::MutableTensorView(std::shared_ptr<Storage<T>> storage, Shape shape, Strides strides, std::size_t offset)
        :layout_(std::move(shape), std::move(strides), offset),
        storage_(std::move(storage))
    {
        detail::validateView(storage_, layout_);
    }

    template<typename T>
    const Shape& MutableTensorView<T>::getShape() const {
        return layout_.getShape();
    }

    template<typename T>
    const Strides& MutableTensorView<T>::getStrides() const {
        return layout_.getStrides();
    }

    template<typename T>
    std::size_t MutableTensorView<T>::getDim() const{
        return layout_.getDim();
    }

    template<typename T>
    std::size_t MutableTensorView<T>::numElements() const{
        return layout_.numElements();
    }

    template<typename T>
    bool MutableTensorView<T>::isContiguous() const{
        return layout_.isContiguous();
    }

    template<typename T>
    T& MutableTensorView<T>::at(const Indices& indices){
        return detail::at(*storage_, layout_, indices);
    }

    template<typename T>
    const T& MutableTensorView<T>::at(const Indices& indices) const {
        const Storage<T>& storage = *storage_;
        return detail::at(storage, layout_, indices);
    }
}
