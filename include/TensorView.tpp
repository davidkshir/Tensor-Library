#pragma once

#include<memory>
#include<utility>
#include "details/TensorHelpers.hpp"

namespace tensor {

    template<typename T>
    TensorView<T>::TensorView(std::shared_ptr<Storage<T>> storage, Shape shape, Strides strides, std::size_t offset)
        :layout_(std::move(shape), std::move(strides), offset),
        storage_(std::move(storage))
    {
        detail::validateView(storage_, layout_);
    }

    template<typename T>
    TensorView<T>::TensorView(std::shared_ptr<Storage<T>> storage, TensorLayout layout)
        :layout_(std::move(layout)),
        storage_(std::move(storage))
    {
        detail::validateView(storage_, layout_);
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

    template<typename T>
    TensorView<T> TensorView<T>::slice(const Slices& slices) const {
        TensorLayout newLayout = layout_.slice(slices);
        return TensorView<T>(storage_, std::move(newLayout));
    }

}
