#pragma once

#include "Storage.hpp"
#include "TensorLayout.hpp"
#include "TensorView.hpp"
#include <memory>

namespace tensor {

    template<typename T>
    class MutableTensorView {
    public:
        MutableTensorView(std::shared_ptr<Storage<T>> storage, Shape shape, Strides strides, std::size_t offset);
        MutableTensorView(std::shared_ptr<Storage<T>> storage, TensorLayout layout);
        [[nodiscard]] const Shape& getShape() const;
        [[nodiscard]] const Strides& getStrides() const;
        [[nodiscard]] std::size_t getDim() const;
        [[nodiscard]] std::size_t numElements() const;
        [[nodiscard]] bool isContiguous() const;
        [[nodiscard]] T& at(const Indices& indices);
        [[nodiscard]] const T& at(const Indices& indices) const;
        [[nodiscard]] TensorView<T> slice(const Slices& slices) const;
        [[nodiscard]] MutableTensorView<T> mutableSlice(const Slices& slices);


    private:
        TensorLayout layout_;
        std::shared_ptr<Storage<T>> storage_;
    };
}

#include "MutableTensorView.tpp"