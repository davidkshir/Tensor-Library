#pragma once

#include "Storage.hpp"
#include "Types.hpp"
#include "TensorLayout.hpp"
#include <memory>

namespace tensor {

    template<typename T>
    class Tensor;

    template<typename T>
    class TensorView {
    public:
        using valueType = T;
        TensorView(std::shared_ptr<Storage<T>> storage, Shape shape, Strides strides, std::size_t offset);
        TensorView(std::shared_ptr<Storage<T>> storage, TensorLayout layout);
        [[nodiscard]] const Shape& getShape() const;
        [[nodiscard]] const Strides& getStrides() const;
        [[nodiscard]] std::size_t getDim() const;
        [[nodiscard]] std::size_t numElements() const;
        [[nodiscard]] bool isContiguous() const;
        [[nodiscard]] const T& at(const Indices& indices) const;
        [[nodiscard]] TensorView<T> slice(const Slices& slices) const;
        [[nodiscard]] Tensor<T> clone() const;
        [[nodiscard]] TensorView<T> reshape(const Shape& newShape) const;
        [[nodiscard]] TensorView<T> permute(const Axes& newAxes) const;
        [[nodiscard]] TensorView<T> transpose(std::size_t axis1, std::size_t axis2) const;
        [[nodiscard]] const Storage<T>& getStorage() const;
        [[nodiscard]] const TensorLayout& getLayout() const;
    private:
        TensorLayout layout_;
        std::shared_ptr<Storage<T>> storage_;
    };
}
#include "TensorView.tpp"