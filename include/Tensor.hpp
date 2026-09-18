#pragma once

#include "Storage.hpp"
#include "TensorLayout.hpp"
#include "TensorView.hpp"
#include "MutableTensorView.hpp"
#include "Types.hpp"
#include <memory>

namespace tensor {

    template<typename T>
    class Tensor {
    public:
        explicit Tensor(Shape shape);
        Tensor(Shape shape, const std::vector<T>& values); // copies data
        Tensor(Shape shape, std::vector<T>&& values); // moves data
        [[nodiscard]] const Shape& getShape() const;
        [[nodiscard]] const Strides& getStrides() const;
        [[nodiscard]] std::size_t getDim() const;
        [[nodiscard]] std::size_t numElements() const;
        [[nodiscard]] bool isContiguous() const;
        [[nodiscard]] T& at(const Indices& indices);
        [[nodiscard]] const T& at(const Indices& indices) const; // Allows for immutable tensors
        [[nodiscard]] TensorView<T> slice(const Slices& slices) const;
        [[nodiscard]] MutableTensorView<T> mutableSlice(const Slices& slices);

    private:
        TensorLayout layout_;
        std::shared_ptr<Storage<T>> storage_;
    };
}

#include "Tensor.tpp"