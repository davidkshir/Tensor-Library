#pragma once

#include "Storage.hpp"
#include "Types.hpp"
#include "TensorLayout.hpp"
#include <memory>

namespace tensor {

    template<typename T>
    class TensorView {
    public:
        TensorView(std::shared_ptr<Storage<T>> storage, Shape shape, Strides strides, std::size_t offset);
        TensorView(std::shared_ptr<Storage<T>> storage, TensorLayout layout);
        [[nodiscard]] const Shape& getShape() const;
        [[nodiscard]] const Strides& getStrides() const;
        [[nodiscard]] std::size_t getDim() const;
        [[nodiscard]] std::size_t numElements() const;
        [[nodiscard]] bool isContiguous() const;
        [[nodiscard]] const T& at(const Indices& indices) const;
        [[nodiscard]] TensorView<T> slice(const Slices& slices) const;
    private:
        TensorLayout layout_;
        std::shared_ptr<Storage<T>> storage_;
    };
}
#include "TensorView.tpp"