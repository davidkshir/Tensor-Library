#pragma once

#include "Storage.hpp"
#include "TensorLayout.hpp"
#include <memory>

namespace tensor {

    template<typename T>
    class TensorView {
    public:
        TensorView(std::shared_ptr<Storage<T>> storage, Shape shape, Strides strides, std::size_t offset);
        [[nodiscard]] const Shape& getShape() const;
        [[nodiscard]] const Strides& getStrides() const;
        [[nodiscard]] std::size_t getDim() const;
        [[nodiscard]] std::size_t numElements() const;
        [[nodiscard]] bool isContiguous() const;
        [[nodiscard]] const T& at(const Indices& indices) const;
    private:
        TensorLayout layout_;
        std::shared_ptr<Storage<T>> storage_;
    };
}
#include "TensorView.tpp"