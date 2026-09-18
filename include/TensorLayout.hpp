#pragma once

#include "Types.hpp"
#include <cstddef>

namespace tensor {

    class TensorLayout {
    public:
        explicit TensorLayout(Shape shape);
        TensorLayout(Shape shape, Strides strides, std::size_t offset);
        [[nodiscard]] std::size_t numElements() const;
        [[nodiscard]] const Shape& getShape() const;
        [[nodiscard]] const Strides& getStrides() const;
        [[nodiscard]] std::size_t getOffset() const;
        [[nodiscard]] std::size_t getDim() const;
        [[nodiscard]] std::size_t getStorageIndex(const Indices& indices) const;
        [[nodiscard]] bool isContiguous() const;
        [[nodiscard]] TensorLayout slice(const Slices& slices) const;

    private:
        Shape shape_;
        Strides strides_;
        std::size_t offset_;

    };
}