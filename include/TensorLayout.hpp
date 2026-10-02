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
        void advanceStorage(std::size_t& physicalIndex, std::size_t advancedAxis) const;
        [[nodiscard]] TensorLayout reshape(const Shape& newShape) const;
        [[nodiscard]] TensorLayout permute(const Axes& newAxes) const;
        [[nodiscard]] TensorLayout transpose(std::size_t axis1, std::size_t axis2) const;
        [[nodiscard]] TensorLayout broadcastTo(const Shape& targetShape) const;

    private:
        Shape shape_;
        Strides strides_;
        std::size_t offset_;

    };
}
