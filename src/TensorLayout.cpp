#include "TensorLayout.hpp"
#include <utility>
#include <stdexcept>
#include <string>

namespace tensor {
    TensorLayout::TensorLayout(Shape shape)
        :shape_(std::move(shape)),
        strides_(shape_.size()),
        offset_(0)
    {
        std::size_t stride = 1;
        for (std::size_t i = shape_.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    TensorLayout::TensorLayout(Shape shape, Strides strides, const std::size_t offset)
        :shape_(std::move(shape)),
        strides_(std::move(strides)),
        offset_(offset)
    {
        if (shape_.size() != strides_.size()) {
            throw std::invalid_argument("Shape and strides must have the same dimensions.");
        }
    }

    std::size_t TensorLayout::numElements() const{
        std::size_t size = 1;
        for (const std::size_t elem : shape_) {
            size *= elem;
        }
        return size;
    }

    const Shape& TensorLayout::getShape() const {
        return shape_;
    }

    const Strides& TensorLayout::getStrides() const {
        return strides_;
    }

    std::size_t TensorLayout::getOffset() const {
        return offset_;
    }

    std::size_t TensorLayout::getDim() const {
        return shape_.size();
    }

    std::size_t TensorLayout::getStorageIndex(const Indices& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Invalid index dimensions.");
        }

        for (std::size_t i = 0; i < indices.size(); i++) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("One or more index out of bounds.");
            }
        }

        std::size_t storageIndex = offset_;
        for (std::size_t i = 0; i < indices.size(); i ++) {
            storageIndex += indices[i] * strides_[i];
        }

        return storageIndex;
    }

    bool TensorLayout::isContiguous() const {
        std::size_t expectedStride = 1;

        for (std::size_t i = shape_.size(); i-- > 0;) {
            if (strides_[i] != expectedStride) {
                return false;
            }

            expectedStride *= shape_[i];
        }

        return true;
    }
}
