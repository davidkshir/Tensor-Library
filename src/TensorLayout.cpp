#include "TensorLayout.hpp"
#include "Types.hpp"
#include <utility>
#include <stdexcept>
#include <unordered_set>

namespace tensor {
    TensorLayout::TensorLayout(Shape shape) // Automatically creates strides based on shape and no offset
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
        for (const std::size_t dim : shape_) {
            size *= dim;
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
    TensorLayout TensorLayout::slice(const Slices& slices) const {

        Shape newShape = shape_;
        Strides newStrides = strides_;
        std::size_t newOffset = offset_;

        std::unordered_set<std::size_t> usedAxes;
        for (const Slice& slice : slices) {
            if (slice.axis >= shape_.size()) {
                throw std::out_of_range("Axis out of bounds.");
            }
            if (slice.start > shape_[slice.axis]) {
                throw std::out_of_range("Start cannot exceed length of axis.");
            }
            if (slice.end > shape_[slice.axis]) {
                throw std::out_of_range("End cannot exceed length of axis.");
            }
            if (slice.start > slice.end) {
                throw std::invalid_argument("End cannot be before start.");
            }
            if (slice.step == 0) {
                throw std::invalid_argument("Step cannot be 0.");
            }
            if (!usedAxes.insert(slice.axis).second) {
                throw std::invalid_argument("Axis cannot be sliced multiple times.");
            }
            newShape[slice.axis] = ((slice.end - slice.start) + slice.step - 1) / slice.step;
            newStrides[slice.axis] = strides_[slice.axis] * slice.step;
            newOffset += slice.start * strides_[slice.axis];
        }

        TensorLayout layout(newShape, newStrides, newOffset);
        return layout;
    }
}
