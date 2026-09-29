#include "TensorLayout.hpp"

#include <numeric>
#include "Types.hpp"
#include "details/TensorHelpers.hpp"
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

    void TensorLayout::advanceStorage(std::size_t& physicalIndex, std::size_t const advancedAxis) const {
        if (advancedAxis >= shape_.size()) {
            throw std::out_of_range("Advanced axis is out of range.");
        }

        physicalIndex += strides_[advancedAxis];

        for (std::size_t axis = advancedAxis + 1; axis < shape_.size(); ++axis) {

            physicalIndex -= (shape_[axis] - 1) * strides_[axis];
        }
    }

    TensorLayout TensorLayout::reshape(const Shape& newShape) const {
        if (!isContiguous()) {
            throw std::invalid_argument("Original layout must be contiguous.");
        }

        std::size_t newShapeSize = 1;
        for (const std::size_t dim : newShape) {
            newShapeSize *= dim;
        }

        if (newShapeSize != numElements()) {
            throw std::invalid_argument("New shape must have the same amount of elements as original layout.");
        }

        Strides newStrides(newShape.size());

        std::size_t stride = 1;

        for (std::size_t i = newShape.size(); i-- > 0;) {
            newStrides[i] = stride;
            stride *= newShape[i];
        }

        return TensorLayout(newShape, std::move(newStrides), offset_);
    }

    TensorLayout TensorLayout::permute(const Axes &newAxes) const {
        if (newAxes.size() != shape_.size()) {
            throw std::invalid_argument("Must have same number of Axes as original.");
        }

        if (const std::unordered_set<std::size_t> uniqueAxes(newAxes.begin(), newAxes.end());
            uniqueAxes.size() != newAxes.size()) {

            throw std::invalid_argument("Cannot have duplicate axes");
        }

        for (std::size_t const newAxis : newAxes) {
            if (newAxis >= shape_.size()) {
                throw std::out_of_range("Axis out of range.");
            }
        }

        Shape newShape(shape_.size());
        for (std::size_t i = 0; i < shape_.size(); i++) {
            newShape[i] = shape_[newAxes[i]];
        }

        Strides newStrides(shape_.size());
        for (std::size_t i = 0; i < strides_.size(); i++) {
            newStrides[i] = strides_[newAxes[i]];
        }

        return TensorLayout(std::move(newShape), std::move(newStrides), offset_);
    }

    TensorLayout TensorLayout::transpose(const std::size_t axis1, const std::size_t axis2) const {
        if (axis1 >= shape_.size() || axis2 >= shape_.size()) {
            throw std::out_of_range("Axis out of range.");
        }

        Axes newAxes(shape_.size());
        std::iota(newAxes.begin(), newAxes.end(), 0);
        std::swap(newAxes[axis1], newAxes[axis2]);

        return permute(newAxes);
    }
}
