#pragma once

#include "Storage.hpp"
#include "TensorLayout.hpp"

namespace tensor::detail {
    template<typename T>
    T& at(Storage<T>& storage, const TensorLayout& layout, const Indices& indices) {
        return storage[layout.getStorageIndex(indices)];
    }
    template<typename T>
    const T& at(const Storage<T>& storage, const TensorLayout& layout, const Indices& indices){
        return storage[layout.getStorageIndex(indices)];
    }

    template<typename T>
    void validateView(const std::shared_ptr<Storage<T>>&  storage, const TensorLayout& layout) {
        if (storage == nullptr) {
            throw std::invalid_argument("Storage pointer cannot be null.");
        }
        if (layout.numElements() == 0) {
            if (layout.getOffset() > storage->size()) {
                throw std::out_of_range("Offset is out of storage range.");
            }
        }
        else {
            Indices maxIndex;

            for (std::size_t axis : layout.getShape()) {
                maxIndex.push_back(axis - 1);
            }
            if (layout.getStorageIndex(maxIndex) >= storage->size()) {
                throw std::invalid_argument("Tensor view exceeds max storage.");
            }
        }
    }

    template<typename T>
    std::vector<T> cloneData(const Storage<T>& source, const TensorLayout& layout) {
        const std::size_t size = layout.numElements();

        std::vector<T> clonedData(size);

        if (size == 0) {
            return clonedData;
        }

        Indices indices(layout.getShape().size(), 0); // Both logical & physical traversal expect indices have 0s for all dimensions

        std::size_t sourceIndex = layout.getOffset();
        std::size_t destinationIndex = 0;

        while (true) {
            clonedData[destinationIndex] = source[sourceIndex];

            const std::size_t advancedAxis = advanceIndices(indices, layout.getShape());

            if (advancedAxis == layout.getShape().size()) {
                break; // Traversal over
            }

            layout.advanceStorage(sourceIndex, advancedAxis);
            ++destinationIndex;
        }
        return clonedData;
    }

}