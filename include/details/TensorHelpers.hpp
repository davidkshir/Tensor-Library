#pragma once

#include <memory>
#include "Storage.hpp"
#include "TensorLayout.hpp"

namespace tensor::detail {

    std::size_t advanceIndices(Indices& indices, const Shape& shape);
    
    template<typename T>
    T& at(Storage<T>& storage, const TensorLayout& layout, const Indices& indices);

    template<typename T>
    const T& at(const Storage<T>& storage, const TensorLayout& layout, const Indices& indices);

    template<typename T>
    void validateView(const std::shared_ptr<Storage<T>>&  storage, const TensorLayout& layout);

    template<typename T>
    std::vector<T> cloneData(const Storage<T>& source, const TensorLayout& layout);
}
#include "TensorHelpers.tpp"
