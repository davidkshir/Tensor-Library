#pragma once

#include <memory>
#include "Storage.hpp"
#include "TensorLayout.hpp"

namespace tensor::detail {
    template<typename T>
    T& at(Storage<T>& storage, const TensorLayout& layout, const Indices& indices);

    template<typename T>
    const T& at(const Storage<T>& storage, const TensorLayout& layout, const Indices& indices);

    template<typename T>
    void validateView(const std::shared_ptr<Storage<T>>&  storage, const TensorLayout& layout);

}
#include "TensorHelpers.tpp"
