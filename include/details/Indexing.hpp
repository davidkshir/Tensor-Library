#pragma once
#include "Storage.hpp"
#include "TensorLayout.hpp"

namespace tensor::detail {
    template<typename T>
    T& at(Storage<T>& storage, const TensorLayout& layout, const Indices& indices);

    template<typename T>
    const T& at(const Storage<T>& storage, const TensorLayout& layout, const Indices& indices);

}
#include "Indexing.tpp"