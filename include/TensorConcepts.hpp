#pragma once

#include <concepts>

namespace tensor {
    class TensorLayout;

    template<typename T>
    concept ReadableTensor = requires(const T& tensor){
        typename T::valueType;
        {tensor.getLayout()} -> std::same_as<const TensorLayout&>;
        {tensor.getStorage()} -> std::same_as<const Storage<typename T::valueType>&>;
    };

    template<typename T>
    concept NumericTensor = ReadableTensor<T> && std::is_arithmetic_v<typename T::valueType> && !std::same_as<typename T::valueType, bool>;
}
