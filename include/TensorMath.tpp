#pragma once

#include "Tensor.hpp"
#include "TensorMath.hpp"
#include <stdexcept>
#include <type_traits>
#include <limits>

#include "details/MathHelpers.hpp"

namespace tensor {
    template<typename T, NumericScalar Scalar>
    Tensor<detail::PromoteType<T, Scalar>> add(const Tensor<T>& tensor, Scalar scalar) {
        using ResultType = detail::PromoteType<T, Scalar>;

        Tensor<ResultType> result(tensor.getShape());

        Indices indices(tensor.getDim(), 0);

        for (std::size_t i = 0; i < tensor.numElements(); ++i){
            result.at(indices) = (static_cast<ResultType>(tensor.at(indices)) + static_cast<ResultType>(scalar));
            detail::advanceIndices(indices, tensor.getShape());
        }
        return result;
    }

    template<typename T, typename U>
    Tensor<detail::PromoteType<T, U>> add(const Tensor<T>& tensor1, const Tensor<U>&  tensor2) {
        using ResultType = detail::PromoteType<T, U>;
        auto [layout1, layout2] = detail::broadcast(tensor1.getLayout(), tensor2.getLayout()); // Unpacks pair

        Tensor<ResultType> result(layout1.getShape());

        Indices indices(layout1.getDim(), 0);

        for (std::size_t i = 0; i < layout1.numElements(); ++i){

            result.at(indices) = static_cast<ResultType>(tensor1.getStorage()[layout1.getStorageIndex(indices)]) +
                    static_cast<ResultType>(tensor2.getStorage()[layout2.getStorageIndex(indices)]);

            detail::advanceIndices(indices, layout1.getShape());
        }
        return result;
    }


    template<typename T, NumericScalar Scalar>
    Tensor<detail::PromoteType<T, Scalar>> subtract(const Tensor<T>& tensor, Scalar scalar) {
        using ResultType = detail::PromoteType<T, Scalar>;

        Tensor<ResultType> result(tensor.getShape());

        Indices indices(tensor.getDim(), 0);

        for (std::size_t i = 0; i < tensor.numElements(); ++i){
            result.at(indices) = (static_cast<ResultType>(tensor.at(indices)) - static_cast<ResultType>(scalar));
            detail::advanceIndices(indices, tensor.getShape());
        }
        return result;
    }

    template<typename T, typename U>
    Tensor<detail::PromoteType<T, U>> subtract(const Tensor<T>& tensor1, const Tensor<U>&  tensor2) {
        using ResultType = detail::PromoteType<T, U>;
        auto [layout1, layout2] = detail::broadcast(tensor1.getLayout(), tensor2.getLayout()); // Unpacks pair

        Tensor<ResultType> result(layout1.getShape());

        Indices indices(layout1.getDim(), 0);

        for (std::size_t i = 0; i < layout1.numElements(); ++i){

            result.at(indices) = static_cast<ResultType>(tensor1.getStorage()[layout1.getStorageIndex(indices)]) -
                    static_cast<ResultType>(tensor2.getStorage()[layout2.getStorageIndex(indices)]);

            detail::advanceIndices(indices, layout1.getShape());
        }
        return result;
    }

    template<typename T, NumericScalar Scalar>
    Tensor<detail::PromoteType<T, Scalar>> multiply(const Tensor<T>& tensor, Scalar scalar) {
        using ResultType = detail::PromoteType<T, Scalar>;

        Tensor<ResultType> result(tensor.getShape());

        Indices indices(tensor.getDim(), 0);

        for (std::size_t i = 0; i < tensor.numElements(); ++i){
            result.at(indices) = (static_cast<ResultType>(tensor.at(indices)) * static_cast<ResultType>(scalar));
            detail::advanceIndices(indices, tensor.getShape());
        }
        return result;
    }

    template<typename T, typename U>
    Tensor<detail::PromoteType<T, U>> multiply(const Tensor<T>& tensor1, const Tensor<U>&  tensor2) {
        using ResultType = detail::PromoteType<T, U>;
        auto [layout1, layout2] = detail::broadcast(tensor1.getLayout(), tensor2.getLayout()); // Unpacks pair

        Tensor<ResultType> result(layout1.getShape());

        Indices indices(layout1.getDim(), 0);

        for (std::size_t i = 0; i < layout1.numElements(); ++i){

            result.at(indices) = static_cast<ResultType>(tensor1.getStorage()[layout1.getStorageIndex(indices)]) *
                    static_cast<ResultType>(tensor2.getStorage()[layout2.getStorageIndex(indices)]);

            detail::advanceIndices(indices, layout1.getShape());
        }
        return result;
    }

    template<typename T, NumericScalar Scalar>
    Tensor<detail::PromoteType<T, Scalar>> divide(const Tensor<T>& tensor, Scalar scalar) {
        if (scalar == 0) {
            throw std::domain_error("Division by zero");
        }

        using ResultType = detail::PromoteType<T, Scalar>;

        Tensor<ResultType> result(tensor.getShape());

        Indices indices(tensor.getDim(), 0);

        for (std::size_t i = 0; i < tensor.numElements(); ++i){
            auto tensorValue = static_cast<ResultType>(tensor.at(indices));
            auto scalarValue = static_cast<ResultType>(scalar);

            if constexpr (std::is_integral_v<ResultType> && std::is_signed_v<ResultType>) {  // Checks if integer is even possible before next check

                if (tensorValue == std::numeric_limits<ResultType>::min() && scalarValue == -1) { // Checks for integer division
                    throw std::overflow_error("Integer division overflow");
                }
            }

            result.at(indices) = (tensorValue / scalarValue);
            detail::advanceIndices(indices, tensor.getShape());
        }
        return result;
    }

    template<typename T, typename U>
    Tensor<detail::PromoteType<T, U>> divide(const Tensor<T>& tensor1, const Tensor<U>&  tensor2) {
        using ResultType = detail::PromoteType<T, U>;
        auto [layout1, layout2] = detail::broadcast(tensor1.getLayout(), tensor2.getLayout()); // Unpacks pair

        Tensor<ResultType> result(layout1.getShape());

        Indices indices(layout1.getDim(), 0);

        for (std::size_t i = 0; i < layout1.numElements(); ++i){
            // Gets both values from tensors
            const auto dividend = static_cast<ResultType>(tensor1.getStorage()[layout1.getStorageIndex(indices)]);
            const auto divisor = static_cast<ResultType>(tensor2.getStorage()[layout2.getStorageIndex(indices)]);

            if (divisor == 0) {
                throw std::domain_error("Division by zero");
            }

            if constexpr (std::is_integral_v<ResultType> && std::is_signed_v<ResultType>) { // Checks if integer is even possible before next check
                if (dividend == std::numeric_limits<ResultType>::min() && divisor == -1) {
                    throw std::overflow_error("Integer division overflow");
                    }
              }

            result.at(indices) =  dividend / divisor;

            detail::advanceIndices(indices, layout1.getShape());
        }
        return result;
    }


}




