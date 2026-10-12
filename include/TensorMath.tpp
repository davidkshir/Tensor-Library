#pragma once

#include "Tensor.hpp"
#include "TensorMath.hpp"
#include <stdexcept>
#include <type_traits>
#include <limits>

#include "details/MathHelpers.hpp"

namespace tensor {
    template<ReadableTensor T, NumericScalar Scalar>
    Tensor<detail::PromoteType<typename T::valueType, Scalar>> add(const T& tensor, Scalar scalar) {
        using ResultType = detail::PromoteType<typename T::valueType, Scalar>;

        Tensor<ResultType> result(tensor.getShape());

        Indices indices(tensor.getDim(), 0);

        for (std::size_t i = 0; i < tensor.numElements(); ++i){
            result.at(indices) = (static_cast<ResultType>(tensor.at(indices)) + static_cast<ResultType>(scalar));
            detail::advanceIndices(indices, tensor.getShape());
        }
        return result;
    }

    template<ReadableTensor T, ReadableTensor U>
    Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> add(const T& tensor1, const U&  tensor2) {
        using ResultType = detail::PromoteType<typename T::valueType, typename U::valueType>;
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


    template<ReadableTensor T, NumericScalar Scalar>
    Tensor<detail::PromoteType<typename T::valueType, Scalar>> subtract(const T& tensor, Scalar scalar) {
        using ResultType = detail::PromoteType<typename T::valueType, Scalar>;

        Tensor<ResultType> result(tensor.getShape());

        Indices indices(tensor.getDim(), 0);

        for (std::size_t i = 0; i < tensor.numElements(); ++i){
            result.at(indices) = (static_cast<ResultType>(tensor.at(indices)) - static_cast<ResultType>(scalar));
            detail::advanceIndices(indices, tensor.getShape());
        }
        return result;
    }

    template<ReadableTensor T, ReadableTensor U>
    Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> subtract(const T& tensor1, const U&  tensor2) {
        using ResultType = detail::PromoteType<typename T::valueType, typename U::valueType>;
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

    template<ReadableTensor T, NumericScalar Scalar>
    Tensor<detail::PromoteType<typename T::valueType, Scalar>> multiply(const T& tensor, Scalar scalar) {
        using ResultType = detail::PromoteType<typename T::valueType, Scalar>;

        Tensor<ResultType> result(tensor.getShape());

        Indices indices(tensor.getDim(), 0);

        for (std::size_t i = 0; i < tensor.numElements(); ++i){
            result.at(indices) = (static_cast<ResultType>(tensor.at(indices)) * static_cast<ResultType>(scalar));
            detail::advanceIndices(indices, tensor.getShape());
        }
        return result;
    }

    template<ReadableTensor T, ReadableTensor U>
    Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> multiply(const T& tensor1, const U&  tensor2) {
        using ResultType = detail::PromoteType<typename T::valueType, typename U::valueType>;
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

    template<ReadableTensor T, NumericScalar Scalar>
    Tensor<detail::PromoteType<typename T::valueType, Scalar>> divide(const T& tensor, Scalar scalar) {
        if (scalar == 0) {
            throw std::domain_error("Division by zero");
        }

        using ResultType = detail::PromoteType<typename T::valueType, Scalar>;

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

    template<ReadableTensor T, ReadableTensor U>
    Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> divide(const T& tensor1, const U&  tensor2) {
        using ResultType = detail::PromoteType<typename T::valueType, typename U::valueType>;
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

    template<ReadableTensor T, ReadableTensor U>
    Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> matMul(const T& tensor1, const U&  tensor2) {

        if (tensor1.getShape().size() < 2 || tensor2.getShape().size() < 2) {
            throw std::invalid_argument("Both tensors must have 2 or more dimensions.");
        }

        if (tensor1.getShape().back() != tensor2.getShape()[tensor2.getShape().size() - 2]) {
            throw std::invalid_argument("Incompatible matrix dimensions.");
        }

        using ResultType = detail::PromoteType<typename T::valueType, typename U::valueType>;

        // Gets the batch dimensions by excluding the last 2 dimensions
        const Shape batchShape1(tensor1.getShape().begin(), tensor1.getShape().end() - 2);
        const Shape batchShape2(tensor2.getShape().begin(), tensor2.getShape().end() - 2);

        // Broadcasts batch dimensions
        auto [batchLayout1, batchLayout2] = detail::broadcast(TensorLayout(batchShape1), TensorLayout(batchShape2));

        Shape batchShape = batchLayout1.getShape();

        // Create resulting tensor
        Shape resultShape = batchLayout1.getShape();
        resultShape.push_back(tensor1.getShape()[tensor1.getDim() - 2]); // M
        resultShape.push_back(tensor2.getShape().back()); // N
        Tensor<ResultType> result(resultShape);

        const std::size_t innerDim = tensor1.getShape().back();

        Indices indices(result.getDim(), 0);

        // Creates new layouts for both tensors that have proper batch dimensions, but keep their own final 2 dimensions
        Shape targetShape1 = resultShape;
        Shape targetShape2 = resultShape;
        targetShape1.back() = tensor1.getShape().back();
        targetShape2[targetShape2.size() - 2] = tensor2.getShape()[tensor2.getDim() - 2];

        // Use these logical coordinates to calculate physical locations in original tensors
        auto layout1 = tensor1.getLayout().broadcastTo(targetShape1);
        auto layout2 = tensor2.getLayout().broadcastTo(targetShape2);

        for (std::size_t i = 0; i < result.numElements(); ++i) { // Calculates values for resulting tensor
            ResultType sum = 0;

            Indices indices1 = indices;
            Indices indices2 = indices;

            for (std::size_t k = 0; k < innerDim; ++k) { // Accumulates dot product
                // Gets logical coordinates
                indices1.back() = k;
                indices2[indices2.size() - 2] = k;

                // Gets physical location
                const std::size_t storageIndex1 = layout1.getStorageIndex(indices1);
                const std::size_t storageIndex2 = layout2.getStorageIndex(indices2);

                sum += static_cast<ResultType>(tensor1.getStorage()[storageIndex1]) * static_cast<ResultType>(tensor2.getStorage()[storageIndex2]);
            }

            result.at(indices) = sum;
            detail::advanceIndices(indices, result.getShape());
        }
        return result;
    }

}




