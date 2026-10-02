#pragma once
#include <utility>
#include "TensorLayout.hpp"

namespace tensor::detail {

    std::pair<TensorLayout, TensorLayout> broadcast(const TensorLayout& a, const TensorLayout& b);

}