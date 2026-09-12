#pragma once

#include <cstddef>
#include <vector>

namespace tensor {
    using Strides = std::vector<std::size_t>;
    using Shape = std::vector<std::size_t>;
    using Indices = std::vector<std::size_t>;
}
