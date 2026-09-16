#pragma once

#include <vector>
#include <cstddef>
#include <utility>

namespace tensor {

    template<typename T>
    class Storage {
    public:
        explicit Storage(std::size_t size);
        explicit Storage(const std::vector<T>& values); // copies data
        explicit Storage(std::vector<T>&& values); // moves data (takes ownership)

        T& operator[](std::size_t index);

        const T& operator[](std::size_t index) const;

        [[nodiscard]] std::size_t size() const;


    private:
        std::vector<T> storage_;
    };
    
}
#include "Storage.tpp"