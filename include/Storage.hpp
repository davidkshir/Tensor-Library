#pragma once

#include <vector>
#include <cstddef>
#include <utility>

namespace tensor {

    template<typename T>
    class Storage {
    public:
        explicit Storage(std::size_t size); // Allocation constructor
        explicit Storage(const std::vector<T>& values); // Constructor to copy data
        explicit Storage(std::vector<T>&& values); // Constructor to take ownership of data

        T& operator[](std::size_t index);

        const T& operator[](std::size_t index) const;

        [[nodiscard]] std::size_t size() const;


    private:
        std::vector<T> storage;
    };
}
#include "Storage.tpp"