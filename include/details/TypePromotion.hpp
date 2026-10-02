#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace tensor::details {

    // Checks to see if container is big enough to store a value
    template<typename Container, typename Value>
    struct CanFit {
        static constexpr bool value =
            !(!std::is_signed_v<Container> && std::is_signed_v<Value>) &&
            (std::numeric_limits<Container>::digits >= std::numeric_limits<Value>::digits);
    };

    template<typename... Types>
    struct TypeList {};

    using SignedTypes = TypeList<
        std::int8_t,
        std::int16_t,
        std::int32_t,
        std::int64_t
    >;

    using UnsignedTypes = TypeList<
        std::uint8_t,
        std::uint16_t,
        std::uint32_t,
        std::uint64_t
    >;

    template<typename List, typename A, typename B>
    struct FindFirstFit;

    template<bool Fits, typename Candidate, typename RestList, typename A, typename B>
    struct FitSelector;

    template<typename Candidate, typename RestList, typename A, typename B>
    struct FitSelector<true, Candidate, RestList, A, B> {
        using type = Candidate;
    };

    template<typename Candidate, typename RestList, typename A, typename B>
    struct FitSelector<false, Candidate, RestList, A, B> {
        using type = typename FindFirstFit<RestList, A, B>::type;
    };

    // Determines integer container size
    template<typename Candidate, typename... Rest, typename A, typename B>
    struct FindFirstFit<TypeList<Candidate, Rest...>, A, B> {

        using type = typename FitSelector<
            CanFit<Candidate, A>::value && CanFit<Candidate, B>::value,
            Candidate,
            TypeList<Rest...>,A,B>::type;
    };

    template<typename A, typename B>
    struct FindFirstFit<TypeList<>, A, B> {
        static_assert(
            sizeof(A) == 0,
            "TensorLib: no type can represent both input types."
        );
    };

    template<typename A, typename B>
    struct FloatingPromotion;

    // Determines floating-point container size
    template<typename A, typename B>
    struct FloatingPromotion {
        using type = std::conditional_t<
            std::same_as<A, long double> || std::same_as<B, long double>, long double,
            std::conditional_t<std::same_as<A, double> || std::same_as<B, double>, double,
                float>>;
    };

    template<bool HasFloatingPoint, typename A, typename B>
    struct PromotionSelector;

    // Floating-point case
    template<typename A, typename B>
    struct PromotionSelector<true, A, B> {
        using type = typename FloatingPromotion<A, B>::type;
    };

    // Integer case
    template<typename A, typename B>
    struct PromotionSelector<false, A, B> {
        using Candidates = std::conditional_t<
            !std::is_signed_v<A> && !std::is_signed_v<B>,
            UnsignedTypes,
            SignedTypes
        >;

        using type = typename FindFirstFit<Candidates, A, B>::type;
    };

    template<typename A, typename B>
    struct Promote {
        using type = typename PromotionSelector<std::is_floating_point_v<A> || std::is_floating_point_v<B>, A, B>::type;
    };

    template<typename A, typename B>
    using PromoteType = typename Promote<A, B>::type;

}