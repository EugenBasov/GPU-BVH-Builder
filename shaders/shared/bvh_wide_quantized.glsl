#ifndef BVH_WIDE_QUANTIZED_GLSL
#define BVH_WIDE_QUANTIZED_GLSL

#include "data_bvh_wide.h"

#define IEEE754_FRACTION_MASK 0x7FFFFF
#define IEEE754_EXPONENT_SHIFT 23

uint32_t ceilExponentToBits(float inputFloat) {
    uint32_t bits = floatBitsToUint(inputFloat);
    // sign bit is not cleared and it is expected that diff is positive value
    uint32_t exponent = bits >> IEEE754_EXPONENT_SHIFT;

    // manual exponent ceiling: if the mantissa is not empty, increment exponent to cover the entire range
    // exponent overflow is not checked, because it would mean that `diff == Inf`, which will be undefined behaviour in any case
    if ((bits & IEEE754_FRACTION_MASK) != 0)
        exponent++;

    return exponent;
}

uint32_t computeQuantizationExponentPacked(AABB aabb) {
    // division by 255.0 introduces floating point errors in the exponent quantization, which can lead to loss of precision (for example 1/255.0 -> exponent 0.0078125)
    // there is nothing I can do about it, but it is good to know, that this is happening and in some cases, the effective quantization of an AABB is only 7 bits

    // compute absolute value, just in case of some edge cases
    vec3 rawExponent = abs(aabb.max - aabb.min) / 255.0f;

    return (ceilExponentToBits(rawExponent.x)) |
        (ceilExponentToBits(rawExponent.y) << 8) |
        (ceilExponentToBits(rawExponent.z) << 16);
}

vec3 unpackQuantizationExponents(uint32_t packed) {
    return vec3(uintBitsToFloat((packed & 0xFF) << IEEE754_EXPONENT_SHIFT),
        uintBitsToFloat(((packed >> 8) & 0xFF) << IEEE754_EXPONENT_SHIFT),
        uintBitsToFloat(((packed >> 16) & 0xFF) << IEEE754_EXPONENT_SHIFT)
    );
}

uint32_t packChildCount(uint totalCount, uint leafCount) {
    return ((leafCount & 0xF) << 24) | ((totalCount & 0xF) << 28);
}

void unpackChildCount(uint32_t packedData, out uint totalCount, out uint leafCount) {
    totalCount = packedData >> 28;
    leafCount = (packedData >> 24) & 0xF;
}

void quantizeBv(vec3 origin, vec3 quantizationExponents, AABB bv, out uint32_t packedBvXY, out uint32_t packedBvZ) {
    // packUnorm4x8 does the rounding, but here it is necessary to use floor/ceiling to ensure that the BV only grows
    vec3 minQuantized = floor((bv.min - origin) / quantizationExponents);
    vec3 maxQuantized = ceil((bv.max - origin) / quantizationExponents);

    ivec4 xyInt = ivec4(minQuantized.xy, maxQuantized.xy) & 0xFF;
    ivec2 zInt = ivec2(minQuantized.z, maxQuantized.z) & 0xFF;

    packedBvXY = (xyInt.x) | (xyInt.y << 8) | (xyInt.z << 16) | (xyInt.w << 24);
    packedBvZ = (zInt.x) | (zInt.y << 8);
}

AABB unquantizeBv(vec3 bvOrigin, vec3 quantizationExponents, uint32_t packedBvXY, uint32_t packedBvZ) {
    // in some very specific cases, the unquantized BV may be slightly smaller (max a few ULP) due to the floating point errors in the division+multiplication
    vec3 quantizedMin = vec3(packedBvXY & 0xFF, (packedBvXY >> 8) & 0xFF, packedBvZ & 0xFF);
    vec3 quantizedMax = vec3((packedBvXY >> 16) & 0xFF, (packedBvXY >> 24) & 0xFF, (packedBvZ >> 8) & 0xFF);

    return AABB(
        bvOrigin + quantizationExponents * quantizedMin,
        bvOrigin + quantizationExponents * quantizedMax
    );
}

uint32_t packLeafSize(int32_t triangleCount[BVH_ARITY], uint childCount) {
    uint32_t packedLeafSize = 0;

    // encode from the last child to the first -> first child will be in the LSB
    for (int i = int(childCount - 1); i >= 0; i--) {
        packedLeafSize = (packedLeafSize << 4) | (triangleCount[i] - 1);
    }

    return packedLeafSize;
}

#endif
