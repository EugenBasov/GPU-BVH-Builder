// All bit encodings in this file are written from the LSB to the MSB

#ifndef BVH_WIDE_DATA_H
#define BVH_WIDE_DATA_H

#include "types.glsl"
#include "bv_aabb.glsl"

#ifndef BVH_ARITY
#    define BVH_ARITY 4
#endif

struct NodeBvhWideTraceQuantized {
    // origin (and the min point) of the quantized world space
    vec3 bvOrigin;
    // packed BV exponent for each axis (mantissa part of float) and child count (uint):
    // | exponent X : 8 | exponent Y : 8 | exponent Z : 8 | count of leaves - 1 : 4 | total count of children - 1 : 4 |
    uint32_t packedBvExponentAndCount;

    // index of the first node child, node children are stored in the continuous memory
    uint32_t internalNodeIndex;
    // index of the first triangle, triangles for node leaves are stored in the continuous memory
    uint32_t triangleIndex;
    // packed 8x4 bits (uint) of (leaf size - 1), from the least significant nibble
    uint32_t packedLeafSize;

    // children arrangement in the node: starting with the leaves, thereafter internal nodes and unallocated space in the end

    // packed 4x8 bits (uint): | min X : 8 | min Y : 8 | max X : 8 | max Y : 8 |
    uint32_t packedBvXY[BVH_ARITY];
    // packed 4x8 bits (uint), shared between two children: | min Z1 : 8 | max Z1 : 8 | min Z2 : 8 | max Z2 : 8 |
    uint32_t packedBvZ[BVH_ARITY / 2];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer NodeBvhWideTraceQuantizedRef {
    NodeBvhWideTraceQuantized val;
};

#endif // BVH_WIDE_DATA_H
