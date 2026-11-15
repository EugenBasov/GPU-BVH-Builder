#ifndef DATA_BVH_H
#define DATA_BVH_H

#include "types.glsl"
#include "bv_aabb.glsl"

struct NodeBVH2_AABB {
    AABB bv;
    i32 size;
    i32 parent;
    i32 c0;
    i32 c1;
};

struct BvhTriangleIndex {
    u32 nodeId;
    u32 triangleId;
};

struct BvhTriangle {
    vec4 v0;
    vec4 v1;
    vec4 v2;
};

layout(buffer_reference, scalar) buffer BVH2_AABB {
    NodeBVH2_AABB node[];
};
layout(buffer_reference, scalar) buffer BvhTriangles {
    BvhTriangle t[];
};
layout(buffer_reference, scalar) buffer BvhTriangleIndices {
    BvhTriangleIndex val[];
};

#endif
