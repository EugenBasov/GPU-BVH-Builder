#ifndef DATA_FUSED_H
#define DATA_FUSED_H

#include "types.glsl"
#include "bv_aabb.glsl"

#define PR_QUEUE_SIZE (1024 * 1)
#define IA_ARRAY_OFFSET (PR_QUEUE_SIZE * 2)

#ifndef BVH_ARITY
#    define BVH_ARITY 4
#endif

struct NodeCluster {
    AABB bv;
    i32 c[BVH_ARITY / 2];
};

struct NodeTopology {
    i32 cl;
    i32 c[BVH_ARITY];
};

struct NodeInnerMap {
    u32 finalIdx;
};
struct NodeLeafMap {
    u32 finalIdx;
};
struct NodeBaseIds {
    u32 innerBaseIdx;
    u32 leafBaseIdx;
};

layout(buffer_reference, scalar) buffer CLUSTER {
    NodeCluster node[];
};
layout(buffer_reference, scalar) buffer TOPOLOGY {
    NodeTopology node[];
};
layout(buffer_reference, scalar) buffer INNERMAP {
    NodeInnerMap node[];
};
layout(buffer_reference, scalar) buffer LEAFMAP {
    NodeLeafMap node[];
};
layout(buffer_reference, scalar) buffer BASEIDS {
    NodeBaseIds node[];
};

#endif
