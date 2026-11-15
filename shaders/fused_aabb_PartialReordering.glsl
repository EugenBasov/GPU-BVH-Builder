#extension GL_GOOGLE_include_directive : enable

#include "shared/types.glsl"
#include "shared/data_fused.h"
#include "shared/data_plocpp.h"
#include "shared/bvh_wide_quantized.glsl"

layout(local_size_x_id = 0) in;

layout(push_constant, scalar) uniform uPushConstant {
    PC_PartialReordering data;
} pc;

layout(buffer_reference, scalar) buffer RuntimeData {
    FUSEDData ploc;
};

shared ivec2 queue[PR_QUEUE_SIZE];
shared u32 queueSize;
shared u32 nodeCounter;

void main() {
    u32 globalId = gl_GlobalInvocationID.x;
    RuntimeData data = RuntimeData(pc.data.runtimeDataAddress);
    if (globalId >= gl_WorkGroupSize.x) return;
    if (globalId == 0) {
        queue[0] = ivec2(data.ploc.wideCounter - 1 + IA_ARRAY_OFFSET, 0);
        queueSize = 1;
        nodeCounter = 1;
    }
    const u32 indexPairArrSize = (PR_QUEUE_SIZE + gl_WorkGroupSize.x - 1) / gl_WorkGroupSize.x;
    ivec2 indexPairArr[indexPairArrSize];
    u32 indexPairCount = 0;
    NodeBvhWideTraceQuantizedRef wideNodes = NodeBvhWideTraceQuantizedRef(pc.data.bvhWideAddress);
    NodeBvhWideTraceQuantizedRef node[indexPairArrSize];
    u32 outBaseIndex[indexPairArrSize];
    barrier();
    while (queueSize != 0 && queueSize <= PR_QUEUE_SIZE) {
        indexPairCount = 0;
        for (u32 i = globalId; i < queueSize; i += gl_WorkGroupSize.x) {
            indexPairArr[indexPairCount] = queue[i];
            indexPairCount++;
        }
        barrier();
        if (gl_LocalInvocationID.x == 0) {
            queueSize = 0;
        }
        barrier();
        for (u32 i = 0; i < indexPairCount; ++i) {
            node[i] = wideNodes[indexPairArr[i].x];
            u32 childCount;
            u32 leafCount;
            unpackChildCount(node[i].val.packedBvExponentAndCount, childCount, leafCount);
            u32 innerNodes = childCount - leafCount;
            outBaseIndex[i] = atomicAdd(nodeCounter, innerNodes);
            u32 queueIdx = atomicAdd(queueSize, innerNodes);
            if (queueIdx + innerNodes < PR_QUEUE_SIZE) {
                for (u32 j = 0; j < innerNodes; ++j) {
                    queue[queueIdx + j] = ivec2(node[i].val.internalNodeIndex + j, outBaseIndex[i] + j);
                }
            }
        }
        barrier();
        for (u32 i = 0; i < indexPairCount; ++i) {
            ivec2 indexPair = indexPairArr[i];
            if (queueSize <= PR_QUEUE_SIZE) {
                node[i].val.internalNodeIndex = outBaseIndex[i];
            }
            wideNodes[indexPair.y].val.bvOrigin = node[i].val.bvOrigin;
            wideNodes[indexPair.y].val.packedBvExponentAndCount = node[i].val.packedBvExponentAndCount;
            wideNodes[indexPair.y].val.internalNodeIndex = node[i].val.internalNodeIndex;
            wideNodes[indexPair.y].val.triangleIndex = node[i].val.triangleIndex;
            wideNodes[indexPair.y].val.packedLeafSize = node[i].val.packedLeafSize;
            for (u32 j = 0; j < BVH_ARITY; ++j) {
                wideNodes[indexPair.y].val.packedBvXY[j] = node[i].val.packedBvXY[j];
            }
            for (u32 j = 0; j < (BVH_ARITY / 2); ++j) {
                wideNodes[indexPair.y].val.packedBvZ[j] = node[i].val.packedBvZ[j];
            }
            node[i].val.packedBvExponentAndCount = 0; // mark as empty
        }
    }
}