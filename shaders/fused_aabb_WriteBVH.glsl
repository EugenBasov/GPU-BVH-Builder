#extension GL_GOOGLE_include_directive : enable

#include "shared/types.glsl"
#include "shared/data_fused.h"
#include "shared/data_plocpp.h"
#include "shared/bvh_wide_quantized.glsl"

layout(local_size_x_id = 0) in;

layout(push_constant, scalar) uniform uPushConstant {
    PC_WriteBVH data;
} pc;

layout(buffer_reference, scalar) buffer RuntimeData {
    FUSEDData ploc;
};

void main() {
    u32 globalId = gl_GlobalInvocationID.x;
    RuntimeData data = RuntimeData(pc.data.runtimeDataAddress);
    if (globalId >= data.ploc.wideCounter) return;

    NodeBvhWideTraceQuantizedRef bvhWide = NodeBvhWideTraceQuantizedRef(pc.data.bvhWideAddress);

    u32 trianglesCount = data.ploc.triangleCount;
    u32 innerNodeCount = 0;
    u32 leafCount = 0;
    TOPOLOGY topoArr = TOPOLOGY(pc.data.wideTopologyAddress);
    NodeTopology topo = topoArr.node[globalId];
    CLUSTER clusterArr = CLUSTER(pc.data.clusterArrayAddress);
    NodeCluster cluster = clusterArr.node[topo.cl];
    INNERMAP innerMap = INNERMAP(pc.data.clusterMapAddress);
    BASEIDS baseIds = BASEIDS(pc.data.baseIdsAddress);
    NodeBaseIds bIdx = baseIds.node[globalId];
    NodeBvhWideTraceQuantizedRef nodeOut = bvhWide[innerMap.node[topo.cl].finalIdx];
    for (u32 i = 0; i < BVH_ARITY; ++i) {
        if (topo.c[i] == -1)
            break;
        if (topo.c[i] < trianglesCount) {
            leafCount++;
        }
        else {
            innerNodeCount++;
        }
    }

    u32 childCount = innerNodeCount + leafCount;
    nodeOut.val.bvOrigin = cluster.bv.min;
    uint32_t packedExponent = computeQuantizationExponentPacked(cluster.bv);

    nodeOut.val.packedBvExponentAndCount = packedExponent | packChildCount(childCount, leafCount);
    vec3 quantizationExponents = unpackQuantizationExponents(packedExponent);

    // triangle variables
    int32_t triPerLeafCount[BVH_ARITY];

    // write BVs and process leaves triangles
    uint32_t tmpPackedBvZ = 0;
    for (uint i = 0; i < childCount; i++) {
        uint cId = topo.c[i];
        AABB bv = clusterArr.node[cId].bv;

        uint32_t currentPackedBvZ;
        quantizeBv(cluster.bv.min, quantizationExponents, bv, nodeOut.val.packedBvXY[i], currentPackedBvZ);

        if (i % 2 == 1)
            nodeOut.val.packedBvZ[i / 2] = (currentPackedBvZ << 16) | tmpPackedBvZ;
        else
            tmpPackedBvZ = currentPackedBvZ;

        if (i < leafCount) {
            triPerLeafCount[i] = 1;
        }
        else
            triPerLeafCount[i] = 0;
    }

    // store the last half-full quantized Z axis
    if (childCount % 2 == 1)
        nodeOut.val.packedBvZ[childCount / 2] = tmpPackedBvZ;

    // write indices
    nodeOut.val.internalNodeIndex = bIdx.innerBaseIdx;
    nodeOut.val.triangleIndex = bIdx.leafBaseIdx;
    nodeOut.val.packedLeafSize = packLeafSize(triPerLeafCount, childCount);
}
