#extension GL_GOOGLE_include_directive : enable

#include "shared/types.glsl"
#include "shared/compute.glsl"
#include "shared/data_fused.h"
#include "shared/data_plocpp.h"

layout(local_size_x_id = 0) in;

layout(push_constant, scalar) uniform uPushConstant {
    PC_IndexAssignment data;
} pc;

layout(buffer_reference, scalar) buffer RuntimeData {
    FUSEDData ploc;
};

void main() {
    u32 globalId = gl_GlobalInvocationID.x;
    RuntimeData data = RuntimeData(pc.data.runtimeDataAddress);
    if (globalId >= data.ploc.wideCounter) return;

    u32 trianglesCount = data.ploc.triangleCount;
    u32 innerNodeCount = 0;
    u32 leafCount = 0;
    CLUSTER clusterArr = CLUSTER(pc.data.clusterArrayAddress);
    TOPOLOGY topoArr = TOPOLOGY(pc.data.wideTopologyAddress);
    NodeTopology topo = topoArr.node[globalId];
    INNERMAP innerMap = INNERMAP(pc.data.clusterMapAddress);
    LEAFMAP leafMap = LEAFMAP(pc.data.triangleMapAddress);
    BASEIDS baseIds = BASEIDS(pc.data.baseIdsAddress);
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
    u32 innerBaseIdx = atomicAdd(data.ploc.innerNodeCounter, innerNodeCount) + IA_ARRAY_OFFSET;
    u32 leafBaseIdx = atomicAdd(data.ploc.leafCounter, leafCount);
    baseIds.node[globalId].innerBaseIdx = innerBaseIdx;
    baseIds.node[globalId].leafBaseIdx = leafBaseIdx;
    for (u32 i = 0; i < BVH_ARITY; ++i) {
        if (topo.c[i] == -1)
            break;
        if (topo.c[i] < trianglesCount) {
            leafMap.node[u32(clusterArr.node[topo.c[i]].c[0])].finalIdx = leafBaseIdx++;
        } else {
            innerMap.node[u32(topo.c[i])].finalIdx = innerBaseIdx++;
        }
    }
    if (globalId == (data.ploc.wideCounter - 1)) {
        innerMap.node[u32(topo.cl)].finalIdx = data.ploc.wideCounter - 1 + IA_ARRAY_OFFSET;
    }
}
