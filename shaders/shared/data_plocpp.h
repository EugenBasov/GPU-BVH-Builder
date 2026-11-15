#ifndef DATA_PLOCPP_H
#define DATA_PLOCPP_H

#include "types.glsl"

// Push constants and runtime data shared with the host

struct PC_MortonGlobal {
    vec3 sceneAabbCubedMin;
    f32 sceneAabbNormalizationScale;
    u64 mortonAddress;
    u64 bvhAddress;
    u64 bvhTrianglesAddress;
    u64 bvhTriangleIndicesAddress;
    u64 auxBufferAddress;
};

struct PC_MortonPerGeometry {
    u64 idxAddress;
    u64 vtxAddress;
    u32 globalTriangleIdBase;
    u32 sceneNodeId;
    u32 triangleCount;
};

struct PC_ClusterArray {
    u64 clusterArrayAddress;
};

struct PC_PlocppIterationIndirect {
    u64 bvhAddress;
    u64 nodeId0Address;
    u64 nodeId1Address;
    u64 dlWorkBufAddress;
    u64 runtimeDataAddress;
    u64 auxBufferAddress;
    u64 idbAddress;
    u64 parentIDsAddress;
    u32 triangleCount;
};

struct PC_FusedCollapsing {
    u64 clusterArrayAddress;
    u64 wideTopologyAddress;
    u64 nodeId0Address;
    u64 nodeId1Address;
    u64 dlWorkBufAddress;
    u64 runtimeDataAddress;
    u64 auxBufferAddress;
    u64 idbAddress;
    u64 parentIDsAddress;
};

struct PC_IndexAssignment {
    u64 clusterArrayAddress;
    u64 wideTopologyAddress;
    u64 clusterMapAddress;
    u64 triangleMapAddress;
    u64 baseIdsAddress;
    u64 runtimeDataAddress;
};

struct PC_WriteBVH {
    u64 clusterArrayAddress;
    u64 wideTopologyAddress;
    u64 clusterMapAddress;
    u64 triangleMapAddress;
    u64 baseIdsAddress;
    u64 runtimeDataAddress;
    u64 bvhWideAddress;
};

struct PC_PartialReordering {
    u64 runtimeDataAddress;
    u64 bvhWideAddress;
};

struct PC_TriangleTransfer {
    u64 triangleMapAddress;
    u64 runtimeDataAddress;
    u64 inputTrianglesAddress;
    u64 inputTriangleIndicesAddress;
    u64 outputTrianglesAddress;
    u64 outputTriangleIndicesAddress;
};

struct Morton32KeyVal {
    u32 key;
    u32 mortonCode;
};

// State of the framework task scheduler, it precedes PLOCData in the runtime data buffer
struct Scheduler {
    u32 phase;
    u32 idStart;
    u32 idEnd;
    u32 idToAssign;
    u32 numFinished;
};

struct PLOCData {
    u32 bvOffset;
    u32 iterationClusterCount;
    u32 iterationTaskCount;
    u32 fill;
    u32 iterationCounter;
};

struct FUSEDData {
    u32 bvOffset;
    u32 wideCounter;
    u32 innerNodeCounter;
    u32 leafCounter;
    u32 triangleCount;
};

layout(buffer_reference, scalar) buffer Morton32KeyVals {
    Morton32KeyVal keyval[];
};

#endif
