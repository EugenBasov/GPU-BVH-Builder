#extension GL_EXT_shader_atomic_int64 : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require

#include "shared/types.glsl"
#include "shared/compute.glsl"
#include "shared/bv_aabb.glsl"
#include "shared/data_fused.h"
#include "shared/data_plocpp.h"

layout(local_size_x_id = 0) in;
layout(constant_id = 1) const u32 sc_SubgroupSize = 32;

layout(push_constant, scalar) uniform uPushConstant {
    PC_FusedCollapsing data;
} pc;

shared u64 nn[gl_WorkGroupSize.x];

layout(buffer_reference, scalar) buffer RuntimeData {
    FUSEDData ploc;
};

// Lowest set bit of a 64-bit mask, -1 if there is none
i32 findLSB64(u64 mask)
{
    uvec2 parts = unpackUint2x32(mask);

    i32 lsb = findLSB(parts.x);
    if (lsb != -1) {
        return lsb;
    }
    lsb = findLSB(parts.y);
    if (lsb != -1) {
        return lsb + 32;
    }

    return -1;
}

// Position of the n-th set bit (counted from 1), -1 if there is none
u32 findNSB(u32 mask, u32 n) {
    if (n == 0 || n > bitCount(mask)) return -1;

    u32 pos = 0;
    u32 count;

    count = bitCount(mask & 0xFFFFu);
    if (n > count) { n -= count; mask >>= 16; pos += 16; }

    count = bitCount(mask & 0xFFu);
    if (n > count) { n -= count; mask >>= 8; pos += 8; }

    count = bitCount(mask & 0xFu);
    if (n > count) { n -= count; mask >>= 4; pos += 4; }

    count = bitCount(mask & 0x3u);
    if (n > count) { n -= count; mask >>= 2; pos += 2; }

    count = mask & 0x1u;
    if (n > count) { pos += 1; }

    return pos;
}

u32 findNSB64(u64 mask, u32 n) {
    if (n == 0) return -1;

    u32 low = uint(mask & 0xFFFFFFFFu);
    u32 pop_low = bitCount(low);

    if (n <= pop_low) {
        return findNSB(low, n);
    }
    else {
        u32 high = uint(mask >> 32);
        u32 pop_high = bitCount(high);

        if (n > pop_low + pop_high) return -1;

        return findNSB(high, n - pop_low) + 32;
    }
}

u32 mergeClusters(CLUSTER clusterArray, RuntimeData data, u32 triangleCount, u32 numPrims, u64 NN,
    inout u32 CI, inout AABB myBB, inout u32 my_ref_count, inout i32 my_refs[BVH_ARITY/2])
{
    u32 memoryIdLocal = gl_SubgroupInvocationID;

    u32 myNeighbour = u32(subgroupShuffle(u32(NN), memoryIdLocal));
    myNeighbour = (myNeighbour == INVALID_ID) ? 0 : myNeighbour;
    u32 hisNeighbour = u32(subgroupShuffle(u32(NN), myNeighbour));
    bool symmetricMatchFound = (memoryIdLocal == hisNeighbour);
    bool laneIsLeftNeighbor = (memoryIdLocal < myNeighbour);

    const u32 myNodeId = CI;
    const u32 rightNodeId = subgroupShuffle(CI, myNeighbour);

    bool laneHasCluster = (memoryIdLocal < numPrims);
    bool laneIsCreatingNode = ((laneHasCluster) && (symmetricMatchFound) && (laneIsLeftNeighbor));
    const uvec4 ballot = subgroupBallot(laneIsCreatingNode);
    u32 mergedId;
    if (memoryIdLocal == 0)
        mergedId = atomicAdd(data.ploc.bvOffset, subgroupBallotBitCount(ballot));
    mergedId = subgroupBroadcastFirst(mergedId) + subgroupBallotExclusiveBitCount(ballot);

    u32 newCI = INVALID_ID;
    if (laneHasCluster) {
        newCI = CI;
        if (symmetricMatchFound) {
            AABB rightBB;
            rightBB.min = subgroupShuffle(myBB.min, myNeighbour);
            rightBB.max = subgroupShuffle(myBB.max, myNeighbour);
            i32 right_refs[BVH_ARITY / 2];
            for (u32 i = 0; i < BVH_ARITY / 2; i++) {
                right_refs[i] = subgroupShuffle(my_refs[i], myNeighbour);
            }
            if (laneIsLeftNeighbor) {
                AABB bvC0 = myBB;
                bvFit(bvC0, rightBB);
                clusterArray.node[mergedId].bv = bvC0;
                u32 refsCount = 0;
                i32 R[BVH_ARITY];

                // Read references
                if (myNodeId < triangleCount) {
                    R[refsCount] = i32(myNodeId);
                    refsCount++;
                }
                else{
                    for(u32 i = 0; i < BVH_ARITY / 2; i++){
                        if (my_refs[i] == INVALID_ID)
                            break;
                        R[refsCount] = my_refs[i];
                        refsCount++;
                    }
                }
                if (rightNodeId < triangleCount) {
                    R[refsCount] = i32(rightNodeId);
                    refsCount++;
                }
                else{
                    for(u32 i = 0; i < BVH_ARITY / 2; i++){
                        if (right_refs[i] == INVALID_ID)
                            break;
                        R[refsCount] = right_refs[i];
                        refsCount++;
                    }
                }

                // Put leaves at the start of the array
                u32 nextNonLeafPos = 0;
                i32 temp = 0;
                for (u32 i = 0; i < refsCount; i++) {
                    if (R[i] < triangleCount) {
                        if(nextNonLeafPos == i){
                            nextNonLeafPos++;
                            continue;
                        }
                        temp = R[nextNonLeafPos];
                        R[nextNonLeafPos] = R[i];
                        R[i] = temp;
                        nextNonLeafPos++;
                    }
                }

                // Too many references for one cluster (or the root): emit a wide node
                if (refsCount > BVH_ARITY / 2 || mergedId == (data.ploc.triangleCount * 2 - 2)) {
                    u32 wideIdx = atomicAdd(data.ploc.wideCounter, 1);
                    TOPOLOGY wideNodes = TOPOLOGY(pc.data.wideTopologyAddress);
                    wideNodes.node[wideIdx].cl = i32(mergedId);
                    for (u32 i = 0; i < BVH_ARITY; i++) {
                        if(i < refsCount)
                            wideNodes.node[wideIdx].c[i] = R[i];
                        else
                            wideNodes.node[wideIdx].c[i] = INVALID_ID;
                    }

                    refsCount = 1;
                    R[0] = i32(mergedId);
                    R[1] = INVALID_ID;
                }
                for (u32 i = 0; i < BVH_ARITY / 2; i++) {
                    if (i < refsCount) {
                        my_refs[i] = R[i];
                        clusterArray.node[mergedId].c[i] = R[i];
                    } else {
                        my_refs[i] = INVALID_ID;
                        clusterArray.node[mergedId].c[i] = INVALID_ID;
                    }
                }

                my_ref_count = refsCount;

                myBB = bvC0;
                newCI = mergedId;
            }
            else
                newCI = INVALID_ID;
        }
    }

    // Compact nodes
    const u32 flag = (newCI != INVALID_ID) ? 1 : 0;
    const u32 ps = laneHasCluster ? flag : 0;
    const u32 total_reduction = subgroupBallotBitCount(subgroupBallot(bool(ps)));
    const u32 compactionShuffle = findNSB64(packUint2x32(subgroupBallot(bool(ps)).xy), memoryIdLocal + 1);
    const u32 compactionSafeShuffle = (compactionShuffle == u32(-1)) ? 0 : compactionShuffle;

    CI = newCI != INVALID_ID ? newCI : CI;
    CI = subgroupShuffle(CI, compactionSafeShuffle);
    myBB.min = subgroupShuffle(myBB.min, compactionSafeShuffle);
    myBB.max = subgroupShuffle(myBB.max, compactionSafeShuffle);
    my_ref_count = subgroupShuffle(my_ref_count, compactionSafeShuffle);
    for (u32 i = 0; i < BVH_ARITY / 2; i++) {
        my_refs[i] = subgroupShuffle(my_refs[i], compactionSafeShuffle);
    }
    if (compactionShuffle == u32(-1)) {
        CI = INVALID_ID;
    }
    return total_reduction;
}

u64 nearest_neighbour(u32 clusterCount, AABB myBB, u32 my_ref_count){
    atomicExchange(nn[gl_LocalInvocationID.x], INVALID_ID);
    u32 memoryIdLocal = gl_SubgroupInvocationID;
    AABB neighborBv;
    u32 neighbourRefCount;
    f32 alphaFactor = 1.0;

    u64 minValue = INVALID_ID;
    for (u32 cnt = 0; cnt < 16; cnt++) {
        u32 cacheId = memoryIdLocal + cnt + 1;
        neighborBv.min = subgroupShuffle(myBB.min, cacheId);
        neighborBv.max = subgroupShuffle(myBB.max, cacheId);
        neighbourRefCount = subgroupShuffle(my_ref_count, cacheId);
        // Prefer pairs with the same number of references
        if (my_ref_count != neighbourRefCount)
            alphaFactor = 1.3;
        else
            alphaFactor = 1.0;

        if(cacheId < clusterCount){
            bvFit(neighborBv, myBB);
            u64 encoded = (u64(floatBitsToUint(bvArea(neighborBv) * alphaFactor)) << 32);
            minValue = min(minValue, encoded | (cacheId));
            atomicMin(nn[gl_LocalInvocationID.x + cnt + 1], encoded | (memoryIdLocal));
        }
    }
    atomicMin(nn[gl_LocalInvocationID.x], minValue);
    return nn[gl_LocalInvocationID.x];
}

u32 loadIndices(u32 start, u32 end, u32_buf I, inout u32 CI, u32 offset){
    u32 numIndices = min(end - start, u32(sc_SubgroupSize / 2));
    i32 indexID = i32(gl_SubgroupInvocationID - offset);
    bool laneActive = ((indexID >= 0) && (indexID < numIndices));
    if (laneActive) {
        memoryBarrierBuffer();
        CI = atomicExchange(I.val[start + indexID], INVALID_ID, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire);
        memoryBarrierBuffer();
    }
    u32 numValid = subgroupBallotBitCount(subgroupBallot(laneActive && (CI != INVALID_ID)));
    numIndices = min(numIndices, numValid);
    return numIndices;
}

void hploc_merge(u32 laneID, u32 L, u32 R, u32 split, bool is_final, u32 triangleCount)
{
    CLUSTER clusterArray = CLUSTER(pc.data.clusterArrayAddress);

    u32_buf I = u32_buf(pc.data.nodeId0Address);
    RuntimeData data = RuntimeData(pc.data.runtimeDataAddress);
    u32 Lstart = subgroupBroadcast(L, laneID);
    u32 Lend = subgroupBroadcast(split, laneID);
    u32 Rstart = subgroupBroadcast(split, laneID);
    u32 Rend = subgroupBroadcast(R, laneID) + 1;

    u32 CI = INVALID_ID;
    u32 numLeft = loadIndices(Lstart, Lend, I, CI, 0);
    u32 numRight = loadIndices(Rstart, Rend, I, CI, numLeft);
    u32 numPrims = numLeft + numRight;
    AABB myBV;
    u32 my_ref_count = 0;

    i32 my_refs[BVH_ARITY / 2];
    for (u32 i = 0; i < BVH_ARITY / 2; i++) {
        my_refs[i] = INVALID_ID;
    }

    if (CI != INVALID_ID){
        myBV = clusterArray.node[CI].bv;
        for(u32 i = 0; i < BVH_ARITY / 2; i++){
            my_refs[i] = clusterArray.node[CI].c[i];
            if (my_refs[i] != INVALID_ID)
                my_ref_count++;
            else break;
        }
    }
    else{
        myBV = dummyAABB();
        my_ref_count = 0;
    }

    u32 THRESHOLD = subgroupBroadcast(is_final, laneID) == true ? 1 : sc_SubgroupSize / 2;
    while (numPrims > THRESHOLD) {
        u64 NN = nearest_neighbour(numPrims, myBV, my_ref_count);
        numPrims = mergeClusters(clusterArray, data, triangleCount, numPrims, NN, CI, myBV, my_ref_count, my_refs);
    }

    if (gl_SubgroupInvocationID < (numLeft + numRight)){
        memoryBarrierBuffer();
        atomicStore(I.val[Lstart + gl_SubgroupInvocationID], CI, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease);
        memoryBarrierBuffer();
    }
}

i32 delta(i32 NodeID, u32 N, Morton32KeyVals morton32) {
    if (NodeID < 0 || NodeID >= i32(N - 1)) {
        return -1;
    }

    u32 key1 = morton32.keyval[NodeID].mortonCode;
    u32 key2 = morton32.keyval[NodeID + 1].mortonCode;

    if (key1 == key2) {
        return 32 + 31 - findMSB(uint(NodeID ^ (NodeID + 1)));
    }

    return 31 - findMSB(key1 ^ key2);
}

u32 findParentID(u32 L, u32 R, u32 N, Morton32KeyVals morton32) {
    if (L == 0 || (R != N - 1 && delta(i32(R), N, morton32) > delta(i32(L) - 1, N, morton32))) {
        return R;
    } else {
        return L - 1;
    }
}

void buildBVH2(){
    i32_buf pID = i32_buf(pc.data.parentIDsAddress);
    Morton32KeyVals morton32 = Morton32KeyVals(pc.data.nodeId1Address);
    i32 L, R;
    i32 previousID, split;
    RuntimeData data = RuntimeData(pc.data.runtimeDataAddress);
    u32 N = data.ploc.triangleCount;
    L = R = i32(gl_GlobalInvocationID.x);
    bool laneActive = gl_GlobalInvocationID.x < N;
    while(subgroupBallot(laneActive) != uvec4(0)){
        split = -1;
        if(laneActive){
            previousID = -1;
            if(findParentID(L, R, N, morton32) == R){
                memoryBarrierBuffer();
                previousID = atomicExchange(pID.val[R], L, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
                memoryBarrierBuffer();
                if(previousID != -1){
                    split = R + 1;
                    R = previousID;
                }
            }
            else{
                memoryBarrierBuffer();
                previousID = atomicExchange(pID.val[L - 1], R, gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
                memoryBarrierBuffer();
                if(previousID != -1){
                    split = L;
                    L = previousID;
                }
            }
            if(previousID == -1)
                laneActive = false;
        }
        u32 size = R - L + 1;
        bool is_final = laneActive && (size == N);
        u64 waveMask = packUint2x32(subgroupBallot((laneActive && (size > (sc_SubgroupSize / 2))) || is_final).xy);
        while(waveMask != 0){
            u32 laneID = findLSB64(waveMask);
            hploc_merge(laneID, L, R, split, is_final, N);
            waveMask = waveMask & (waveMask-1);
        }
    }
}

void main() {
    RuntimeData data = RuntimeData(pc.data.runtimeDataAddress);

    if (gl_GlobalInvocationID.x == 0) {
        u32_buf clusterCount = u32_buf(pc.data.idbAddress);
        atomicExchange(data.ploc.bvOffset, clusterCount.val[0]);
    }

    buildBVH2();
}