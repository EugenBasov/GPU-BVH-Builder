# GPU-BVH-Builder — wide BVH construction in Vulkan compute shaders

GLSL compute shaders from my master's thesis
[*Construction of bounding volume hierarchies on the GPU using the H-PLOC method*](https://hdl.handle.net/10467/182610)
(CTU in Prague, 2026, supervisor Jaroslav Sloup). They build bounding volume hierarchies (BVH) for ray tracing
on the GPU:

- **H-PLOC** — my implementation of H-PLOC [1]. It builds a binary BVH in a single kernel launch.
- **Fused Collapsing** — my modification of H-PLOC that builds a compressed 4-wide or 8-wide BVH directly.
  No binary BVH is stored, so the usual binary-to-wide conversion pass is not needed.

Only the shaders are published. The host code (Vulkan setup, radix sort, ray tracing) belongs to the Orchard
framework used for the thesis; what it has to do is described in [Host side](#host-side).

## H-PLOC

The triangles are sorted by their Morton codes. Every thread starts at one leaf and walks up an implicit
linear BVH: the parent is chosen from the Morton codes, and an atomic exchange on a parent array decides
which of the two children continues — the first thread that arrives stops, the second one takes over the
merged range of clusters.

When a range spans more than half a subgroup of triangles, the whole subgroup runs PLOC on its clusters.
Every cluster looks for its nearest neighbour among ±16 lanes (the surface area of the merged box), mutual
nearest neighbours are merged, and this repeats until half a subgroup of clusters is left (one cluster at the
root). The whole build is a single dispatch.

The result is a binary BVH with the leaves at indices `0 … N−1` and the inner nodes after them; a child that
is a leaf is stored as `~index`.

## Fused Collapsing

Fused Collapsing keeps the H-PLOC traversal, but a cluster is not a binary node any more. It stores its box
and up to `N/2` references to its future children (triangles or finished wide nodes):

- Merging two clusters concatenates their references, leaves first.
- When the result has more than `N/2` references, or the cluster is the root, a wide node with all of them
  (at most `N`) is emitted, and the merged cluster continues with a single reference to this node.
- The nearest-neighbour cost is multiplied by 1.3 for clusters with a different number of references, which
  favours pairs of the same size.

After this kernel the wide nodes exist only as a topology. Four short kernels then assign the final indices,
write the compressed nodes and reorder the triangles.

| Stage | Shader | Threads | What it does |
|---|---|---|---|
| 1 | `fused_aabb_InitialClusters_{4,8}` | one per triangle | cluster with the triangle box, 30-bit Morton code, triangle for the Woop test |
| – | radix sort (host) | | sorts the Morton codes |
| 2 | `fused_aabb_FusedCollapsing_{4,8}` | one per triangle | builds the topology of the wide BVH in a single launch |
| 3 | `fused_aabb_IndexAssignment_{4,8}` | one per wide node | reserves contiguous indices for the children of every node |
| 4 | `fused_aabb_WriteBVH_{4,8}` | one per wide node | writes the compressed nodes with 8-bit quantized child boxes [2] |
| 5 | `fused_aabb_PartialReordering_{4,8}` | one workgroup | moves the upper levels to the start of the buffer in breadth-first order, root at index 0 |
| 6 | `fused_aabb_TrianglesTransfer` | one per triangle | reorders the triangles to match the leaves |

## Results

Measured in the thesis on 10 scenes (72 K – 6.7 M triangles) on an NVIDIA GeForce RTX 3050 Ti Laptop GPU.
The reference is the optimized PLOC++ already implemented in Orchard; wide BVHs from PLOC++ and H-PLOC are
made by the usual conversion of the binary BVH.

- **H-PLOC** performs on par with PLOC++: build time, SAH cost and trace time differ by a few percent, and
  neither algorithm is clearly better.
- **Fused Collapsing** builds the wide BVH 5–50 % faster than the methods with a separate conversion step,
  the most in large scenes, and its nodes have 10–70 % more children on average.
- The thesis version has no leaf collapsing: every leaf holds one triangle, so the tree is deeper and tracing
  visits more nodes. Tracing gets slower and build + trace time grows by 20–50 %. Adding a leaf collapsing
  phase is the next step suggested in the thesis.

Detailed tables are in chapter 6 of the thesis.

## Host side

N is the number of triangles. The sizes are the ones used in the thesis.

| Buffer | Size | Before the build |
|---|---|---|
| Morton key-values (two copies for the radix sort) | 8 B × N | written by stage 1 |
| Clusters (`NodeCluster`) | (2N − 1) clusters | written by stage 1 |
| Parent IDs | 4 B × N | −1 |
| Topology (`NodeTopology`) | N nodes | — |
| Inner map, leaf map, base indices | 4 B × (2N − 1), 4 B × N, 8 B × N | — |
| Wide BVH | (1.5 N for 4-wide, 1.25 N for 8-wide) + 2048 nodes | zeros |
| Runtime data (`FUSEDData`) | 20 B | zeros, `triangleCount` = N |
| Triangles and triangle IDs | 48 B × N, 8 B × N (input and output) | written by stage 1 |

- Stage 1 is dispatched for every mesh. After the radix sort, the sorted cluster IDs are copied to a separate
  buffer (`nodeId0`); the sorted key-values are read through `nodeId1`.
- Stages 2–6 run back to back with a compute-to-compute memory barrier between them. The number of wide
  nodes is known only on the GPU, so stages 3, 4 and 6 are dispatched for N threads and the extra threads
  exit; stage 5 is a single workgroup.
- Specialization constants: `0` is the workgroup size, `1` the subgroup size. Stage 2 needs 8 B of shared
  memory per thread.
- H-PLOC uses the same preparation (initial clusters, sorting, parent IDs set to −1) and then a single
  dispatch with one thread per triangle.

The shaders use buffer device addresses for all data; every buffer address comes in push constants.
Required device features: `bufferDeviceAddress`, `scalarBlockLayout`, `shaderInt64`,
`shaderSharedInt64Atomics` and subgroup ballot and shuffle operations.

## Files

```
shaders/hploc_aabb_HPLOC.comp            H-PLOC, binary BVH
shaders/fused_aabb_*.glsl                Fused Collapsing stages, *_4.comp and *_8.comp select the BVH width
shaders/fused_aabb_TrianglesTransfer.comp
shaders/shared/                          types, data layouts and helper functions
```

Every shader compiles on its own, for example:

```bash
glslangValidator -V --target-env vulkan1.3 shaders/fused_aabb_FusedCollapsing_8.comp -o fused_8.spv
```

## Credits

- [1] C. Benthin, D. Meister, J. Barczak, R. Mehalwal, J. Tsakok, A. Kensler: *H-PLOC: Hierarchical Parallel
  Locally-Ordered Clustering for Bounding Volume Hierarchy Construction.* HPG 2024.
- [2] H. Ylitie, T. Karras, S. Laine: *Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide
  BVHs.* HPG 2017.
- The quantized wide BVH node (`shared/data_bvh_wide.h`, `shared/bvh_wide_quantized.glsl`) comes from
  Lukáš Cezner, [*Wide Bounding Volume Hierarchies for Ray Tracing*](https://dcgi.fel.cvut.cz/en/theses/2025/cezneluk/),
  master's thesis, CTU in Prague, 2025.
- The other headers in `shared/` and the initial clusters stage are based on the Orchard framework.

## License

MIT for the H-PLOC and Fused Collapsing shaders, see [LICENSE](LICENSE). The files listed in
[Credits](#credits) belong to their authors.
