#ifndef TYPES_GLSL
#define TYPES_GLSL

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define i32 int32_t
#define u32 uint32_t
#define f32 float
#define i64 int64_t
#define u64 uint64_t

layout(buffer_reference, scalar) buffer vec3_buf {
    vec3 val[];
};
layout(buffer_reference, scalar) buffer uvec3_buf {
    uvec3 val[];
};
layout(buffer_reference, scalar) buffer i32_buf {
    i32 val[];
};
layout(buffer_reference, scalar) buffer u32_buf {
    u32 val[];
};

#endif
