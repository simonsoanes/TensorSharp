// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML.Interop
{
    // Raw P/Invoke bindings for the ggml C API. The symbols live in the same
    // GgmlOps native library that hosts the legacy TSGgml_* kernels: ggml is
    // statically linked into GgmlOps and its public API is re-exported via
    // TensorSharp.GGML.Native/ggml_api_exports.def (Windows) or default symbol
    // visibility (Linux/macOS). Binding the ggml API directly lets managed code
    // build and run ggml compute graphs without going through the C++ glue
    // layer; the compute kernels themselves remain ggml's native ones, so this
    // adds no math to the managed side.
    //
    // Types mirror the vendored ggml headers (ExternalProjects/ggml, v0.15.x):
    //   size_t  -> nuint
    //   int64_t -> long
    //   enums   -> int (constants below)
    //   bool    -> 1-byte C bool (marshalled as I1)
    // Handle types (ggml_context*, ggml_tensor*, ggml_backend_t, ...) are
    // opaque IntPtrs except ggml_tensor, whose struct layout is replicated in
    // GgmlTensor for direct field access (validated at runtime against
    // ggml_tensor_overhead, see GgmlManagedRuntime.ValidateAbi).
    internal static unsafe class GgmlApi
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention Cdecl = CallingConvention.Cdecl;

        static GgmlApi()
        {
            // Reuse GgmlNative's resolver so both the TSGgml_* entry points and
            // the raw ggml API resolve to the same loaded module.
            GgmlNative.EnsureImportResolverRegistered();
        }

        // --- ggml type / status / enum constants ---

        internal const int GGML_TYPE_F32 = 0;
        internal const int GGML_TYPE_F16 = 1;
        internal const int GGML_TYPE_Q4_0 = 2;
        internal const int GGML_TYPE_Q8_0 = 8;
        internal const int GGML_TYPE_I32 = 26;
        internal const int GGML_STATUS_SUCCESS = 0;

        internal const int GGML_BACKEND_DEVICE_TYPE_CPU = 0;
        internal const int GGML_BACKEND_DEVICE_TYPE_GPU = 1;
        internal const int GGML_BACKEND_DEVICE_TYPE_IGPU = 2;

        internal const int GGML_BACKEND_BUFFER_USAGE_ANY = 0;
        internal const int GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1;
        internal const int GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2;

        internal const long GGML_DEFAULT_GRAPH_SIZE = 2048;

        // GGML_ROPE_TYPE_* (ggml.h)
        internal const int GGML_ROPE_TYPE_NEOX = 2;
        internal const int GGML_ROPE_TYPE_MROPE = 8;
        internal const int GGML_ROPE_TYPE_VISION = 24;

        // --- structs ---

        [StructLayout(LayoutKind.Sequential)]
        internal struct ggml_init_params
        {
            public nuint mem_size;
            public IntPtr mem_buffer;
            [MarshalAs(UnmanagedType.I1)] public bool no_alloc;
        }

        // Mirrors struct ggml_tensor (ggml.h). GGML_MAX_DIMS=4, GGML_MAX_OP_PARAMS=64,
        // GGML_MAX_SRC=10, GGML_MAX_NAME=64. Layout is validated at runtime.
        [StructLayout(LayoutKind.Sequential)]
        internal struct GgmlTensor
        {
            public int type;
            public IntPtr buffer;
            public fixed long ne[4];
            public fixed ulong nb[4];
            public int op;
            public fixed int op_params[16];
            public int flags;
            public IntPtr src0, src1, src2, src3, src4, src5, src6, src7, src8, src9;
            public IntPtr view_src;
            public nuint view_offs;
            public IntPtr data;
            public fixed byte name[64];
            public IntPtr extra;
            public fixed byte padding[8];
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct ggml_backend_dev_caps
        {
            [MarshalAs(UnmanagedType.I1)] public bool async;
            [MarshalAs(UnmanagedType.I1)] public bool host_buffer;
            [MarshalAs(UnmanagedType.I1)] public bool buffer_from_host_ptr;
            [MarshalAs(UnmanagedType.I1)] public bool events;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct ggml_backend_dev_props
        {
            public IntPtr name;
            public IntPtr description;
            public nuint memory_free;
            public nuint memory_total;
            public int type;
            public IntPtr device_id;
            public ggml_backend_dev_caps caps;
        }

        // --- context ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_init(ggml_init_params @params);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_free(IntPtr ctx);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_used_mem(IntPtr ctx);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_tensor_overhead();
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_graph_overhead();
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_graph_overhead_custom(nuint size, [MarshalAs(UnmanagedType.I1)] bool grads);

        // --- type info ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_type_size(int type);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern long ggml_blck_size(int type);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_row_size(int type, long ne);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_nbytes(GgmlTensor* tensor);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern long ggml_nelements(GgmlTensor* tensor);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_type_name(int type);
        [DllImport(DllName, CallingConvention = Cdecl)] [return: MarshalAs(UnmanagedType.I1)] internal static extern bool ggml_is_contiguous(GgmlTensor* tensor);
        [DllImport(DllName, CallingConvention = Cdecl)] [return: MarshalAs(UnmanagedType.I1)] internal static extern bool ggml_is_quantized(int type);

        // --- tensor creation / views / layout ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_new_tensor(IntPtr ctx, int type, int n_dims, long* ne);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_new_tensor_1d(IntPtr ctx, int type, long ne0);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_new_tensor_2d(IntPtr ctx, int type, long ne0, long ne1);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_new_tensor_3d(IntPtr ctx, int type, long ne0, long ne1, long ne2);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_new_tensor_4d(IntPtr ctx, int type, long ne0, long ne1, long ne2, long ne3);

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_view_1d(IntPtr ctx, GgmlTensor* a, long ne0, nuint offset);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_view_2d(IntPtr ctx, GgmlTensor* a, long ne0, long ne1, nuint nb1, nuint offset);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_view_3d(IntPtr ctx, GgmlTensor* a, long ne0, long ne1, long ne2, nuint nb1, nuint nb2, nuint offset);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_view_4d(IntPtr ctx, GgmlTensor* a, long ne0, long ne1, long ne2, long ne3, nuint nb1, nuint nb2, nuint nb3, nuint offset);

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_reshape_1d(IntPtr ctx, GgmlTensor* a, long ne0);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_reshape_2d(IntPtr ctx, GgmlTensor* a, long ne0, long ne1);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_reshape_3d(IntPtr ctx, GgmlTensor* a, long ne0, long ne1, long ne2);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_reshape_4d(IntPtr ctx, GgmlTensor* a, long ne0, long ne1, long ne2, long ne3);

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_permute(IntPtr ctx, GgmlTensor* a, int axis0, int axis1, int axis2, int axis3);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_transpose(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_cont(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_cpy(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_dup(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_cast(IntPtr ctx, GgmlTensor* a, int type);

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_set_input(GgmlTensor* tensor);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_set_output(GgmlTensor* tensor);

        // --- unary ops ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_neg(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_exp(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_log(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_sqrt(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_sqr(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_relu(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_sigmoid(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_tanh(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_silu(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_step(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_abs(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_sgn(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_gelu(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_silu_back(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);

        // --- binary / scalar ops ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_add(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_sub(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_mul(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_div(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_scale(IntPtr ctx, GgmlTensor* a, float s);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_scale_bias(IntPtr ctx, GgmlTensor* a, float s, float b);

        // --- reductions / indexing ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_sum(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_sum_rows(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_mean(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_argmax(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_argsort(IntPtr ctx, GgmlTensor* a, int order);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_top_k(IntPtr ctx, GgmlTensor* a, int k);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_get_rows(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_set_rows(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, GgmlTensor* c);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_repeat(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_concat(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, int dim);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_acc(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, nuint nb1, nuint nb2, nuint nb3, nuint offset);

        // --- matmul ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_mul_mat(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_mul_mat_id(IntPtr ctx, GgmlTensor* @as, GgmlTensor* b, GgmlTensor* ids);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_out_prod(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);

        // --- norm / softmax / rope / attention ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_norm(IntPtr ctx, GgmlTensor* a, float eps);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_rms_norm(IntPtr ctx, GgmlTensor* a, float eps);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_rms_norm_back(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, float eps);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_soft_max(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_soft_max_ext(IntPtr ctx, GgmlTensor* a, GgmlTensor* mask, float scale, float max_bias);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_soft_max_ext_back(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, float scale, float max_bias);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_rope_ext(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, GgmlTensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_rope_multi(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, GgmlTensor* c, int n_dims, int* sections, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_rope_ext_back(IntPtr ctx, GgmlTensor* a, GgmlTensor* b, GgmlTensor* c, int n_dims, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_flash_attn_ext(IntPtr ctx, GgmlTensor* q, GgmlTensor* k, GgmlTensor* v, GgmlTensor* mask, float scale, float max_bias, float logit_softcap);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_flash_attn_ext_set_prec(GgmlTensor* a, int prec);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_diag_mask_inf(IntPtr ctx, GgmlTensor* a, int n_past);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_clamp(IntPtr ctx, GgmlTensor* a, float min, float max);

        // --- fused GLU ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_swiglu(IntPtr ctx, GgmlTensor* a);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_swiglu_split(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern GgmlTensor* ggml_geglu_split(IntPtr ctx, GgmlTensor* a, GgmlTensor* b);

        // --- graph ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_new_graph(IntPtr ctx);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_new_graph_custom(IntPtr ctx, nuint size, [MarshalAs(UnmanagedType.I1)] bool grads);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_build_forward_expand(IntPtr cgraph, GgmlTensor* tensor);

        // --- backend / device ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_cpu_init();
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_cpu_set_n_threads(IntPtr backend_cpu, int n_threads);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_dev_by_type(int type);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_dev_init(IntPtr device, IntPtr @params);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_dev_get_props(IntPtr device, ggml_backend_dev_props* props);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_dev_buffer_type(IntPtr device);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_dev_buffer_from_host_ptr(IntPtr device, IntPtr ptr, nuint size, nuint max_tensor_size);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_get_device(IntPtr backend);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_free(IntPtr backend);
        [DllImport(DllName, CallingConvention = Cdecl)] [return: MarshalAs(UnmanagedType.I1)] internal static extern bool ggml_backend_supports_op(IntPtr backend, GgmlTensor* op);

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern int ggml_backend_vk_get_device_count();
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_vk_init(nuint dev_num);

        // --- backend buffers ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_get_default_buffer_type(IntPtr backend);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_backend_buft_get_alignment(IntPtr buft);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_backend_buft_get_max_size(IntPtr buft);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_backend_buft_get_alloc_size(IntPtr buft, GgmlTensor* tensor);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_buft_alloc_buffer(IntPtr buft, nuint size);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_buffer_free(IntPtr buffer);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_buffer_get_base(IntPtr buffer);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_backend_buffer_get_size(IntPtr buffer);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_backend_buffer_get_alloc_size(IntPtr buffer, GgmlTensor* tensor);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_buffer_set_usage(IntPtr buffer, int usage);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_buffer_clear(IntPtr buffer, byte value);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern int ggml_backend_tensor_alloc(IntPtr buffer, GgmlTensor* tensor, IntPtr addr);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_backend_alloc_ctx_tensors(IntPtr ctx, IntPtr backend);

        // --- compute / transfer ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern int ggml_backend_graph_compute(IntPtr backend, IntPtr cgraph);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_synchronize(IntPtr backend);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_tensor_set(GgmlTensor* tensor, IntPtr data, nuint offset, nuint size);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_tensor_get(GgmlTensor* tensor, IntPtr data, nuint offset, nuint size);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_tensor_set_async(IntPtr backend, GgmlTensor* tensor, IntPtr data, nuint offset, nuint size);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_backend_tensor_get_async(IntPtr backend, GgmlTensor* tensor, IntPtr data, nuint offset, nuint size);

        // --- graph allocator ---

        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern IntPtr ggml_gallocr_new(IntPtr buft);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern void ggml_gallocr_free(IntPtr galloc);
        [DllImport(DllName, CallingConvention = Cdecl)] [return: MarshalAs(UnmanagedType.I1)] internal static extern bool ggml_gallocr_alloc_graph(IntPtr galloc, IntPtr graph);
        [DllImport(DllName, CallingConvention = Cdecl)] internal static extern nuint ggml_gallocr_get_buffer_size(IntPtr galloc, int buffer_id);
    }
}
