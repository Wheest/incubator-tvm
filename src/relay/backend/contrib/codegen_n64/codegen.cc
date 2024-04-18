/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <sstream>
#include <string>

#include "../../../op/op_common.h"
#include "../../../qnn/utils.h"
#include "../../../transforms/compiler_function_utils.h"
#include "../../utils.h"
#include "codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

/*! \brief Return the "n64" Target instance to use to guide compilation. */
Target GetN64CompilerTarget() {
  Target target = Target::Current(/*allow_not_defined=*/true);
  if (!target.defined() || target->kind->name != "n64") {
    // Use the default compilation options if no specific "n64compiler" target was given
    // in the overall targets list. In that case target_hooks.cc will invoke the custom pass
    // without pushing any target instance onto the implicit target stack.
    target = Target("n64");
  }
  return target;
}

/*!
 * \brief Emits C/C++ code for a single function.
 *
 * For testing and demonstration only, only a few binary operators are supported.
 */
class CodegenN64 : public backend::MemoizedExprTranslator<std::vector<Output>>,
                   public CodegenCBase {
 public:
  CodegenN64(std::unordered_map<std::string, runtime::NDArray>* const_name_to_constant,
             Array<String>* const_names, bool* needs_extra_headers, std::string ext_func_id)
      : const_name_to_constant_(const_name_to_constant),
        const_names_(const_names),
        needs_extra_headers_(needs_extra_headers),
        ext_func_id_(std::move(ext_func_id)) {}

  /*!
   * \brief Emit the source code that invokes C compiler compatible wrappers.
   *
   * \return The emitted code.
   */
  std::string JIT(const std::vector<Output>& out) override {
    // Write function macros
    for (auto decl : func_decl_) {
      code_stream_ << decl << "\n";
    }
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  }

 private:
  std::vector<Output> VisitExprDefault_(const Object* op) override {
    LOG(FATAL) << "C codegen doesn't support: " << op->GetTypeKey();
  }

  std::vector<Output> VisitExpr_(const VarNode* node) override {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const TupleNode* node) override {
    std::vector<Output> outs;
    for (auto field : node->fields) {
      auto res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
      outs.push_back(res[0]);
    }
    return outs;
  }

  std::vector<Output> VisitExpr_(const TupleGetItemNode* op) override {
    auto res = VisitExpr(op->tuple);
    ICHECK_GT(res.size(), static_cast<size_t>(op->index));

    // Only keep the item we want for the child node.
    // FIXME(@comaniac): The other items should still be requried for the primary outputs.
    return {res[op->index]};
  }

  std::vector<Output> VisitExpr_(const ConstantNode* cn) override {
    // Remember we'll need some extra headers to support the runtime constants array.
    *needs_extra_headers_ = true;

    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    Output output;
    // Get const: static_cast<float*>(gcc_0_consts[0]->data)
    size_t const_id = const_name_to_constant_->size();
    output.name = CreateDataReference(ext_func_id_, const_id);
    const auto* type_node = cn->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);

    // Generate the global variable for needed ndarrays
    if (const_array_name_.empty()) {
      *needs_extra_headers_ = true;
      const_array_name_ = CreateNDArrayPool(ext_func_id_);
      std::string checker = CreateInitChecker(ext_func_id_);
      ext_func_body_.insert(ext_func_body_.begin(), checker);
    }

    ICHECK(dtype == "float" || dtype == "int") << "Only float and int are supported for now.";
    output.dtype = dtype;

    std::string const_var_name = CreateConstVar(ext_func_id_, const_id);
    const_name_to_constant_->emplace(const_var_name, cn->data);
    const_names_->push_back(const_var_name);

    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) override {
    std::ostringstream macro_stream;
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    if (backend::IsOp(call, "nn.conv2d")) {
      std::cout << "Wagwan, we have a conv2d op!" << std::endl;
      return Conv2d(call);
    }

    std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);

    // Make function declaration
    macro_stream << "CSOURCE_BINARY_OP_" << call->args.size() << "D(" << func_name << ", ";

    if (backend::IsOp(call, "add")) {
      macro_stream << "+";
    } else if (backend::IsOp(call, "subtract")) {
      macro_stream << "-";
    } else if (backend::IsOp(call, "multiply")) {
      macro_stream << "*";
    } else {
      LOG(FATAL) << "Unrecognized op";
    }

    auto in_shape = backend::GetShape(call->args[0]->checked_type());
    for (size_t i = 0; i < in_shape.size(); ++i) {
      macro_stream << ", " << in_shape[i];
    }

    const auto* type_node = call->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);
    macro_stream << ", " << dtype;

    macro_stream << ");";
    func_decl_.push_back(macro_stream.str());

    // Make function call when visiting arguments
    bool first = true;
    decl_stream << func_name << "(";
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (auto out : res) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        decl_stream << out.name;
      }
    }

    std::string out = "buf_" + std::to_string(buf_idx_++);
    auto out_shape = backend::GetShape(call->checked_type());
    int out_size = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
    }
    buf_stream << dtype << "* " << out << " = (" << dtype << "*)malloc(4 * " << out_size << ");";
    buf_decl_.push_back(buf_stream.str());

    decl_stream << ", " << out << ");";
    ext_func_body_.push_back(decl_stream.str());

    // Update output buffer
    // Note C codegen only handles TensorType. Therefore, we don't flatten
    // tuples and only return a single vaule.
    Output output;
    output.name = out;
    output.dtype = dtype;
    output.need_copy = true;
    output.size = out_size;
    return {output};
  }

  std::vector<Output> Conv2d(const CallNode* call) {
    std::ostringstream macro_stream;
    std::ostringstream decl_stream;
    std::ostringstream buf_stream;

    // Setup function
    std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);
    const auto* conv_attr = call->attrs.as<Conv2DAttrs>();
    ICHECK(conv_attr);

    // Distinguish between normal and depth-wise convolution
    // if (conv_attr->channels.defined() &&
    //     tvm::tir::ExprDeepEqual()(conv_attr->channels, conv_attr->groups) &&
    //     conv_attr->groups != 1) {
    //   ICHECK(conv_attr->kernel_layout == "IHWO")
    //       << "Kernel layout must be IHWO, has the module been pre-processed correctly?";
    // } else {
    //   // Raise an error that regular conv2d is not supported
    //   LOG(FATAL) << "Regular conv2d is not supported for RSP (yet!)";
    // }

    IndexExpr pad_h, pad_w;
    int pad;
    GetPaddingHeightWidth(conv_attr->padding, &pad_h, &pad_w);
    if (pad_h.as<IntImmNode>()->value != pad_w.as<IntImmNode>()->value) {
      LOG(FATAL) << "Asymetric padding is not supported for RSP (yet!)";
    } else {
      pad = pad_h.as<IntImmNode>()->value / 2;
    }
    std::cout << "Our padding is: " << pad << std::endl;

    int stride_h = qnn::get_const_int(conv_attr->strides[0]);
    int stride_w = qnn::get_const_int(conv_attr->strides[1]);
    std::cout << "Our strides are: " << stride_h << ", " << stride_w << std::endl;
    if (stride_h != stride_w) {
      LOG(FATAL) << "Asymetric strides are not supported for RSP (yet!)";
    }

    // Make function declaration
    macro_stream << "CSOURCE_RSP_DEPTH_CONV2D_OP(" << func_name << ", ";
    auto in_shape = backend::GetShape(call->args[0]->checked_type());
    const auto* type_node = call->checked_type().as<TensorTypeNode>();
    ICHECK(type_node);
    const auto& dtype = GetDtypeString(type_node);

    auto out_shape = backend::GetShape(call->checked_type());
    std::vector<int> out_shape_vec;
    int out_size = 1;
    for (size_t i = 0; i < out_shape.size(); ++i) {
      out_size *= out_shape[i];
      out_shape_vec.push_back(out_shape[i]);
    }

    int input_partition_height = 4;
    int output_partition_height = 4;
    macro_stream << input_partition_height << ", " << in_shape[1] << ", " << in_shape[2] << ", "
                 << in_shape[3] << ", " << out_shape[1] << ", " << out_shape[2] << ", 3, 3, "
                 << output_partition_height << ", " << pad << ", " << stride_h;
    macro_stream << ");";
    func_decl_.push_back(macro_stream.str());

    // Make function call when visiting arguments
    bool first = true;
    decl_stream << func_name << "(";
    std::string out_buff = "out0";
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (auto out : res) {
        if (!first) {
          decl_stream << ", ";
        }
        first = false;
        decl_stream << out.name;
      }
    }

    // std::string out = "buf_" + std::to_string(buf_idx_++);

    // buf_stream << dtype << "* " << out << " = (" << dtype << "*)malloc(4 * " << out_size << ");";
    // buf_decl_.push_back(buf_stream.str());

    decl_stream << ", " << out_buff << ");";

    decl_stream << "\n  printf(\"Hello, World!\\n\");";
    ext_func_body_.push_back(decl_stream.str());

    // Update output buffer
    // Note C codegen only handles TensorType. Therefore, we don't flatten
    // tuples and only return a single vaule.
    Output output;
    output.name = out_buff;
    output.dtype = dtype;
    std::cout << "Our output dtype is: " << dtype << std::endl;
    output.need_copy = false;
    output.size = out_size;
    return {output};
    // LOG(FATAL) << "Unrecognized op";
  }

  /*!
   * \brief The accumulated constant name to constant mapping. Shared between all generated
   * functions.
   */
  std::unordered_map<std::string, runtime::NDArray>* const_name_to_constant_;
  /*! \brief The accumulated constant names, in the order they were generated. */
  Array<String>* const_names_;
  /*!
   * \brief Set to true if the ndarray and packed function headers are required to declare and
   * manage the constants array.
   */
  bool* needs_extra_headers_;
  /*! \brief Name of the global function currently being compiled. */
  std::string ext_func_id_;

  /*! \brief The index of the next available wrapped C function. */
  int func_idx = 0;
  /*! \brief The index of the next available allocated buffers. */
  int buf_idx_ = 0;
  /*! \brief The arguments of a C compiler compatible function. */
  Array<Var> ext_func_args_;
  /*! \brief The statements of a C compiler compatible function. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration statements of a C compiler compatible function. */
  std::vector<std::string> func_decl_;
  /*! \brief The declaration statements of buffers. */
  std::vector<std::string> buf_decl_;
};

/*! \brief Emits C/C++ code for a module. */
class CodegenN64Module {
 public:
  CodegenN64Module(Target target, IRModule mod)
      : target_(std::move(target)), mod_(std::move(mod)) {}

  runtime::Module CreateCSourceModule() {
    std::cout << "Hey we are in CreateCSourceModule" << std::endl;
    for (const auto& kv : mod_->functions) {
      if (const auto* function_node = GetN64CompilerFunctionNode(kv.second)) {
        GenCFunc(GetRef<Function>(function_node));
      }
    }
    return Finalize();
  }

  /*! \brief Returns the accumulated constant name to constant mapping. */
  const std::unordered_map<std::string, runtime::NDArray>& const_name_to_constant() const {
    return const_name_to_constant_;
  }

 private:
  /*! \brief Emits the standard C/C++ header into \p os. */
  void EmitPreamble(std::ostringstream& os) {
    // Custom header, if any.
    Optional<String> header = target_->GetAttr<String>("header");
    if (header.defined() && !header.value().empty()) {
      os << header.value().c_str() << "\n";
    }

    // Standard includes.
    os << "#include <stdio.h>\n";
    os << "#include <stdlib.h>\n";
    os << "#include <string.h>\n";
    os << "#include <tvm/runtime/c_runtime_api.h>\n";
    os << "#include <tvm/runtime/c_backend_api.h>\n";
    os << "#include <libdragon.h>\n";

    if (needs_extra_headers_) {
      // This segment would be generated in C++ because of the usage
      // of tvm::runtime::Array. This is not ideal, but this to demonstrate
      // constant copying process used packed imports in other external
      // codegen. Moreover, in microTVM we dont expect this part to be generated.
      os << "#ifdef __cplusplus\n";
      os << "#include <tvm/runtime/ndarray.h>\n";
      os << "#include <tvm/runtime/packed_func.h>\n";
      os << "#endif\n";
    }

    // Define some macros to help operator implementations.
    const char* operator_macro = R"op_macro(
    #define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_, p_DTYPE)       \
      void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {    \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                        \
          out[i] = a[i] p_OP_ b[i];                                    \
        }                                                              \
      }

    #define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_, p_DTYPE)  \
      void p_ID_(p_DTYPE* a, p_DTYPE* b, p_DTYPE* out) {        \
        for (int64_t i = 0; i < p_DIM1_; ++i) {                            \
          for (int64_t j = 0; j < p_DIM2_; ++j) {                          \
            int64_t k = i * p_DIM2_ + j;                                   \
            out[k] = a[k] p_OP_ b[k];                                      \
          }                                                                \
        }                                                                  \
      }

    #define CSOURCE_RSP_DEPTH_CONV2D_OP(p_ID_, p_INPUT_PARTITION_HEIGHT_, p_IN_H_, p_IN_W_, p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_, p_OUTPUT_PARTITION_HEIGHT_, p_PADDING_, p_STRIDE_) \
      void p_ID_(int16_t *dest, int16_t *input_data, int16_t *weights) {   \
        RSPDepthConvTiledPadded(dest, input_data, weights, p_INPUT_PARTITION_HEIGHT_, p_IN_H_, p_IN_W_, p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_, p_OUTPUT_PARTITION_HEIGHT_, p_PADDING_, p_STRIDE_); \
      }
    )op_macro";

    os << operator_macro << "\n\n";

    Conv2dRSPHelpers(os);
  }

  void Conv2dRSPHelpers(std::ostringstream& os) {
    // Emit the helper code for the conv2d operator
    const char* rsp_ucode_macro = R"ucode_macro(
DEFINE_RSP_UCODE(rsp_simple);
uint32_t vec_id;
enum {
  DMAWeights = 0x0,
  DMAInputs = 0x1,
  DepthConv = 0x2,
};
void vec_init() {
  rspq_init();
  vec_id = rspq_overlay_register(&rsp_simple);
}
void vec_close() { rspq_overlay_unregister(vec_id); }
)ucode_macro";

    os << rsp_ucode_macro << "\n\n";

    const char* copy_slice_to_full_with_stride_macro = R"copy_macro(
void copy_slice_to_full_with_stride(int16_t *dest, int16_t *output_pad_part,
                                    int output_partition_height, int out_w,
                                    int slice_count, int in_depth,
                                    int depth_slice_num, size_t slice_depth,
                                    int max_output_parition_height) {
  // Copies a slice into the final destination with the appropriate stride
  // e.g., a slice of size 4x4x8 could be copied into a 4x4x16 destination
  // slice_depth should probably be the inner loop size (e.g., 8)
  for (int y = 0; y < max_output_parition_height; y++) {
    for (int x = 0; x < out_w; x++) {
      // Calculate the initial destination address with offsets and strided gaps
      int16_t *dest_ptr =
          &dest[slice_count * output_partition_height * out_w * in_depth +
                (y * out_w + x) * in_depth + (depth_slice_num * slice_depth)];
      for (int c = 0; c < slice_depth; c++) {
        // Perform the copy operation with stride
        dest_ptr[c] =
            output_pad_part[y * out_w * slice_depth + x * slice_depth + c];
      }
    }
  }
}
)copy_macro";

    os << copy_slice_to_full_with_stride_macro << "\n\n";

    const char* min_macro = R"min_macro(
#define MIN(a, b) ((a) < (b) ? (a) : (b))
)min_macro";

    os << min_macro << "\n\n";

    const char* generate_padded_slices_with_depth_slice_macro = R"generate_macro(
void generate_padded_slices_with_depth_slice(
    int16_t *data, int16_t *padded_input_partition, int start_h, int in_h,
    int in_w, int in_depth, int max_slice_height, int padding, int overlap,
    int depth_slice // New parameter indicating which set of 8 channels to copy
) {
  // Calculate the size of the padded area (height)
  int len = MIN(start_h + max_slice_height, in_h) - start_h;

  // Calculate depth start and end based on the depth_slice
  int depth_start =
      depth_slice * 8; // Start at the depth_slice set of 8 channels
  int depth_end =
      MIN(depth_start + 8, in_depth); // Ensure we don't go beyond in_depth

  // Clear the slice area
  memset(padded_input_partition, 0,
         sizeof(int16_t) * max_slice_height * (in_w + 2 * padding) * 8);

  // Determine slice dimensions
  int actual_start = (start_h == 0) ? start_h : start_h - padding;
  int actual_end = start_h + len;

  int target_start_h = (start_h == 0) ? padding : 0;

  // Copy data to the padded slice
  for (int h = actual_start; h < actual_end; ++h) {
    for (int w = 0; w < in_w; ++w) {
      for (int d = depth_start; d < depth_end; ++d) {
        int16_t value = data[(h * in_w * in_depth) + (w * in_depth) + d];
        int target_row = target_start_h + h - actual_start;
        int target_col = w + padding;
        int target_depth = d - depth_start;

        padded_input_partition[(target_row * (in_w + 2 * padding) * 8) +
                               (target_col * 8) + target_depth] = value;
      }
    }
  }
}
)generate_macro";

    os << generate_padded_slices_with_depth_slice_macro << "\n\n";

    const char* rsp_depth_conv_tiled_padded_macro = R"rsp_macro(
static inline void
RSPDepthConvTiledPadded(int16_t *dest, int16_t *input_data, int16_t *weights,
                        int input_partition_height, int in_h, int in_w,
                        int in_c, int out_h, int out_w, int k_h, int k_w,
                        int output_partition_height, int padding, int stride) {
  // Requires that weights have been reshaped offline to be
  // (out_c // 8, kernel_height * kernel_width, 8)
  extern uint32_t vec_id;

  // raise not implemented error if in_c is not 8
  if (in_c % 8 != 0) {
    printf("Error: in_c must be divisible by 8\n");
    return;
  }

  int16_t *input_pad_part = malloc_uncached_aligned(
      8, input_partition_height * (in_w + 2 * padding) * 8 * sizeof(int16_t));
  int16_t *output_pad_part = malloc_uncached_aligned(
      8, output_partition_height * out_w * 8 * sizeof(int16_t));
  int out_part_size = output_partition_height * out_w * 8;

  int overlap = k_h - stride;
  for (int depth_slice = 0; depth_slice < in_c / 8; depth_slice++) {
    int slice_count = 0;
    int remaining_out_values = out_h * out_w * 8; // Remaining values to copy
    int max_output_partition_height = output_partition_height;

    // Copy weights to the RSP once
    rspq_write(vec_id, DMAWeights,
               PhysicalAddr(&weights[depth_slice * k_h * k_w * 8]));

    // Generate first slice
    generate_padded_slices_with_depth_slice(
        input_data, (int16_t *)input_pad_part, 0, in_h, in_w, in_c,
        input_partition_height, padding, overlap, depth_slice);
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part));
    int start_h_next = input_partition_height - overlap;

    for (int start_h = 0; start_h < in_h;
         start_h += input_partition_height - overlap) {

      rspq_wait();
      // Process the padded partition on the RSP
      rspq_write(vec_id, DepthConv, PhysicalAddr(output_pad_part));
      // Generate next slice while the current one is being
      // processed
      if (start_h_next <= in_h) {
        generate_padded_slices_with_depth_slice(
            input_data, (int16_t *)input_pad_part, start_h_next, in_h, in_w,
            in_c, input_partition_height, padding, overlap, depth_slice);
        start_h_next += input_partition_height - overlap;
      }

      // Copy back the processed partition to the final outputs,
      // DMA the new input slice to the RSP at the same time
      rspq_wait();
      rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part));

      if (remaining_out_values < out_part_size) {
        // Cover the case where our final output slice is larger than required
        max_output_partition_height = remaining_out_values / (out_w * 8);
      }
      copy_slice_to_full_with_stride(
          dest, output_pad_part, output_partition_height, out_w, slice_count,
          in_c, depth_slice, 8, max_output_partition_height);
      remaining_out_values -= output_partition_height * out_w * 8;
      slice_count++;
    }
  }
}
)rsp_macro";

    os << rsp_depth_conv_tiled_padded_macro << "\n\n";
  }

  void GenCFunc(const Function& function) {
    ICHECK(function.defined()) << "Input error: expect a Relay function.";
    std::string ext_func_id = backend::GetExtSymbol(function);
    CodegenN64 builder(&const_name_to_constant_, &const_names_, &needs_extra_headers_, ext_func_id);
    std::vector<Output> out = builder.VisitExpr(function->body);
    code_stream_ << builder.JIT(out);
    func_names_.push_back(ext_func_id);
  }

  /*! \brief Returns function if it is tagged with "Compiler=n64compiler". */
  static const FunctionNode* GetN64CompilerFunctionNode(const Expr& expr) {
    if (const auto* function_node = expr.as<FunctionNode>()) {
      Optional<String> opt_compiler = function_node->GetAttr<String>(attr::kCompiler);
      if (opt_compiler.defined() && opt_compiler.value() == "n64") {
        return function_node;
      }
    }
    return nullptr;
  }

  runtime::Module Finalize() {
    std::ostringstream os;
    EmitPreamble(os);
    os << code_stream_.str();
    std::string code = os.str();

    VLOG(1) << "CodegenN64Module generated:" << std::endl << code;

    // Create a CSource module
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code, "c", func_names_, const_names_);
  }

  /*! \brief "n64" Target with compilation options to use. */
  Target target_;
  /*! \brief Module we are compiling. */
  IRModule mod_;

  /*! \brief True if we need to include the ndarray and packed function headers. */
  bool needs_extra_headers_ = false;
  /*! \brief The accumulated constant name to constant mapping. */
  std::unordered_map<std::string, runtime::NDArray> const_name_to_constant_;
  /*! \brief The accumulated constant names, in the order they were generated. */
  Array<String> const_names_;
  /*! \brief The accumulated function names. */
  Array<String> func_names_;
  /*!
   * \brief The accumulated code stream containing all function definitions.
   * (Does not include the preamble.)
   */
  std::ostringstream code_stream_;
};

/*! \brief The actual translation pass. */
tvm::transform::Pass N64CompilerImpl() {
  auto pass_func = [=](IRModule mod, const tvm::transform::PassContext& pass_ctx) {
    VLOG(1) << "N64CompilerImpl input:" << std::endl << PrettyPrint(mod);
    std::cout << "We are doing... something..." << std::endl;
    Target target = GetN64CompilerTarget();

    // Emit the C/C++ code and package it as a CSourceModule.
    CodegenN64Module codegen(target, mod);
    runtime::Module runtime_mod = codegen.CreateCSourceModule();

    // Capture the new runtime module.
    Array<runtime::Module> external_mods =
        mod->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});
    external_mods.push_back(runtime_mod);

    // Capture the new constants.
    Map<String, runtime::NDArray> const_name_to_constant =
        mod->GetAttr<Map<String, runtime::NDArray>>(tvm::attr::kConstNameToConstant).value_or({});
    for (const auto& kv : codegen.const_name_to_constant()) {
      ICHECK_EQ(const_name_to_constant.count(kv.first), 0);
      const_name_to_constant.Set(kv.first, kv.second);
    }

    return WithAttrs(mod, {{tvm::attr::kExternalMods, external_mods},
                           {tvm::attr::kConstNameToConstant, const_name_to_constant}});
  };
  std::cout << "I think we are doing... something..." << std::endl;
  return tvm::transform::CreateModulePass(pass_func, 0, "N64CompilerImpl", {});
}

tvm::transform::Pass N64CompilerPass() {
  return transform::Sequential({transform::OutlineCompilerFunctionsWithExistingGlobalSymbols("n64"),
                                N64CompilerImpl(),
                                transform::MarkCompilerFunctionsAsExtern("n64")});
}

TVM_REGISTER_GLOBAL("relay.ext.n64").set_body_typed(N64CompilerImpl);
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
