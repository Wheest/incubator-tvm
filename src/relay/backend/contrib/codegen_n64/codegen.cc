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

#include "../../../../target/source/codegen_c_host.h"
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
                   public CodegenN64Base {
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

    ICHECK(dtype == "float" || dtype == "int32_t")
        << "Only int32_t is supported for now (your type:" << dtype << ")";
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

    if (backend::IsOp(call, "qnn.conv2d")) {
      return Conv2d(call);
    } else if (backend::IsOp(call, "nn.conv2d")) {
      return Conv2d(call);
      // LOG(FATAL) << "Not quantized conv2d is not supported for RSP (yet!)";
    }

    std::string func_name = ext_func_id_ + "_" + std::to_string(func_idx++);

    // Make function declaration
    LOG(FATAL) << "Unrecognized op";

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

  std::tuple<int, int, int, int> get_partition_size(std::vector<int> in_shape,
                                                    std::tuple<int, int, int, int> padding,
                                                    int stride, int out_w, const int in_dbytes) {
    const int in_w = in_shape[2];
    const int in_h = in_shape[1];
    const int out_dbytes = 4;
    const int kdim_h = 3;
    const int kdim_w = 3;
    const int kmem = kdim_h * kdim_w * 8 * in_dbytes;
    const int overlap = kdim_w - stride;
    const int pad_t = std::get<0>(padding);
    const int pad_l = std::get<1>(padding);
    const int pad_b = std::get<2>(padding);
    const int pad_r = std::get<3>(padding);
    const int input_part_width_base = in_w + (pad_l + pad_r);
    // const int overhead = 544;  // Overhead of other data structures in the RSP memory
    const int overhead = 700;
    // Overhead of other data structures in the RSP memory
    // const int overhead = 2500;
    // Overhead of other data structures in the RSP memory
    const int max_mem = (4 * 1024) - overhead;  // Total available memory - overhead

    const int max_height = in_h + pad_t + pad_b;
    int input_part_height = 3;                     // start at 3 (kh) and increase
    int input_part_width = input_part_width_base;  // Default to the full width
    int output_part_width = out_w;

    const int out_w_test = (in_w + pad_l + pad_r - kdim_w) / stride + 1;
    if (out_w_test != out_w) {
      LOG(FATAL) << "Output width is not correct, likely a padding issue (should be " << out_w
                 << " but is " << out_w_test << ", padding is " << pad_t << ", " << pad_b << ", "
                 << pad_l << ", " << pad_r << ")";
    }
    int input_mem = input_part_height * input_part_width * 8 * in_dbytes;
    int output_part_height = (input_part_height - kdim_h + pad_t + pad_r) / stride + 1;
    int omem = output_part_height * output_part_width * 8 * out_dbytes;

    int spare_room = max_mem - (kmem + omem);

    if (input_mem > spare_room) {
      int split_factor = 1;
      // If we can't fit even one horizontal strip of input in the RSP memory
      // then we need to reduce the width of the input partition
      do {
        split_factor++;
        // input_part_width = (input_part_width_base + overlap) / split_factor;
        input_part_width = static_cast<int>(
            std::floor((input_part_width_base + overlap) / static_cast<double>(split_factor)));
        output_part_width = (input_part_width - kdim_w) / stride + 1;
        input_mem = input_part_height * input_part_width * 8 * in_dbytes;
        output_part_height = (input_part_height - kdim_h + pad_t + pad_r) / stride + 1;
        omem = output_part_height * output_part_width * 8 * out_dbytes;
        spare_room = max_mem - (kmem + omem);
      } while (input_mem > spare_room);
    }

    int loop_count = 0;
    while ((kmem + input_mem + omem) <= max_mem && input_part_height < max_height) {
      input_part_height += stride;
      if (input_part_height > max_height) {
        input_part_height = max_height;
      }
      input_mem = input_part_height * input_part_width * 8 * in_dbytes;
      output_part_height = (std::min((max_height), input_part_height) - kdim_h) / stride + 1;
      omem = output_part_height * output_part_width * 8 * out_dbytes;
      loop_count++;
      if (loop_count > 1000) {
        LOG(FATAL) << "Loop count exceeded 1000";
      }
    }

    std::cout << "Initial max height: " << input_part_height << std::endl;

    // walk-back if we over-did it
    loop_count = 0;
    while ((kmem + input_mem + omem) > max_mem) {
      input_part_height -= stride;
      if (input_part_height > max_height) {
        input_part_height = max_height;
      }
      input_mem = input_part_height * input_part_width * 8 * in_dbytes;
      output_part_height = (std::min(max_height, input_part_height) - kdim_h) / stride + 1;
      omem = output_part_height * output_part_width * 8 * out_dbytes;
      if (loop_count > 1000) {
        LOG(FATAL) << "Loop count exceeded 1000";
      }
    }

    std::cout << "Initial max height: " << input_part_height << std::endl;

    // Check if our h partitioning could be smaller (i.e. our last partition could have a lot of
    // useless computation)
    auto num_h_partitions = static_cast<int>(std::ceil(
        static_cast<float>(in_h + pad_t + pad_r - overlap) / (input_part_height - overlap)));
    auto num_h_partitions_2 = static_cast<int>(std::ceil(
        static_cast<float>(in_h + pad_t + pad_r - overlap) / (input_part_height - 1 - overlap)));

    std::cout << "Num h partitions: " << num_h_partitions << std::endl;
    if (num_h_partitions == num_h_partitions_2) {
      // get the smallest partition size which keeps the same number of partitions
      input_part_height =
          static_cast<int>(std::floor(static_cast<float>(max_height / num_h_partitions)) + overlap);
      input_part_height = std::min(input_part_height, max_height);

      input_mem = input_part_height * input_part_width * 8 * in_dbytes;
      output_part_height = (std::min(max_height, input_part_height) - kdim_h) / stride + 1;
      omem = output_part_height * output_part_width * 8 * out_dbytes;
    }

    // Example output to verify variables at the end
    std::cout << "Final max height: " << input_part_height << std::endl;
    std::cout << "Final output_height: " << output_part_height << std::endl;
    std::cout << "Final input memory: " << input_mem << std::endl;
    std::cout << "Final output memory: " << omem << std::endl;
    std::cout << "Final total memory: " << kmem + input_mem + omem << std::endl;
    if ((kmem + input_mem + omem) > max_mem) {
      LOG(FATAL) << "Memory usage exceeds maximum available memory";
    } else if (input_part_height < 3) {
      LOG(FATAL) << "Input partition height is less than 3";
    }
    if (input_part_height < 3) {
      LOG(FATAL) << "Input partition height is less than 3";
    }
    if (input_part_width < 3) {
      LOG(FATAL) << "Input partition width is less than 3";
    }
    if (input_part_height > (in_h + pad_t + pad_b)) {
      LOG(FATAL) << "Input partition height exceeds input height";
    }
    if (input_part_width > (in_w + pad_l + pad_r)) {
      LOG(FATAL) << "Input partition width exceeds input width";
    }

    return std::make_tuple(input_part_height, input_part_width, output_part_height,
                           output_part_width);
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

    // Extract the quantization params from the arguments
    int input_zero_point;
    int kernel_zero_point;
    int output_zero_point;
    AsConstant(call->args[2], &input_zero_point);
    AsConstant(call->args[3], &kernel_zero_point);
    AsConstant(call->args[4], &output_zero_point);
    input_zero_point *= -1;  // Negate the zero point

    auto padding = std::tuple{conv_attr->padding[0].as<IntImmNode>()->value,
                              conv_attr->padding[1].as<IntImmNode>()->value,
                              conv_attr->padding[2].as<IntImmNode>()->value,
                              conv_attr->padding[3].as<IntImmNode>()->value};

    int stride_h = qnn::get_const_int(conv_attr->strides[0]);
    int stride_w = qnn::get_const_int(conv_attr->strides[1]);

    if (stride_h != stride_w) {
      LOG(FATAL) << "Asymetric strides are not supported for RSP (yet!)";
    }

    // Make function declaration
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

    // if any paddings are non-zero
    // TODO always true
    if (true || std::get<0>(padding) != 0 || std::get<1>(padding) != 0 ||
        std::get<2>(padding) != 0 || std::get<3>(padding) != 0) {
      std::tuple<int, int, int, int> partitions =
          get_partition_size(in_shape, padding, stride_h, out_shape[2], 2);
      macro_stream << "CSOURCE_RSP_DEPTH_CONV2D_OP(" << func_name << ", ";
      macro_stream << "/*input_partition_height=*/" << std::get<0>(partitions)
                   << ", /*input_partition_width=*/" << std::get<1>(partitions)
                   << ", /*output_partition_height=*/" << std::get<2>(partitions)
                   << ", /*output_partition_width=*/" << std::get<3>(partitions) << ", /*in_h=*/"
                   << in_shape[1] << ", /*in_w=*/" << in_shape[2] << ", /*in_c=*/" << in_shape[3]
                   << ", /*out_h=*/" << out_shape[1] << ", /*out_w=*/" << out_shape[2]
                   << ", /*k_h=*/3, /*k_w=*/3, /*pad_t=*/" << std::get<0>(padding) << ", /*pad_b=*/"
                   << std::get<2>(padding) << ", /*pad_l=*/" << std::get<1>(padding)
                   << ", /*pad_r=*/" << std::get<3>(padding) << ", /*stride=*/" << stride_h
                   << ", /*input_zero_point=*/" << input_zero_point;
      macro_stream << ");";
    } else {
      std::tuple<int, int, int, int> partitions =
          get_partition_size(in_shape, padding, stride_h, out_shape[2], 1);
      macro_stream << "CSOURCE_RSP_DEPTH_CONV2D_NO_PAD_OP(" << func_name << ", ";
      macro_stream << "/*input_partition_height=*/" << std::get<0>(partitions)
                   << ", /*input_partition_width=*/" << std::get<1>(partitions)
                   << ", /*output_partition_height=*/" << std::get<2>(partitions)
                   << ", /*output_partition_width=*/" << std::get<3>(partitions) << ", /*in_h=*/"
                   << in_shape[1] << ", /*in_w=*/" << in_shape[2] << ", /*in_c=*/" << in_shape[3]
                   << ", /*out_h=*/" << out_shape[1] << ", /*out_w=*/" << out_shape[2]
                   << ", /*k_h=*/3, /*k_w=*/3, /*stride=*/" << stride_h << ", /*input_zero_point=*/"
                   << input_zero_point;
      macro_stream << ");";
    }
    func_decl_.push_back(macro_stream.str());

    // Make function call when visiting arguments
    bool first = true;
    decl_stream << func_name << "(";
    std::string out_buff = "out0";
    for (size_t i = 0; i < 2; ++i) {
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

  runtime::Module CreateN64SourceModule() {
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

#define CSOURCE_RSP_DEPTH_CONV2D_OP(                                                           \
    p_ID_, p_INPUT_PARTITION_HEIGHT_, p_INPUT_PARTITION_WIDTH_, p_OUTPUT_PARTITION_HEIGHT_,    \
    p_OUTPUT_PARTITION_WIDTH_, p_IN_H_, p_IN_W_, p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_,  \
    p_PAD_T_, p_PAD_B_, p_PAD_L_, p_PAD_R_, p_STRIDE_, p_INPUT_ZERO_POINT_)                    \
  void p_ID_(int8_t* input_data, int8_t* weights, int32_t* dest) {                             \
    RSPDepthConvTiledPadded(dest, input_data, weights, p_INPUT_PARTITION_HEIGHT_, p_IN_H_,     \
                            p_IN_W_, p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_,              \
                            p_OUTPUT_PARTITION_HEIGHT_, p_INPUT_PARTITION_WIDTH_,              \
                            p_OUTPUT_PARTITION_WIDTH_, p_PAD_T_, p_PAD_B_, p_PAD_L_, p_PAD_R_, \
                            p_STRIDE_, p_INPUT_ZERO_POINT_);                                   \
  }

#define CSOURCE_RSP_DEPTH_CONV2D_NO_PAD_OP(                                                    \
    p_ID_, p_INPUT_PARTITION_HEIGHT_, p_INPUT_PARTITION_WIDTH_, p_OUTPUT_PARTITION_HEIGHT_,    \
    p_OUTPUT_PARTITION_WIDTH_, p_IN_H_, p_IN_W_, p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_,  \
    p_STRIDE_, p_INPUT_ZERO_POINT_)                                                            \
  void p_ID_(int8_t* input_data, int8_t* weights, int32_t* dest) {                             \
    RSPDepthConvTiled(dest, input_data, weights, p_INPUT_PARTITION_HEIGHT_, p_IN_H_, p_IN_W_,  \
                      p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_, p_OUTPUT_PARTITION_HEIGHT_, \
                      p_INPUT_PARTITION_WIDTH_, p_OUTPUT_PARTITION_WIDTH_, p_STRIDE_,          \
                      p_INPUT_ZERO_POINT_);                                                    \
  }

    )op_macro";

    os << operator_macro << "\n\n";

    Conv2dRSPHelpers(os);
  }

  void Conv2dRSPHelpers(std::ostringstream& os) {
    // Find code in depth_conv.c and copy it here
    const char* c_depth_conv_code = R"(
#define MIN(a, b) ((a) < (b) ? (a) : (b))

extern uint32_t vec_id;
enum {
  DMAWeights = 0x0,
  SetArgs = 0x1,
  DMAInputs = 0x2,
  DepthConvPad = 0x3,
  DepthConv = 0x4,
};

void copy_output_slice_to_full(int32_t* dest, const int32_t* output_pad_part,
                               const int output_partition_height, const int output_partition_width,
                               const int out_w, const int h_slice_num, const int w_slice_num,
                               const int in_depth, const int depth_slice_num,
                               const int max_output_partition_height) {
  /*
  Copies a slice into the final destination with the appropriate stride
  using for-loops.
  */
  const int slice_depth = 8;  // 8 channels per slice
  for (int y = 0; y < max_output_partition_height; y++) {
    for (int x = 0; x < output_partition_width; x++) {
      // Calculate the initial destination index with offsets and
      // strided gaps
      int dest_idx_y = h_slice_num * output_partition_height + y;
      int dest_idx_x = w_slice_num * output_partition_width + x;

      int32_t* dest_ptr = &dest[(dest_idx_y * out_w * in_depth) + (dest_idx_x * in_depth) +
                                (depth_slice_num * slice_depth)];
      // cast to uint16_t as the RSP puts the upper 16 bits of all 8
      // elements in the first 64 bits and the lower 16 bits in the
      // second 64 bits
      uint16_t* src_ptr =
          (uint16_t*)&output_pad_part[y * output_partition_width * slice_depth + x * slice_depth];
      for (int c = 0; c < slice_depth; c++) {
        dest_ptr[c] = ((uint32_t)src_ptr[c] << 16) |
                      src_ptr[c + 8];  // reconstruct from upper and lower 2 bytes
      }
    }
  }
}

void generate_padded_slices_with_depth_slice(int8_t* data, int16_t* padded_input_partition,
                                             const int h_slice_num, const int w_slice_num,
                                             const int depth_slice_num, const int in_h,
                                             const int in_w, const int in_depth,
                                             const int slice_height, const int slice_width,
                                             const int pad_t, const int pad_l, const int overlap,
                                             const int input_zero_point) {
  // start_[h/w]_pad are the starting indices of the slice in the
  // (virtual) fully padded input data.  We then need to deterimine the
  // starting indices in the actual input data, which may be less than
  // the padded input data.
  typedef int16_t input_slice_t;
  const int start_h_pad = h_slice_num * (slice_height - overlap);
  const int start_w_pad = w_slice_num * (slice_width - overlap);

  // The starting indices of the non-padded values in the slice
  const int slice_start_h = (start_h_pad == 0) ? pad_t : 0;
  const int slice_start_w = (start_w_pad == 0) ? pad_l : 0;
  const int num_h_elems = slice_height - slice_start_h;
  const int num_w_elems = slice_width - slice_start_w;

  // Calculate depth start and end based on the depth_slice
  const int depth_start = depth_slice_num * 8;           // Start at the depth slice of 8 channels
  const int depth_end = MIN(depth_start + 8, in_depth);  // Ensure we don't go beyond in_depth

  // Clear the slice area
  memset(padded_input_partition, 0, sizeof(input_slice_t) * slice_height * slice_width * 8);

  // The starting indices in the full non-padded input data
  int h_start = (start_h_pad == 0) ? start_h_pad : start_h_pad - pad_t;
  int h_end = MIN(h_start + num_h_elems, in_h);

  int w_start = (start_w_pad == 0) ? start_w_pad : start_w_pad - pad_l;
  int w_end = MIN(w_start + num_w_elems, in_w);

  // Copy data to the padded slice
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      for (int d = depth_start; d < depth_end; ++d) {
        input_slice_t value = (input_slice_t)data[(h * in_w * in_depth) + (w * in_depth) + d];
        value += input_zero_point;
        int slice_row = slice_start_h + (h - h_start);
        int slice_col = slice_start_w + (w - w_start);
        int slice_depth = d - depth_start;
        padded_input_partition[(slice_row * slice_width * 8) + (slice_col * 8) + slice_depth] =
            value;
      }
    }
  }
}

void generate_slices_with_depth_slice(int8_t* data, int16_t* padded_input_partition,
                                      const int h_slice_num, const int w_slice_num,
                                      const int depth_slice_num, const int in_h, const int in_w,
                                      const int in_depth, const int slice_height,
                                      const int slice_width, const int overlap) {
  // No padded version
  // start_[h/w]_pad are the starting indices of the slice in the
  // (virtual) fully padded input data.  We then need to deterimine the
  // starting indices in the actual input data, which may be less than
  // the padded input data.
  typedef int8_t input_slice_t;
  const int pad_t, pad_l = 0;
  const int start_h_pad = h_slice_num * (slice_height - overlap);
  const int start_w_pad = w_slice_num * (slice_width - overlap);

  // The starting indices of the non-padded values in the slice
  const int slice_start_h = (start_h_pad == 0) ? pad_t : 0;
  const int slice_start_w = (start_w_pad == 0) ? pad_l : 0;
  const int num_h_elems = slice_height - slice_start_h;
  const int num_w_elems = slice_width - slice_start_w;

  // Calculate depth start and end based on the depth_slice
  const int depth_start = depth_slice_num * 8;           // Start at the depth slice of 8 channels
  const int depth_end = MIN(depth_start + 8, in_depth);  // Ensure we don't go beyond in_depth

  // Clear the slice area
  memset(padded_input_partition, 0, sizeof(input_slice_t) * slice_height * slice_width * 8);

  // The starting indices in the full non-padded input data
  int h_start = (start_h_pad == 0) ? start_h_pad : start_h_pad - pad_t;
  int h_end = MIN(h_start + num_h_elems, in_h);

  int w_start = (start_w_pad == 0) ? start_w_pad : start_w_pad - pad_l;
  int w_end = MIN(w_start + num_w_elems, in_w);

  // Copy data to the padded slice
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      for (int d = depth_start; d < depth_end; ++d) {
        input_slice_t value = (input_slice_t)data[(h * in_w * in_depth) + (w * in_depth) + d];
        int slice_row = slice_start_h + (h - h_start);
        int slice_col = slice_start_w + (w - w_start);
        int slice_depth = d - depth_start;
        padded_input_partition[(slice_row * slice_width * 8) + (slice_col * 8) + slice_depth] =
            value;
      }
    }
  }
}

static inline void RSPDepthConvTiledPadded(
    int32_t* dest, int8_t* input_data, int8_t* weights, const int input_partition_height,
    const int in_h, const int in_w, const int in_c, const int out_h, const int out_w, const int k_h,
    const int k_w, const int output_partition_height, const int input_partition_width,
    const int output_partition_width, const int pad_t, const int pad_b, const int pad_l,
    const int pad_r, const int stride, const int input_zero_point) {
  // pad_r and pad_b are implicit from the
  // input_partition_{height,width}

  // Requires that weights have been reshaped offline to be
  // (out_c // 8, kernel_height * kernel_width, 8)
  extern uint32_t vec_id;

  // raise not implemented error if in_c is not 8
  if (in_c % 8 != 0) {
    printf("Error: in_c must be divisible 8\n");
    return;
  }

  /* printf("Weights reshape\n"); */
  /* printInt8ArrayHWC(weights, 3, 3, 8); */

  const int wbytes = sizeof(int8_t);
  // TODO we are doing int16 conversion on CPU for now
  // because of input_zero_point and how it interacts with padded values
  // The non-padded version should be faster
  typedef int16_t input_slice_t;
  const int in_bytes = sizeof(input_slice_t);
  const int out_bytes = sizeof(int32_t);

  const int in_part_size = input_partition_height * input_partition_width * 8 * in_bytes;
  input_slice_t* input_pad_part = malloc_uncached_aligned(8, in_part_size);

  const int out_part_size = output_partition_height * output_partition_width * 8;
  int32_t* output_pad_part = malloc_uncached_aligned(8, out_part_size * out_bytes);

  const int w_part_size = k_h * k_w * 8 * wbytes;

  const int overlap = k_h - stride;
  // the number of bytes to offset our input data pointer by between
  // elements in the same window.  I.e., if our pointer is at 0, how
  // does it get to 5?
  // |*0, 1, 2,| 3, 4
  // | 5, 6, 7,| 8, 9
  // |10,11,12,|13,14
  //  15,16,17,18,19
  const int w_slide_byte_offset = (8 * in_bytes * input_partition_width);

  // when we slide right for a new window, the number of bytes to offset
  // our input data pointer
  const int w_window_stride = 8 * in_bytes * stride;
  // when we slide  for a new window, the number of bytes to offset our
  // input data pointer
  const int h_window_stride = (8 * in_bytes) *                  /*account for our depth of 8*/
                              ((input_partition_width * stride) /*move the pointer vertically down*/
                               - /*move the pointer back left to the start of the row*/
                               (output_partition_width * stride));

  rspq_write(vec_id, SetArgs, output_partition_height, output_partition_width, w_window_stride,
             w_slide_byte_offset, h_window_stride, w_part_size, in_part_size);

  const int num_h_partitions =
      ceil((float)(in_h + pad_t + pad_b - overlap) / (input_partition_height - overlap));

  const int num_w_partitions =
      ceil((float)(in_w + pad_l + pad_r - overlap) / (input_partition_width - overlap));

  for (int depth_slice = 0; depth_slice < in_c / 8; depth_slice++) {
    int remaining_out_values = out_h * out_w * 8;  // Remaining values to copy for this depth slice
    int max_output_partition_height = output_partition_height;

    // Copy weights for this depth slice to the RSP once
    rspq_write(vec_id, DMAWeights, PhysicalAddr(&weights[depth_slice * k_h * k_w * 8]),
               w_part_size);

    /* printf("Weight slice %d: \n", depth_slice); */
    /* printInt8ArrayHWC(&weights[depth_slice * k_h * k_w * 8], k_h,
     * k_w, 8); */

    // Generate first input  slice
    generate_padded_slices_with_depth_slice(
        input_data, input_pad_part, 0, 0, depth_slice, in_h, in_w, in_c, input_partition_height,
        input_partition_width, pad_t, pad_l, overlap, input_zero_point);

    // Copy-in the first slice
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);

    for (int h_slice_count = 0; h_slice_count < num_h_partitions; h_slice_count++) {
      for (int w_slice_count = 0; w_slice_count < num_w_partitions; w_slice_count++) {
        rspq_wait();

        /* printf("Input slice %d, %d : \n", h_slice_count,
         * w_slice_count); */
        /* printInt8ArrayHWC(input_pad_part, input_partition_height, */
        /*                   input_partition_width, 8); */

        // Process the padded partition on the RSP
        rspq_write(vec_id, DepthConvPad, PhysicalAddr(output_pad_part), out_part_size * out_bytes);

        // Generate next slice while the current one is being
        // processed
        if ((w_slice_count + 1) < num_w_partitions) {
          generate_padded_slices_with_depth_slice(input_data, input_pad_part, h_slice_count,
                                                  w_slice_count + 1, depth_slice, in_h, in_w, in_c,
                                                  input_partition_height, input_partition_width,
                                                  pad_t, pad_l, overlap, input_zero_point);
        } else if ((h_slice_count + 1) < num_h_partitions) {
          generate_padded_slices_with_depth_slice(input_data, input_pad_part, h_slice_count + 1, 0,
                                                  depth_slice, in_h, in_w, in_c,
                                                  input_partition_height, input_partition_width,
                                                  pad_t, pad_l, overlap, input_zero_point);
        }

        // Copy back the processed partition to the final outputs,
        // DMA the new input slice to the RSP at the same time
        rspq_wait();
        rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);
        /* printf("DMA inputs again!\n"); */

        copy_output_slice_to_full(dest, output_pad_part, output_partition_height,
                                  output_partition_width, out_w, h_slice_count, w_slice_count, in_c,
                                  depth_slice, max_output_partition_height);

        remaining_out_values -= output_partition_height * output_partition_width * 8;
      }

      // May need to adjust the max_output_partition_height for the
      // final slice(s)
      if (remaining_out_values < (out_part_size * num_w_partitions)) {
        // Cover the case where our final output slice is larger than
        // required
        max_output_partition_height = remaining_out_values / (out_w * 8);
      }
    }
  }
  free_uncached(input_pad_part);
  free_uncached(output_pad_part);
}

static inline void RSPDepthConvTiled(
    int32_t* dest, int8_t* input_data, int8_t* weights, const int input_partition_height,
    const int in_h, const int in_w, const int in_c, const int out_h, const int out_w, const int k_h,
    const int k_w, const int output_partition_height, const int input_partition_width,
    const int output_partition_width, const int stride, const int input_zero_point) {
  // Requires that weights have been reshaped offline to be
  // (out_c // 8, kernel_height * kernel_width, 8)
  extern uint32_t vec_id;

  // raise not implemented error if in_c is not 8
  if (in_c % 8 != 0) {
    printf("Error: in_c must be divisible 8\n");
    return;
  }

  /* printf("Weights reshape\n"); */
  /* printInt8ArrayHWC(weights, 3, 3, 8); */

  const int wbytes = sizeof(int8_t);

  typedef int8_t input_slice_t;
  const int in_bytes = sizeof(input_slice_t);
  const int out_bytes = sizeof(int32_t);

  const int in_part_size = input_partition_height * input_partition_width * 8 * in_bytes;
  input_slice_t* input_pad_part = malloc_uncached_aligned(8, in_part_size);

  const int out_part_size = output_partition_height * output_partition_width * 8;
  int32_t* output_pad_part = malloc_uncached_aligned(8, out_part_size * out_bytes);

  const int w_part_size = k_h * k_w * 8 * wbytes;

  const int overlap = k_h - stride;
  // the number of bytes to offset our input data pointer by between
  // elements in the same window.  I.e., if our pointer is at 0, how
  // does it get to 5?
  // |*0, 1, 2,| 3, 4
  // | 5, 6, 7,| 8, 9
  // |10,11,12,|13,14
  //  15,16,17,18,19
  const int w_slide_byte_offset = (8 * in_bytes * input_partition_width);

  // when we slide right for a new window, the number of bytes to offset
  // our input data pointer
  const int w_window_stride = 8 * in_bytes * stride;
  // when we slide  for a new window, the number of bytes to offset our
  // input data pointer
  const int h_window_stride = (8 * in_bytes) *                  /*account for our depth of 8*/
                              ((input_partition_width * stride) /*move the pointer vertically down*/
                               - /*move the pointer back left to the start of the row*/
                               (output_partition_width * stride));

  rspq_write(vec_id, SetArgs, output_partition_height, output_partition_width, w_window_stride,
             w_slide_byte_offset, h_window_stride, w_part_size, in_part_size);

  const int num_h_partitions = ceil((float)(in_h - overlap) / (input_partition_height - overlap));

  const int num_w_partitions = ceil((float)(in_w - overlap) / (input_partition_width - overlap));

  for (int depth_slice = 0; depth_slice < in_c / 8; depth_slice++) {
    int remaining_out_values = out_h * out_w * 8;  // Remaining values to copy for this depth slice
    int max_output_partition_height = output_partition_height;

    // Copy weights for this depth slice to the RSP once
    rspq_write(vec_id, DMAWeights, PhysicalAddr(&weights[depth_slice * k_h * k_w * 8]),
               w_part_size);

    /* printf("Weight slice %d: \n", depth_slice); */
    /* printInt8ArrayHWC(&weights[depth_slice * k_h * k_w * 8], k_h,
     * k_w, 8); */

    // Generate first input  slice
    generate_slices_with_depth_slice(input_data, input_pad_part, 0, 0, depth_slice, in_h, in_w,
                                     in_c, input_partition_height, input_partition_width, overlap);

    // Copy-in the first slice
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);

    for (int h_slice_count = 0; h_slice_count < num_h_partitions; h_slice_count++) {
      for (int w_slice_count = 0; w_slice_count < num_w_partitions; w_slice_count++) {
        rspq_wait();

        /* printf("Input slice %d, %d : \n", h_slice_count,
         * w_slice_count); */
        /* printInt8ArrayHWC(input_pad_part, input_partition_height, */
        /*                   input_partition_width, 8); */

        // Process the padded partition on the RSP
        rspq_write(vec_id, DepthConv, PhysicalAddr(output_pad_part), out_part_size * out_bytes,
                   input_zero_point);

        // Generate next slice while the current one is being
        // processed
        if ((w_slice_count + 1) < num_w_partitions) {
          generate_slices_with_depth_slice(input_data, input_pad_part, h_slice_count,
                                           w_slice_count + 1, depth_slice, in_h, in_w, in_c,
                                           input_partition_height, input_partition_width, overlap);
        } else if ((h_slice_count + 1) < num_h_partitions) {
          generate_slices_with_depth_slice(input_data, input_pad_part, h_slice_count + 1, 0,
                                           depth_slice, in_h, in_w, in_c, input_partition_height,
                                           input_partition_width, overlap);
        }

        // Copy back the processed partition to the final outputs,
        // DMA the new input slice to the RSP at the same time
        rspq_wait();
        rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);
        /* printf("DMA inputs again!\n"); */

        copy_output_slice_to_full(dest, output_pad_part, output_partition_height,
                                  output_partition_width, out_w, h_slice_count, w_slice_count, in_c,
                                  depth_slice, max_output_partition_height);

        remaining_out_values -= output_partition_height * output_partition_width * 8;
      }

      // May need to adjust the max_output_partition_height for the
      // final slice(s)
      if (remaining_out_values < (out_part_size * num_w_partitions)) {
        // Cover the case where our final output slice is larger than
        // required
        max_output_partition_height = remaining_out_values / (out_w * 8);
      }
    }
  }
  free_uncached(input_pad_part);
  free_uncached(output_pad_part);
}

)";
    os << c_depth_conv_code << "\n\n";
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
    ICHECK(pf != nullptr) << "Cannot find N64source module to create the external runtime module";

    return codegen::CSourceModuleCreate(code, "c", Array<String>{func_names_});
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
    Target target = GetN64CompilerTarget();

    // Emit the C/C++ code and package it as a CSourceModule.
    CodegenN64Module codegen(target, mod);
    runtime::Module runtime_mod = codegen.CreateN64SourceModule();

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
