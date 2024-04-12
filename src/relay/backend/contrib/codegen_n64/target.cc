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
#include <tvm/target/target.h>

#include "./codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief This external codegen target emits C/C++ with specific extensions for the N64
 *  - Patterns: None, functions must be explicitly marked as "Primitive" and "Compiler=n64".
 *  - Custom compiler: relay/backend/contrib/codegen_n64/codegen.cc
 */
TVM_REGISTER_TARGET_KIND("n64", kDLCPU)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true))
    .set_attr<relay::transform::FTVMRelayToTIR>(tvm::attr::kRelayToTIR, N64CompilerPass())
    // Value is prepended to every output CModule.
    .add_attr_option<String>("header", String(""));

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
