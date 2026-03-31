<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM Bazel Build System

Native Bazel BUILD files for Apache TVM with LLVM codegen support.

## Quick Start

### Option 1: Using bzlmod (recommended)

In your `MODULE.bazel`:

```python
bazel_dep(name = "tvm", version = "0.0.0")

# Point to your TVM checkout
local_path_override(
    module_name = "tvm",
    path = "path/to/tvm-bazel/utils/bazel",
)

# Configure TVM overlay
tvm_overlay = use_extension("@tvm//:extensions.bzl", "tvm_overlay")
tvm_overlay.configure(use_llvm = True)
use_repo(tvm_overlay, "tvm-project")

# Provide your hermetic LLVM (required)
bazel_dep(name = "llvm-project", version = "19.1.0")
# ... configure LLVM as needed
```

### Option 2: Using WORKSPACE (legacy)

See `WORKSPACE.bazel` for example configuration.

## Building

```bash
# Build runtime only
bazel build @tvm-project//:libtvm_runtime

# Build full compiler with LLVM
bazel build @tvm-project//:libtvm

# Build Python bindings
bazel build @tvm-project//python:tvm
```

## LLVM Requirement

TVM requires LLVM 15.0+. Users must provide their own hermetic LLVM dependency.
This allows for:

- Clean integration via standard Bazel deps
- No system LLVM dependency
- Proper version control and reproducibility

Example LLVM setup in MODULE.bazel:

```python
http_archive(
    name = "llvm-raw",
    build_file_content = "# Empty",
    sha256 = "...",
    strip_prefix = "llvm-project-...",
    urls = ["https://github.com/llvm/llvm-project/archive/llvmorg-19.1.0.tar.gz"],
)

llvm_project_overlay = use_extension("@llvm-raw//utils/bazel:extensions.bzl", "llvm_project_overlay")
llvm_project_overlay.configure(targets = ["X86", "AArch64"])
use_repo(llvm_project_overlay, "llvm-project")
```

## Structure

```
utils/bazel/
├── MODULE.bazel     # Bzlmod entry point
├── WORKSPACE.bazel  # Legacy workspace support
├── configure.bzl    # Repository rule for overlay
├── extensions.bzl   # Module extension
├── overlays/        # BUILD files overlaid onto TVM source
│   ├── BUILD.bazel  # Root: libtvm, libtvm_runtime
│   ├── src/
│   │   ├── runtime/BUILD.bazel
│   │   ├── ir/BUILD.bazel
│   │   └── target/llvm/BUILD.bazel
│   └── 3rdparty/
│       └── tvm-ffi/BUILD.bazel
└── patches/
    └── libinfo_bazel.patch  # Python library discovery fix
```

## Python Bindings

For Python bindings to work in Bazel's sandbox, apply the patch:

```bash
cd /path/to/tvm
git apply utils/bazel/patches/libinfo_bazel.patch
```

This adds RUNFILES_DIR support for finding shared libraries.

## Configuration Options

The `tvm_overlay.configure()` tag accepts:

- `use_llvm`: Enable LLVM codegen (default: True)
- `use_rpc`: Enable RPC support (default: True)

## Contributing

This Bazel overlay is designed to be upstream-friendly. The BUILD files live
in `utils/bazel/overlays/` to minimise diff with upstream TVM.
