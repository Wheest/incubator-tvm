# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Module extension for configuring TVM."""

load(":configure.bzl", "tvm_configure")

def _tvm_configure_extension_impl(ctx):
    """Configure the TVM project overlay."""
    # Get configuration from tags
    use_llvm = True
    use_rpc = True
    tvm_source = None
    tvm_ffi_source = None
    dlpack_source = None
    libbacktrace_source = None

    for module in ctx.modules:
        for config in module.tags.configure:
            if hasattr(config, "use_llvm"):
                use_llvm = config.use_llvm
            if hasattr(config, "use_rpc"):
                use_rpc = config.use_rpc
            # Labels resolved in caller's context where repos are visible
            if hasattr(config, "tvm_source") and config.tvm_source:
                tvm_source = config.tvm_source
            if hasattr(config, "tvm_ffi_source") and config.tvm_ffi_source:
                tvm_ffi_source = config.tvm_ffi_source
            if hasattr(config, "dlpack_source") and config.dlpack_source:
                dlpack_source = config.dlpack_source
            if hasattr(config, "libbacktrace_source") and config.libbacktrace_source:
                libbacktrace_source = config.libbacktrace_source

    tvm_configure(
        name = "tvm-project",
        use_llvm = use_llvm,
        use_rpc = use_rpc,
        tvm_source = tvm_source,
        tvm_ffi_source = tvm_ffi_source,
        dlpack_source = dlpack_source,
        libbacktrace_source = libbacktrace_source,
    )

tvm_overlay = module_extension(
    doc = """Configure the TVM project.

    This extension creates a configured TVM repository with BUILD files
    overlaid on the TVM source tree.

    Example usage in MODULE.bazel (local mode):

        tvm_overlay = use_extension("@tvm//:extensions.bzl", "tvm_overlay")
        tvm_overlay.configure(use_llvm = True)
        use_repo(tvm_overlay, "tvm-project")

    Remote mode (http_archive doesn't fetch submodules, so fetch separately):

        http_archive(name = "tvm-raw", ...)
        http_archive(name = "tvm-ffi-raw", ...)
        http_archive(name = "dlpack-raw", ...)
        http_archive(name = "libbacktrace-raw", ...)

        tvm_overlay = use_extension("@tvm//:extensions.bzl", "tvm_overlay")
        tvm_overlay.configure(
            use_llvm = True,
            tvm_source = "@tvm-raw//:LICENSE",
            tvm_ffi_source = "@tvm-ffi-raw//:LICENSE",
            dlpack_source = "@dlpack-raw//:LICENSE",
            libbacktrace_source = "@libbacktrace-raw//:LICENSE",
        )
        use_repo(tvm_overlay, "tvm-project")
    """,
    implementation = _tvm_configure_extension_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {
                "use_llvm": attr.bool(default = True),
                "use_rpc": attr.bool(default = True),
                "tvm_source": attr.label(
                    default = None,
                    doc = "Label to a file in TVM source repo (for remote mode).",
                ),
                "tvm_ffi_source": attr.label(
                    default = None,
                    doc = "Label to a file in tvm-ffi repo (for remote mode).",
                ),
                "dlpack_source": attr.label(
                    default = None,
                    doc = "Label to a file in dlpack repo (for remote mode).",
                ),
                "libbacktrace_source": attr.label(
                    default = None,
                    doc = "Label to a file in libbacktrace repo (for remote mode).",
                ),
            },
        ),
    },
)
