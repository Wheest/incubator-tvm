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
"""TVM repository configuration rule.

This creates an overlay repository that combines TVM source with Bazel BUILD files.
"""

# Directories where we need to place BUILD files (within src/)
_SRC_OVERLAY_DIRS = [
    "runtime",
    "support",
    "ir",
    "node",
    "arith",
    "te",
    "tirx",
    "s_tir",
    "topi",
    "script",
    "relax",
    "target",
    "target/llvm",
]

def _tvm_configure_impl(repository_ctx):
    """Implementation of the tvm_configure repository rule."""
    use_llvm = repository_ctx.attr.use_llvm
    use_rpc = repository_ctx.attr.use_rpc

    # Find TVM source root
    if repository_ctx.attr.tvm_source:
        # Remote mode (explicit): label passed from extension points to TVM source
        tvm_root = repository_ctx.path(repository_ctx.attr.tvm_source).dirname
    else:
        # Local mode OR loaded from @tvm-raw context
        # When loaded from @tvm-raw, Label("//...") resolves relative to @tvm-raw
        # When loaded from @tvm (local), we need to walk up to find TVM root

        # First, find where this rule was loaded from
        # We use //:LICENSE as a marker - if it exists, we're in TVM source root
        module_path = repository_ctx.path(Label("//:MODULE.bazel")).dirname

        # Check if we're at TVM source root (has src/, include/, LICENSE)
        if module_path.get_child("src").exists and module_path.get_child("include").exists:
            # Loaded from @tvm-raw or similar - we're already at TVM root
            tvm_root = module_path
        else:
            # Loaded from @tvm (utils/bazel) - walk up to TVM root
            tvm_bazel_path = module_path
            result = repository_ctx.execute(["readlink", str(tvm_bazel_path)])
            if result.return_code == 0:
                real_bazel_path = result.stdout.strip()
                tvm_root = repository_ctx.path(real_bazel_path).dirname.dirname
            else:
                tvm_root = tvm_bazel_path.dirname.dirname

    # Get overlay path - depends on how we were loaded
    # When loaded from @tvm-raw: overlays at //:utils/bazel/overlays/
    # When loaded from @tvm: overlays at @tvm//:overlays/
    if tvm_root.get_child("utils").get_child("bazel").get_child("overlays").exists:
        # Loaded from @tvm-raw context
        overlay_path = tvm_root.get_child("utils").get_child("bazel").get_child("overlays")
    else:
        # Loaded from @tvm module
        overlay_path = repository_ctx.path(Label("@tvm//:overlays/BUILD.bazel")).dirname

    # Symlink top-level files
    for item in ["LICENSE", "NOTICE"]:
        src = tvm_root.get_child(item)
        if src.exists:
            repository_ctx.symlink(src, item)

    # Symlink include and python directories directly
    for item in ["include", "python"]:
        src = tvm_root.get_child(item)
        if src.exists:
            repository_ctx.symlink(src, item)

    # Set up src/ directory
    _setup_src_directory(repository_ctx, tvm_root, overlay_path)

    # Set up 3rdparty/ directory - special handling for tvm-ffi
    _setup_3rdparty_directory(repository_ctx, tvm_root, overlay_path)

    # Root BUILD.bazel
    root_build = overlay_path.get_child("BUILD.bazel")
    repository_ctx.symlink(root_build, "BUILD.bazel")

    # Generate config header
    repository_ctx.file(
        "tvm_config.h",
        content = _generate_config_header(use_llvm, use_rpc),
    )

def _setup_src_directory(repository_ctx, tvm_root, overlay_path):
    """Set up src/ directory with symlinked sources and BUILD files.

    For directories that need BUILD files, we symlink individual files
    rather than the whole directory, so we can add our BUILD.bazel.
    """
    src_root = tvm_root.get_child("src")
    if not src_root.exists:
        return

    # Build set of top-level directories that need BUILD files
    top_level_overlay_dirs = {}
    nested_overlay_dirs = {}
    for d in _SRC_OVERLAY_DIRS:
        if "/" in d:
            parts = d.split("/")
            top_level_overlay_dirs[parts[0]] = True
            nested_overlay_dirs[d] = True
        else:
            top_level_overlay_dirs[d] = True

    # Process each subdirectory in src/
    for entry in src_root.readdir():
        dest = "src/" + entry.basename
        if entry.basename in top_level_overlay_dirs:
            # This directory needs a BUILD file - symlink contents individually
            _symlink_dir_contents_flat(repository_ctx, entry, dest, entry.basename, nested_overlay_dirs)
        else:
            # No BUILD file needed - symlink the whole directory
            repository_ctx.symlink(entry, dest)

    # Create src/src symlink for TVM's broken relative includes like "../../src/arith/..."
    # These resolve to src/src/arith/... from src/tirx/transform/, so we need src/src -> src
    repository_ctx.symlink(src_root, "src/src")

    # Add BUILD files for src subdirectories
    for subdir in _SRC_OVERLAY_DIRS:
        overlay_build = overlay_path.get_child("src").get_child(subdir).get_child("BUILD.bazel")
        if overlay_build.exists:
            content = repository_ctx.read(overlay_build)
            dest = "src/" + subdir + "/BUILD.bazel"
            repository_ctx.file(dest, content)

def _symlink_dir_contents_flat(repository_ctx, src_dir, dest_dir, parent_name, nested_dirs):
    """Symlink directory contents, handling one level of nesting for nested overlay dirs."""
    for entry in src_dir.readdir():
        dest = dest_dir + "/" + entry.basename
        # Skip any existing BUILD files in source
        if entry.basename == "BUILD.bazel" or entry.basename == "BUILD":
            continue
        # Check if this is a nested subdir that needs a BUILD file (e.g., target/llvm)
        nested_key = parent_name + "/" + entry.basename
        if nested_key in nested_dirs:
            # Symlink contents of this nested dir
            for nested_entry in entry.readdir():
                if nested_entry.basename not in ["BUILD.bazel", "BUILD"]:
                    repository_ctx.symlink(nested_entry, dest + "/" + nested_entry.basename)
        else:
            repository_ctx.symlink(entry, dest)

def _setup_3rdparty_directory(repository_ctx, tvm_root, overlay_path):
    """Set up 3rdparty/ directory with special handling for tvm-ffi and its deps.

    For remote mode, tvm-ffi and its submodules (dlpack, libbacktrace) can be
    provided as separate http_archive repos via the tvm_ffi_source, dlpack_source,
    and libbacktrace_source attributes.
    """
    third_party = tvm_root.get_child("3rdparty")
    if not third_party.exists:
        return

    # Symlink non-tvm-ffi directories directly
    for entry in third_party.readdir():
        if entry.basename != "tvm-ffi":
            repository_ctx.symlink(entry, "3rdparty/" + entry.basename)

    # Set up tvm-ffi - either from tvm_root or from separate repo
    if repository_ctx.attr.tvm_ffi_source:
        # Remote mode: use separate tvm-ffi repo
        tvm_ffi_root = repository_ctx.path(repository_ctx.attr.tvm_ffi_source).dirname
    else:
        # Local mode: use tvm-ffi from tvm_root
        tvm_ffi_root = third_party.get_child("tvm-ffi")

    if tvm_ffi_root.exists:
        # Symlink tvm-ffi contents (excluding 3rdparty which we handle specially)
        for entry in tvm_ffi_root.readdir():
            if entry.basename == "3rdparty":
                continue  # Handle separately below
            repository_ctx.symlink(entry, "3rdparty/tvm-ffi/" + entry.basename)

        # Set up tvm-ffi's 3rdparty (dlpack, libbacktrace)
        _setup_tvm_ffi_3rdparty(repository_ctx, tvm_ffi_root)

        # Add BUILD file for tvm-ffi
        overlay_build = overlay_path.get_child("3rdparty").get_child("tvm-ffi").get_child("BUILD.bazel")
        if overlay_build.exists:
            content = repository_ctx.read(overlay_build)
            repository_ctx.file("3rdparty/tvm-ffi/BUILD.bazel", content)

def _setup_tvm_ffi_3rdparty(repository_ctx, tvm_ffi_root):
    """Set up tvm-ffi's 3rdparty deps (dlpack, libbacktrace).

    These can come from tvm-ffi's submodules or from separate http_archive repos.
    For remote mode (separate repos), we must copy files because Bazel's glob
    doesn't follow symlinks to external repos.
    """
    tvm_ffi_3rdparty = tvm_ffi_root.get_child("3rdparty")

    # dlpack
    if repository_ctx.attr.dlpack_source:
        # Remote mode: copy from external repo (glob doesn't follow external symlinks)
        dlpack_root = repository_ctx.path(repository_ctx.attr.dlpack_source).dirname
        _copy_tree(repository_ctx, dlpack_root, "3rdparty/tvm-ffi/3rdparty/dlpack")
    elif tvm_ffi_3rdparty.exists:
        # Local mode: symlink from submodule
        dlpack_root = tvm_ffi_3rdparty.get_child("dlpack")
        if dlpack_root.exists:
            repository_ctx.symlink(dlpack_root, "3rdparty/tvm-ffi/3rdparty/dlpack")

    # libbacktrace
    if repository_ctx.attr.libbacktrace_source:
        # Remote mode: copy from external repo
        libbacktrace_root = repository_ctx.path(repository_ctx.attr.libbacktrace_source).dirname
        _copy_tree(repository_ctx, libbacktrace_root, "3rdparty/tvm-ffi/3rdparty/libbacktrace")
    elif tvm_ffi_3rdparty.exists:
        # Local mode: symlink from submodule
        libbacktrace_root = tvm_ffi_3rdparty.get_child("libbacktrace")
        if libbacktrace_root.exists:
            repository_ctx.symlink(libbacktrace_root, "3rdparty/tvm-ffi/3rdparty/libbacktrace")

def _copy_tree(repository_ctx, src, dest):
    """Copy a directory tree using shell cp command.

    This is needed because repository_ctx.symlink to external repos doesn't work
    well with glob patterns during analysis.
    """
    # Create parent directory if it doesn't exist
    parent = "/".join(dest.split("/")[:-1])
    if parent:
        repository_ctx.execute(["mkdir", "-p", parent], quiet = True)

    result = repository_ctx.execute(
        ["cp", "-r", str(src), dest],
        quiet = True,
    )
    if result.return_code != 0:
        fail("Failed to copy {} to {}: {}".format(src, dest, result.stderr))

    # Remove any BUILD/REPO.bazel files from the copied tree - they create package
    # boundaries that prevent glob from finding files
    for pattern in ["BUILD.bazel", "BUILD", "REPO.bazel"]:
        repository_ctx.execute(
            ["find", dest, "-name", pattern, "-delete"],
            quiet = True,
        )

def _generate_config_header(use_llvm, use_rpc):
    """Generate TVM configuration header."""
    lines = [
        "// Auto-generated TVM configuration for Bazel build",
        "#ifndef TVM_CONFIG_H_",
        "#define TVM_CONFIG_H_",
        "",
        "#define TVM_INDEX_DEFAULT_I64 1",
        "",
    ]

    if use_llvm:
        lines.append("#define TVM_LLVM_VERSION 190")  # LLVM 19.x

    if use_rpc:
        lines.append("#define USE_RPC 1")

    lines.extend([
        "",
        "#endif  // TVM_CONFIG_H_",
        "",
    ])

    return "\n".join(lines)

tvm_configure = repository_rule(
    implementation = _tvm_configure_impl,
    attrs = {
        "use_llvm": attr.bool(default = True),
        "use_rpc": attr.bool(default = True),
        "tvm_source": attr.label(
            default = None,
            doc = "Label to a file in TVM source repository (for remote mode).",
        ),
        "tvm_ffi_source": attr.label(
            default = None,
            doc = "Label to a file in tvm-ffi repository (for remote mode without submodules).",
        ),
        "dlpack_source": attr.label(
            default = None,
            doc = "Label to a file in dlpack repository (for remote mode without submodules).",
        ),
        "libbacktrace_source": attr.label(
            default = None,
            doc = "Label to a file in libbacktrace repository (for remote mode without submodules).",
        ),
    },
    doc = """Configures the TVM project with Bazel BUILD files.

    Args:
        use_llvm: Enable LLVM codegen support (requires @llvm-project).
        use_rpc: Enable RPC support.
        tvm_source: Label to a file in the TVM source repo (remote mode).
                    If None, auto-detects based on load context (local mode).
        tvm_ffi_source: Label to tvm-ffi repo (remote mode, when submodules unavailable).
        dlpack_source: Label to dlpack repo (remote mode, when submodules unavailable).
        libbacktrace_source: Label to libbacktrace repo (remote mode, when submodules unavailable).
    """,
)
