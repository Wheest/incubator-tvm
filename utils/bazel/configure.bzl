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

    # Find TVM source root using readlink to follow symlinks
    tvm_bazel_path = repository_ctx.path(Label("@tvm//:MODULE.bazel")).dirname
    result = repository_ctx.execute(["readlink", str(tvm_bazel_path)])
    if result.return_code == 0:
        real_bazel_path = result.stdout.strip()
        tvm_root = repository_ctx.path(real_bazel_path).dirname.dirname
    else:
        tvm_root = tvm_bazel_path.dirname.dirname

    # Get overlay path
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
    """Set up 3rdparty/ directory with special handling for tvm-ffi."""
    third_party = tvm_root.get_child("3rdparty")
    if not third_party.exists:
        return

    # Symlink non-tvm-ffi directories directly
    for entry in third_party.readdir():
        if entry.basename != "tvm-ffi":
            repository_ctx.symlink(entry, "3rdparty/" + entry.basename)

    # For tvm-ffi, symlink its contents so we can add BUILD.bazel
    tvm_ffi = third_party.get_child("tvm-ffi")
    if tvm_ffi.exists:
        for entry in tvm_ffi.readdir():
            repository_ctx.symlink(entry, "3rdparty/tvm-ffi/" + entry.basename)

        # Add BUILD file for tvm-ffi
        overlay_build = overlay_path.get_child("3rdparty").get_child("tvm-ffi").get_child("BUILD.bazel")
        if overlay_build.exists:
            content = repository_ctx.read(overlay_build)
            repository_ctx.file("3rdparty/tvm-ffi/BUILD.bazel", content)

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
    },
    doc = """Configures the TVM project with Bazel BUILD files.""",
)
