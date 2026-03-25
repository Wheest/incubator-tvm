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

    print("DEBUG: tvm_root = " + str(tvm_root))
    print("DEBUG: tvm_root exists = " + str(tvm_root.exists))

    # Get overlay path
    overlay_path = repository_ctx.path(Label("@tvm//:overlays/BUILD.bazel")).dirname
    print("DEBUG: overlay_path = " + str(overlay_path))

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
            print("DEBUG: symlinked " + item)

    # Set up src/ directory
    _setup_src_directory(repository_ctx, tvm_root, overlay_path)

    # Set up 3rdparty/ directory - special handling for tvm-ffi
    _setup_3rdparty_directory(repository_ctx, tvm_root, overlay_path)

    # Root BUILD.bazel
    root_build = overlay_path.get_child("BUILD.bazel")
    print("DEBUG: root BUILD exists = " + str(root_build.exists))
    repository_ctx.symlink(root_build, "BUILD.bazel")

    # Generate config header
    repository_ctx.file(
        "tvm_config.h",
        content = _generate_config_header(use_llvm, use_rpc),
    )

def _setup_src_directory(repository_ctx, tvm_root, overlay_path):
    """Set up src/ directory with symlinked sources and BUILD files."""
    src_root = tvm_root.get_child("src")
    if not src_root.exists:
        print("DEBUG: src_root does not exist: " + str(src_root))
        return

    print("DEBUG: setting up src directory from " + str(src_root))

    # Symlink each subdirectory in src/
    for entry in src_root.readdir():
        dest = "src/" + entry.basename
        repository_ctx.symlink(entry, dest)
        print("DEBUG: symlinked " + dest)

    # Add BUILD files for src subdirectories
    for subdir in _SRC_OVERLAY_DIRS:
        overlay_build = overlay_path.get_child("src").get_child(subdir).get_child("BUILD.bazel")
        if overlay_build.exists:
            content = repository_ctx.read(overlay_build)
            dest = "src/" + subdir + "/BUILD.bazel"
            repository_ctx.file(dest, content)
            print("DEBUG: wrote BUILD to " + dest)
        else:
            print("DEBUG: overlay BUILD not found: " + str(overlay_build))

def _setup_3rdparty_directory(repository_ctx, tvm_root, overlay_path):
    """Set up 3rdparty/ directory with special handling for tvm-ffi."""
    third_party = tvm_root.get_child("3rdparty")
    if not third_party.exists:
        print("DEBUG: 3rdparty does not exist")
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
            print("DEBUG: wrote BUILD to 3rdparty/tvm-ffi/BUILD.bazel")

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
