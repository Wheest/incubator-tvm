"""Module extension for configuring TVM."""

load(":configure.bzl", "tvm_configure")

def _tvm_configure_extension_impl(ctx):
    """Configure the TVM project overlay."""
    # Get configuration from tags
    use_llvm = True
    use_rpc = True

    for module in ctx.modules:
        for config in module.tags.configure:
            if hasattr(config, "use_llvm"):
                use_llvm = config.use_llvm
            if hasattr(config, "use_rpc"):
                use_rpc = config.use_rpc

    # Build repo_mapping for injected repos
    repo_mapping = {}

    # Check if llvm-project was injected
    if hasattr(ctx, "injected_repos") and "llvm-project" in ctx.injected_repos:
        repo_mapping["llvm-project"] = ctx.injected_repos["llvm-project"]

    tvm_configure(
        name = "tvm-project",
        use_llvm = use_llvm,
        use_rpc = use_rpc,
    )

tvm_overlay = module_extension(
    doc = """Configure the TVM project.

    This extension creates a configured TVM repository with BUILD files
    overlaid on the TVM source tree.

    Example usage in MODULE.bazel:

        tvm_overlay = use_extension("@tvm//:extensions.bzl", "tvm_overlay")
        tvm_overlay.configure(use_llvm = True)
        use_repo(tvm_overlay, "tvm-project")
    """,
    implementation = _tvm_configure_extension_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {
                "use_llvm": attr.bool(default = True),
                "use_rpc": attr.bool(default = True),
            },
        ),
    },
)
