load("@rules_python//python:defs.bzl", "py_library")
load("//third_party/bazel_doctest:doctest.bzl", "py_doc_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "anchors",
    srcs = ["anchors.py"],
)

py_doc_test(
    name = "anchors_doctest",
    deps = [":anchors"],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
)

py_library(
    name = "postprocess",
    srcs = ["postprocess.py"],
    deps = [":regression"]
)

py_doc_test(
    name = "postprocess_doctest",
    deps = [":postprocess"],
)

py_library(
    name = "regression",
    srcs = ["regression.py"],
)

py_doc_test(
    name = "regression_doctest",
    deps = [":regression"],
)

py_library(
    name = "retinanet_head",
    srcs = ["retinanet_head.py"],
)
