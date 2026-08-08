#!/usr/bin/env bash

_EXROMA_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXROMA_ROOT="$(cd -- "${_EXROMA_SCRIPT_DIR}/.." && pwd)"
export EXROMA_HOME="$(cd -- "${EXROMA_ROOT}/.." && pwd)"
export ISAACSIM_PATH="${EXROMA_HOME}/tools/isaacsim-4.5-v23"
export EXROMA_ISAACLAB_ROOT="${EXROMA_HOME}/IsaacLab-v2.3"
export EXROMA_PYTHON="${ISAACSIM_PATH}/python.sh"
export EXROMA_ASSET_ROOT="${EXROMA_HOME}/ExRoMa-Assets"
export EXROMA_DUAL_PIPER_SOURCE="urdf"
export EXROMA_FAST_HEADLESS_EXIT="1"
export EXROMA_CONVERTED_ASSET_ROOT="${EXROMA_HOME}/cache/isaacsim-4.5-v23"
export CUDA_HOME="${EXROMA_HOME}/tools/cuda-12.8"
export BLENDER_ROOT="${EXROMA_HOME}/tools/blender-4.5.3"
export EXROMA_CACHE_HOME="${EXROMA_HOME}/cache"
export TMPDIR="$(cd -- "${EXROMA_HOME}/.." && pwd)/tmp"
export PIP_CACHE_DIR="${TMPDIR}/pip-cache"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_DRIVER_FILES="/etc/vulkan/icd.d/nvidia_icd.json"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export EXROMA_SIM_GPU="${EXROMA_SIM_GPU:-0}"
export EXROMA_MULTI_GPU="${EXROMA_MULTI_GPU:-0}"
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export SETUPTOOLS_SCM_PRETEND_VERSION="0.7.8"
export PYTHONEXE="${ISAACSIM_PATH}/kit/python/bin/python3"
if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM=xterm
fi

_EXROMA_SITE_PACKAGES="${ISAACSIM_PATH}/kit/python/lib/python3.10/site-packages"
_EXROMA_TORCH_LIB="${_EXROMA_SITE_PACKAGES}/torch/lib"
_EXROMA_CUDA_RUNTIME_LIB="${_EXROMA_SITE_PACKAGES}/nvidia/cuda_runtime/lib"
_EXROMA_CUDA_TARGET="${CUDA_HOME}/targets/x86_64-linux"

export PATH="${EXROMA_HOME}/tools/bin:${PATH}"
export PYTHONPATH="${EXROMA_ROOT}/source:${PYTHONPATH:-}"
export CPATH="${_EXROMA_CUDA_TARGET}/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${_EXROMA_CUDA_TARGET}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${_EXROMA_CUDA_TARGET}/lib:${CUDA_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${_EXROMA_TORCH_LIB}:${_EXROMA_CUDA_RUNTIME_LIB}:${_EXROMA_CUDA_TARGET}/lib:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

exroma() {
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER TERM="${TERM}" \
    XDG_CACHE_HOME="${EXROMA_CACHE_HOME}" PYTHONEXE="${PYTHONEXE}" \
    "${EXROMA_PYTHON}" -m exroma_bench.cli "$@"
}

isaaclab45() {
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER TERM=xterm \
    PYTHONEXE="${PYTHONEXE}" "${EXROMA_ISAACLAB_ROOT}/isaaclab.sh" "$@"
}
