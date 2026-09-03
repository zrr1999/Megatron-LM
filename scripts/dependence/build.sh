#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
export megatron_dir=/workspace/Megatron-LM
mkdir -p /workspace/build_logs
export log_path=/workspace/build_logs
mkdir -p /workspace/upload
upload_path=/workspace/upload

python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --user set global.trusted-host pypi.tuna.tsinghua.edu.cn

megatron_tar (){
    cd /workspace
    # Megatron-LM.tar only include the main branch
    if [ -n "$TARGET_COMMIT" ]; then
        echo "TARGET_COMMIT=$TARGET_COMMIT specified, skip refreshing Megatron-LM.tar.gz"
    elif [ -n "$BRANCH" ] && [ "$BRANCH" = "main" ]; then
        echo "Checkout branch $BRANCH"
        tar -zcf Megatron-LM.tar.gz Megatron-LM/
        mv Megatron-LM.tar.gz ${upload_path}/
    else
        echo "No BRANCH specified, skip checkout"
    fi
}

megatron_build (){
    cd $megatron_dir
    rm -rf di/
    rm -rf dist/
    rm -rf megatron_core.egg-info/

    python -m pip install --upgrade pip
    python -m pip install build "setuptools>=80" pybind11 packaging
    NO_VCS_VERSION=1 python -m build --wheel --no-isolation

    echo "install_megatron_develop_whl"
    python -m pip install --ignore-installed dist/megatron_core-*.whl --no-cache-dir --force-reinstall --no-dependencies
    
    commit=${COMMIT_ID:-unknown}
    commit=${commit:0:7}

    whl_file=$(ls $megatron_dir/dist/megatron_core-*.whl)
    base_name=$(basename $whl_file)
    new_name=$(echo $base_name | sed "s/^\(megatron_core-[0-9.]*\)-/\1+${commit}-/")
    echo "commit whl: $new_name"
    cp "$whl_file" "${upload_path}/${new_name}"

    zero_name=$(echo $base_name | sed "s/^megatron_core-[^-]*-/megatron_core-0.0.0-/")
    if [ "${UPDATE_LATEST:-true}" = "true" ]; then
        echo "latest whl: $base_name"
        cp "$whl_file" "${upload_path}/${base_name}"
        echo "0.0.0 whl: $zero_name"
        cp "$whl_file" "${upload_path}/${zero_name}"
    else
        echo "UPDATE_LATEST=$UPDATE_LATEST, skip publishing $base_name and $zero_name"
    fi
}

# main
cd ${megatron_dir}
echo -e "\033[32m ---- make Megatron-LM.tar.gz  \033[0m"
megatron_tar
echo -e "\033[32m ---- build Megatron-LM whl  \033[0m"
megatron_build

