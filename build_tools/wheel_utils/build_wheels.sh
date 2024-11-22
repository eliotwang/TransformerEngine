# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

PLATFORM=${1:-manylinux_2_28_x86_64}
<<<<<<< HEAD
BUILD_COMMON=${2:-true}
BUILD_JAX=${3:-true}
BUILD_PYTORCH=${4:-true}
BUILD_PADDLE=${5:-true}
=======
BUILD_METAPACKAGE=${2:-true}
BUILD_COMMON=${3:-true}
BUILD_PYTORCH=${4:-true}
BUILD_JAX=${5:-true}
BUILD_PADDLE=${6:-true}
>>>>>>> upstream/release_v1.11

export NVTE_RELEASE_BUILD=1
export TARGET_BRANCH=${TARGET_BRANCH:-}
mkdir -p /wheelhouse/logs

# Generate wheels for common library.
git config --global --add safe.directory /TransformerEngine
cd /TransformerEngine
git checkout $TARGET_BRANCH
git submodule update --init --recursive

<<<<<<< HEAD
if $BUILD_COMMON ; then
        /opt/python/cp38-cp38/bin/python setup.py bdist_wheel --verbose --python-tag=py3 --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/common.txt
        whl_name=$(basename dist/*)
        IFS='-' read -ra whl_parts <<< "$whl_name"
        whl_name_target="${whl_parts[0]}-${whl_parts[1]}-py3-none-${whl_parts[4]}"
        mv dist/"$whl_name" /wheelhouse/"$whl_name_target"
=======
if $BUILD_METAPACKAGE ; then
        cd /TransformerEngine
        NVTE_BUILD_METAPACKAGE=1 /opt/python/cp310-cp310/bin/python setup.py bdist_wheel 2>&1 | tee /wheelhouse/logs/metapackage.txt
        mv dist/* /wheelhouse/
fi

if $BUILD_COMMON ; then
        VERSION=`cat build_tools/VERSION.txt`
        WHL_BASE="transformer_engine-${VERSION}"

        # Create the wheel.
        /opt/python/cp38-cp38/bin/python setup.py bdist_wheel --verbose --python-tag=py3 --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/common.txt

        # Repack the wheel for cuda specific package, i.e. cu12.
        /opt/python/cp38-cp38/bin/wheel unpack dist/*
        # From python 3.10 to 3.11, the package name delimiter in metadata got changed from - (hyphen) to _ (underscore).
        sed -i "s/Name: transformer-engine/Name: transformer-engine-cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        sed -i "s/Name: transformer_engine/Name: transformer_engine_cu12/g" "transformer_engine-${VERSION}/transformer_engine-${VERSION}.dist-info/METADATA"
        mv "${WHL_BASE}/${WHL_BASE}.dist-info" "${WHL_BASE}/transformer_engine_cu12-${VERSION}.dist-info"
        /opt/python/cp38-cp38/bin/wheel pack ${WHL_BASE}

        # Rename the wheel to make it python version agnostic.
        whl_name=$(basename dist/*)
        IFS='-' read -ra whl_parts <<< "$whl_name"
        whl_name_target="${whl_parts[0]}_cu12-${whl_parts[1]}-py3-none-${whl_parts[4]}"
        rm -rf $WHL_BASE dist
        mv *.whl /wheelhouse/"$whl_name_target"
>>>>>>> upstream/release_v1.11
fi

if $BUILD_PYTORCH ; then
	cd /TransformerEngine/transformer_engine/pytorch
	/opt/python/cp38-cp38/bin/pip install torch
	/opt/python/cp38-cp38/bin/python setup.py sdist 2>&1 | tee /wheelhouse/logs/torch.txt
	cp dist/* /wheelhouse/
fi

if $BUILD_JAX ; then
	cd /TransformerEngine/transformer_engine/jax
<<<<<<< HEAD
	/opt/python/cp38-cp38/bin/pip install jax jaxlib
	/opt/python/cp38-cp38/bin/python setup.py sdist 2>&1 | tee /wheelhouse/logs/jax.txt
=======
	/opt/python/cp310-cp310/bin/pip install "jax[cuda12_local]" jaxlib
	/opt/python/cp310-cp310/bin/python setup.py sdist 2>&1 | tee /wheelhouse/logs/jax.txt
>>>>>>> upstream/release_v1.11
	cp dist/* /wheelhouse/
fi

if $BUILD_PADDLE ; then
        if [ "$PLATFORM" == "manylinux_2_28_x86_64" ] ; then
                dnf -y remove --allowerasing cudnn9-cuda-12
                dnf -y install libcudnn8-devel.x86_64 libcudnn8.x86_64
                cd /TransformerEngine/transformer_engine/paddle

<<<<<<< HEAD
                /opt/python/cp38-cp38/bin/pip install /wheelhouse/*.whl
                /opt/python/cp38-cp38/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp38-cp38/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp38.txt
                /opt/python/cp38-cp38/bin/pip uninstall -y transformer-engine paddlepaddle-gpu

                /opt/python/cp39-cp39/bin/pip install /wheelhouse/*.whl
                /opt/python/cp39-cp39/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp39-cp39/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp39.txt
                /opt/python/cp39-cp39/bin/pip uninstall -y transformer-engine paddlepaddle-gpu

                /opt/python/cp310-cp310/bin/pip install /wheelhouse/*.whl
                /opt/python/cp310-cp310/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp310-cp310/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp310.txt
                /opt/python/cp310-cp310/bin/pip uninstall -y transformer-engine paddlepaddle-gpu

                /opt/python/cp311-cp311/bin/pip install /wheelhouse/*.whl
                /opt/python/cp311-cp311/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp311-cp311/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp311.txt
                /opt/python/cp311-cp311/bin/pip uninstall -y transformer-engine paddlepaddle-gpu

                /opt/python/cp312-cp312/bin/pip install /wheelhouse/*.whl
                /opt/python/cp312-cp312/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp312-cp312/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp312.txt
                /opt/python/cp312-cp312/bin/pip uninstall -y transformer-engine paddlepaddle-gpu
=======
                /opt/python/cp38-cp38/bin/pip install /wheelhouse/*.whl --no-deps
                /opt/python/cp38-cp38/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp38-cp38/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp38.txt
                /opt/python/cp38-cp38/bin/pip uninstall -y transformer-engine transformer-engine-cu12 paddlepaddle-gpu

                /opt/python/cp39-cp39/bin/pip install /wheelhouse/*.whl --no-deps
                /opt/python/cp39-cp39/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp39-cp39/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp39.txt
                /opt/python/cp39-cp39/bin/pip uninstall -y transformer-engine transformer-engine-cu12 paddlepaddle-gpu

                /opt/python/cp310-cp310/bin/pip install /wheelhouse/*.whl --no-deps
                /opt/python/cp310-cp310/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp310-cp310/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp310.txt
                /opt/python/cp310-cp310/bin/pip uninstall -y transformer-engine transformer-engine-cu12 paddlepaddle-gpu

                /opt/python/cp311-cp311/bin/pip install /wheelhouse/*.whl --no-deps
                /opt/python/cp311-cp311/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp311-cp311/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp311.txt
                /opt/python/cp311-cp311/bin/pip uninstall -y transformer-engine transformer-engine-cu12 paddlepaddle-gpu

                /opt/python/cp312-cp312/bin/pip install /wheelhouse/*.whl --no-deps
                /opt/python/cp312-cp312/bin/pip install paddlepaddle-gpu==2.6.1
                /opt/python/cp312-cp312/bin/python setup.py bdist_wheel --verbose --plat-name=$PLATFORM 2>&1 | tee /wheelhouse/logs/paddle_cp312.txt
                /opt/python/cp312-cp312/bin/pip uninstall -y transformer-engine transformer-engine-cu12 paddlepaddle-gpu
>>>>>>> upstream/release_v1.11

                mv dist/* /wheelhouse/
	fi
fi
