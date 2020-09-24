#!/bin/bash -e
#*******************************************************************************
# Copyright 2019-2020 FUJITSU LIMITED
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
#*******************************************************************************/

cd src/cpu/aarch64/xbyak_translator_aarch64/translator/third_party/
mkdir build_xed_aarch64
cd build_xed_aarch64/
../xed/mfile.py --shared examples install
cd kits/
XED=`ls | grep install`
ln -sf $XED xed
cd xed/
XED_ROOT=`pwd`
cd bin/
CI_XED_PATH=`pwd`
cd ../lib/
CI_XED_LIB=`pwd`
cd ../../../../../../../../../../
export LD_LIBRARY_PATH=${CI_XED_LIB}:${LD_LIBRARY_PATH}
export XED_ROOT_DIR=${XED_ROOT}
