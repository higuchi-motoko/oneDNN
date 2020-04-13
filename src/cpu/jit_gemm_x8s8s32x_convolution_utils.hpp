/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_JIT_GEMM_X8S8S32X_CONVOLUTION_UTILS_HPP
#define CPU_JIT_GEMM_X8S8S32X_CONVOLUTION_UTILS_HPP

#include "gemm_x8s8s32x_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace gemm_x8s8s32x_convolution_utils {

pp_ker_t *jit_pp_ker_create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp);

}
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif