/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_CPU_ISA_TRAITS_HPP
#define CPU_AARCH64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR

#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64.h"
#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

enum cpu_isa_bit_t : unsigned {
    simdfp_bit = 1u << 0,
    sve_128_bit = 1u << 1,
    sve_256_bit = 1u << 2,
    sve_384_bit = 1u << 3,
    sve_512_bit = 1u << 4,
};

enum cpu_isa_t : unsigned {
    isa_any = 0u,
    simdfp = simdfp_bit,
    sve_128 = sve_128_bit | simdfp,
    sve_256 = sve_256_bit | simdfp,
    sve_384 = sve_384_bit | simdfp,
    sve_512 = sve_512_bit | simdfp,
    isa_all = ~0u,
};

const char *get_isa_info();

cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

// pack struct so it can fit into a single 64-byte cache line
#pragma pack(push, 1)
struct palette_config_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
};
#pragma pack(pop)

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_all;
    static constexpr const char *user_option_env = "ALL";
};

template <>
struct cpu_isa_traits<simdfp> {
    typedef Xbyak_aarch64::VReg4S Vmm;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_simdfp;
    static constexpr const char *user_option_env = "SIMD&FP";
};

template <>
struct cpu_isa_traits<sve_512> {
    typedef Xbyak_aarch64::ZRegS Vmm;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sve_512;
    static constexpr const char *user_option_env = "SVE_512";
};

inline const Xbyak_aarch64::util::Cpu &cpu() {
    const static Xbyak_aarch64::util::Cpu cpu_;
    return cpu_;
}

namespace {

static inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak_aarch64::util;

    unsigned cpu_isa_mask = aarch64::get_max_cpu_isa_mask(soft);
    if ((cpu_isa_mask & cpu_isa) != cpu_isa) return false;

    switch (cpu_isa) {
        case simdfp: return cpu().has(Cpu::tADVSIMD) && cpu().has(Cpu::tFP);
        case sve_128:
            return cpu().has(Cpu::tADVSIMD) && cpu().has(Cpu::tFP)
                    && cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_128;
        case sve_256:
            return cpu().has(Cpu::tADVSIMD) && cpu().has(Cpu::tFP)
                    && cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_256;
        case sve_384:
            return cpu().has(Cpu::tADVSIMD) && cpu().has(Cpu::tFP)
                    && cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_384;
        case sve_512:
            return cpu().has(Cpu::tADVSIMD) && cpu().has(Cpu::tFP)
                    && cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_512;
        case isa_any: return true;
        case isa_all: return false;
    }
    return false;
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return false;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_any ? prefix STRINGIFY(any) : \
    ((isa) == simdfp ? prefix STRINGIFY(simdfp) : \
    ((isa) == sve_512 ? prefix STRINGIFY(sve_512) : \
    prefix suffix_if_any))))
/* clang-format on */

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
