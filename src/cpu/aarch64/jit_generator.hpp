/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_GENERATOR_HPP
#define CPU_AARCH64_JIT_GENERATOR_HPP

#include <limits.h>

#include "common/bit_cast.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

#include "cpu/aarch64/jit_utils/jit_utils.hpp"

#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { return STRINGIFY(jit_name); } \
    const char *source_file() const override { return __FILE__; }

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// TODO: move this to jit_generator class?
namespace {

typedef enum {
    MAX_CODE_SIZE = 256 * 1024,
} max_code_size_t;

// TODO: move this somewhere else? Although this is only used by jit kernels
// (Roma)
static inline int float2int(float x) {
    return utils::bit_cast<int>(x);
}

static inline void tc_configure_tile(
        palette_config_t *tc, int t, int rows, int cols) {
    tc->rows[t] = rows;
    tc->cols[t] = cols;
}

// Callee-saved registers
constexpr Xbyak_aarch64::Operand::Code abi_save_gpr_regs[]
        = {Xbyak_aarch64::Operand::X19, Xbyak_aarch64::Operand::X20,
                Xbyak_aarch64::Operand::X21, Xbyak_aarch64::Operand::X22,
                Xbyak_aarch64::Operand::X23, Xbyak_aarch64::Operand::X24,
                Xbyak_aarch64::Operand::X25, Xbyak_aarch64::Operand::X26,
                Xbyak_aarch64::Operand::X27, Xbyak_aarch64::Operand::X28};

// See "Procedure Call Standsard for the ARM 64-bit Architecture (AArch64)"
static const Xbyak_aarch64::XReg abi_param1(Xbyak_aarch64::Operand::X0),
        abi_param2(Xbyak_aarch64::Operand::X1),
        abi_param3(Xbyak_aarch64::Operand::X2),
        abi_param4(Xbyak_aarch64::Operand::X3),
        abi_param5(Xbyak_aarch64::Operand::X4),
        abi_param6(Xbyak_aarch64::Operand::X5),
        abi_param7(Xbyak_aarch64::Operand::X6),
        abi_param8(Xbyak_aarch64::Operand::X7),
        abi_not_param1(Xbyak_aarch64::Operand::X15);
} // namespace

class jit_generator : public Xbyak_aarch64::CodeGenerator, public c_compatible {
public:
    using c_compatible::operator new;
    using c_compatible::operator new[];
    using c_compatible::operator delete;
    using c_compatible::operator delete[];

private:
    const size_t xmm_len = 16;
#ifdef _WIN32
    const size_t xmm_to_preserve_start = 6;
    const size_t xmm_to_preserve = 10;
#else
    const size_t xmm_to_preserve_start = 0;
    const size_t xmm_to_preserve = 0;
#endif

    const size_t num_abi_save_gpr_regs
            = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    const size_t size_of_abi_save_regs
            = num_abi_save_gpr_regs * rax.getBit() / 8
            + xmm_to_preserve * xmm_len;

public:
    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    Xbyak::Reg64 param1 = abi_param1;
    const int EVEX_max_8b_offt = 0x200;
    const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

    inline size_t get_size_of_abi_save_regs() { return size_of_abi_save_regs; }

    void preamble() {
        if (xmm_to_preserve) {
            sub(rsp, xmm_to_preserve * xmm_len);
            for (size_t i = 0; i < xmm_to_preserve; ++i)
                movdqu(ptr[rsp + i * xmm_len],
                        Xbyak::Xmm(xmm_to_preserve_start + i));
        }
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
            push(Xbyak::Reg64(abi_save_gpr_regs[i]));
        if (mayiuse(avx512_common)) {
            mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
        }
    }

    void postamble() {
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
            pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
        if (xmm_to_preserve) {
            for (size_t i = 0; i < xmm_to_preserve; ++i)
                movdqu(Xbyak::Xmm(xmm_to_preserve_start + i),
                        ptr[rsp + i * xmm_len]);
            add(rsp, xmm_to_preserve * xmm_len);
        }
        uni_vzeroupper();
        ret();
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak::Label &label) { Xbyak::CodeGenerator::L(label); }

    void L_aligned(Xbyak::Label &label, int alignment = 16) {
        align(alignment);
        L(label);
    }

    /*
      Saturation facility functions. enable to prepare the register
      holding the saturation upperbound and apply the saturation on
      the floating point register
     */
    template <typename Vmm>
    void init_saturate_f32(Vmm vmm_lbound, Vmm vmm_ubound, Xbyak::Reg64 reg_tmp,
            data_type_t idt, data_type_t odt) {
        using namespace data_type;
        if (!((idt == f32) && utils::one_of(odt, u8, s8, s32))) return;

        assert(IMPLICATION(
                idt == u8, vmm_lbound.getIdx() != vmm_ubound.getIdx()));
        // No need to saturate on lower bound for signed integer types, as
        // the conversion to int would return INT_MIN, and then proper
        // saturation will happen in store_data
        if (odt == u8) uni_vpxor(vmm_lbound, vmm_lbound, vmm_lbound);

        Xbyak::Xmm tmp(vmm_ubound.getIdx());
        float saturation_ubound = types::max_value<float>(odt);
        mov(reg_tmp, float2int(saturation_ubound));
        uni_vmovq(tmp, reg_tmp);
        if (vmm_ubound.isYMM() || vmm_ubound.isZMM())
            uni_vbroadcastss(vmm_ubound, tmp);
        else
            uni_vshufps(vmm_ubound, tmp, tmp, 0);
    }

    // This function is used to saturate to odt in f32 before converting to s32
    // in order to avoid bad saturation due to cvtps2dq behavior (it returns
    // INT_MIN if the f32 is out of the s32 range)
    template <typename Vmm>
    void saturate_f32(const Vmm &vmm, const Vmm &vmm_lbound,
            const Vmm &vmm_ubound, const Vmm &vmm_tmp, data_type_t odt) {
        using namespace data_type;
        if (!utils::one_of(odt, u8, s8, s32)) return;

        // no need to apply lower saturation bound when odt is signed, as
        // cvtps2dq will return MIN_INT if the value does not fit.
        // The comment below for a certain order applied for maxps instruction
        // as well. No changes here since NaN with positive sign was not met
        // yet.
        if (odt == u8) {
            if (mayiuse(avx))
                vmaxps(vmm, vmm, vmm_lbound);
            else
                maxps(vmm, vmm_lbound);
        }

        // Order matters for minps due to peculiar behavior of the instruction
        // with NaNs:
        //     if (SRC1 == NaN)
        //         return SRC2;
        //     else if (SRC2 == NaN)
        //         return SRC2;
        // that's why we keep user's data at SRC2 reg to pass NaNs further to
        // cvtps2dq which handles them properly.
        if (mayiuse(avx))
            vminps(vmm, vmm_ubound, vmm);
        else {
            movups(vmm_tmp, vmm_ubound);
            minps(vmm_tmp, vmm);
            movups(vmm, vmm_tmp);
        }
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator);

public:
    jit_generator(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true)
        : Xbyak::CodeGenerator(code_size,
                (code_ptr == nullptr && use_autogrow) ? Xbyak::AutoGrow
                                                      : code_ptr) {}
    virtual ~jit_generator() {}

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    void register_jit_code(const Xbyak::uint8 *code, size_t code_size) const {
        jit_utils::register_jit_code(code, code_size, name(), source_file());
    }

    const Xbyak::uint8 *jit_ker() const { return jit_ker_; }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual status_t create_kernel() {
        generate();
        jit_ker_ = getCode();
        return (jit_ker_) ? status::success : status::runtime_error;
    }

private:
    const Xbyak::uint8 *getCode() {
        this->ready();
        if (!is_initialized()) return nullptr;
        const Xbyak::uint8 *code = CodeGenerator::getCode();
        register_jit_code(code, getSize());
        return code;
    }

    static inline bool is_initialized() {
        return Xbyak::GetError() == Xbyak::ERR_NONE;
    }

protected:
    virtual void generate() = 0;
    const Xbyak::uint8 *jit_ker_ = nullptr;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
