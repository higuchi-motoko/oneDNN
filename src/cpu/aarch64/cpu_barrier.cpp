/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include <assert.h>

#include "cpu/aarch64/cpu_barrier.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace simple_barrier {

void generate(jit_generator &code, Xbyak_aarch64::XReg reg_ctx,
        Xbyak_aarch64::XReg reg_nthr) {
#define BAR_CTR_OFF offsetof(ctx_t, ctr)
#define BAR_SENSE_OFF offsetof(ctx_t, sense)
#define IDX(a) static_cast<uint32_t>(a.getIdx())
    using namespace Xbyak_aarch64;

    XReg reg_tmp = [&]() {
        /* returns register which is neither reg_ctx nor reg_nthr */
        const int regs[] = {0, 3, 1};
        for (size_t i = 0; i < sizeof(regs) / sizeof(regs[0]); ++i)
            if (!utils::one_of(i, reg_ctx.getIdx(), reg_nthr.getIdx()))
                return XReg(i);
        return XReg(0); /* should not happen */
    }();

    const XReg x_tmp_0 = code.X_TMP_0;
    const XReg x_tmp_1 = code.X_TMP_1;
    const XReg x_tmp_2 = code.X_TMP_2;
    const XReg x_tmp_sp = code.X_TMP_3;
    const XReg sp = code.sp;

    Label barrier_exit_label, barrier_exit_restore_label, spin_label;
    Label debug0, debug1;

#if 0
    code.CodeGenerator::L(debug0);
    code.nop();
    code.b(debug0);
#endif

    code.mov(x_tmp_sp, sp);
    code.cmp(reg_nthr, 1);
    code.b(EQ, barrier_exit_label);

    code.sub(x_tmp_sp, x_tmp_sp, 8);
    code.str(reg_tmp, ptr(x_tmp_sp));

    /* take and save current sense */
    code.add_imm(x_tmp_0, reg_ctx, BAR_SENSE_OFF, x_tmp_0);
    code.ldr(reg_tmp, ptr(x_tmp_0));
    code.sub(x_tmp_sp, x_tmp_sp, 8);
    code.str(reg_tmp, ptr(x_tmp_sp));
    code.mov(WReg(IDX(reg_tmp)), 1);

#if 0
    if (mayiuse(avx512_mic)) {
        code.prefetchwt1(code.ptr[reg_ctx + BAR_CTR_OFF]);
        code.prefetchwt1(code.ptr[reg_ctx + BAR_CTR_OFF]);
    }
#endif // #if 0

    code.add_imm(x_tmp_1, reg_ctx, BAR_CTR_OFF, x_tmp_2);
    code.ldaddal(reg_tmp, reg_tmp, ptr(x_tmp_1));
    code.adds(reg_tmp, reg_tmp, 1);
    code.cmp(reg_tmp, reg_nthr);
    code.ldr(reg_tmp, ptr(x_tmp_sp));
    code.add(x_tmp_sp, x_tmp_sp, 8);
    code.b(NE, spin_label);

    /* the last thread {{{ */
    code.mov_imm(x_tmp_2, 0);
    code.str(x_tmp_2, ptr(x_tmp_1));

    // notify waiting threads
    code.mvn(reg_tmp, reg_tmp);
    code.str(reg_tmp, ptr(x_tmp_0));
    code.b(barrier_exit_restore_label);
    /* }}} the last thread */

    code.CodeGenerator::L(spin_label);
    code.ldr(x_tmp_1, ptr(x_tmp_0));
    code.cmp(reg_tmp, x_tmp_1);
    code.b(EQ, spin_label);

    //#ifdef DNNL_INDIRECT_JIT_AARCH64
    code.CodeGenerator::dmb(ISH);
    //#endif //#ifdef DNNL_INDIRECT_JIT_AARCH64

    code.CodeGenerator::L(barrier_exit_restore_label);
    code.ldr(reg_tmp, ptr(x_tmp_sp));
    code.add(x_tmp_sp, x_tmp_sp, 8);

    code.CodeGenerator::L(barrier_exit_label);
    //    code.mov(sp, x_tmp_sp);
#undef BAR_CTR_OFF
#undef BAR_SENSE_OFF

#if 0
    code.CodeGenerator::L(debug1);
    code.nop();
    code.b(debug1);
#endif
}

/** jit barrier generator */
struct jit_t : public jit_generator {

    void generate() override {
        simple_barrier::generate(*this, abi_param1, abi_param2);
        ret();
    }

    // TODO: Need to check status
    jit_t() { create_kernel(); }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_t)
};

void barrier(ctx_t *ctx, int nthr) {
    static jit_t j; /* XXX: constructed on load ... */
    j(ctx, nthr);
}

} // namespace simple_barrier

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
