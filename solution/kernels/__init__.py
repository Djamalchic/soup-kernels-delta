"""Our Triton kernels (all written by team Delta).

v8 C1, one launcher for all of them (flag USE_UNIFIED_SIG in solution/__init__.py).

Triton runs every compiled kernel through a small C launcher that it generates from the kernel's
list of argument types, builds with gcc and caches as `__triton_launcher*.so`, keyed by the hash of
that C source. Pointer dtypes do not enter the source (every pointer is parsed as a Python object),
so two kernels with the same sequence of argument kinds share one launcher; a different count, order
or scalar type, or a constexpr parameter at another position, makes a new source and a new gcc
build. (From memory of triton/backends/nvidia/driver.py make_launcher and
triton/runtime/build.py compile_module_from_src, Triton 3.2-3.4; no Triton source on this machine.
The v7 Kaggle run's triton_cache, 6 *.cubin + 6 __triton_launcher*.so + 1 cuda_utils*.so for our
six kernels, fits this model; tests/test_compile_probe.py checks it on Triton 3.6.)

With C1 every kernel we launch takes the same twelve runtime arguments and no constexpr parameter:

    slot   0 .. 6        7 .. 10      11
    kind   pointer x 7   int32 x 4    fp32

- Unused pointer slots get a tensor that the launch already passes, and are in do_not_specialize,
  so their alignment never splits a variant. Unused int slots get 0, the float slot 0.0.
- Modes, row offsets and flags are runtime int32 values in do_not_specialize, so the value 1 never
  turns into a constexpr (which would change both the compiled variant and the launcher source).
  The only specialised int slots are the ones v7 already specialised for vector alignment:
  head_ce `n_cols` and the SwiGLU element count, multiples of 16 in training.
- Block sizes and widths are module-level `tl.constexpr` globals instead of parameters.

Launched entries and their bodies:
  addnorm.soup_addnorm_fwd_usig   -> soup_addnorm_fwd_kernel   (K3)
  addnorm.soup_addnorm_bwd_usig   -> soup_addnorm_bwd_kernel   (K3)
  head_ce.soup_head_ce_usig       -> soup_head_ce_kernel       (K2)
  nf4.soup_nf4_dequant_usig       -> soup_nf4_dequant_kernel   (K1)
  swiglu.soup_swiglu_kernel       its own body (v8 C2)

Each *_usig entry calls the v7 kernel as a Triton device function with its own arguments unchanged,
the v7 constexpr values and the v7 launch options (num_warps, enable_fp_fusion). Triton inlines the
call, and the pointer and int slots v7 specialised are specialised the same way here, so the
compiled body sees the same values and the same alignment facts: same layouts, same reduction
order, same bits (the tests compare with torch.equal). Each kernel still compiles its own cubin;
C1 removes only the per-kernel gcc launcher build.
"""
