Add an opt-in HIP device backend for AMD GPUs. Build with
``build_lib.py --hip`` (optionally ``--hip-arch`` and ``--rocm-path``) to compile
the device sources with ``hipcc`` for ROCm. ``Device.is_hip`` reports whether a
device is driven by the HIP backend; ``Device.is_cuda`` remains ``True`` for
those devices, so existing code keeps working. CUDA builds are unaffected.
