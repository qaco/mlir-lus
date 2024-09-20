include ../makefile.config

CUDA=1
VULKAN=1

IREE_STATIC_OPTS=-DCPUINFO_SUPPORTED_PLATFORM=1 -DIREE_HAL_HAVE_CUDA_DRIVER_MODULE=$(CUDA) \
    -DIREE_HAL_HAVE_DYLIB_DRIVER_MODULE=1 -DIREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE=1 \
    -DIREE_HAL_HAVE_EMBEDDED_LIBRARY_LOADER=1 -DIREE_HAL_HAVE_SYSTEM_LIBRARY_LOADER=1 \
    -DIREE_HAL_HAVE_VMVX_DRIVER_MODULE=1 -DIREE_HAL_HAVE_VMVX_MODULE_LOADER=1 \
    -DIREE_HAL_HAVE_VULKAN_DRIVER_MODULE=$(VULKAN) -DIREE_HAVE_EMITC_DIALECT

IREE_INCLUDES=-I$(IREE_DIR)/third_party/cpuinfo/include \
    -I$(IREE_DIR)/third_party/vulkan_headers/include \
    -isystem $(IREE_DIR) \
    -isystem . \
    -isystem $(IREE_DIR)/third_party/cuda/include \
    -isystem iree/base/internal \
    -isystem $(IREE_DIR)/third_party/flatcc/include \
    -isystem iree/schemas \
    -isystem $(IREE_DIR)/third_party/vulkan_memory_allocator/include \
    -I$(IREE_DIR)/third_party/flatcc/include/ \
    -I$(IREE_DIR)/third_party/flatcc/include/flatcc/reflection/ \

CFLAGS=-O3 -fPIE \
    -Wall -Wno-ambiguous-member-template -Wno-char-subscripts -Wno-deprecated-declarations \
    -Wno-extern-c-compat -Wno-gnu-alignof-expression -Wno-gnu-variable-sized-type-not-at-end \
    -Wno-ignored-optimization-argument -Wno-invalid-offsetof -Wno-invalid-source-encoding \
    -Wno-mismatched-tags -Wno-pointer-sign -Wno-reserved-user-defined-literal \
    -Wno-return-type-c-linkage -Wno-self-assign-overloaded -Wno-sign-compare \
    -Wno-signed-unsigned-wchar -Wno-strict-overflow -Wno-trigraphs -Wno-unknown-pragmas \
    -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-unused-const-variable \
    -Wno-unused-function -Wno-unused-local-typedef -Wno-unused-private-field \
    -Wno-user-defined-warnings -Wctad-maybe-unsupported -Wfloat-overflow-conversion \
    -Wfloat-zero-conversion -Wfor-loop-analysis -Wformat-security -Wgnu-redeclared-enum \
    -Wimplicit-fallthrough -Winfinite-recursion -Wliteral-conversion -Wnon-virtual-dtor \
    -Woverloaded-virtual -Wself-assign -Wstring-conversion -Wtautological-overlap-compare \
    -Wthread-safety -Wthread-safety-beta -Wunused-comparison -Wvla -Wno-strict-prototypes \
    -Wno-shadow-uncaptured-local -Wno-gnu-zero-variadic-macro-arguments \
    -Wno-shadow-field-in-constructor -Wno-unreachable-code-return \
    -Wno-missing-variable-declarations -Wno-gnu-label-as-value -Wno-unused-parameter \
    -Wno-unused-variable -Wno-undef \
    -fvisibility=hidden -std=gnu11 -MD \

LDFLAGS=-O2 -g \

LIBS= \
    -lm -lpthread \
    $(IREE_BUILD_DIR)/iree/runtime/libiree_runtime_impl.a \
    $(IREE_BUILD_DIR)/iree/hal/drivers/libiree_hal_drivers_drivers.a \
    $(IREE_BUILD_DIR)/iree/hal/utils/libiree_hal_utils_deferred_command_buffer.a \
    $(IREE_BUILD_DIR)/iree/hal/dylib/registration/libiree_hal_dylib_registration_registration.a \
    $(IREE_BUILD_DIR)/iree/hal/local/loaders/libiree_hal_local_loaders_embedded_library_loader.a \
    $(IREE_BUILD_DIR)/iree/hal/local/elf/libiree_hal_local_elf_elf_module.a \
    $(IREE_BUILD_DIR)/iree/hal/local/elf/libiree_hal_local_elf_arch.a \
    $(IREE_BUILD_DIR)/iree/hal/local/elf/libiree_hal_local_elf_platform.a \
    $(IREE_BUILD_DIR)/iree/hal/dylib/registration/libiree_hal_dylib_registration_sync.a \
    $(IREE_BUILD_DIR)/iree/hal/local/loaders/libiree_hal_local_loaders_system_library_loader.a \
    $(IREE_BUILD_DIR)/iree/hal/local/libiree_hal_local_sync_driver.a \
    $(IREE_BUILD_DIR)/iree/hal/vmvx/registration/libiree_hal_vmvx_registration_registration.a \
    $(IREE_BUILD_DIR)/iree/hal/local/libiree_hal_local_task_driver.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_arena.a \
    $(IREE_BUILD_DIR)/iree/hal/local/libiree_hal_local_event_pool.a \
    $(IREE_BUILD_DIR)/iree/task/libiree_task_api.a \
    $(IREE_BUILD_DIR)/iree/task/libiree_task_task.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_wait_handle.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_atomic_slist.a \
    $(IREE_BUILD_DIR)/third_party/cpuinfo/libcpuinfo.a \
    -pthread \
    $(IREE_BUILD_DIR)/third_party/cpuinfo/deps/clog/libclog.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_threading.a \
    $(IREE_BUILD_DIR)/iree/hal/local/loaders/libiree_hal_local_loaders_vmvx_module_loader.a \
    $(IREE_BUILD_DIR)/iree/hal/local/libiree_hal_local_local.a \
    $(IREE_BUILD_DIR)/iree/modules/vmvx/libiree_modules_vmvx_vmvx.a \
    $(IREE_BUILD_DIR)/iree/base/libiree_base_cc.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_dynamic_library.a \
    -ldl \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_file_path.a \
    $(IREE_BUILD_DIR)/iree/base/libiree_base_logging.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_flags.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_file_io.a \
    $(IREE_BUILD_DIR)/iree/modules/hal/libiree_modules_hal_hal.a \
    $(IREE_BUILD_DIR)/iree/hal/libiree_hal_hal.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_synchronization.a \
    $(IREE_BUILD_DIR)/iree/vm/libiree_vm_bytecode_module.a \
    $(IREE_BUILD_DIR)/iree/vm/libiree_vm_impl.a \
    $(IREE_BUILD_DIR)/build_tools/third_party/flatcc/libflatcc_runtime.a \
    $(IREE_BUILD_DIR)/build_tools/third_party/flatcc/libflatcc_parsing.a \
    -lm  \
    -lpthread \
    $(IREE_BUILD_DIR)/iree/base/libiree_base_base.a \
    $(IREE_BUILD_DIR)/iree/base/internal/libiree_base_internal_fpu_state.a \

SPECIFIC_LIBS= \
	$(IREE_BUILD_DIR)/iree/hal/vmvx/registration/libiree_hal_vmvx_registration_sync.a \
	$(IREE_BUILD_DIR)/iree/hal/cuda/registration/libiree_hal_cuda_registration_registration.a \
	$(IREE_BUILD_DIR)/iree/hal/cuda/libiree_hal_cuda_cuda.a \
	$(IREE_BUILD_DIR)/iree/hal/cuda/libiree_hal_cuda_dynamic_symbols.a \
	$(IREE_BUILD_DIR)/iree/hal/vulkan/registration/libiree_hal_vulkan_registration_registration.a \
	$(IREE_BUILD_DIR)/iree/hal/vulkan/libiree_hal_vulkan_vulkan.a \
	$(IREE_BUILD_DIR)/iree/hal/vulkan/libiree_hal_vulkan_dynamic_symbols.a \
	$(IREE_BUILD_DIR)/iree/hal/vulkan/util/libiree_hal_vulkan_util_arena.a \
	$(IREE_BUILD_DIR)/iree/hal/vulkan/builtin/libiree_hal_vulkan_builtin_builtin.a \

.PRECIOUS: kws_lstm.mhlo.mlir

all: dylib cuda vulkan

run-dylib: dylib
	./kws_lstm kws_lstm-dylib.vmfb dylib

run-cuda: cuda
	./kws_lstm kws_lstm-cuda.vmfb cuda

run-vulkan: vulkan
	./kws_lstm kws_lstm-vulkan.vmfb vulkan

dylib: kws_lstm kws_lstm-dylib.vmfb

cuda: kws_lstm kws_lstm-cuda.vmfb

vulkan: kws_lstm kws_lstm-vulkan.vmfb

kws_lstm: main.c.o tick.c.o
	$(LD) $(LDFLAGS) $^ $(LIBS) $(SPECIFIC_LIBS) $(LIBS) -o $@

main.c.o: main.c
	$(CC) $(IREE_STATIC_OPTS) $(IREE_INCLUDES) $(CFLAGS) -o $@ -c $<

tick.c.o: tick.c
	$(CC) $(CFLAGS) -o $@ -c $<

kws_lstm-dylib.vm.mlir: kws_lstm.mhlo.mlir
	$(IREE_OPT) $< \
		--iree-hal-target-backends=dylib-llvm-aot \
		--iree-mhlo-to-mhlo-preprocessing \
		--iree-mhlo-input-transformation-pipeline \
		--iree-transformation-pipeline -o $@

kws_lstm-cuda.vm.mlir: kws_lstm.mhlo.mlir
	$(IREE_OPT) $< \
		--iree-hal-target-backends=cuda \
		--iree-mhlo-to-mhlo-preprocessing \
		--iree-mhlo-input-transformation-pipeline \
		--iree-transformation-pipeline -o $@

kws_lstm-vulkan.vm.mlir: kws_lstm.mhlo.mlir
	$(IREE_OPT) $< \
		--iree-hal-target-backends=vulkan \
		--iree-mhlo-to-mhlo-preprocessing \
		--iree-mhlo-input-transformation-pipeline \
		--iree-transformation-pipeline -o $@

%.vmfb: %.vm.mlir
	$(IREE_TRANSLATE) $< \
		-iree-vm-ir-to-bytecode-module \
		--iree-vm-bytecode-module-optimize -o $@

kws_lstm.mhlo.mlir: kws_lstm.py
	$(PY) $< | \
	$(MLIRLUS) --normalize --invert-control --sync-to-std | \
	$(TF_OPT) --tf-standard-pipeline \
		--xla-legalize-tf=allow-partial-conversion > $@

clean:
	rm -rf *.c.o
	rm -rf *.c.d
	rm -rf *.vmfb
	rm -rf *.vm.mlir
	rm -rf *.mhlo.mlir
	rm -rf kws_lstm
