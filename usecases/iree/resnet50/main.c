#include <stdio.h>
#include "iree/runtime/api.h"
#include "tick.h"

iree_status_t resnet_step(iree_runtime_session_t *session,
			  float input[224][224][3],
			  float output[1][1000]) {

  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.resnet50_step"), &call));

  iree_hal_buffer_view_t* input_buffer = NULL;
  const iree_hal_dim_t input_shape[3] = {224, 224, 3};
  const size_t input_size = 224 * 224 * 3 * sizeof(float);

  iree_status_t status;
  // input_buffer
  status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), input_shape,
        IREE_ARRAYSIZE(input_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)(input), input_size),
        &input_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, input_buffer);
  }
  iree_hal_buffer_view_release(input_buffer);

  // call
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }
  
  // output_buffer
  iree_hal_buffer_view_t* output_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(&call, &output_buffer);
  }
  iree_hal_buffer_mapping_t output_buffer_mapping;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(iree_hal_buffer_view_buffer(output_buffer),
                                       IREE_HAL_MEMORY_ACCESS_READ, 0,
                                       IREE_WHOLE_BUFFER, &output_buffer_mapping);
  }
  if (iree_status_is_ok(status)) {
    memcpy(output, output_buffer_mapping.contents.data, 1 * 1000 * sizeof(float));
  }
  iree_hal_buffer_unmap_range(&output_buffer_mapping);
  iree_hal_buffer_view_release(output_buffer);

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t run_sample(iree_string_view_t bytecode_module_path,
                         iree_string_view_t driver_name) {
  
  iree_status_t status = iree_ok_status();

  //===-------------------------------------------------------------------===//
  // Instance configuration (this should be shared across sessions).
  fprintf(stdout, "Configuring IREE runtime instance and '%s' device\n",
          driver_name.data);
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_create(&instance_options,
                                          iree_allocator_system(), &instance);
  }
  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_try_create_default_device(
        instance, driver_name, &device);
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Session configuration (one per loaded module to hold module state).
  fprintf(stdout, "Creating IREE runtime session\n");
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }
  iree_hal_device_release(device);

  fprintf(stdout, "Loading bytecode module at '%s'\n",
          bytecode_module_path.data);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(
        session, bytecode_module_path.data);
  }

  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Call functions to manipulate the resnet
  fprintf(stdout, "Calling functions\n\n");

  init_time();
  int i = 0;
  while(iree_status_is_ok(status)) {
    float input[224][224][3];
    float output[1][1000];
    status = resnet_step(session, input, output);
    tick();
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Cleanup.
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  //===-------------------------------------------------------------------===//

  return status;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(
        stderr,
        "Usage: resnet50 </path/to/resnet50.vmfb> <driver_name>\n");
    fprintf(stderr, "  (See the README for this sample for details)\n ");
    return -1;
  }

  iree_string_view_t bytecode_module_path = iree_make_cstring_view(argv[1]);
  iree_string_view_t driver_name = iree_make_cstring_view(argv[2]);

  iree_status_t result = run_sample(bytecode_module_path, driver_name);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  fprintf(stdout, "\nSuccess!\n");
  return 0;
}
