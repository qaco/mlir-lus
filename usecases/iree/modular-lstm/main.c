#include <stdio.h>
#include "iree/runtime/api.h"
#include "tick.h"

iree_status_t get_next_output(iree_status_t status,
			      iree_runtime_call_t* call,
			      size_t buffer_size,
			      void *output) {
  iree_hal_buffer_view_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(call,
							     &buffer);
  }
  iree_hal_buffer_mapping_t buffer_mapping;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(iree_hal_buffer_view_buffer(buffer),
                                       IREE_HAL_MEMORY_ACCESS_READ, 0,
                                       IREE_WHOLE_BUFFER, &buffer_mapping);
  }
  if (iree_status_is_ok(status)) {
    memcpy(output, buffer_mapping.contents.data, buffer_size);
  }

  iree_hal_buffer_unmap_range(&buffer_mapping);
  iree_hal_buffer_view_release(buffer);
  
  return status;
}

iree_status_t load_next_input(iree_status_t status,
			      iree_runtime_session_t* session,
			      iree_runtime_call_t* call,
			      size_t *shape,
			      const size_t ndims,
			      size_t elt_size,
			      void* input) {

  iree_hal_buffer_view_t* buffer = NULL;
  iree_hal_dim_t buffer_shape[ndims];
  size_t buffer_size = elt_size;
  for (size_t i = 0; i < ndims; i++) {
    buffer_shape[i] = shape[i];
    buffer_size *= shape[i];
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), buffer_shape,
        IREE_ARRAYSIZE(buffer_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span(input, buffer_size),
        &buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(call, buffer);
  }
  iree_hal_buffer_view_release(buffer);
  return status;
}

iree_status_t rnn_reset(iree_runtime_session_t* session,
			int time[1],
			float state0[1][4],
			float state1[1][4]) {

  const size_t time_size = 1*sizeof(int);
  const size_t state0_size = 1*4*sizeof(float);
  const size_t state1_size = 1*4*sizeof(float);
  
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.model_reset"), &call));

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  status = get_next_output(status, &call, time_size, (void*)time);
  status = get_next_output(status, &call, state0_size, (void*)state0);
  status = get_next_output(status, &call, state1_size, (void*)state1);

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t rnn_step(iree_runtime_session_t *session,
		       float input[1][40],
		       float output[1][4],
		       int time[1],
		       float state0[1][4],
		       float state1[1][4]) {

  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(session,
							    iree_make_cstring_view("module.model_step"),
							    &call));

  iree_status_t status;
  status = iree_ok_status();
  
  size_t input_shape[2] = {1,40};
  const size_t input_elt_size = sizeof(float);
  status = load_next_input(status, session, &call, input_shape, 2, input_elt_size, input);
  
  size_t time_shape[1] = {1};
  const size_t time_elt_size = sizeof(int);
  const size_t time_size = 1*time_elt_size;
  status = load_next_input(status, session, &call, time_shape, 1, time_elt_size, time);

  size_t state0_shape[2] = {1,4};
  const size_t state0_elt_size = sizeof(float);
  const size_t state0_size = 1*4*state0_elt_size;
  status = load_next_input(status, session, &call, state0_shape, 2, state0_elt_size, state0);

  size_t state1_shape[2] = {1,4};
  const size_t state1_elt_size = sizeof(float);
  const size_t state1_size = 1*4*state1_elt_size;
  status = load_next_input(status, session, &call, state1_shape, 2, state1_elt_size, state1);

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  const size_t output_size = 1*4*sizeof(float);
  status = get_next_output(status, &call, output_size, (void*)output);

  status = get_next_output(status, &call, time_size, (void*)time);
  status = get_next_output(status, &call, state0_size, (void*)state0);
  status = get_next_output(status, &call, state1_size, (void*)state1);

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
  // Call functions to manipulate the rnn
  fprintf(stdout, "Calling functions\n\n");

  init_time();
  int time[1];
  float state0[1][4];
  float state1[1][4];
  if (iree_status_is_ok(status)) {
    rnn_reset(session, time, state0, state1);
  }
  while(iree_status_is_ok(status)) {
    float input[1][40];
    float output[1][4];
    status = rnn_step(session, input, output, time, state0, state1);
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
        "Usage: rnn </path/to/rnn.vmfb> <driver_name>\n");
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
