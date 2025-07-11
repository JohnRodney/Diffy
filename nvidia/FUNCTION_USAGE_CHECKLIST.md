# NVIDIA Directory Function Usage Checklist

**Purpose**: Systematically verify which functions are actually called to identify dead code before making changes.

**Process**:

1. Go through each function one by one
2. Search the entire codebase for calls to that function
3. Check the box if the function IS called somewhere
4. Leave unchecked if the function is NOT called anywhere

---

## nvidia/experiments/parameter_sweep.py

### ParameterSweep class methods:

- [ ] `__init__(self, base_output_dir=None)`
- [ ] `define_parameter_grid(self)`
- [ ] `estimate_experiment_time(self, params, num_epochs=10000)`
- [ ] `run_single_experiment(self, params, experiment_id, max_epochs=10000)`
- [ ] `load_real_color_data(self, vector_length)`
- [ ] `color_to_vector(self, color_data, tokenizer, vector_length)`
- [ ] `create_dummy_training_data(self, num_items, vector_length)`
- [ ] `run_time_limited_sweep(self, max_runtime_hours=20)`
- [ ] `save_aggregate_results(self)`
- [ ] `analyze_results(self)`
- [ ] `create_downloadable_summary(self)`

### Standalone functions:

- [ ] `main()`

---

## nvidia/gpu_networks/gpu_text_autoencoder.py

### Stub functions (seem to be import placeholders):

- [ ] `launch_elementwise_kernel(func, *args)`
- [ ] `launch_matrix_multiply(a, b, c)`
- [ ] `launch_2d_kernel(func, *args)`
- [ ] `gpu_copy_array(a, b)`
- [ ] `gpu_add_bias(a, b, c)`
- [ ] `gpu_leaky_relu_forward(a, b, c)`
- [ ] `gpu_mse_gradient(a, b, c)`
- [ ] `gpu_gradient_clip(a, b)`
- [ ] `gpu_update_weights(a, b, c)`
- [ ] `gpu_leaky_relu_backward(a, b, c, d)`
- [ ] `gpu_mse_loss(a, b, c)`
- [ ] `gpu_scale_weights(a, b)`
- [ ] `launch_matrix_multiply_rtx5090(a, b, c)`
- [ ] `launch_elementwise_rtx5090(a, b, c, op)`
- [ ] `validate_rtx5090_utilization(a, b)`

### GPUTextAutoencoder class methods:

- [ ] `__init__(self, vector_length, hidden_layer_count, bottleneck_size, alpha=0.01)`
- [ ] `_calculate_layer_sizes(self)`
- [ ] `_initialize_weights_on_gpu(self)`
- [ ] `_allocate_gpu_memory(self)`
- [ ] `cleanup_gpu_memory(self)`
- [ ] `_count_parameters(self)`
- [ ] `forward_gpu(self, input_batch_gpu)`
- [ ] `backward_gpu(self, input_batch_gpu, target_batch_gpu, learning_rate, grad_clip_norm)`
- [ ] `train_batch_gpu_efficient(self, batch_data_cpu, learning_rate, grad_clip_norm)`
- [ ] `backward_gpu_safe(self, input_batch_gpu, target_batch_gpu, learning_rate, grad_clip_norm)`
- [ ] `train_batch_gpu(self, input_batch_cpu, target_batch_cpu, learning_rate, grad_clip_norm)`
- [ ] `predict_gpu(self, input_batch_cpu)`
- [ ] `get_weights_cpu(self)`
- [ ] `set_weights_cpu(self, weights_cpu, biases_cpu)`
- [ ] `train_batch_gpu_only(self, batch_data_gpu, learning_rate, grad_clip_norm)`

### Standalone functions:

- [ ] `test_gpu_autoencoder()`

---

## nvidia/kernels/matrix_kernels.py

### CUDA Kernel functions:

- [ ] `gpu_matrix_multiply(A, B, C)`
- [ ] `gpu_add_bias(matrix, bias, output)`
- [ ] `gpu_leaky_relu_forward(input_data, output, alpha)`
- [ ] `gpu_leaky_relu_backward(input_data, grad_output, grad_input, alpha)`
- [ ] `gpu_mse_loss(predicted, target, loss_output)`
- [ ] `gpu_mse_gradient(predicted, target, gradient)`
- [ ] `gpu_gradient_clip(gradient, max_norm)`
- [ ] `gpu_update_weights(weights, gradients, learning_rate)`
- [ ] `gpu_copy_array(source, dest)`
- [ ] `gpu_elementwise_add(A, B, C)`
- [ ] `gpu_elementwise_multiply(A, B, C)`
- [ ] `gpu_elementwise_activate(A, C)`
- [ ] `gpu_elementwise_decay(A, C, decay_factor)`
- [ ] `gpu_reduce_sum(A, result)`
- [ ] `gpu_fused_linear_relu(input_data, weights, bias, output, alpha)`

### RTX 5090 Optimization functions:

- [ ] `launch_matrix_multiply_rtx5090(A, B, C)`
- [ ] `launch_elementwise_rtx5090(A, B, C, operation_type='add')`
- [ ] `launch_reduce_rtx5090(A, result)`
- [ ] `get_rtx5090_config(matrix_size)`
- [ ] `validate_rtx5090_utilization(blocks_per_grid, threads_per_block)`

---

## Summary Stats

- **Total functions**: 49
- **Functions called**: 0 (to be filled as we check)
- **Functions not called**: 0 (to be filled as we check)
- **Percentage used**: 0% (to be calculated)

---

## Notes

- Functions marked as "stub functions" appear to be import placeholders
- Class methods need to be searched as `self.method_name` or `instance.method_name`
- Look for both direct calls and dynamic calls (getattr, etc.)
- Check imports and from statements too
