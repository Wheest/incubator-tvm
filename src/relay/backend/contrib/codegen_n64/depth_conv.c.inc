
#define CSOURCE_RSP_DEPTH_CONV2D_OP(                                                           \
    p_ID_, p_INPUT_PARTITION_HEIGHT_, p_INPUT_PARTITION_WIDTH_, p_OUTPUT_PARTITION_HEIGHT_,    \
    p_OUTPUT_PARTITION_WIDTH_, p_IN_H_, p_IN_W_, p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_,  \
    p_PAD_T_, p_PAD_B_, p_PAD_L_, p_PAD_R_, p_STRIDE_, p_INPUT_ZERO_POINT_)                    \
  void p_ID_(int8_t* input_data, int8_t* weights, int32_t* dest) {                             \
    RSPDepthConvTiledPadded(dest, input_data, weights, p_INPUT_PARTITION_HEIGHT_, p_IN_H_,     \
                            p_IN_W_, p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_,              \
                            p_OUTPUT_PARTITION_HEIGHT_, p_INPUT_PARTITION_WIDTH_,              \
                            p_OUTPUT_PARTITION_WIDTH_, p_PAD_T_, p_PAD_B_, p_PAD_L_, p_PAD_R_, \
                            p_STRIDE_, p_INPUT_ZERO_POINT_);                                   \
  }

#define CSOURCE_RSP_DEPTH_CONV2D_NO_PAD_OP(                                                    \
    p_ID_, p_INPUT_PARTITION_HEIGHT_, p_INPUT_PARTITION_WIDTH_, p_OUTPUT_PARTITION_HEIGHT_,    \
    p_OUTPUT_PARTITION_WIDTH_, p_IN_H_, p_IN_W_, p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_,  \
    p_STRIDE_, p_INPUT_ZERO_POINT_)                                                            \
  void p_ID_(int8_t* input_data, int8_t* weights, int32_t* dest) {                             \
    RSPDepthConvTiled(dest, input_data, weights, p_INPUT_PARTITION_HEIGHT_, p_IN_H_, p_IN_W_,  \
                      p_IN_C_, p_OUT_H_, p_OUT_W_, p_K_H_, p_K_W_, p_OUTPUT_PARTITION_HEIGHT_, \
                      p_INPUT_PARTITION_WIDTH_, p_OUTPUT_PARTITION_WIDTH_, p_STRIDE_,          \
                      p_INPUT_ZERO_POINT_);                                                    \
  }

#define MIN(a, b) ((a) < (b) ? (a) : (b))

extern uint32_t vec_id;
enum {
  DMAWeights = 0x0,
  SetArgs = 0x1,
  DMAInputs = 0x2,
  DepthConvPad = 0x3,
  DepthConv = 0x4,
};

void copy_output_slice_to_full(int32_t* dest, const int32_t* output_pad_part,
                               const int output_partition_height, const int output_partition_width,
                               const int out_w, const int h_slice_num, const int w_slice_num,
                               const int in_depth, const int depth_slice_num,
                               const int max_output_partition_height) {
  /*
  Copies a slice into the final destination with the appropriate stride
  using for-loops.
  */
  const int slice_depth = 8;  // 8 channels per slice
  for (int y = 0; y < max_output_partition_height; y++) {
    for (int x = 0; x < output_partition_width; x++) {
      // Calculate the initial destination index with offsets and
      // strided gaps
      int dest_idx_y = h_slice_num * output_partition_height + y;
      int dest_idx_x = w_slice_num * output_partition_width + x;

      int32_t* dest_ptr = &dest[(dest_idx_y * out_w * in_depth) + (dest_idx_x * in_depth) +
                                (depth_slice_num * slice_depth)];
      // cast to uint16_t as the RSP puts the upper 16 bits of all 8
      // elements in the first 64 bits and the lower 16 bits in the
      // second 64 bits
      uint16_t* src_ptr =
          (uint16_t*)&output_pad_part[y * output_partition_width * slice_depth + x * slice_depth];
      for (int c = 0; c < slice_depth; c++) {
        dest_ptr[c] = ((uint32_t)src_ptr[c] << 16) |
                      src_ptr[c + 8];  // reconstruct from upper and lower 2 bytes
      }
    }
  }
}

void generate_padded_slices_with_depth_slice(int8_t* data, int16_t* padded_input_partition,
                                             const int h_slice_num, const int w_slice_num,
                                             const int depth_slice_num, const int in_h,
                                             const int in_w, const int in_depth,
                                             const int slice_height, const int slice_width,
                                             const int pad_t, const int pad_l, const int overlap,
                                             const int input_zero_point) {
  // start_[h/w]_pad are the starting indices of the slice in the
  // (virtual) fully padded input data.  We then need to deterimine the
  // starting indices in the actual input data, which may be less than
  // the padded input data.
  typedef int16_t input_slice_t;
  const int start_h_pad = h_slice_num * (slice_height - overlap);
  const int start_w_pad = w_slice_num * (slice_width - overlap);

  // The starting indices of the non-padded values in the slice
  const int slice_start_h = (start_h_pad == 0) ? pad_t : 0;
  const int slice_start_w = (start_w_pad == 0) ? pad_l : 0;
  const int num_h_elems = slice_height - slice_start_h;
  const int num_w_elems = slice_width - slice_start_w;

  // Calculate depth start and end based on the depth_slice
  const int depth_start = depth_slice_num * 8;           // Start at the depth slice of 8 channels
  const int depth_end = MIN(depth_start + 8, in_depth);  // Ensure we don't go beyond in_depth

  // Clear the slice area
  memset(padded_input_partition, 0, sizeof(input_slice_t) * slice_height * slice_width * 8);

  // The starting indices in the full non-padded input data
  int h_start = (start_h_pad == 0) ? start_h_pad : start_h_pad - pad_t;
  int h_end = MIN(h_start + num_h_elems, in_h);

  int w_start = (start_w_pad == 0) ? start_w_pad : start_w_pad - pad_l;
  int w_end = MIN(w_start + num_w_elems, in_w);

  // Copy data to the padded slice
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      for (int d = depth_start; d < depth_end; ++d) {
        input_slice_t value = (input_slice_t)data[(h * in_w * in_depth) + (w * in_depth) + d];
        value += input_zero_point;
        int slice_row = slice_start_h + (h - h_start);
        int slice_col = slice_start_w + (w - w_start);
        int slice_depth = d - depth_start;
        padded_input_partition[(slice_row * slice_width * 8) + (slice_col * 8) + slice_depth] =
            value;
      }
    }
  }
}

void generate_slices_with_depth_slice(int8_t* data, int16_t* padded_input_partition,
                                      const int h_slice_num, const int w_slice_num,
                                      const int depth_slice_num, const int in_h, const int in_w,
                                      const int in_depth, const int slice_height,
                                      const int slice_width, const int overlap) {
  // No padded version
  // start_[h/w]_pad are the starting indices of the slice in the
  // (virtual) fully padded input data.  We then need to deterimine the
  // starting indices in the actual input data, which may be less than
  // the padded input data.
  typedef int8_t input_slice_t;
  const int pad_t, pad_l = 0;
  const int start_h_pad = h_slice_num * (slice_height - overlap);
  const int start_w_pad = w_slice_num * (slice_width - overlap);

  // The starting indices of the non-padded values in the slice
  const int slice_start_h = (start_h_pad == 0) ? pad_t : 0;
  const int slice_start_w = (start_w_pad == 0) ? pad_l : 0;
  const int num_h_elems = slice_height - slice_start_h;
  const int num_w_elems = slice_width - slice_start_w;

  // Calculate depth start and end based on the depth_slice
  const int depth_start = depth_slice_num * 8;           // Start at the depth slice of 8 channels
  const int depth_end = MIN(depth_start + 8, in_depth);  // Ensure we don't go beyond in_depth

  // Clear the slice area
  memset(padded_input_partition, 0, sizeof(input_slice_t) * slice_height * slice_width * 8);

  // The starting indices in the full non-padded input data
  int h_start = (start_h_pad == 0) ? start_h_pad : start_h_pad - pad_t;
  int h_end = MIN(h_start + num_h_elems, in_h);

  int w_start = (start_w_pad == 0) ? start_w_pad : start_w_pad - pad_l;
  int w_end = MIN(w_start + num_w_elems, in_w);

  // Copy data to the padded slice
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      for (int d = depth_start; d < depth_end; ++d) {
        input_slice_t value = (input_slice_t)data[(h * in_w * in_depth) + (w * in_depth) + d];
        int slice_row = slice_start_h + (h - h_start);
        int slice_col = slice_start_w + (w - w_start);
        int slice_depth = d - depth_start;
        padded_input_partition[(slice_row * slice_width * 8) + (slice_col * 8) + slice_depth] =
            value;
      }
    }
  }
}

static inline void RSPDepthConvTiledPadded(
    int32_t* dest, int8_t* input_data, int8_t* weights, const int input_partition_height,
    const int in_h, const int in_w, const int in_c, const int out_h, const int out_w, const int k_h,
    const int k_w, const int output_partition_height, const int input_partition_width,
    const int output_partition_width, const int pad_t, const int pad_b, const int pad_l,
    const int pad_r, const int stride, const int input_zero_point) {
  // pad_r and pad_b are implicit from the
  // input_partition_{height,width}

  // Requires that weights have been reshaped offline to be
  // (out_c // 8, kernel_height * kernel_width, 8)
  extern uint32_t vec_id;

  // raise not implemented error if in_c is not 8
  if (in_c % 8 != 0) {
    printf("Error: in_c must be divisible 8\n");
    return;
  }

  /* printf("Weights reshape\n"); */
  /* printInt8ArrayHWC(weights, 3, 3, 8); */

  const int wbytes = sizeof(int8_t);
  // TODO we are doing int16 conversion on CPU for now
  // because of input_zero_point and how it interacts with padded values
  // The non-padded version should be faster
  typedef int16_t input_slice_t;
  const int in_bytes = sizeof(input_slice_t);
  const int out_bytes = sizeof(int32_t);

  const int in_part_size = input_partition_height * input_partition_width * 8 * in_bytes;
  input_slice_t* input_pad_part = malloc_uncached_aligned(8, in_part_size);

  const int out_part_size = output_partition_height * output_partition_width * 8;
  int32_t* output_pad_part = malloc_uncached_aligned(8, out_part_size * out_bytes);

  const int w_part_size = k_h * k_w * 8 * wbytes;

  const int overlap = k_h - stride;
  // the number of bytes to offset our input data pointer by between
  // elements in the same window.  I.e., if our pointer is at 0, how
  // does it get to 5?
  // |*0, 1, 2,| 3, 4
  // | 5, 6, 7,| 8, 9
  // |10,11,12,|13,14
  //  15,16,17,18,19
  const int w_slide_byte_offset = (8 * in_bytes * input_partition_width);

  // when we slide right for a new window, the number of bytes to offset
  // our input data pointer
  const int w_window_stride = 8 * in_bytes * stride;
  // when we slide  for a new window, the number of bytes to offset our
  // input data pointer
  const int h_window_stride = (8 * in_bytes) *                  /*account for our depth of 8*/
                              ((input_partition_width * stride) /*move the pointer vertically down*/
                               - /*move the pointer back left to the start of the row*/
                               (output_partition_width * stride));

  rspq_write(vec_id, SetArgs, output_partition_height, output_partition_width, w_window_stride,
             w_slide_byte_offset, h_window_stride, w_part_size, in_part_size);

  const int num_h_partitions =
      ceil((float)(in_h + pad_t + pad_b - overlap) / (input_partition_height - overlap));

  const int num_w_partitions =
      ceil((float)(in_w + pad_l + pad_r - overlap) / (input_partition_width - overlap));

  for (int depth_slice = 0; depth_slice < in_c / 8; depth_slice++) {
    int remaining_out_values = out_h * out_w * 8;  // Remaining values to copy for this depth slice
    int max_output_partition_height = output_partition_height;

    // Copy weights for this depth slice to the RSP once
    rspq_write(vec_id, DMAWeights, PhysicalAddr(&weights[depth_slice * k_h * k_w * 8]),
               w_part_size);

    /* printf("Weight slice %d: \n", depth_slice); */
    /* printInt8ArrayHWC(&weights[depth_slice * k_h * k_w * 8], k_h,
     * k_w, 8); */

    // Generate first input  slice
    generate_padded_slices_with_depth_slice(
        input_data, input_pad_part, 0, 0, depth_slice, in_h, in_w, in_c, input_partition_height,
        input_partition_width, pad_t, pad_l, overlap, input_zero_point);

    // Copy-in the first slice
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);

    for (int h_slice_count = 0; h_slice_count < num_h_partitions; h_slice_count++) {
      for (int w_slice_count = 0; w_slice_count < num_w_partitions; w_slice_count++) {
        rspq_wait();

        /* printf("Input slice %d, %d : \n", h_slice_count,
         * w_slice_count); */
        /* printInt8ArrayHWC(input_pad_part, input_partition_height, */
        /*                   input_partition_width, 8); */

        // Process the padded partition on the RSP
        rspq_write(vec_id, DepthConvPad, PhysicalAddr(output_pad_part), out_part_size * out_bytes);

        // Generate next slice while the current one is being
        // processed
        if ((w_slice_count + 1) < num_w_partitions) {
          generate_padded_slices_with_depth_slice(input_data, input_pad_part, h_slice_count,
                                                  w_slice_count + 1, depth_slice, in_h, in_w, in_c,
                                                  input_partition_height, input_partition_width,
                                                  pad_t, pad_l, overlap, input_zero_point);
        } else if ((h_slice_count + 1) < num_h_partitions) {
          generate_padded_slices_with_depth_slice(input_data, input_pad_part, h_slice_count + 1, 0,
                                                  depth_slice, in_h, in_w, in_c,
                                                  input_partition_height, input_partition_width,
                                                  pad_t, pad_l, overlap, input_zero_point);
        }

        // Copy back the processed partition to the final outputs,
        // DMA the new input slice to the RSP at the same time
        rspq_wait();
        rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);
        /* printf("DMA inputs again!\n"); */

        copy_output_slice_to_full(dest, output_pad_part, output_partition_height,
                                  output_partition_width, out_w, h_slice_count, w_slice_count, in_c,
                                  depth_slice, max_output_partition_height);

        remaining_out_values -= output_partition_height * output_partition_width * 8;
      }

      // May need to adjust the max_output_partition_height for the
      // final slice(s)
      if (remaining_out_values < (out_part_size * num_w_partitions)) {
        // Cover the case where our final output slice is larger than
        // required
        max_output_partition_height = remaining_out_values / (out_w * 8);
      }
    }
  }
  free_uncached(input_pad_part);
  free_uncached(output_pad_part);
}

static inline void RSPDepthConvTiled(
    int32_t* dest, int8_t* input_data, int8_t* weights, const int input_partition_height,
    const int in_h, const int in_w, const int in_c, const int out_h, const int out_w, const int k_h,
    const int k_w, const int output_partition_height, const int input_partition_width,
    const int output_partition_width, const int stride, const int input_zero_point) {
  // Requires that weights have been reshaped offline to be
  // (out_c // 8, kernel_height * kernel_width, 8)
  extern uint32_t vec_id;

  // raise not implemented error if in_c is not 8
  if (in_c % 8 != 0) {
    printf("Error: in_c must be divisible 8\n");
    return;
  }

  /* printf("Weights reshape\n"); */
  /* printInt8ArrayHWC(weights, 3, 3, 8); */

  const int wbytes = sizeof(int8_t);

  typedef int8_t input_slice_t;
  const int in_bytes = sizeof(input_slice_t);
  const int out_bytes = sizeof(int32_t);

  const int in_part_size = input_partition_height * input_partition_width * 8 * in_bytes;
  input_slice_t* input_pad_part = malloc_uncached_aligned(8, in_part_size);

  const int out_part_size = output_partition_height * output_partition_width * 8;
  int32_t* output_pad_part = malloc_uncached_aligned(8, out_part_size * out_bytes);

  const int w_part_size = k_h * k_w * 8 * wbytes;

  const int overlap = k_h - stride;
  // the number of bytes to offset our input data pointer by between
  // elements in the same window.  I.e., if our pointer is at 0, how
  // does it get to 5?
  // |*0, 1, 2,| 3, 4
  // | 5, 6, 7,| 8, 9
  // |10,11,12,|13,14
  //  15,16,17,18,19
  const int w_slide_byte_offset = (8 * in_bytes * input_partition_width);

  // when we slide right for a new window, the number of bytes to offset
  // our input data pointer
  const int w_window_stride = 8 * in_bytes * stride;
  // when we slide  for a new window, the number of bytes to offset our
  // input data pointer
  const int h_window_stride = (8 * in_bytes) *                  /*account for our depth of 8*/
                              ((input_partition_width * stride) /*move the pointer vertically down*/
                               - /*move the pointer back left to the start of the row*/
                               (output_partition_width * stride));

  rspq_write(vec_id, SetArgs, output_partition_height, output_partition_width, w_window_stride,
             w_slide_byte_offset, h_window_stride, w_part_size, in_part_size);

  const int num_h_partitions = ceil((float)(in_h - overlap) / (input_partition_height - overlap));

  const int num_w_partitions = ceil((float)(in_w - overlap) / (input_partition_width - overlap));

  for (int depth_slice = 0; depth_slice < in_c / 8; depth_slice++) {
    int remaining_out_values = out_h * out_w * 8;  // Remaining values to copy for this depth slice
    int max_output_partition_height = output_partition_height;

    // Copy weights for this depth slice to the RSP once
    rspq_write(vec_id, DMAWeights, PhysicalAddr(&weights[depth_slice * k_h * k_w * 8]),
               w_part_size);

    /* printf("Weight slice %d: \n", depth_slice); */
    /* printInt8ArrayHWC(&weights[depth_slice * k_h * k_w * 8], k_h,
     * k_w, 8); */

    // Generate first input  slice
    generate_slices_with_depth_slice(input_data, input_pad_part, 0, 0, depth_slice, in_h, in_w,
                                     in_c, input_partition_height, input_partition_width, overlap);

    // Copy-in the first slice
    rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);

    for (int h_slice_count = 0; h_slice_count < num_h_partitions; h_slice_count++) {
      for (int w_slice_count = 0; w_slice_count < num_w_partitions; w_slice_count++) {
        rspq_wait();

        /* printf("Input slice %d, %d : \n", h_slice_count,
         * w_slice_count); */
        /* printInt8ArrayHWC(input_pad_part, input_partition_height, */
        /*                   input_partition_width, 8); */

        // Process the padded partition on the RSP
        rspq_write(vec_id, DepthConv, PhysicalAddr(output_pad_part), out_part_size * out_bytes,
                   input_zero_point);

        // Generate next slice while the current one is being
        // processed
        if ((w_slice_count + 1) < num_w_partitions) {
          generate_slices_with_depth_slice(input_data, input_pad_part, h_slice_count,
                                           w_slice_count + 1, depth_slice, in_h, in_w, in_c,
                                           input_partition_height, input_partition_width, overlap);
        } else if ((h_slice_count + 1) < num_h_partitions) {
          generate_slices_with_depth_slice(input_data, input_pad_part, h_slice_count + 1, 0,
                                           depth_slice, in_h, in_w, in_c, input_partition_height,
                                           input_partition_width, overlap);
        }

        // Copy back the processed partition to the final outputs,
        // DMA the new input slice to the RSP at the same time
        rspq_wait();
        rspq_write(vec_id, DMAInputs, PhysicalAddr(input_pad_part), in_part_size);
        /* printf("DMA inputs again!\n"); */

        copy_output_slice_to_full(dest, output_pad_part, output_partition_height,
                                  output_partition_width, out_w, h_slice_count, w_slice_count, in_c,
                                  depth_slice, max_output_partition_height);

        remaining_out_values -= output_partition_height * output_partition_width * 8;
      }

      // May need to adjust the max_output_partition_height for the
      // final slice(s)
      if (remaining_out_values < (out_part_size * num_w_partitions)) {
        // Cover the case where our final output slice is larger than
        // required
        max_output_partition_height = remaining_out_values / (out_w * 8);
      }
    }
  }
  free_uncached(input_pad_part);
  free_uncached(output_pad_part);
}
