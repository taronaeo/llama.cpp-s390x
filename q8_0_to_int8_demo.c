/*
 * Example: Converting Q8_0 to raw int8 values
 * This demonstrates how to extract the actual int8 data from Q8_0 blocks
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Simplified definitions for demonstration
#define QK8_0 32
typedef uint16_t ggml_half;  // Simplified for demo

typedef struct {
    ggml_half d;       // delta (scaling factor)
    int8_t qs[QK8_0];  // quantized values
} block_q8_0;

// Helper function to convert half to float (simplified)
float fp16_to_fp32(ggml_half h) {
    // This is a simplified conversion - actual implementation is more complex
    union { uint32_t i; float f; } u;
    u.i = ((uint32_t)h << 16);
    return u.f;
}

// Function to extract raw int8 values from Q8_0
void extract_int8_from_q8_0(const block_q8_0 * q8_blocks, 
                            int8_t * int8_output, 
                            float * scaling_factors,
                            int64_t num_elements) {
    
    const int nb = num_elements / QK8_0;  // Number of blocks
    
    for (int i = 0; i < nb; i++) {
        // Extract the scaling factor for this block
        if (scaling_factors) {
            scaling_factors[i] = fp16_to_fp32(q8_blocks[i].d);
        }
        
        // Copy the raw int8 values
        for (int j = 0; j < QK8_0; j++) {
            int8_output[i * QK8_0 + j] = q8_blocks[i].qs[j];
        }
    }
}

// Function to reconstruct floats from extracted int8 + scaling factors
void reconstruct_from_int8(const int8_t * int8_data,
                          const float * scaling_factors,
                          float * output,
                          int64_t num_elements) {
    
    const int nb = num_elements / QK8_0;
    
    for (int i = 0; i < nb; i++) {
        const float scale = scaling_factors[i];
        
        for (int j = 0; j < QK8_0; j++) {
            output[i * QK8_0 + j] = int8_data[i * QK8_0 + j] * scale;
        }
    }
}

// Function to convert Q8_0 to a different int8 quantization scheme
void q8_0_to_uniform_int8(const block_q8_0 * q8_blocks,
                         int8_t * uniform_int8,
                         float * global_scale,
                         int64_t num_elements) {
    
    const int nb = num_elements / QK8_0;
    
    // First pass: find global min/max after dequantization
    float global_min = INFINITY, global_max = -INFINITY;
    
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(q8_blocks[i].d);
        
        for (int j = 0; j < QK8_0; j++) {
            float val = q8_blocks[i].qs[j] * d;
            if (val < global_min) global_min = val;
            if (val > global_max) global_max = val;
        }
    }
    
    // Calculate global scaling factor
    float range = global_max - global_min;
    *global_scale = range / 255.0f;  // Map to 0-255 range for uint8, or -128 to 127 for int8
    float offset = global_min;
    
    // Second pass: quantize to uniform int8
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(q8_blocks[i].d);
        
        for (int j = 0; j < QK8_0; j++) {
            float val = q8_blocks[i].qs[j] * d;
            // Quantize to -128 to 127 range
            int quantized = roundf((val - offset) / (*global_scale) - 128.0f);
            uniform_int8[i * QK8_0 + j] = (int8_t)fmaxf(-128, fminf(127, quantized));
        }
    }
}

int main() {
    printf("Q8_0 to int8 conversion examples:\n\n");
    
    printf("Method 1: Extract raw int8 values + scaling factors\n");
    printf("- Each block has its own scaling factor\n");
    printf("- Preserves the block-wise quantization structure\n");
    printf("- Most accurate reconstruction\n\n");
    
    printf("Method 2: Convert to uniform int8 quantization\n");
    printf("- Single global scaling factor\n");
    printf("- Traditional int8 quantization\n");
    printf("- May lose some precision\n\n");
    
    printf("Raw int8 values in Q8_0 are already int8_t!\n");
    printf("The 'magic' is in the per-block scaling factors.\n");
    
    return 0;
}
