#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Headers/stb_image.h"
#include "../Headers/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // Required for std::max and std::min
#include <time.h>
#include <arm_neon.h> // Include NEON header
// #include <omp.h>      // Removed OpenMP header
using namespace std;

// Helper function for 16-byte aligned memory allocation (for NEON performance)
template <typename T>
std::vector<T> create_aligned_vector(size_t size) {
    // For proper alignment with NEON (16-byte for float32x4_t),
    // we should use aligned allocators or ensure std::vector's internal buffer is aligned.
    // std::vector might not guarantee 16-byte alignment by default on all systems for float.
    // However, for typical Raspberry Pi OS builds, it often works well enough or the compiler
    // optimizes around it. For maximum safety/performance, consider custom allocator
    // or posix_memalign for raw arrays if needed.
    return std::vector<T>(size);
}

// Separable Box Filter (Scalar inner loops)
void box_filter_separable_neon(const vector<float>& src, vector<float>& dst, int width, int height, int r) {
    // Intermediate buffer for horizontal pass results
    vector<float> temp_h_pass = create_aligned_vector<float>(width * height);

    // Horizontal Pass (Scalar inner loop)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            int count = 0;
            for (int dx = -r; dx <= r; dx++) {
                int x = j + dx;
                if (x >= 0 && x < width) {
                    sum += src[i * width + x];
                    count++;
                }
            }
            temp_h_pass[i * width + j] = sum / (float)count;
        }
    }

    // Vertical Pass (Scalar inner loop)
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            float sum = 0.0f;
            int count = 0;
            for (int dy = -r; dy <= r; dy++) {
                int y = i + dy;
                if (y >= 0 && y < height) {
                    sum += temp_h_pass[y * width + j];
                    count++;
                }
            }
            dst[i * width + j] = sum / (float)count;
        }
    }
}

// Guided Filter: variant from He et al.'s paper
// Using float for processing, assuming conversion happens in main
void guided_filter_neon(const vector<float>& I, const vector<float>& p, vector<float>& q, int width, int height, int r, float eps) {
    int N = width * height;

    vector<float> mean_I = create_aligned_vector<float>(N);
    vector<float> mean_p = create_aligned_vector<float>(N);
    vector<float> corr_I = create_aligned_vector<float>(N);
    vector<float> corr_Ip = create_aligned_vector<float>(N);
    vector<float> var_I = create_aligned_vector<float>(N);
    vector<float> cov_Ip = create_aligned_vector<float>(N);
    vector<float> a = create_aligned_vector<float>(N);
    vector<float> b = create_aligned_vector<float>(N);
    vector<float> mean_a = create_aligned_vector<float>(N);
    vector<float> mean_b = create_aligned_vector<float>(N);

    // Apply separable box filter
    box_filter_separable_neon(I, mean_I, width, height, r);
    box_filter_separable_neon(p, mean_p, width, height, r);

    // NEON optimized element-wise operations (process 8 floats at a time)
    for (int i = 0; i < N; i += 8) {
        if (i + 7 < N) { // Process two float32x4_t vectors (8 floats)
            float32x4_t I_vec1 = vld1q_f32(&I[i]);
            float32x4_t p_vec1 = vld1q_f32(&p[i]);
            float32x4_t I_vec2 = vld1q_f32(&I[i+4]);
            float32x4_t p_vec2 = vld1q_f32(&p[i+4]);

            float32x4_t corr_I_vec1 = vmulq_f32(I_vec1, I_vec1); // I * I
            float32x4_t corr_Ip_vec1 = vmulq_f32(I_vec1, p_vec1); // I * p
            float32x4_t corr_I_vec2 = vmulq_f32(I_vec2, I_vec2);
            float32x4_t corr_Ip_vec2 = vmulq_f32(I_vec2, p_vec2);

            vst1q_f32(&corr_I[i], corr_I_vec1);
            vst1q_f32(&corr_Ip[i], corr_Ip_vec1);
            vst1q_f32(&corr_I[i+4], corr_I_vec2);
            vst1q_f32(&corr_Ip[i+4], corr_Ip_vec2);
        } else { // Handle remaining elements (tail) with scalar ops
            for (int k = i; k < N; ++k) {
                corr_I[k] = I[k] * I[k];
                corr_Ip[k] = I[k] * p[k];
            }
        }
    }
    
    // Apply separable box filter on corr_I and corr_Ip
    box_filter_separable_neon(corr_I, corr_I, width, height, r);
    box_filter_separable_neon(corr_Ip, corr_Ip, width, height, r);

    for (int i = 0; i < N; i += 8) {
        if (i + 7 < N) {
            float32x4_t mean_I_vec1 = vld1q_f32(&mean_I[i]);
            float32x4_t mean_p_vec1 = vld1q_f32(&mean_p[i]);
            float32x4_t corr_I_vec1 = vld1q_f32(&corr_I[i]);
            float32x4_t corr_Ip_vec1 = vld1q_f32(&corr_Ip[i]);
            
            float32x4_t mean_I_vec2 = vld1q_f32(&mean_I[i+4]);
            float32x4_t mean_p_vec2 = vld1q_f32(&mean_p[i+4]);
            float32x4_t corr_I_vec2 = vld1q_f32(&corr_I[i+4]);
            float32x4_t corr_Ip_vec2 = vld1q_f32(&corr_Ip[i+4]);

            float32x4_t var_I_vec1 = vsubq_f32(corr_I_vec1, vmulq_f32(mean_I_vec1, mean_I_vec1));
            float32x4_t cov_Ip_vec1 = vsubq_f32(corr_Ip_vec1, vmulq_f32(mean_I_vec1, mean_p_vec1));
            float32x4_t var_I_vec2 = vsubq_f32(corr_I_vec2, vmulq_f32(mean_I_vec2, mean_I_vec2));
            float32x4_t cov_Ip_vec2 = vsubq_f32(corr_Ip_vec2, vmulq_f32(mean_I_vec2, mean_p_vec2));

            vst1q_f32(&var_I[i], var_I_vec1);
            vst1q_f32(&cov_Ip[i], cov_Ip_vec1);
            vst1q_f32(&var_I[i+4], var_I_vec2);
            vst1q_f32(&cov_Ip[i+4], cov_Ip_vec2);
        } else {
            for (int k = i; k < N; ++k) {
                var_I[k] = corr_I[k] - mean_I[k] * mean_I[k];
                cov_Ip[k] = corr_Ip[k] - mean_I[k] * mean_p[k];
            }
        }
    }

    float32x4_t eps_vec = vdupq_n_f32(eps);

    for (int i = 0; i < N; i += 8) {
        if (i + 7 < N) {
            float32x4_t var_I_vec1 = vld1q_f32(&var_I[i]);
            float32x4_t cov_Ip_vec1 = vld1q_f32(&cov_Ip[i]);
            float32x4_t mean_p_vec1 = vld1q_f32(&mean_p[i]);
            float32x4_t mean_I_vec1 = vld1q_f32(&mean_I[i]);
            
            float32x4_t var_I_vec2 = vld1q_f32(&var_I[i+4]);
            float32x4_t cov_Ip_vec2 = vld1q_f32(&cov_Ip[i+4]);
            float32x4_t mean_p_vec2 = vld1q_f32(&mean_p[i+4]);
            float32x4_t mean_I_vec2 = vld1q_f32(&mean_I[i+4]);

            float32x4_t a_vec1 = vdivq_f32(cov_Ip_vec1, vaddq_f32(var_I_vec1, eps_vec));
            float32x4_t b_vec1 = vsubq_f32(mean_p_vec1, vmulq_f32(a_vec1, mean_I_vec1));
            float32x4_t a_vec2 = vdivq_f32(cov_Ip_vec2, vaddq_f32(var_I_vec2, eps_vec));
            float32x4_t b_vec2 = vsubq_f32(mean_p_vec2, vmulq_f32(a_vec2, mean_I_vec2));

            vst1q_f32(&a[i], a_vec1);
            vst1q_f32(&b[i], b_vec1);
            vst1q_f32(&a[i+4], a_vec2);
            vst1q_f32(&b[i+4], b_vec2);
        } else {
            for (int k = i; k < N; ++k) {
                a[k] = cov_Ip[k] / (var_I[k] + eps);
                b[k] = mean_p[k] - a[k] * mean_I[k];
            }
        }
    }

    // Apply separable box filter on a and b
    box_filter_separable_neon(a, mean_a, width, height, r);
    box_filter_separable_neon(b, mean_b, width, height, r);

    for (int i = 0; i < N; i += 8) {
        if (i + 7 < N) {
            float32x4_t mean_a_vec1 = vld1q_f32(&mean_a[i]);
            float32x4_t I_vec1 = vld1q_f32(&I[i]);
            float32x4_t mean_b_vec1 = vld1q_f32(&mean_b[i]);
            
            float32x4_t mean_a_vec2 = vld1q_f32(&mean_a[i+4]);
            float32x4_t I_vec2 = vld1q_f32(&I[i+4]);
            float32x4_t mean_b_vec2 = vld1q_f32(&mean_b[i+4]);

            float32x4_t q_vec1 = vaddq_f32(vmulq_f32(mean_a_vec1, I_vec1), mean_b_vec1);
            float32x4_t q_vec2 = vaddq_f32(vmulq_f32(mean_a_vec2, I_vec2), mean_b_vec2);

            vst1q_f32(&q[i], q_vec1);
            vst1q_f32(&q[i+4], q_vec2);
        } else {
            for (int k = i; k < N; ++k) {
                q[k] = mean_a[k] * I[k] + mean_b[k];
            }
        }
    }
}

int main() {
    int width, height, channels_actual;
    const int desired_channels = 1;

    clock_t start;
    clock_t end;
    double time_taken;

    unsigned char* guide_image_data = stbi_load("Images/Input/target.png", &width, &height, &channels_actual, desired_channels);
    if (!guide_image_data) {
        cerr << "Eroare la citire target.png\n";
        return 1;
    }
    cout << "Imagine de ghidare citita: " << width << " x " << height << " cu " << channels_actual << " canale (cerut: " << desired_channels << ")." << endl;

    unsigned char* process_image_data = stbi_load("Images/Input/input.png", &width, &height, &channels_actual, desired_channels);
    if (!process_image_data) {
        cerr << "Eroare la citire input.png\n";
        stbi_image_free(guide_image_data);
        return 1;
    }
    cout << "Imagine de procesat citita: " << width << " x " << height << " cu " << channels_actual << " canal (cerut: " << desired_channels << ")." << endl;

    int N = width * height;

    // Use float vectors for NEON processing. Adjusted size for 8-element processing.
    vector<float> I_grayscale_float = create_aligned_vector<float>(N);
    vector<float> p_grayscale_float = create_aligned_vector<float>(N);

    // Normalize images from 0-255 to 0.0-1.0 with NEON (8 elements at a time)
    float32x4_t scale_vec_0_1 = vdupq_n_f32(1.0f / 255.0f); 

    for (int i = 0; i < N; i += 8) {
        if (i + 7 < N) {
            // Load 8 uint8_t values
            uint8x8_t guide_u8 = vld1_u8(guide_image_data + i);
            uint8x8_t process_u8 = vld1_u8(process_image_data + i);

            // Widen to uint16x8_t (8 * 16-bit integers)
            uint16x8_t guide_u16 = vmovl_u8(guide_u8);
            uint16x8_t process_u16 = vmovl_u8(process_u8);

            // Split into two uint16x4_t and convert to uint32x4_t
            uint32x4_t guide_u32_low = vmovl_u16(vget_low_u16(guide_u16));
            uint32x4_t guide_u32_high = vmovl_u16(vget_high_u16(guide_u16));
            uint32x4_t process_u32_low = vmovl_u16(vget_low_u16(process_u16));
            uint32x4_t process_u32_high = vmovl_u16(vget_high_u16(process_u16));

            // Convert uint32x4_t to float32x4_t and scale
            float32x4_t guide_float_vec_low = vmulq_f32(vcvtq_f32_u32(guide_u32_low), scale_vec_0_1);
            float32x4_t guide_float_vec_high = vmulq_f32(vcvtq_f32_u32(guide_u32_high), scale_vec_0_1);
            float32x4_t process_float_vec_low = vmulq_f32(vcvtq_f32_u32(process_u32_low), scale_vec_0_1);
            float32x4_t process_float_vec_high = vmulq_f32(vcvtq_f32_u32(process_u32_high), scale_vec_0_1);

            vst1q_f32(&I_grayscale_float[i], guide_float_vec_low);
            vst1q_f32(&I_grayscale_float[i+4], guide_float_vec_high);
            vst1q_f32(&p_grayscale_float[i], process_float_vec_low);
            vst1q_f32(&p_grayscale_float[i+4], process_float_vec_high);
        } else { // Handle remaining elements (tail) with scalar ops
            for (int k = i; k < N; ++k) {
                I_grayscale_float[k] = guide_image_data[k] / 255.0f;
                p_grayscale_float[k] = process_image_data[k] / 255.0f;
            }
        }
    }

    float r_float = 1.0f;
    float eps_float = 0.1f;

    vector<float> q_grayscale_float = create_aligned_vector<float>(N);

    cout << "Applying guided filter on grayscale images (NEON)..." << endl;
    start = clock();
    guided_filter_neon(I_grayscale_float, p_grayscale_float, q_grayscale_float, width, height, static_cast<int>(r_float), eps_float);
    end = clock();
    time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;

    vector<unsigned char> output_image_data(N);

    // Convert float [0,1] to unsigned char [0,255] with NEON (8 elements at a time)
    float32x4_t scale_255_vec = vdupq_n_f32(255.0f);
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    float32x4_t max_vec = vdupq_n_f32(255.0f);

    for (int i = 0; i < N; i += 8) {
        if (i + 7 < N) {
            // Load two float32x4_t vectors
            float32x4_t q_vec_low = vld1q_f32(&q_grayscale_float[i]);
            float32x4_t q_vec_high = vld1q_f32(&q_grayscale_float[i+4]);

            // Scale and clamp
            q_vec_low = vmaxq_f32(vminq_f32(vmulq_f32(q_vec_low, scale_255_vec), max_vec), zero_vec);
            q_vec_high = vmaxq_f32(vminq_f32(vmulq_f32(q_vec_high, scale_255_vec), max_vec), zero_vec);

            // Convert to uint32x4_t
            uint32x4_t q_u32_low = vcvtq_u32_f32(q_vec_low);
            uint32x4_t q_u32_high = vcvtq_u32_f32(q_vec_high);

            // Narrow to uint16x4_t (saturating)
            uint16x4_t q_u16_low = vqmovn_u32(q_u32_low);
            uint16x4_t q_u16_high = vqmovn_u32(q_u32_high);

            // Combine to uint16x8_t for next narrowing step
            uint16x8_t q_u16_combined = vcombine_u16(q_u16_low, q_u16_high);

            // Narrow to uint8x8_t (saturating)
            uint8x8_t q_u8_result = vqmovn_u16(q_u16_combined);

            // Store the 8 uint8_t values
            vst1_u8(&output_image_data[i], q_u8_result);
        } else { // Handle remaining elements (tail)
            for (int k = i; k < N; ++k) {
                output_image_data[k] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, q_grayscale_float[k] * 255.0f)));
            }
        }
    }

    stbi_write_png("Images/Output/output_grayscale_neon.png", width, height, 1, output_image_data.data(), width);

    cout << "Filter applied successfully. Result: Images/Output/output_grayscale_neon.png\n";
    cout << "Filter application time: " << time_taken << "s\n";

    stbi_image_free(guide_image_data);
    stbi_image_free(process_image_data);
    return 0;
}
