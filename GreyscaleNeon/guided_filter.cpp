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
using namespace std;

// Helper function for 16-byte aligned memory allocation (for NEON performance)
// This is good practice for NEON, though std::vector might manage alignment on its own depending on compiler/libstdc++ version.
template <typename T>
std::vector<T> create_aligned_vector(size_t size) {
    std::vector<T> vec(size);
    // On some systems/compilers, std::vector might already be aligned for its element type.
    // For explicit alignment, one would typically use aligned_alloc or similar.
    // For simplicity and common use cases, we'll rely on std::vector default behavior here
    // but be aware that for maximum performance, explicit alignment might be needed.
    return vec;
}

// box filter simplu: media locala
// Optimized for NEON (using float instead of double)
void box_filter_neon(const vector<float>& src, vector<float>& dst, int width, int height, int r) {
    int r_sq = (2 * r + 1) * (2 * r + 1); // Area of the box filter, used for division

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            int count = 0; // count is not strictly needed if we always divide by r_sq, but for boundary cases it is.
                           // For NEON, handling boundaries and non-full vectors is complex.
                           // Here, we simplify and assume full box is typically within bounds for NEON path.
                           // A more robust NEON box filter often uses integral images or separate horizontal/vertical passes.

            // Standard scalar loop for simplicity in the inner box calculation,
            // as vectorizing this small inner loop with bounds checking is very complex for NEON.
            // The outer loops (i, j) are the primary targets for SIMD/parallelization in typical image filters.
            // For a true NEON box filter, separable filters (horizontal then vertical) or integral images are more common.
            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {
                    int y = i + dy;
                    int x = j + dx;
                    if (y >= 0 && y < height && x >= 0 && x < width) {
                        sum += src[y * width + x];
                        count++;
                    }
                }
            }
            dst[i * width + j] = sum / (float)count; // Divide by actual count due to boundaries
        }
    }
}


// guided filter: varianta paper He et al.
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

    // medii - these still use the (scalar) box_filter_neon
    box_filter_neon(I, mean_I, width, height, r);
    box_filter_neon(p, mean_p, width, height, r);
    box_filter_neon(corr_I, corr_I, width, height, r); // Apply on corr_I
    box_filter_neon(corr_Ip, corr_Ip, width, height, r); // Apply on corr_Ip
    box_filter_neon(a, mean_a, width, height, r); // Apply on a
    box_filter_neon(b, mean_b, width, height, r); // Apply on b

    // NEON optimized element-wise operations (process 4 floats at a time)
    for (int i = 0; i < N; i += 4) {
        if (i + 3 < N) { // Process full vectors
            float32x4_t I_vec = vld1q_f32(&I[i]);
            float32x4_t p_vec = vld1q_f32(&p[i]);

            float32x4_t corr_I_vec = vmulq_f32(I_vec, I_vec); // I * I
            float32x4_t corr_Ip_vec = vmulq_f32(I_vec, p_vec); // I * p

            vst1q_f32(&corr_I[i], corr_I_vec);
            vst1q_f32(&corr_Ip[i], corr_Ip_vec);
        } else { // Handle remaining elements (tail)
            for (int k = i; k < N; ++k) {
                corr_I[k] = I[k] * I[k];
                corr_Ip[k] = I[k] * p[k];
            }
        }
    }

    for (int i = 0; i < N; i += 4) {
        if (i + 3 < N) {
            float32x4_t mean_I_vec = vld1q_f32(&mean_I[i]);
            float32x4_t mean_p_vec = vld1q_f32(&mean_p[i]);
            float32x4_t corr_I_vec = vld1q_f32(&corr_I[i]);
            float32x4_t corr_Ip_vec = vld1q_f32(&corr_Ip[i]);

            float32x4_t var_I_vec = vsubq_f32(corr_I_vec, vmulq_f32(mean_I_vec, mean_I_vec)); // corr_I - mean_I * mean_I
            float32x4_t cov_Ip_vec = vsubq_f32(corr_Ip_vec, vmulq_f32(mean_I_vec, mean_p_vec)); // corr_Ip - mean_I * mean_p

            vst1q_f32(&var_I[i], var_I_vec);
            vst1q_f32(&cov_Ip[i], cov_Ip_vec);
        } else {
            for (int k = i; k < N; ++k) {
                var_I[k] = corr_I[k] - mean_I[k] * mean_I[k];
                cov_Ip[k] = corr_Ip[k] - mean_I[k] * mean_p[k];
            }
        }
    }

    float32x4_t eps_vec = vdupq_n_f32(eps); // Load epsilon into a NEON register

    for (int i = 0; i < N; i += 4) {
        if (i + 3 < N) {
            float32x4_t var_I_vec = vld1q_f32(&var_I[i]);
            float32x4_t cov_Ip_vec = vld1q_f32(&cov_Ip[i]);
            float32x4_t mean_p_vec = vld1q_f32(&mean_p[i]);
            float32x4_t mean_I_vec = vld1q_f32(&mean_I[i]);

            float32x4_t a_vec = vdivq_f32(cov_Ip_vec, vaddq_f32(var_I_vec, eps_vec)); // cov_Ip / (var_I + eps)
            float32x4_t b_vec = vsubq_f32(mean_p_vec, vmulq_f32(a_vec, mean_I_vec)); // mean_p - a * mean_I

            vst1q_f32(&a[i], a_vec);
            vst1q_f32(&b[i], b_vec);
        } else {
            for (int k = i; k < N; ++k) {
                a[k] = cov_Ip[k] / (var_I[k] + eps);
                b[k] = mean_p[k] - a[k] * mean_I[k];
            }
        }
    }

    for (int i = 0; i < N; i += 4) {
        if (i + 3 < N) {
            float32x4_t mean_a_vec = vld1q_f32(&mean_a[i]);
            float32x4_t I_vec = vld1q_f32(&I[i]);
            float32x4_t mean_b_vec = vld1q_f32(&mean_b[i]);

            float32x4_t q_vec = vaddq_f32(vmulq_f32(mean_a_vec, I_vec), mean_b_vec); // mean_a * I + mean_b

            vst1q_f32(&q[i], q_vec);
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

    // Use float vectors for NEON processing
    vector<float> I_grayscale_float = create_aligned_vector<float>(N);
    vector<float> p_grayscale_float = create_aligned_vector<float>(N);

    // Normalizăm ambele imagini de la 0-255 la 0.0-1.0 cu NEON
    for (int i = 0; i < N; i += 4) {
        if (i + 3 < N) {
            uint8x8_t guide_u8 = vld1_u8(guide_image_data + i); // Load 8 uint8 values
            uint8x8_t process_u8 = vld1_u8(process_image_data + i);

            // Convert uint8 to float32. This is a bit tricky with direct NEON intrinsics for float,
            // often involving intermediate int32 conversions or scalar loops.
            // For simplicity, we'll convert 4 at a time using specific conversions.
            // Load 4 uint8s, convert to int16, then int32, then float32.
            uint8x8_t guide_low = vget_low_u8(guide_u8);
            uint8x8_t process_low = vget_low_u8(process_u8);

            uint16x4_t guide_u16_low = vmovl_u8(guide_low);
            uint16x4_t process_u16_low = vmovl_u8(process_low);

            uint32x4_t guide_u32_low = vmovl_u16(guide_u16_low);
            uint32x4_t process_u32_low = vmovl_u16(process_u16_low);

            float32x4_t guide_float_vec = vcvtq_f32_u32(guide_u32_low);
            float32x4_t process_float_vec = vcvtq_f32_u32(process_u32_low);

            float32x4_t scale_vec = vdupq_n_f32(1.0f / 255.0f);
            guide_float_vec = vmulq_f32(guide_float_vec, scale_vec);
            process_float_vec = vmulq_f32(process_float_vec, scale_vec);

            vst1q_f32(&I_grayscale_float[i], guide_float_vec);
            vst1q_f32(&p_grayscale_float[i], process_float_vec);

            // If N is not a multiple of 4, handle the remaining 0-3 elements
            if (i + 4 >= N && N % 4 != 0) { // Check if this is the last chunk and it's not full
                for (int k = i + 4; k < N; ++k) {
                    I_grayscale_float[k] = guide_image_data[k] / 255.0f;
                    p_grayscale_float[k] = process_image_data[k] / 255.0f;
                }
            }
        } else { // Handle remaining elements (tail) if N is not a multiple of 4
            for (int k = i; k < N; ++k) {
                I_grayscale_float[k] = guide_image_data[k] / 255.0f;
                p_grayscale_float[k] = process_image_data[k] / 255.0f;
            }
        }
    }


    float r_float = 1.0f;
    float eps_float = 0.1f;

    vector<float> q_grayscale_float = create_aligned_vector<float>(N);

    cout << "Aplicare filtru ghidat pe imagini grayscale (NEON)..." << endl;
    start = clock();
    guided_filter_neon(I_grayscale_float, p_grayscale_float, q_grayscale_float, width, height, static_cast<int>(r_float), eps_float);
    end = clock();
    time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;

    vector<unsigned char> output_image_data(N);

    // Convert float [0,1] to unsigned char [0,255] with NEON
    float32x4_t scale_255_vec = vdupq_n_f32(255.0f);
    for (int i = 0; i < N; i += 4) {
        if (i + 3 < N) {
            float32x4_t q_vec = vld1q_f32(&q_grayscale_float[i]);
            q_vec = vmulq_f32(q_vec, scale_255_vec);

            // Clamp values (0-255)
            float32x4_t zero_vec = vdupq_n_f32(0.0f);
            float32x4_t max_vec = vdupq_n_f32(255.0f);
            q_vec = vmaxq_f32(q_vec, zero_vec); // max(q, 0)
            q_vec = vminq_f32(q_vec, max_vec);   // min(q, 255)

            // Convert to unsigned integer, then store as unsigned char
            uint32x4_t q_u32_vec = vcvtq_u32_f32(q_vec);
            uint16x4_t q_u16_vec = vqmovn_u32(q_u32_vec); // Narrow to 16-bit, saturating
            uint8x8_t q_u8_vec = vqmovn_u16(vcombine_u16(q_u16_vec, q_u16_vec)); // Narrow to 8-bit, saturating (need 8 for vst1_u8)

            vst1_u8(&output_image_data[i], vget_low_u8(q_u8_vec)); // Store lower 4
            // For full 8, need to load 8, convert 8, then store 8.
            // Simplified here to just handle 4 elements since we load 4 at a time.
        } else {
            for (int k = i; k < N; ++k) {
                output_image_data[k] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, q_grayscale_float[k] * 255.0f)));
            }
        }
    }

    stbi_write_png("Images/Output/output_grayscale_neon.png", width, height, 1, output_image_data.data(), width);

    cout << "Filtru aplicat cu succes. Rezultat: Images/Output/output_grayscale_neon.png\n";
    cout << "Timp aplicare filtru: " << time_taken << "s\n";

    stbi_image_free(guide_image_data);
    stbi_image_free(process_image_data);
    return 0;
}