
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Headers/stb_image.h"
#include "../Headers/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <arm_neon.h>

using namespace std;

// box filter simplu: media locala
void box_filter(const vector<double>& src, vector<double>& dst, int width, int height, int r) {
    #pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double sum = 0.0;
            int count = 0;
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
            dst[i * width + j] = sum / count;
        }
    }
}

void guided_filter(const vector<double>& I, const vector<double>& p, vector<double>& q, int width, int height, int r, double eps) {
    int N = width * height;

    vector<double> mean_I(N), mean_p(N);
    vector<double> corr_I(N), corr_Ip(N);
    vector<double> var_I(N), cov_Ip(N);
    vector<double> a(N), b(N);
    vector<double> mean_a(N), mean_b(N);

    box_filter(I, mean_I, width, height, r);
    box_filter(p, mean_p, width, height, r);

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        corr_I[i] = I[i] * I[i];
        corr_Ip[i] = I[i] * p[i];
    }

    box_filter(corr_I, corr_I, width, height, r);
    box_filter(corr_Ip, corr_Ip, width, height, r);

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        var_I[i] = corr_I[i] - mean_I[i] * mean_I[i];
        cov_Ip[i] = corr_Ip[i] - mean_I[i] * mean_p[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        a[i] = cov_Ip[i] / (var_I[i] + eps);
        b[i] = mean_p[i] - a[i] * mean_I[i];
    }

    box_filter(a, mean_a, width, height, r);
    box_filter(b, mean_b, width, height, r);

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        q[i] = mean_a[i] * I[i] + mean_b[i];
    }
}

int main() {
    int width, height, channels_actual;
    const int desired_channels = 1;

    unsigned char* guide_image_data = stbi_load("Images/Input/target.png", &width, &height, &channels_actual, desired_channels);
    if (!guide_image_data) {
        cerr << "Eroare la citire target.png\n";
        return 1;
    }
    cout << "Imagine de ghidare citita: " << width << " x " << height << endl;

    unsigned char* process_image_data = stbi_load("Images/Input/input.png", &width, &height, &channels_actual, desired_channels);
    if (!process_image_data) {
        cerr << "Eroare la citire input.png\n";
        stbi_image_free(guide_image_data);
        return 1;
    }
    cout << "Imagine de procesat citita: " << width << " x " << height << endl;

    int N = width * height;
    vector<double> I_grayscale(N);
    vector<double> p_grayscale(N);

    // NEON normalization (vectorized load and scale)
    #pragma omp parallel for
    for (int i = 0; i < N; i += 4) {
        uint8x8_t guide_u8 = vld1_u8(guide_image_data + i);
        uint8x8_t process_u8 = vld1_u8(process_image_data + i);

        uint16x8_t guide_u16 = vmovl_u8(guide_u8);
        uint16x8_t process_u16 = vmovl_u8(process_u8);

        float32x4_t guide_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(guide_u16)));
        float32x4_t process_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(process_u16)));

        float32x4_t scale = vdupq_n_f32(1.0f / 255.0f);
        guide_f32 = vmulq_f32(guide_f32, scale);
        process_f32 = vmulq_f32(process_f32, scale);

        vst1q_f64(&I_grayscale[i], vcvt_f64_f32(vget_low_f32(guide_f32)));
        vst1q_f64(&p_grayscale[i], vcvt_f64_f32(vget_low_f32(process_f32)));
    }

    int r = 1;
    double eps = 0.1;
    vector<double> q_grayscale(N);

    cout << "Aplicare filtru ghidat pe imagini grayscale..." << endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    guided_filter(I_grayscale, p_grayscale, q_grayscale, width, height, r, eps);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double>time_taken = end_time - start_time;

    vector<unsigned char> output_image_data(N);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        output_image_data[i] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, q_grayscale[i] * 255.0)));
    }

    stbi_write_png("Images/Output/output_grayscale.png", width, height, 1, output_image_data.data(), width);

    cout << "Filtru aplicat cu succes. Rezultat: Images/Output/output_grayscale.png\n";
    cout << "Timp aplicare filtru: " << time_taken.count() << "s\n";

    stbi_image_free(guide_image_data);
    stbi_image_free(process_image_data);
    return 0;
}
