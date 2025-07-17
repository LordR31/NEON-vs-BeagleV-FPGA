#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Headers/stb_image.h"
#include "../Headers/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <arm_neon.h>

using namespace std;

// box filter simplu cu NEON pe linie (vectorizat pe X)
void box_filter_line_neon(const float* src, float* dst, int width, int height, int r) {
    for (int i = 0; i < height; i++) {
        const float* row = src + i * width;
        float* dst_row = dst + i * width;

        for (int j = 0; j < width; j++) {
            int x0 = max(0, j - r);
            int x1 = min(width - 1, j + r);

            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            int x = x0;

            // vectorizat pe 4
            for (; x + 3 <= x1; x += 4) {
                float32x4_t vals = vld1q_f32(row + x);
                sum_vec = vaddq_f32(sum_vec, vals);
            }

            float sum = vgetq_lane_f32(sum_vec, 0) +
                        vgetq_lane_f32(sum_vec, 1) +
                        vgetq_lane_f32(sum_vec, 2) +
                        vgetq_lane_f32(sum_vec, 3);

            // scalar pt pix rest
            for (; x <= x1; x++) {
                sum += row[x];
            }

            int count = x1 - x0 + 1;
            dst_row[j] = sum / count;
        }
    }
}

// box filter complet 2D: aplicat pe randuri + pe coloane
void box_filter(const vector<float>& src, vector<float>& dst, int width, int height, int r) {
    vector<float> temp(width * height);

    // pe randuri
    box_filter_line_neon(src.data(), temp.data(), width, height, r);

    // pe coloane (transpus)
    for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
            int y0 = max(0, i - r);
            int y1 = min(height - 1, i + r);

            float sum = 0.0f;
            for (int y = y0; y <= y1; y++) {
                sum += temp[y * width + j];
            }
            int count = y1 - y0 + 1;
            dst[i * width + j] = sum / count;
        }
    }
}

// guided filter NEON
void guided_filter(const vector<float>& I, const vector<float>& p, vector<float>& q, int width, int height, int r, float eps) {
    int N = width * height;

    vector<float> mean_I(N), mean_p(N);
    vector<float> corr_I(N), corr_Ip(N);
    vector<float> var_I(N), cov_Ip(N);
    vector<float> a(N), b(N);
    vector<float> mean_a(N), mean_b(N);

    box_filter(I, mean_I, width, height, r);
    box_filter(p, mean_p, width, height, r);

    // NEON vector multiply I*I si I*p
    int i = 0;
    for (; i + 3 < N; i += 4) {
        float32x4_t vi = vld1q_f32(I.data() + i);
        float32x4_t vp = vld1q_f32(p.data() + i);
        vst1q_f32(corr_I.data() + i, vmulq_f32(vi, vi));
        vst1q_f32(corr_Ip.data() + i, vmulq_f32(vi, vp));
    }
    for (; i < N; i++) {
        corr_I[i] = I[i] * I[i];
        corr_Ip[i] = I[i] * p[i];
    }

    box_filter(corr_I, corr_I, width, height, r);
    box_filter(corr_Ip, corr_Ip, width, height, r);

    // NEON pt var_I, cov_Ip, a, b
    i = 0;
    for (; i + 3 < N; i += 4) {
        float32x4_t mcI = vld1q_f32(corr_I.data() + i);
        float32x4_t mI = vld1q_f32(mean_I.data() + i);
        float32x4_t mp = vld1q_f32(mean_p.data() + i);
        float32x4_t mcIp = vld1q_f32(corr_Ip.data() + i);

        float32x4_t mI2 = vmulq_f32(mI, mI);
        float32x4_t varI = vsubq_f32(mcI, mI2);
        float32x4_t covIp = vsubq_f32(mcIp, vmulq_f32(mI, mp));

        vst1q_f32(var_I.data() + i, varI);
        vst1q_f32(cov_Ip.data() + i, covIp);

        // a = cov / (var + eps)
        float32x4_t epsv = vdupq_n_f32(eps);
        float32x4_t a_v = vdivq_f32(covIp, vaddq_f32(varI, epsv));
        float32x4_t b_v = vsubq_f32(mp, vmulq_f32(a_v, mI));

        vst1q_f32(a.data() + i, a_v);
        vst1q_f32(b.data() + i, b_v);
    }

    for (; i < N; i++) {
        var_I[i] = corr_I[i] - mean_I[i] * mean_I[i];
        cov_Ip[i] = corr_Ip[i] - mean_I[i] * mean_p[i];
        a[i] = cov_Ip[i] / (var_I[i] + eps);
        b[i] = mean_p[i] - a[i] * mean_I[i];
    }

    box_filter(a, mean_a, width, height, r);
    box_filter(b, mean_b, width, height, r);

    // final q = mean_a * I + mean_b
    i = 0;
    for (; i + 3 < N; i += 4) {
        float32x4_t ma = vld1q_f32(mean_a.data() + i);
        float32x4_t mb = vld1q_f32(mean_b.data() + i);
        float32x4_t vi = vld1q_f32(I.data() + i);
        vst1q_f32(q.data() + i, vaddq_f32(vmulq_f32(ma, vi), mb));
    }
    for (; i < N; i++) {
        q[i] = mean_a[i] * I[i] + mean_b[i];
    }
}

int main() {
    int width, height, channels_actual;

    // We'll load both images as grayscale (1 channel)
    const int desired_channels = 1;

    unsigned char* guide_image_data = stbi_load("Images/Input/target.png", &width, &height, &channels_actual, desired_channels);
    if (!guide_image_data) {
        cerr << "Eroare la citire target.png\n";
        return 1;
    }

    unsigned char* process_image_data = stbi_load("Images/Input/input.png", &width, &height, &channels_actual, desired_channels);
    if (!process_image_data) {
        cerr << "Eroare la citire input.png\n";
        stbi_image_free(guide_image_data);
        return 1;
    }

    // Since we're loading both as grayscale, their dimensions should match if they originated from the same source
    // No need for a separate width_I, height_I etc.
    int N = width * height;

    vector<float> I_grayscale(N); // Guide image (target.png)
    vector<float> p_grayscale(N); // Input image to be filtered (input.png)

    for (int i = 0; i < N; i++) {
        I_grayscale[i] = guide_image_data[i] / 255.0f;
        p_grayscale[i] = process_image_data[i] / 255.0f;
    }

    int r = 5;
    float eps = 0.1f;

    vector<float> q_grayscale(N); // Output grayscale image

    cout << "Aplicare filtru ghidat (grayscale)..." << endl;
    start = clock();
    guided_filter(I_grayscale, p_grayscale, q_grayscale, width, height, r, eps);
    end = clock();
    time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;

    vector<unsigned char> output_image_data(N); // Output will be 1 channel (grayscale)
    for (int i = 0; i < N; i++) {
        // Clamp and convert back to 0-255 range
        output_image_data[i] = static_cast<unsigned char>(max(0.0f, min(255.0f, q_grayscale[i] * 255.0f)));
    }

    // Write the grayscale output image
    stbi_write_png("Images/Output/output_grayscale.png", width, height, 1, output_image_data.data(), width);

    cout << "Filtru NEON aplicat cu succes (grayscale): Images/Output/output_grayscale.png\n";
    cout << "Timp aplicare filtru: " << time_taken << "s\n";
    
    stbi_image_free(guide_image_data);
    stbi_image_free(process_image_data);
    return 0;
}