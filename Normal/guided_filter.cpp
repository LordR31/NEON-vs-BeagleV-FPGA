#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// box filter simplu: media locala
void box_filter(const vector<double>& src, vector<double>& dst, int width, int height, int r) {
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

// guided filter: varianta paper He et al.
void guided_filter(const vector<double>& I, const vector<double>& p, vector<double>& q, int width, int height, int r, double eps) {
    int N = width * height;

    vector<double> mean_I(N), mean_p(N);
    vector<double> corr_I(N), corr_Ip(N);
    vector<double> var_I(N), cov_Ip(N);
    vector<double> a(N), b(N);
    vector<double> mean_a(N), mean_b(N);

    // medii
    box_filter(I, mean_I, width, height, r);
    box_filter(p, mean_p, width, height, r);

    for (int i = 0; i < N; i++) {
        corr_I[i] = I[i] * I[i];
        corr_Ip[i] = I[i] * p[i];
    }

    box_filter(corr_I, corr_I, width, height, r);
    box_filter(corr_Ip, corr_Ip, width, height, r);

    for (int i = 0; i < N; i++) {
        var_I[i] = corr_I[i] - mean_I[i] * mean_I[i];
        cov_Ip[i] = corr_Ip[i] - mean_I[i] * mean_p[i];
    }

    for (int i = 0; i < N; i++) {
        a[i] = cov_Ip[i] / (var_I[i] + eps);
        b[i] = mean_p[i] - a[i] * mean_I[i];
    }

    box_filter(a, mean_a, width, height, r);
    box_filter(b, mean_b, width, height, r);

    for (int i = 0; i < N; i++) {
        q[i] = mean_a[i] * I[i] + mean_b[i];
    }
}

int main() {
    int width, height, channels;

    // citire imagine input.png in grayscale
    unsigned char* input = stbi_load("input.png", &width, &height, &channels, 1);
    if (!input) {
        cerr << "Eroare la citire input.png\n";
        return 1;
    }

    cout << "Imagine citita: " << width << " x " << height << endl;

    int N = width * height;
    vector<double> I(N), p(N), q(N);

    for (int i = 0; i < N; i++) {
        I[i] = input[i] / 255.0;
        p[i] = I[i];
    }

    int r = 5;
    double eps = 0.01;

    guided_filter(I, p, q, width, height, r, eps);

    vector<unsigned char> output(N);
    for (int i = 0; i < N; i++) {
        double val = q[i] * 255.0;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        output[i] = static_cast<unsigned char>(val);
    }

    stbi_write_png("output.png", width, height, 1, output.data(), width);

    cout << "Filtru aplicat cu succes. Rezultat: output.png\n";

    stbi_image_free(input);
    return 0;
}
