#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Headers/stb_image.h"
#include "../Headers/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // Required for std::max and std::min
#include <time.h>
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
    int width, height, channels_actual; // Use single width/height as both images must match
    const int desired_channels = 1; // We want 1 channel for grayscale

    clock_t start;
    clock_t end;
    double time_taken;

    // CALEA CĂTRE IMAGINEA DE GHIDARE (ACUM GRAYSCALE)
    unsigned char* guide_image_data = stbi_load("Images/Input/target.png", &width, &height, &channels_actual, desired_channels);
    if (!guide_image_data) {
        cerr << "Eroare la citire target.png\n";
        return 1;
    }
    cout << "Imagine de ghidare citita: " << width << " x " << height << " cu " << channels_actual << " canale (cerut: " << desired_channels << ")." << endl;

    // CALEA CĂTRE IMAGINEA DE PROCESAT (ACUM GRAYSCALE)
    unsigned char* process_image_data = stbi_load("Images/Input/input.png", &width, &height, &channels_actual, desired_channels);
    if (!process_image_data) {
        cerr << "Eroare la citire input.png\n";
        stbi_image_free(guide_image_data);
        return 1;
    }
    cout << "Imagine de procesat citita: " << width << " x " << height << " cu " << channels_actual << " canal (cerut: " << desired_channels << ")." << endl;

    // No need to check dimensions, as stbi_load (when loading the second image) would overwrite
    // width and height if they were different. We'll assume for simplicity that
    // both images are of the same dimensions for this grayscale-only version.
    // In a robust application, you'd load both images separately and then compare dimensions.

    int N = width * height; // Numarul total de pixeli

    // Vectori pentru imaginea de ghidare și de procesat (ambele grayscale)
    vector<double> I_grayscale(N);
    vector<double> p_grayscale(N);

    // Normalizăm ambele imagini de la 0-255 la 0.0-1.0
    for (int i = 0; i < N; i++) {
        I_grayscale[i] = guide_image_data[i] / 255.0;
        p_grayscale[i] = process_image_data[i] / 255.0;
    }

    int r = 1;          // Raza filtrului (am mărit-o un pic, 0.5 e cam mic pentru o rază în pixeli)
    double eps = 0.1; // Parametru de regularizare (valoare tipică)

    // Vector pentru imaginea filtrată de ieșire (grayscale)
    vector<double> q_grayscale(N);

    // APELAREA FILTRULUI GHIDAT PENTRU IMAGINILE GRAYSCALE
    cout << "Aplicare filtru ghidat pe imagini grayscale..." << endl;
    start = clock();
    guided_filter(I_grayscale, p_grayscale, q_grayscale, width, height, r, eps);
    end = clock();
    time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    // Acum trebuie să convertim q_grayscale înapoi la unsigned char
    vector<unsigned char> output_image_data(N); // 1 canal pentru imaginea de iesire grayscale

    for (int i = 0; i < N; i++) {
        // Conversie din double [0,1] la unsigned char [0,255] si clamping
        output_image_data[i] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, q_grayscale[i] * 255.0)));
    }

    // Salvare imagine de ieșire (grayscale)
    stbi_write_png("Images/Output/output_grayscale.png", width, height, 1, output_image_data.data(), width);

    cout << "Filtru aplicat cu succes. Rezultat: Images/Output/output_grayscale.png\n";
    cout << "Timp aplicare filtru: " << time_taken << "s\n";
    // Eliberare memorie
    stbi_image_free(guide_image_data);
    stbi_image_free(process_image_data);
    return 0;
}