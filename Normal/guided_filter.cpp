#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> // Required for std::max and std::min
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
    int width_I, height_I, channels_I_actual;
    int width_p, height_p, channels_p_actual;

    const int desired_channels_I = 3; // We want 3 channels for guide image (RGB for conversion)
    const int desired_channels_p = 1; // We want 1 channel for process image (grayscale)

    // CALEA CĂTRE IMAGINEA DE GHIDARE COLOR
    unsigned char* guide_image_data = stbi_load("Images/Input/target.png", &width_I, &height_I, &channels_I_actual, desired_channels_I);
    if (!guide_image_data) {
        cerr << "Eroare la citire target.png\n";
        return 1;
    }
    cout << "Imagine de ghidare citita: " << width_I << " x " << height_I << " cu " << channels_I_actual << " canale (cerute: " << desired_channels_I << ")." << endl;

    // CALEA CĂTRE IMAGINEA DE PROCESAT GRAYSCALE
    unsigned char* process_image_data = stbi_load("Images/Input/input.png", &width_p, &height_p, &channels_p_actual, desired_channels_p);
    if (!process_image_data) {
        cerr << "Eroare la citire input.png\n";
        stbi_image_free(guide_image_data);
        return 1;
    }
    cout << "Imagine de procesat citita: " << width_p << " x " << height_p << " cu " << channels_p_actual << " canal (cerut: " << desired_channels_p << ")." << endl;

    // Verificăm dacă imaginile au aceleași dimensiuni
    if (width_I != width_p || height_I != height_p) {
        cerr << "Eroare: Imaginile de ghidare si de procesat trebuie sa aiba aceleasi dimensiuni!\n";
        stbi_image_free(guide_image_data);
        stbi_image_free(process_image_data);
        return 1;
    }

    int N = width_I * height_I; // Numarul total de pixeli

    // Vectori pentru canalele Y, U, V ale imaginii de ghidare
    vector<double> I_Y(N), I_U(N), I_V(N);
    // Vector pentru imaginea de procesat (grayscale)
    vector<double> p_grayscale(N);

    // Conversie de la RGB la YUV pentru imaginea de ghidare
    // Normalizăm imaginea de procesat (grayscale)
    for (int i = 0; i < height_I; i++) {
        for (int j = 0; j < width_I; j++) {
            int pixel_idx_flat = i * width_I + j; // Flat index for YUV and grayscale vectors
            int img_data_idx_color = pixel_idx_flat * desired_channels_I; // Index for guide_image_data (RGB)

            // Normalize RGB components from 0-255 to 0.0-1.0
            double R = guide_image_data[img_data_idx_color] / 255.0;
            double G = guide_image_data[img_data_idx_color + 1] / 255.0;
            double B = guide_image_data[img_data_idx_color + 2] / 255.0;

            // Convert RGB to YUV (BT.709)
            I_Y[pixel_idx_flat] = 0.2126 * R + 0.7152 * G + 0.0722 * B;
            I_U[pixel_idx_flat] = -0.0999 * R - 0.3360 * G + 0.4360 * B + 0.5; // +0.5 to shift range
            I_V[pixel_idx_flat] = 0.6150 * R - 0.5586 * G - 0.0563 * B + 0.5; // +0.5 to shift range

            // Normalizăm imaginea grayscale de procesat (input.png)
            p_grayscale[pixel_idx_flat] = process_image_data[pixel_idx_flat] / 255.0;
        }
    }

    int r = 0.5;       // Raza filtrului
    double eps = 1; // Parametru de regularizare

    // Vectori pentru canalele filtrate de iesire (Y, U, și V)
    vector<double> q_Y(N); // New vector for filtered Y
    vector<double> q_U(N), q_V(N);

    // APELAREA FILTRULUI GHIDAT PENTRU FIECARE CANAL
    // Guide (I) is I_Y (luminance from target.png)
    // Input (p) is p_grayscale (luminance from input.png)
    cout << "Aplicare filtru ghidat pe canalul Y (Luminance)..." << endl;
    guided_filter(I_Y, p_grayscale, q_Y, width_I, height_I, r, eps);

    // Guide (I) is p_grayscale (luminance from input.png) for U and V
    // Input (p) is I_U/I_V (chrominance from target.png)
    cout << "Aplicare filtru ghidat pe canalul U (Chrominance Albastru-Galben)..." << endl;
    guided_filter(p_grayscale, I_U, q_U, width_I, height_I, r, eps);
    cout << "Aplicare filtru ghidat pe canalul V (Chrominance Rosu-Verde)..." << endl;
    guided_filter(p_grayscale, I_V, q_V, width_I, height_I, r, eps);

    // Acum trebuie să convertim înapoi q_Y, q_U și q_V la RGB
    vector<unsigned char> output_image_data(N * 3); // 3 canale pentru imaginea de iesire color

    for (int i = 0; i < N; i++) {
        // Obținem valorile Y, U, V filtrate
        double Y = q_Y[i]; // Use the filtered Y channel
        double U = q_U[i] - 0.5;    // Subtract 0.5 to revert U to its original range
        double V = q_V[i] - 0.5;    // Subtract 0.5 to revert V to its original range

        // Convert YUV back to RGB (BT.709)
        double R = Y + 1.28033 * V;
        double G = Y - 0.21482 * U - 0.38059 * V;
        double B = Y + 2.12798 * U;

        // Conversie din double [0,1] la unsigned char [0,255] si clamping
        output_image_data[i * 3]     = static_cast<unsigned char>(std::max(0.0, std::min(255.0, R * 255.0)));
        output_image_data[i * 3 + 1] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, G * 255.0)));
        output_image_data[i * 3 + 2] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, B * 255.0)));
    }

    // Salvare imagine de ieșire
    stbi_write_png("Images/Output/output.png", width_I, height_I, 3, output_image_data.data(), width_I * 3);

    cout << "Filtru aplicat cu succes. Rezultat: Images/Output/output.png\n";

    // Eliberare memorie
    stbi_image_free(guide_image_data);
    stbi_image_free(process_image_data);
    return 0;
}