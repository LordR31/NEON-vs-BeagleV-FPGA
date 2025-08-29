#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../Headers/stb_image.h"
#include "../Headers/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>

using namespace std;

// === USER CONFIGURATION: REPLACE THESE WITH YOUR ACTUAL VALUES ===
// The base address of your box filter IP block in the FPGA memory map.
#define FPGA_BASE_ADDR 0x40000000
// The total size of the mapped memory region.
#define MAP_SIZE 4096

// === REGISTER OFFSETS FOR YOUR CUSTOM IP ===
// These are placeholders; replace them with the correct offsets from your
// SmartHLS generated design. The AXI-Lite interface will have these registers.
#define CONTROL_REG_OFFSET 0x000
#define DATA_IN_OFFSET 0x010 // AXI-Lite address for the DMA start address
#define DATA_OUT_OFFSET 0x014 // AXI-Lite address for the DMA destination address

// AXI-Lite control bits
#define START_BIT 0x1

// === CORE_GPIO CONFIGURATION ===
// The base address of your CoreGPIO block in the FPGA memory map.
#define GPIO_BASE_ADDR 0x40010000 // Placeholder, check your design!
// Offset to the PAD_STATUS register.
#define GPIO_PAD_STATUS_REG 0x14
// The specific pin number tied to the 'finish' signal.
#define GPIO_DONE_PIN_NUM 0 // Placeholder, e.g., GPIO_0

// =================================================================

// Function to handle communication with the FPGA hardware.
void box_filter_hw_accel(
    const vector<double>& src,
    vector<double>& dst,
    int width,
    int height,
    int r)
{
    // The SmartHLS code uses float, so we must cast our doubles to floats
    // for the hardware accelerator and then convert back.
    int N = width * height;
    vector<float> src_float(N);
    vector<float> dst_float(N);

    // Convert input doubles to floats
    for (int i = 0; i < N; ++i) {
        src_float[i] = static_cast<float>(src[i]);
    }

    int mem_fd;
    void *fpga_ptr;
    void *gpio_ptr;
    volatile uint32_t *control_reg;
    volatile uint32_t *data_in_ptr_reg;
    volatile uint32_t *data_out_ptr_reg;
    volatile uint32_t *gpio_pad_status;

    // --- 1. Open /dev/mem to get access to physical memory ---
    if ((mem_fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) {
        perror("Error opening /dev/mem");
        return;
    }

    // --- 2. Map the physical addresses of the IP blocks to virtual addresses ---
    fpga_ptr = mmap(
        NULL,
        MAP_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        mem_fd,
        FPGA_BASE_ADDR
    );
    gpio_ptr = mmap(
        NULL,
        MAP_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        mem_fd,
        GPIO_BASE_ADDR
    );

    if (fpga_ptr == MAP_FAILED || gpio_ptr == MAP_FAILED) {
        perror("Error mapping memory");
        close(mem_fd);
        return;
    }

    // --- 3. Set up pointers to our registers ---
    control_reg = (volatile uint32_t *)(fpga_ptr + CONTROL_REG_OFFSET);
    data_in_ptr_reg = (volatile uint32_t *)(fpga_ptr + DATA_IN_OFFSET);
    data_out_ptr_reg = (volatile uint32_t *)(fpga_ptr + DATA_OUT_OFFSET);
    // Pointer to the CoreGPIO PAD_STATUS register.
    gpio_pad_status = (volatile uint32_t *)(gpio_ptr + GPIO_PAD_STATUS_REG);

    // Get the physical addresses of the input and output vectors
    uint32_t src_phys_addr = (uint32_t)(uintptr_t)src_float.data();
    uint32_t dst_phys_addr = (uint32_t)(uintptr_t)dst_float.data();
    
    // --- 4. Write the physical addresses to the IP's registers ---
    *data_in_ptr_reg = src_phys_addr;
    *data_out_ptr_reg = dst_phys_addr;

    // --- 5. Start the FPGA accelerator ---
    printf("Starting FPGA box filter accelerator...\n");
    *control_reg = START_BIT;

    // --- 6. Poll the GPIO pin until the operation is complete ---
    printf("Waiting for FPGA to complete via GPIO pin...\n");
    // We check if the bit corresponding to the 'done' pin is high.
    while (!(*gpio_pad_status & (1 << GPIO_DONE_PIN_NUM))) {
        // Wait...
    }
    printf("FPGA operation complete! Reading output data.\n");

    // Convert output floats back to doubles
    for (int i = 0; i < N; ++i) {
        dst[i] = static_cast<double>(dst_float[i]);
    }
    
    // --- 7. Clean up memory mapping and file descriptor ---
    if (munmap(fpga_ptr, MAP_SIZE) == -1) {
        perror("Error unmapping memory");
    }
    if (munmap(gpio_ptr, MAP_SIZE) == -1) {
        perror("Error unmapping GPIO memory");
    }
    close(mem_fd);
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
    // ** Replaced the software call with the hardware accelerator call **
    box_filter_hw_accel(I, mean_I, width, height, r);
    box_filter_hw_accel(p, mean_p, width, height, r);

    for (int i = 0; i < N; i++) {
        corr_I[i] = I[i] * I[i];
        corr_Ip[i] = I[i] * p[i];
    }

    // ** Replaced the software call with the hardware accelerator call **
    box_filter_hw_accel(corr_I, corr_I, width, height, r);
    box_filter_hw_accel(corr_Ip, corr_Ip, width, height, r);

    for (int i = 0; i < N; i++) {
        var_I[i] = corr_I[i] - mean_I[i] * mean_I[i];
        cov_Ip[i] = corr_Ip[i] - mean_I[i] * mean_p[i];
    }

    for (int i = 0; i < N; i++) {
        a[i] = cov_Ip[i] / (var_I[i] + eps);
        b[i] = mean_p[i] - a[i] * mean_I[i];
    }

    // ** Replaced the software call with the hardware accelerator call **
    box_filter_hw_accel(a, mean_a, width, height, r);
    box_filter_hw_accel(b, mean_b, width, height, r);

    for (int i = 0; i < N; i++) {
        q[i] = mean_a[i] * I[i] + mean_b[i];
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

    vector<double> I_grayscale(N);
    vector<double> p_grayscale(N);

    for (int i = 0; i < N; i++) {
        I_grayscale[i] = guide_image_data[i] / 255.0;
        p_grayscale[i] = process_image_data[i] / 255.0;
    }

    int r = 1;
    double eps = 0.1;

    vector<double> q_grayscale(N);

    cout << "Aplicare filtru ghidat pe imagini grayscale..." << endl;
    start = clock();
    guided_filter(I_grayscale, p_grayscale, q_grayscale, width, height, r, eps);
    end = clock();
    time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;

    vector<unsigned char> output_image_data(N);

    for (int i = 0; i < N; i++) {
        output_image_data[i] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, q_grayscale[i] * 255.0)));
    }

    stbi_write_png("Images/Output/output_grayscale.png", width, height, 1, output_image_data.data(), width);

    cout << "Filtru aplicat cu succes. Rezultat: Images/Output/output_grayscale.png\n";
    cout << "Timp aplicare filtru: " << time_taken << "s\n";

    stbi_image_free(guide_image_data);
    stbi_image_free(process_image_data);
    return 0;
}
