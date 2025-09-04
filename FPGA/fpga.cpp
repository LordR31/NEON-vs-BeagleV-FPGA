#include <iostream>
#include <fstream>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// A constant for the UIO device path.
// The device number is hardcoded for simplicity, but in a real-world
// application, you would dynamically find the correct device.
const std::string UIO_DEV_PATH = "/dev/uio0";
const size_t UIO_MMAP_SIZE = 0x1000; // The size of our IP core's address space.

// Use an enum to represent the registers for clarity
enum BoxFilterRegisters {
    CONTROL_REG = 0,
    STATUS_REG = 4,
    INPUT_DATA_REG = 8,
    OUTPUT_DATA_REG = 12
};

int main() {
    // 1. Open the UIO device file
    // The device file is a character device that provides access to the hardware
    int uio_fd = open(UIO_DEV_PATH.c_str(), O_RDWR | O_SYNC);
    if (uio_fd < 0) {
        std::cerr << "Error: Cannot open UIO device file: " << UIO_DEV_PATH << std::endl;
        return 1;
    }

    // 2. Map the physical memory of the IP core into the process's virtual memory
    // This is the crucial step that gives us a direct pointer to the hardware registers.
    // The offset is 0 because we are mapping the entire UIO region.
    volatile char* uio_mmap_ptr = (volatile char*)mmap(NULL, UIO_MMAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, uio_fd, 0);
    if (uio_mmap_ptr == MAP_FAILED) {
        std::cerr << "Error: mmap() failed. Check permissions and device tree." << std::endl;
        close(uio_fd);
        return 1;
    }

    // 3. Cast the memory pointer to a volatile unsigned int pointer for register access
    volatile unsigned int* registers = reinterpret_cast<volatile unsigned int*>(uio_mmap_ptr);

    std::cout << "Successfully mapped UIO device memory." << std::endl;
    std::cout << "You can now read and write to your hardware registers." << std::endl;

    // --- Example Register Access ---
    // Here we can read from and write to our hardware registers
    // using the base pointer and the register offsets.

    // A. Read the status register
    unsigned int status = registers[STATUS_REG / sizeof(unsigned int)];
    std::cout << "Current STATUS_REG value: 0x" << std::hex << status << std::endl;

    // B. Write some data to the input register
    unsigned int test_data = 0xABCD1234;
    std::cout << "Writing 0x" << std::hex << test_data << " to INPUT_DATA_REG..." << std::endl;
    registers[INPUT_DATA_REG / sizeof(unsigned int)] = test_data;
    
    // C. Read back the data (if the core supports it, as a simple verification)
    unsigned int read_back_data = registers[INPUT_DATA_REG / sizeof(unsigned int)];
    std::cout << "Read back 0x" << std::hex << read_back_data << " from INPUT_DATA_REG." << std::endl;

    // D. Trigger the box filter operation by writing to the control register
    unsigned int control_value = 1; // Assuming 1 triggers the operation
    std::cout << "Writing 0x" << std::hex << control_value << " to CONTROL_REG to start the operation." << std::endl;
    registers[CONTROL_REG / sizeof(unsigned int)] = control_value;

    // E. Read the final output
    unsigned int output = registers[OUTPUT_DATA_REG / sizeof(unsigned int)];
    std::cout << "Final result from OUTPUT_DATA_REG: 0x" << std::hex << output << std::endl;

    // 4. Clean up: unmap the memory and close the file descriptor
    if (munmap(uio_mmap_ptr, UIO_MMAP_SIZE) < 0) {
        std::cerr << "Error: munmap() failed." << std::endl;
    }

    close(uio_fd);
    std::cout << "Resources released. Program finished." << std::endl;

    return 0;
}
