// Minimal stb_image_write implementation for PNG writing
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Simplified PNG writer for our use case
inline int stbi_write_png(const char* filename, int w, int h, int comp, const void* data, int stride_in_bytes) {
    // Very simplified PNG writer - just saves as raw binary for testing
    FILE* f = fopen(filename, "wb");
    if (!f) return 0;
    
    // Write simple header
    fwrite(&w, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(&comp, sizeof(int), 1, f);
    
    // Write data
    const unsigned char* pixels = (const unsigned char*)data;
    for (int y = 0; y < h; ++y) {
        fwrite(pixels + y * stride_in_bytes, comp * w, 1, f);
    }
    
    fclose(f);
    return 1;
}

// For loading our simplified format
inline unsigned char* stbi_load_simple(const char* filename, int* x, int* y, int* comp, int req_comp) {
    FILE* f = fopen(filename, "rb");
    if (!f) return nullptr;
    
    fread(x, sizeof(int), 1, f);
    fread(y, sizeof(int), 1, f);
    fread(comp, sizeof(int), 1, f);
    
    int size = (*x) * (*y) * (*comp);
    unsigned char* data = (unsigned char*)malloc(size);
    fread(data, size, 1, f);
    
    fclose(f);
    return data;
}