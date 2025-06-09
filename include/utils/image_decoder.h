/**
 * @file image_decoder.h
 * @brief Production-ready image decoders for JPEG and PNG formats
 * 
 * Implements efficient decoders for common image formats without external
 * dependencies. Optimized for real-time traffic image processing with
 * hardware acceleration support where available.
 */
#pragma once

#include "core/tensor.h"
#include <vector>
#include <cstdint>
#include <string>

namespace tacs {
namespace utils {

class ImageDecoder {
public:
    static bool decode_jpeg(const uint8_t* data, size_t size, core::Tensor& output);
    static bool decode_png(const uint8_t* data, size_t size, core::Tensor& output);
    static bool decode_image(const std::string& path, core::Tensor& output);

private:
    // JPEG decoder components
    struct JPEGComponent {
        int id;
        int h_samp_factor;
        int v_samp_factor;
        int quant_table_id;
        int dc_table_id;
        int ac_table_id;
        std::vector<int16_t> data;
    };
    
    struct JPEGDecoder {
        int width;
        int height;
        int num_components;
        std::vector<JPEGComponent> components;
        std::vector<std::vector<uint16_t>> quant_tables;
        std::vector<std::vector<uint8_t>> huffman_dc_tables;
        std::vector<std::vector<uint8_t>> huffman_ac_tables;
        std::vector<std::vector<uint16_t>> huffman_dc_values;
        std::vector<std::vector<uint16_t>> huffman_ac_values;
        
        bool parse_jpeg(const uint8_t* data, size_t size);
        bool decode_scan(const uint8_t* data, size_t size, size_t offset);
        void idct_block(int16_t* block);
        void ycbcr_to_rgb(core::Tensor& output);
    };
    
    // PNG decoder components
    struct PNGChunk {
        uint32_t length;
        char type[5];
        std::vector<uint8_t> data;
        uint32_t crc;
    };
    
    struct PNGDecoder {
        int width;
        int height;
        int bit_depth;
        int color_type;
        int compression;
        int filter;
        int interlace;
        std::vector<uint8_t> image_data;
        
        bool parse_png(const uint8_t* data, size_t size);
        bool decode_idat(const std::vector<PNGChunk>& chunks);
        void unfilter_scanline(uint8_t* scanline, uint8_t* prev_scanline, int filter_type, int bytes_per_pixel, int width);
        void convert_to_rgb(core::Tensor& output);
    };
    
    // Huffman decoding helpers
    static int decode_huffman(const uint8_t*& data, size_t& bit_pos, 
                             const std::vector<uint8_t>& table, 
                             const std::vector<uint16_t>& values);
    
    // CRC calculation for PNG
    static uint32_t calculate_crc(const uint8_t* data, size_t length);
    
    // Decompression helpers
    static bool inflate_data(const uint8_t* compressed, size_t comp_size, 
                            std::vector<uint8_t>& decompressed);
};

}
}