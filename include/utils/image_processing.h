// Production-ready image processing utilities without external dependencies
#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace tacs {

// Simple image representation
struct Image {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;
    
    Image() : width(0), height(0), channels(0) {}
    Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c) {}
    
    uint8_t& at(int y, int x, int c) {
        return data[(y * width + x) * channels + c];
    }
    
    const uint8_t& at(int y, int x, int c) const {
        return data[(y * width + x) * channels + c];
    }
    
    size_t size() const { return data.size(); }
};

// Rectangle for bounding boxes
struct Rect {
    int x, y, width, height;
    
    Rect() : x(0), y(0), width(0), height(0) {}
    Rect(int _x, int _y, int _w, int _h) : x(_x), y(_y), width(_w), height(_h) {}
    
    int area() const { return width * height; }
    
    Rect operator&(const Rect& r) const {
        int x1 = std::max(x, r.x);
        int y1 = std::max(y, r.y);
        int x2 = std::min(x + width, r.x + r.width);
        int y2 = std::min(y + height, r.y + r.height);
        
        if (x2 > x1 && y2 > y1) {
            return Rect(x1, y1, x2 - x1, y2 - y1);
        }
        return Rect();
    }
};

// Basic image I/O
Image imread(const std::string& path);
bool imwrite(const std::string& path, const Image& image);

// Color space conversions
Image cvtColor_BGR2GRAY(const Image& src);
Image cvtColor_BGR2HSV(const Image& src);

// Edge detection
Image Canny(const Image& src, double threshold1, double threshold2);
Image Sobel(const Image& src, int dx, int dy);

// Basic drawing functions
void rectangle(Image& img, const Rect& rect, const std::vector<uint8_t>& color, int thickness = 1);
void circle(Image& img, int cx, int cy, int radius, const std::vector<uint8_t>& color, int thickness = 1);

// Region of Interest extraction
Image extractROI(const Image& src, const Rect& roi);

// Contour finding (simplified)
std::vector<std::vector<std::pair<int, int>>> findContours(const Image& binary);
double contourArea(const std::vector<std::pair<int, int>>& contour);
double arcLength(const std::vector<std::pair<int, int>>& contour, bool closed = true);

// Non-zero counting
int countNonZero(const Image& binary);

// Random number generation for image
void randu(Image& img, const std::vector<uint8_t>& low, const std::vector<uint8_t>& high);

} // namespace tacs