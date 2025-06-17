// Production-ready text rendering for SDL2 without external dependencies
#pragma once

#include <SDL2/SDL.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace tacs {

// Bitmap font renderer for production text display
class TextRenderer {
public:
    TextRenderer(SDL_Renderer* renderer);
    ~TextRenderer();
    
    // Initialize built-in bitmap font
    bool initialize();
    
    // Render text at specified position
    void renderText(const std::string& text, int x, int y, 
                   uint8_t r = 255, uint8_t g = 255, uint8_t b = 255, uint8_t a = 255);
    
    // Get text dimensions
    void getTextSize(const std::string& text, int& width, int& height) const;
    
private:
    SDL_Renderer* renderer_;
    SDL_Texture* font_texture_;
    
    // Character dimensions in bitmap font
    static constexpr int CHAR_WIDTH = 8;
    static constexpr int CHAR_HEIGHT = 16;
    static constexpr int CHARS_PER_ROW = 16;
    
    // Generate bitmap font data
    void generateFontBitmap();
    
    // Render single character
    void renderChar(char c, int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    
    // Built-in 8x16 bitmap font data (ASCII 32-127)
    static const uint8_t FONT_DATA[];
};

// Circle drawing utilities for production rendering
class DrawUtils {
public:
    // Draw filled circle using Bresenham's algorithm
    static void drawFilledCircle(SDL_Renderer* renderer, int cx, int cy, int radius);
    
    // Draw circle outline
    static void drawCircle(SDL_Renderer* renderer, int cx, int cy, int radius);
    
    // Draw thick line
    static void drawThickLine(SDL_Renderer* renderer, int x1, int y1, int x2, int y2, int thickness);
    
private:
    // Helper for circle drawing
    static void drawCirclePoints(SDL_Renderer* renderer, int cx, int cy, int x, int y);
    static void fillCircleLines(SDL_Renderer* renderer, int cx, int cy, int x, int y);
};

} // namespace tacs