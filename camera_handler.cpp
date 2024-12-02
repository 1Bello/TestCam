// camera_handler.cpp
#include <Arduino.h>
#include "esp_camera.h"
#include "camera_handler.h"
#include "tensorflow/lite/micro/micro_log.h"
extern ProcessedImage g_processed_image;

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

bool InitCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;  // Changed to JPEG
    config.frame_size = FRAMESIZE_VGA;     // Increased resolution
    config.jpeg_quality = 12;              // Lower number means higher quality
    config.fb_count = 2;                   // Increased for streaming
    config.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    sensor_t* sensor = esp_camera_sensor_get();
    if (sensor) {
        sensor->set_brightness(sensor, 1);
        sensor->set_contrast(sensor, 1);
        sensor->set_saturation(sensor, 1);
        sensor->set_special_effect(sensor, 0); // No special effect
    }

    Serial.println("Camera initialized successfully");
    return true;
}

bool CaptureImage(float* image_data, int target_width, int target_height) {
    if (!image_data) return false;

    // First get the frame buffer
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return false;
    }

    size_t rgb_len = fb->width * fb->height * 2; // RGB565 uses 2 bytes per pixel
    uint8_t* rgb_buffer = (uint8_t*)malloc(rgb_len);
    if (!rgb_buffer) {
        Serial.println("Memory allocation failed for RGB buffer");
        esp_camera_fb_return(fb);
        return false;
    }

    // Decode JPEG to RGB565
    if (fb->format == PIXFORMAT_JPEG) {
        bool converted = jpg2rgb565(fb->buf, fb->len, rgb_buffer, JPG_SCALE_NONE);
        if (!converted) {
            Serial.println("JPEG conversion failed");
            free(rgb_buffer);
            esp_camera_fb_return(fb);
            return false;
        }
    }

    // Convert to grayscale and normalize
    const uint8_t* source = (fb->format == PIXFORMAT_JPEG) ? rgb_buffer : fb->buf;
    const int src_width = fb->width;
    const int src_height = fb->height;
    
    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            // Calculate source coordinates with proper scaling
            int src_x = x * src_width / target_width;
            int src_y = y * src_height / target_height;
            
            uint8_t r, g, b;
            if (fb->format == PIXFORMAT_JPEG) {
                // For RGB565 from JPEG
                int src_idx = (src_y * src_width + src_x) * 2;
                uint16_t pixel = (source[src_idx + 1] << 8) | source[src_idx];
                r = ((pixel >> 11) & 0x1F) << 3;
                g = ((pixel >> 5) & 0x3F) << 2;
                b = (pixel & 0x1F) << 3;
            } else {
                // For raw RGB/grayscale format
                int src_idx = (src_y * src_width + src_x) * 3; // Assuming RGB888
                r = source[src_idx];
                g = source[src_idx + 1];
                b = source[src_idx + 2];
            }
            
            // Calculate grayscale value and normalize to 0-1
            float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
            image_data[y * target_width + x] = gray;
        }
    }

    // Create JPEG for web display
    uint8_t* gray_buffer = (uint8_t*)heap_caps_malloc(target_width * target_height, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (gray_buffer) {
        for (int i = 0; i < target_width * target_height; i++) {
            gray_buffer[i] = (uint8_t)(image_data[i] * 255.0f);
        }

        uint8_t* jpg_buf = nullptr;
        size_t jpg_len = 0;
        if (fmt2jpg(gray_buffer, target_width * target_height, 
                   target_width, target_height, 
                   PIXFORMAT_GRAYSCALE, 80, 
                   &jpg_buf, &jpg_len)) {
            
            if (xSemaphoreTake(g_processed_image.mutex, portMAX_DELAY) == pdTRUE) {
                if (g_processed_image.jpg_buf != nullptr) {
                    free(g_processed_image.jpg_buf);
                }
                g_processed_image.jpg_buf = jpg_buf;
                g_processed_image.jpg_len = jpg_len;
                g_processed_image.is_new = true;
                xSemaphoreGive(g_processed_image.mutex);
            }
        }
        free(gray_buffer);
    }

    // Cleanup
    if (rgb_buffer) free(rgb_buffer);
    esp_camera_fb_return(fb);

    return true;
}

bool fmt2grayscale(uint8_t* input, size_t input_len, int target_width, int target_height, float* output) {
    if (!input || !output) return false;

    // Source image is 160x120 RGB565
    const int src_width = 160;
    const int src_height = 120;
    
    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            // Calculate source coordinates with simple scaling
            int src_x = x * src_width / target_width;
            int src_y = y * src_height / target_height;
            int src_idx = (src_y * src_width + src_x) * 2;

            if (src_idx + 1 >= input_len) continue;

            // Convert RGB565 to grayscale
            uint16_t pixel = (input[src_idx + 1] << 8) | input[src_idx];
            uint8_t r = ((pixel >> 11) & 0x1F) << 3;
            uint8_t g = ((pixel >> 5) & 0x3F) << 2;
            uint8_t b = (pixel & 0x1F) << 3;

            // Calculate grayscale value and normalize to 0-1
            output[y * target_width + x] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
        }
    }

    return true;
}