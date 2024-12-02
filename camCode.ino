#include "esp_task_wdt.h"
#include "camera_handler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "model.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "WiFi.h"

const char* ssid = "JAD";
const char* password = "andreaytono12";


// Add this global variable after the structure definition
ProcessedImage g_processed_image = {
    .jpg_buf = nullptr,
    .jpg_len = 0,
    .is_new = false,
    .mutex = nullptr
};


httpd_handle_t camera_httpd = NULL;

// JPEG capture and stream handling
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// Stream handler
static esp_err_t stream_handler(httpd_req_t *req) {
    esp_err_t res = ESP_OK;
    char * part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if(res != ESP_OK) {
        return res;
    }

    while(true) {
        if (xSemaphoreTake(g_processed_image.mutex, portMAX_DELAY) == pdTRUE) {
            if (g_processed_image.is_new && g_processed_image.jpg_buf != nullptr) {
                size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, g_processed_image.jpg_len);
                res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
                
                if(res == ESP_OK) {
                    res = httpd_resp_send_chunk(req, (const char *)g_processed_image.jpg_buf, g_processed_image.jpg_len);
                }
                if(res == ESP_OK) {
                    res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
                }
                g_processed_image.is_new = false;
            }
            xSemaphoreGive(g_processed_image.mutex);
        }
        
        if(res != ESP_OK) {
            break;
        }
        delay(100);
    }
    return res;
}

// Index page handler
static esp_err_t index_handler(httpd_req_t *req){
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, 
        "<!DOCTYPE html>"
        "<html>"
        "<head>"
        "<title>ESP32-CAM Stream</title>"
        "<style>"
        ".container { display: flex; justify-content: center; gap: 20px; }"
        ".image-container { text-align: center; }"
        "</style>"
        "</head>"
        "<body>"
        "<center><h1>ESP32-CAM Processed Stream</h1></center>"
        "<div class='container'>"
        "<div class='image-container'>"
        "<h3>Processed Image (Used for Inference)</h3>"
        "<img src=\"/stream\" width=\"320\" height=\"320\">"
        "</div>"
        "</div>"
        "</body>"
        "</html>", -1);
}

// Start the web server
void startCameraServer(){
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &stream_uri);
    }
}

// Increase tensor arena size based on the error message
constexpr int kTensorArenaSize = 100000;  // Increased from 50000
uint8_t* tensor_arena = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
    Serial.begin(115200);
    while(!Serial) delay(100);
    Serial.println("Starting setup...");

    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");

    g_processed_image.mutex = xSemaphoreCreateMutex();
    if (!g_processed_image.mutex) {
        Serial.println("Failed to create mutex");
        while(1) delay(1000);
    }
    
    startCameraServer();
    
    Serial.print("Camera Stream Ready! Go to: http://");
    Serial.println(WiFi.localIP());

    // Initialize PSRAM
    if(!psramInit()) {
        Serial.println("PSRAM initialization failed");
        while(1) delay(1000);
    }

    Serial.printf("Total heap: %d\n", ESP.getHeapSize());
    Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
    Serial.printf("Total PSRAM: %d\n", ESP.getPsramSize());
    Serial.printf("Free PSRAM: %d\n", ESP.getFreePsram());

    // Allocate tensor arena from PSRAM with specific capabilities
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        Serial.println("Failed to allocate tensor arena in PSRAM");
        // Try allocating from regular memory as fallback
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT);
        if (tensor_arena == nullptr) {
            Serial.println("Failed to allocate tensor arena in regular memory");
            while(1) delay(1000);
        }
        Serial.println("Allocated tensor arena in regular memory");
    } else {
        Serial.println("Allocated tensor arena in PSRAM");
    }

    // Initialize the camera
    if (!InitCamera()) {
        Serial.println("Camera initialization failed");
        while(1) delay(1000);
    }

    Serial.println("Loading TFLite model...");
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        while(1) delay(1000);
    }

    // Print model details
    Serial.printf("Model version: %d\n", model->version());
    Serial.printf("Model description: %s\n", model->description()->c_str());

    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddQuantize();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddDequantize();

    // Create interpreter without error reporter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    Serial.println("Allocating tensors...");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        Serial.printf("Arena size requested: %d\n", kTensorArenaSize);
        while(1) delay(1000);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    // Print tensor details
    Serial.println("Model initialized. Tensor details:");
    Serial.printf("Input tensor dims: %d x %d x %d x %d\n", 
        input->dims->data[0], input->dims->data[1],
        input->dims->data[2], input->dims->data[3]);
    Serial.printf("Input type: %d\n", input->type);
    Serial.printf("Output tensor dims: %d\n", output->dims->data[1]);
    
    Serial.println("Setup completed successfully");
}

void loop() {
    const int image_width = input->dims->data[1];
    const int image_height = input->dims->data[2];
    
    float* image_data = (float*)heap_caps_malloc(
        image_width * image_height * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    
    if (!image_data) {
        Serial.println("Failed to allocate image buffer");
        delay(1000);
        return;
    }

    if (!CaptureImage(image_data, image_width, image_height)) {
        Serial.println("Image capture failed");
        heap_caps_free(image_data);
        delay(1000);
        return;
    }

    // Copy image data to input tensor
    memcpy(input->data.f, image_data, image_width * image_height * sizeof(float));
    heap_caps_free(image_data);

    Serial.println("Running inference...");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("Invoke failed");
        delay(1000);
        return;
    }

    // Process results
    int num_classes = output->dims->data[1];
    for (int i = 0; i < num_classes; ++i) {
        float value = output->type == kTfLiteInt8 
            ? (output->data.int8[i] - output->params.zero_point) * output->params.scale 
            : output->data.f[i];
        Serial.printf("Class %d: %.4f\n", i, value);
    }

    delay(1000);
}