/*
 * ================================================================
 *  EV RANGE DASHBOARD — Professional TFT Display Firmware
 * ================================================================
 *  ESP32 + ILI9341/ST7789 240×320 TFT + Multi-Sensor + ML
 *
 *  Sensors (I2C @ SDA=21, SCL=22):
 *    INA219  — Bus voltage, current, power  (addr 0x41)
 *    BMP280  — Altitude, pressure, temp     (addr 0x76)
 *    MPU6050 — 3-axis accel + gyro          (addr 0x68)
 *    HDC1080 — Temperature, humidity        (addr 0x40)
 *
 *  Analog / Digital:
 *    GPIO39  LDR          (ambient light)
 *    GPIO35  Rain sensor  (analog, wet=low)
 *    GPIO34  NTC 10 K     (battery temperature)
 *    GPIO36  Throttle     (Hall, 0–100 %)
 *    GPIO26  Wind Hall 1  (dual sensor @ 0°)
 *    GPIO25  Wind Hall 2  (dual sensor @ 90°)
 *    GPIO12  Push-button  (screen navigation)
 *    GPIO 5  Buzzer       (PWM alerts)
 *
 *  ADS1232 24-bit ADC (weight / load-cell):
 *    GPIO14  DOUT    GPIO27  PDWN
 *    GPIO13  SCLK    SPEED → GND (hardwired)
 *
 *  ML Model: Random Forest 36 trees, 12 features → range (km)
 *
 *  TFT_eSPI: configure User_Setup.h for your display:
 *    #define ILI9341_DRIVER   // or ST7789_DRIVER
 *    #define TFT_WIDTH  240
 *    #define TFT_HEIGHT 320
 *    #define TFT_MOSI 23
 *    #define TFT_SCLK 18
 *    #define TFT_CS   15
 *    #define TFT_DC    2
 *    #define TFT_RST   4
 *    #define SPI_FREQUENCY 40000000
 *
 *  Libraries:
 *    TFT_eSPI, Adafruit_BMP280, Adafruit_MPU6050,
 *    Adafruit_INA219, ClosedCube_HDC1080, Wire
 * ================================================================
 */

// ====================== INCLUDES ================================
#include <Adafruit_BMP280.h>
#include <Adafruit_INA219.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <SPI.h>
#include <TFT_eSPI.h>
#include <Wire.h>
#include <math.h>

#include "ClosedCube_HDC1080.h"

// Model header — copy to this folder if relative include fails
#include "ev_range_model-new-v4.h"

// ====================== PIN MAP =================================
#define LDR_PIN 39
#define RAIN_PIN 35
#define THROTTLE_PIN 36
#define NTC_PIN 34
#define WIND_SENSOR_1 26
#define WIND_SENSOR_2 25
#define BUTTON_PIN 12
#define BUZZER_PIN 5

// ADS1232 load-cell ADC
#define ADS_DOUT_PIN 14
#define ADS_SCLK_PIN 13
#define ADS_PDWN_PIN 27
// SPEED pin hardwired to GND

// ====================== TUNABLES ================================
// Battery pack (13S Li-ion typical for e-scooter / e-bike)
#define BATT_FULL_V 54.6f
#define BATT_EMPTY_V 39.0f
#define BATT_CAPACITY_WH 480.0f

// NTC 10 K thermistor (Steinhart-Hart B-parameter model)
#define NTC_R_SERIES 10000
#define NTC_R_NOMINAL 10000
#define NTC_TEMP_NOM 25
#define NTC_BETA 3950

// Throttle hall-sensor ADC range (calibrate for your sensor)
#define THROT_ADC_MIN 490
#define THROT_ADC_MAX 3500

// Wind anemometer (dual hall sensor on cooling fan)
#define FAN_RADIUS 0.018f  // 1.8 cm (with fins) in meters
#define FAN_CIRCUMFERENCE (2.0f * PI * FAN_RADIUS)

// ADS1232 weight calibration
#define ADS_TARE_VALUE 39172L
#define ADS_SCALE_FACTOR 17667.50f  // raw counts per kg — calibrate!

// Max range shown on gauge (km)
#define GAUGE_MAX_RANGE 50.0f

// Update intervals (ms)
#define SENSOR_INTERVAL 500
#define PREDICT_INTERVAL 5000
#define SCREEN_INTERVAL 250
#define WIND_CALC_MS 1000

// ====================== COLOUR PALETTE (RGB565) =================
#define C_BG 0x0841      // near-black navy
#define C_CARD 0x10A2    // dark card surface
#define C_BORDER 0x2945  // subtle border
#define C_CYAN 0x2F3F    // electric cyan
#define C_GREEN 0x2EC8   // emerald
#define C_AMBER 0xFC00   // warm amber
#define C_RED 0xF800     // alert red
#define C_WHITE 0xFFFF
#define C_LGRAY 0xB596
#define C_GRAY 0x7BEF
#define C_DGRAY 0x2945
#define C_BLUE 0x2B1F

// ====================== LAYOUT ==================================
#define SCR_W 240
#define SCR_H 320
#define NUM_SCREENS 5

#define BAR_H 22            // status-bar height
#define DOT_Y (SCR_H - 14)  // page-dot vertical centre

// ====================== OBJECTS =================================
TFT_eSPI tft = TFT_eSPI();
TFT_eSprite spr = TFT_eSprite(&tft);  // small sprite for numbers

Adafruit_BMP280 bmp;
Adafruit_MPU6050 mpu;
Adafruit_INA219 ina(0x41);  // A0→VDD to avoid HDC1080 conflict
ClosedCube_HDC1080 hdc;

// ====================== SENSOR FLAGS ============================
bool hasINA = false;
bool hasBMP = false;
bool hasMPU = false;
bool hasHDC = false;
bool hasADS = false;

// ====================== STATE ===================================
int currentScreen = 0;
bool screenDirty = true;  // full redraw needed
bool valuesDirty = true;  // partial value update

unsigned long lastSensorMs = 0;
unsigned long lastPredictMs = 0;
unsigned long lastScreenMs = 0;
unsigned long lastWindMs = 0;
unsigned long lastInteractMs = 0;
unsigned long tripStartMs = 0;
unsigned long lastStopMs = 0;

// Smoothed display values
float dispRange = 0.0f;
float dispSOC = 50.0f;

// ---------------------- sensor data struct ----------------------
struct Sens {
    // INA219
    float busV;     // V
    float current;  // A
    float power;    // W
    // BMP280
    float altitude;  // m
    float pressure;  // hPa
    float bmpTemp;   // °C
    // HDC1080
    float humidity;  // %
    float hdcTemp;   // °C
    // MPU6050
    float ax, ay, az;  // m/s²
    float gx, gy, gz;  // °/s
    float tiltPitch;   // °
    float tiltRoll;    // °
    // Analog
    int ldr;
    int rain;
    float battTemp;  // °C (NTC)
    float throttle;  // 0-100 %
    // Wind
    float windKmh;      // wind speed km/h
    int windDirection;  // -1=headwind, 0=cross, +1=tailwind
    // Weight
    float weight;  // kg
    // Derived / cumulative
    float soc;            // %
    float speed;          // km/h  (estimated)
    float energyEff;      // Wh/km
    float energyWh;       // Wh consumed
    float distM;          // metres
    float rangeConsumed;  // km
    float altChange30s;   // m
    float timeSinceStop;  // s
    float rollRes;        // coeff
    float precipMm;       // mm/h est
    // Prediction
    float predRange;  // km
    int replaced;     // fallback count
    unsigned long inferUs;
} S = {};

// ---------------------- history ---------------------------------
#define HIST_LEN 60
float rangeHist[HIST_LEN] = {};
int histIdx = 0;
int histCount = 0;

// Altitude ring-buffer (6 × 5 s = 30 s window)
float altBuf[6] = {};
int altBufIdx = 0;

// ---------------------- wind ISR --------------------------------
volatile unsigned long wind1Pulses = 0;
volatile unsigned long wind2Pulses = 0;
void IRAM_ATTR wind1ISR() { wind1Pulses++; }
void IRAM_ATTR wind2ISR() { wind2Pulses++; }

// ====================== BUZZER ==================================
void buzzerSetup() {
    ledcAttach(BUZZER_PIN, 2000, 8);  // ESP32 core 3.x API
}
void beep(int freq, int ms) {
    ledcWriteTone(BUZZER_PIN, freq);
    delay(ms);
    ledcWriteTone(BUZZER_PIN, 0);
}

// ================================================================
//  UI  HELPERS
// ================================================================

// --- colour interpolation (RGB565) ---
uint16_t lerpColor(uint16_t a, uint16_t b, float t) {
    if (t <= 0) return a;
    if (t >= 1) return b;
    uint8_t r1 = (a >> 11) & 0x1F, g1 = (a >> 5) & 0x3F, b1 = a & 0x1F;
    uint8_t r2 = (b >> 11) & 0x1F, g2 = (b >> 5) & 0x3F, b2 = b & 0x1F;
    uint8_t r = r1 + (int)((r2 - r1) * t);
    uint8_t g = g1 + (int)((g2 - g1) * t);
    uint8_t bl = b1 + (int)((b2 - b1) * t);
    return (r << 11) | (g << 5) | bl;
}

// --- thick arc (radial-line method) ---
//  startDeg / sweepDeg in math degrees (0°=right, CCW positive).
//  Arc sweeps clockwise (decreasing angle).
void drawArc(int cx, int cy, int ro, int ri, float startDeg, float sweepDeg,
             uint16_t col) {
    for (float i = 0; i <= sweepDeg; i += 0.7f) {
        float a = (startDeg - i) * DEG_TO_RAD;
        float ca = cosf(a), sa = sinf(a);
        tft.drawLine(cx + (int)(ro * ca), cy - (int)(ro * sa),
                     cx + (int)(ri * ca), cy - (int)(ri * sa), col);
    }
}

// --- gradient arc for range gauge ---
void drawArcGradient(int cx, int cy, int ro, int ri, float startDeg,
                     float totalSweep, float pct) {
    float valSweep = totalSweep * constrain(pct, 0, 100) / 100.0f;
    for (float i = 0; i <= valSweep; i += 0.7f) {
        float t = i / totalSweep;
        uint16_t c;
        if (t < 0.35f)
            c = lerpColor(C_RED, C_AMBER, t / 0.35f);
        else if (t < 0.65f)
            c = lerpColor(C_AMBER, C_GREEN, (t - 0.35f) / 0.30f);
        else
            c = lerpColor(C_GREEN, C_CYAN, (t - 0.65f) / 0.35f);
        float a = (startDeg - i) * DEG_TO_RAD;
        float ca = cosf(a), sa = sinf(a);
        tft.drawLine(cx + (int)(ro * ca), cy - (int)(ro * sa),
                     cx + (int)(ri * ca), cy - (int)(ri * sa), c);
    }
}

// --- tick marks around gauge ---
void drawGaugeTicks(int cx, int cy, int r, int numTicks, float startDeg,
                    float sweepDeg) {
    for (int i = 0; i <= numTicks; i++) {
        float a = (startDeg - (float)i / numTicks * sweepDeg) * DEG_TO_RAD;
        float ca = cosf(a), sa = sinf(a);
        bool major = (i % (numTicks / 5) == 0);
        int oR = r + 3, iR = major ? r - 5 : r - 2;
        uint16_t c = major ? C_LGRAY : C_DGRAY;
        tft.drawLine(cx + (int)(oR * ca), cy - (int)(oR * sa),
                     cx + (int)(iR * ca), cy - (int)(iR * sa), c);
    }
}

// --- gauge number labels ---
void drawGaugeLabels(int cx, int cy, int r, float startDeg, float sweepDeg,
                     float maxVal) {
    tft.setTextDatum(MC_DATUM);
    tft.setTextColor(C_GRAY, C_BG);
    for (int i = 0; i <= 5; i++) {
        float a = (startDeg - (float)i / 5.0f * sweepDeg) * DEG_TO_RAD;
        int lR = r + 14;
        int x = cx + (int)(lR * cosf(a));
        int y = cy - (int)(lR * sinf(a));
        char b[6];
        dtostrf(maxVal * i / 5.0f, 1, 0, b);
        tft.drawString(b, x, y, 1);
    }
}

// --- info card (rounded rect with value + label) ---
void drawCard(int x, int y, int w, int h, const char* val, const char* lbl,
              uint16_t valCol, uint8_t valFont = 4) {
    tft.fillRoundRect(x, y, w, h, 6, C_CARD);
    tft.drawRoundRect(x, y, w, h, 6, C_BORDER);
    tft.setTextDatum(MC_DATUM);
    tft.setTextPadding(w - 8);
    tft.setTextColor(valCol, C_CARD);
    tft.drawString(val, x + w / 2, y + h / 2 - 7, valFont);
    tft.setTextColor(C_GRAY, C_CARD);
    tft.setTextPadding(w - 8);
    tft.drawString(lbl, x + w / 2, y + h - 11, 1);
}

// --- mini card ---
void drawMini(int x, int y, int w, int h, const char* val, const char* lbl,
              uint16_t c) {
    tft.fillRoundRect(x, y, w, h, 4, C_CARD);
    tft.setTextDatum(MC_DATUM);
    tft.setTextPadding(w - 4);
    tft.setTextColor(c, C_CARD);
    tft.drawString(val, x + w / 2, y + h / 2 - 5, 2);
    tft.setTextColor(C_DGRAY, C_CARD);
    tft.setTextPadding(w - 4);
    tft.drawString(lbl, x + w / 2, y + h - 8, 1);
}

// --- SOC progress bar ---
void drawSOCBar(int x, int y, int w, int h, float soc) {
    tft.fillRoundRect(x, y, w, h, h / 2, C_CARD);
    int fw = max((int)h, (int)(soc / 100.0f * w));
    uint16_t c = soc > 50 ? C_GREEN : soc > 25 ? C_AMBER : C_RED;
    if (fw > 0) tft.fillRoundRect(x, y, fw, h, h / 2, c);
    char b[6];
    sprintf(b, "%d%%", (int)soc);
    tft.setTextDatum(MC_DATUM);
    tft.setTextPadding(0);
    tft.setTextColor(C_WHITE, c);
    tft.drawString(b, x + w / 2, y + h / 2 + 1, 2);
}

// --- throttle bar ---
void drawThrottleBar(int x, int y, int w, int h, float pct) {
    tft.fillRoundRect(x, y, w, h, h / 2, C_CARD);
    int fw = max(0, (int)(pct / 100.0f * w));
    uint16_t c = pct < 50 ? C_GREEN : pct < 80 ? C_AMBER : C_RED;
    if (fw > h) tft.fillRoundRect(x, y, fw, h, h / 2, c);
}

// --- accelerometer bubble ---
void drawBubble(int cx, int cy, int sz, float axG, float ayG) {
    tft.fillRoundRect(cx - sz, cy - sz, sz * 2, sz * 2, 6, C_CARD);
    tft.drawRoundRect(cx - sz, cy - sz, sz * 2, sz * 2, 6, C_BORDER);
    // crosshair
    tft.drawFastHLine(cx - sz + 6, cy, (sz - 6) * 2, C_DGRAY);
    tft.drawFastVLine(cx, cy - sz + 6, (sz - 6) * 2, C_DGRAY);
    tft.drawCircle(cx, cy, sz / 3, C_DGRAY);
    tft.drawCircle(cx, cy, sz * 2 / 3, C_DGRAY);
    // dot  (±2 g range)
    int dx = constrain((int)(axG / 20.0f * sz), -sz + 6, sz - 6);
    int dy = constrain((int)(ayG / 20.0f * sz), -sz + 6, sz - 6);
    float mag = sqrtf(axG * axG + ayG * ayG);
    uint16_t dc = mag < 3 ? C_GREEN : mag < 7 ? C_AMBER : C_RED;
    tft.fillCircle(cx + dx, cy + dy, 5, dc);
}

// --- line graph ---
void drawGraph(int x, int y, int w, int h, float* data, int count, int startIdx,
               uint16_t lineCol) {
    tft.fillRoundRect(x, y, w, h, 4, C_CARD);
    tft.drawRoundRect(x, y, w, h, 4, C_BORDER);
    if (count < 2) {
        tft.setTextDatum(MC_DATUM);
        tft.setTextColor(C_DGRAY, C_CARD);
        tft.drawString("collecting data...", x + w / 2, y + h / 2, 1);
        return;
    }
    int pad = 6;
    int gx = x + pad, gy = y + pad, gw = w - pad * 2, gh = h - pad * 2;
    float mn = 1e9, mx = -1e9;
    for (int i = 0; i < count; i++) {
        int idx = (startIdx + i) % HIST_LEN;
        if (data[idx] < mn) mn = data[idx];
        if (data[idx] > mx) mx = data[idx];
    }
    float rng = mx - mn;
    if (rng < 1) {
        mn -= 0.5;
        mx += 0.5;
        rng = mx - mn;
    }
    mn -= rng * 0.08f;
    mx += rng * 0.08f;
    rng = mx - mn;
    // horizontal grid
    for (int i = 0; i <= 4; i++) {
        int ly = gy + gh - i * gh / 4;
        tft.drawFastHLine(gx, ly, gw, C_DGRAY);
    }
    // plot
    for (int i = 1; i < count; i++) {
        int i0 = (startIdx + i - 1) % HIST_LEN;
        int i1 = (startIdx + i) % HIST_LEN;
        int px0 = gx + (i - 1) * gw / (count - 1);
        int px1 = gx + i * gw / (count - 1);
        int py0 = gy + gh - (int)((data[i0] - mn) / rng * gh);
        int py1 = gy + gh - (int)((data[i1] - mn) / rng * gh);
        tft.drawLine(px0, py0, px1, py1, lineCol);
        // thicken
        tft.drawLine(px0, py0 + 1, px1, py1 + 1, lineCol);
    }
}

// --- page indicator ---
void drawPageDots(int cur, int total) {
    int sp = 14, rad = 4;
    int tw = total * sp;
    int sx = (SCR_W - tw) / 2 + rad;
    for (int i = 0; i < total; i++) {
        int px = sx + i * sp;
        if (i == cur)
            tft.fillCircle(px, DOT_Y, rad, C_CYAN);
        else
            tft.drawCircle(px, DOT_Y, rad - 1, C_DGRAY);
    }
}

// --- status bar ---
void drawStatusBar() {
    tft.fillRect(0, 0, SCR_W, BAR_H, C_CARD);
    tft.drawFastHLine(0, BAR_H, SCR_W, C_BORDER);
    // sensor health dots
    int dx = 8;
    uint16_t cols[] = {hasINA ? C_GREEN : C_RED, hasBMP ? C_GREEN : C_RED,
                       hasMPU ? C_GREEN : C_RED, hasHDC ? C_GREEN : C_RED,
                       hasADS ? C_GREEN : C_RED};
    const char* ids[] = {"I", "B", "M", "H", "W"};
    for (int i = 0; i < 5; i++) {
        tft.fillCircle(dx, BAR_H / 2, 3, cols[i]);
        tft.setTextDatum(ML_DATUM);
        tft.setTextColor(C_LGRAY, C_CARD);
        tft.drawString(ids[i], dx + 6, BAR_H / 2, 1);
        dx += 22;
    }
    // heartbeat
    static bool pulse = false;
    pulse = !pulse;
    tft.fillCircle(SCR_W - 10, BAR_H / 2, 3, pulse ? C_CYAN : C_CARD);
    // fallback warning
    if (S.replaced > 0) {
        char b[6];
        sprintf(b, "!%d", S.replaced);
        tft.setTextDatum(MR_DATUM);
        tft.setTextColor(C_AMBER, C_CARD);
        tft.drawString(b, SCR_W - 20, BAR_H / 2, 2);
    }
}

// --- screen title ---
void drawTitle(const char* t) {
    tft.setTextDatum(TL_DATUM);
    tft.setTextColor(C_CYAN, C_BG);
    tft.setTextPadding(0);
    tft.drawString(t, 8, BAR_H + 6, 2);
    tft.drawFastHLine(8, BAR_H + 24, SCR_W - 16, C_BORDER);
}

// colour for range value
uint16_t rangeColor(float r) {
    if (r < 5) return C_RED;
    if (r < 15) return C_AMBER;
    if (r < 30) return C_WHITE;
    return C_CYAN;
}

// colour for SOC
uint16_t socColor(float s) {
    if (s < 15) return C_RED;
    if (s < 30) return C_AMBER;
    return C_GREEN;
}

// ================================================================
//  SENSOR  DRIVERS
// ================================================================

// --- NTC temperature ---
float readNTC() {
    int raw = analogRead(NTC_PIN);
    if (raw <= 0 || raw >= 4095) return NAN;
    float r = (float)NTC_R_SERIES * (4095.0f / raw - 1.0f);
    float t = logf(r / NTC_R_NOMINAL) / NTC_BETA;
    t += 1.0f / (NTC_TEMP_NOM + 273.15f);
    return 1.0f / t - 273.15f;
}

// --- ADS1232 24-bit read ---
long readADS1232raw() {
    unsigned long t0 = millis();
    while (digitalRead(ADS_DOUT_PIN) == HIGH) {
        if (millis() - t0 > 150) return 0;  // timeout
    }
    long val = 0;
    for (int i = 0; i < 24; i++) {
        digitalWrite(ADS_SCLK_PIN, HIGH);
        delayMicroseconds(1);
        val = (val << 1) | digitalRead(ADS_DOUT_PIN);
        digitalWrite(ADS_SCLK_PIN, LOW);
        delayMicroseconds(1);
    }
    // 25th clock → channel 1
    digitalWrite(ADS_SCLK_PIN, HIGH);
    delayMicroseconds(1);
    digitalWrite(ADS_SCLK_PIN, LOW);
    delayMicroseconds(1);
    // sign-extend 24→32
    if (val & 0x800000) val |= 0xFF000000;
    return val;
}

float readWeight() {
    long raw = readADS1232raw();
    return (float)(raw - ADS_TARE_VALUE) / ADS_SCALE_FACTOR;
}

// --- wind speed & direction (dual hall sensor) ---
void calcWind() {
    unsigned long now = millis();
    unsigned long dt = now - lastWindMs;
    if (dt < 1000) return;  // Calculate every 1 second

    // Capture pulses atomically
    noInterrupts();
    unsigned long p1 = wind1Pulses;
    unsigned long p2 = wind2Pulses;
    wind1Pulses = 0;
    wind2Pulses = 0;
    interrupts();
    lastWindMs = now;

    // Calculate RPM from each sensor
    float rpm1 = (p1 * 60000.0f) / dt;
    float rpm2 = (p2 * 60000.0f) / dt;

    // Convert RPM to linear speed (m/s → km/h)
    float speed1 = (FAN_CIRCUMFERENCE * rpm1 / 60.0f) * 3.6f;
    float speed2 = (FAN_CIRCUMFERENCE * rpm2 / 60.0f) * 3.6f;

    // Average speed
    S.windKmh = (speed1 + speed2) / 2.0f;

    // Direction (90° sensor separation → pulse ratio indicates angle)
    if (p1 > p2 * 1.2f)
        S.windDirection = 1;  // tailwind (sensor 1 leads)
    else if (p2 > p1 * 1.2f)
        S.windDirection = -1;  // headwind (sensor 2 leads)
    else
        S.windDirection = 0;  // crosswind (balanced)
}

// --- derived quantities ---
float estimatePrecip(int rainVal) {
    if (rainVal > 3000) return 0.0f;
    if (rainVal > 2000) return 0.5f;
    if (rainVal > 1000) return 2.0f;
    if (rainVal > 500) return 5.0f;
    return 10.0f;
}

float calcRollRes(int rainVal) { return (rainVal < 1500) ? 0.013f : 0.008f; }

float calcSOC(float v) {
    return constrain((v - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V) * 100.0f,
                     0.0f, 100.0f);
}

// --- sensor init ---
void initSensors() {
    Wire.begin(21, 22);
    Wire.setClock(400000);

    // INA219
    if (ina.begin()) {
        hasINA = true;
        ina.setCalibration_32V_2A();
    }

    // BMP280
    if (bmp.begin(0x76)) {
        hasBMP = true;
        bmp.setSampling(
            Adafruit_BMP280::MODE_NORMAL, Adafruit_BMP280::SAMPLING_X2,
            Adafruit_BMP280::SAMPLING_X16, Adafruit_BMP280::FILTER_X4,
            Adafruit_BMP280::STANDBY_MS_500);
    }

    // MPU6050
    if (mpu.begin(0x68)) {
        hasMPU = true;
        mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
        mpu.setGyroRange(MPU6050_RANGE_500_DEG);
        mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    }

    // HDC1080
    hdc.begin(0x40);
    // Simple connectivity check
    float t = hdc.readTemperature();
    hasHDC = !(isnan(t) || t < -40 || t > 125);

    // ADS1232
    pinMode(ADS_DOUT_PIN, INPUT);
    pinMode(ADS_SCLK_PIN, OUTPUT);
    pinMode(ADS_PDWN_PIN, OUTPUT);
    digitalWrite(ADS_PDWN_PIN, HIGH);  // power on
    digitalWrite(ADS_SCLK_PIN, LOW);
    // SPEED pin hardwired to GND (10 Hz mode)
    delay(100);
    // quick read test
    long test = readADS1232raw();
    hasADS = (test != 0);
}

// --- read all sensors ---
void readSensors() {
    // INA219
    if (hasINA) {
        S.busV = ina.getBusVoltage_V();
        S.current = ina.getCurrent_mA() / 1000.0f;
        S.power = ina.getPower_mW() / 1000.0f;
    }
    // BMP280
    if (hasBMP) {
        S.altitude = bmp.readAltitude(1013.25f);
        S.pressure = bmp.readPressure() / 100.0f;
        S.bmpTemp = bmp.readTemperature();
    }
    // MPU6050
    if (hasMPU) {
        sensors_event_t a, g, t;
        mpu.getEvent(&a, &g, &t);
        S.ax = a.acceleration.x;
        S.ay = a.acceleration.y;
        S.az = a.acceleration.z;
        S.gx = g.gyro.x * RAD_TO_DEG;
        S.gy = g.gyro.y * RAD_TO_DEG;
        S.gz = g.gyro.z * RAD_TO_DEG;
        S.tiltPitch =
            atan2f(S.ax, sqrtf(S.ay * S.ay + S.az * S.az)) * RAD_TO_DEG;
        S.tiltRoll =
            atan2f(S.ay, sqrtf(S.ax * S.ax + S.az * S.az)) * RAD_TO_DEG;
    }
    // HDC1080
    if (hasHDC) {
        S.hdcTemp = hdc.readTemperature();
        S.humidity = hdc.readHumidity();
    }
    // Analog sensors
    S.ldr = analogRead(LDR_PIN);
    S.rain = analogRead(RAIN_PIN);
    S.battTemp = readNTC();
    int rawThr = analogRead(THROTTLE_PIN);
    S.throttle = constrain((float)(rawThr - THROT_ADC_MIN) /
                               (THROT_ADC_MAX - THROT_ADC_MIN) * 100.0f,
                           0.0f, 100.0f);
    // Wind (calculated separately in loop)
    calcWind();

    // Weight
    if (hasADS) S.weight = readWeight();

    // Derived - with bounds checking
    if (hasINA) {
        S.soc = calcSOC(S.busV);
    } else {
        S.soc = 50.0f;  // safe default if no INA219
    }

    S.precipMm = estimatePrecip(S.rain);
    S.rollRes = calcRollRes(S.rain);

    // Validate critical sensor ranges
    if (S.ldr < 0 || S.ldr > 4095) S.ldr = 447;     // median
    if (S.rain < 0 || S.rain > 4095) S.rain = 162;  // median
}

// --- cumulative updates (call each SENSOR_INTERVAL) ---
void updateCumulatives(float dtS) {
    // Speed estimate (rough — replace with wheel sensor if available)
    if (S.throttle < 3.0f)
        S.speed = 0.0f;
    else
        S.speed = S.throttle * 0.35f;  // max ~35 km/h at 100 %

    // Distance
    float ms = S.speed / 3.6f;
    S.distM += ms * dtS;
    S.rangeConsumed = S.distM / 1000.0f;

    // Energy
    if (hasINA && S.power > 0) S.energyWh += S.power * (dtS / 3600.0f);

    // Efficiency
    if (S.distM > 50)
        S.energyEff = S.energyWh / (S.distM / 1000.0f);
    else
        S.energyEff = 8.1f;

    // Stop timer
    if (S.speed < 1.0f) lastStopMs = millis();
    S.timeSinceStop = (millis() - lastStopMs) / 1000.0f;

    // Altitude Δ30 s
    altBuf[altBufIdx] = S.altitude;
    int oldest = (altBufIdx + 1) % 6;
    S.altChange30s = S.altitude - altBuf[oldest];
    altBufIdx = (altBufIdx + 1) % 6;
}

// ================================================================
//  ML  PREDICTION
// ================================================================

void runPrediction() {
    // Feature array - ORDER MUST MATCH MODEL HEADER!
    float feat[EV_MODEL_N_FEATURES];

    // [ 0] soc_percent (0-100%)
    feat[0] = constrain(S.soc, 0.0f, 100.0f);

    // [ 1] energy_efficiency_whkm (typically 5-15 Wh/km)
    feat[1] = constrain(S.energyEff, 0.1f, 50.0f);

    // [ 2] precipitation_mm (0-10+ mm/h)
    feat[2] = constrain(S.precipMm, 0.0f, 20.0f);

    // [ 3] altitude_bmp280_m (-500 to 3000m typical)
    feat[3] = constrain(S.altitude, -500.0f, 3000.0f);

    // [ 4] rain_sensor_value (ADC 0-4095, lower=wetter)
    feat[4] = (float)constrain(S.rain, 0, 4095);

    // [ 5] ldr_value (ADC 0-4095, higher=brighter)
    feat[5] = (float)constrain(S.ldr, 0, 4095);

    // [ 6] rolling_resistance_coeff (0.005-0.015 typical)
    feat[6] = constrain(S.rollRes, 0.004f, 0.020f);

    // [ 7] energy_consumed_wh (0+ Wh)
    feat[7] = max(0.0f, S.energyWh);

    // [ 8] altitude_change_30s_m (-100 to +100m typical)
    feat[8] = constrain(S.altChange30s, -200.0f, 200.0f);

    // [ 9] distance_m (0+ metres)
    feat[9] = max(0.0f, S.distM);

    // [10] range_consumed_km (0+ km)
    feat[10] = max(0.0f, S.rangeConsumed);

    // [11] time_since_last_stop_s (0+ seconds)
    feat[11] = max(0.0f, S.timeSinceStop);

    unsigned long t0 = micros();
    S.predRange = ev_predict_range_km_safe(feat, &S.replaced);
    S.inferUs = micros() - t0;
    S.predRange = constrain(S.predRange, 0, 999);

    // Push to history
    int hSlot = (histIdx + histCount) % HIST_LEN;
    rangeHist[hSlot] = S.predRange;
    if (histCount < HIST_LEN)
        histCount++;
    else
        histIdx = (histIdx + 1) % HIST_LEN;
}

// ================================================================
//  SCREEN  DRAWING
// ================================================================

// -------------------- 0: BOOT SPLASH ---------------------------
void drawBootScreen() {
    tft.fillScreen(C_BG);

    // Title
    tft.setTextDatum(MC_DATUM);
    tft.setTextColor(C_CYAN);
    tft.drawString("EV  RANGE", SCR_W / 2, 60, 4);
    tft.setTextColor(C_WHITE);
    tft.drawString("DASHBOARD", SCR_W / 2, 90, 4);
    tft.setTextColor(C_DGRAY);
    tft.drawString("v1.0   Random Forest ML", SCR_W / 2, 118, 1);

    // decorative line
    tft.drawFastHLine(40, 135, SCR_W - 80, C_BORDER);

    // Sensor init list
    const char* names[] = {"INA219  Pwr", "BMP280  Alt", "MPU6050 IMU",
                           "HDC1080 T/H", "ADS1232 Wgt"};
    bool* flags[] = {&hasINA, &hasBMP, &hasMPU, &hasHDC, &hasADS};

    tft.setTextDatum(TL_DATUM);
    for (int i = 0; i < 5; i++) {
        int y = 150 + i * 24;
        tft.setTextColor(C_GRAY);
        tft.drawString(names[i], 40, y, 2);
        // animate a brief pause
        delay(180);
        bool ok = *flags[i];
        tft.setTextColor(ok ? C_GREEN : C_RED);
        tft.drawString(ok ? "OK" : "FAIL", 190, y, 2);
        if (!ok) beep(800, 60);
    }

    // progress bar
    for (int i = 0; i <= 100; i += 2) {
        int bw = (SCR_W - 80) * i / 100;
        tft.fillRoundRect(40, 290, bw, 8, 4, C_CYAN);
        delay(12);
    }
    delay(400);
}

// -------------------- 1: MAIN RANGE ----------------------------
void drawScreen1(bool full) {
    const int cx = 120, cy = 115;
    const int oR = 74, iR = 60;
    const float sD = 225.0f, tS = 270.0f;

    if (full) {
        tft.fillScreen(C_BG);
        drawStatusBar();
        // gauge ticks + labels (static)
        drawGaugeTicks(cx, cy, oR, 25, sD, tS);
        drawGaugeLabels(cx, cy, oR, sD, tS, GAUGE_MAX_RANGE);
    }

    // Smooth displayed range
    dispRange += (S.predRange - dispRange) * 0.25f;
    dispSOC += (S.soc - dispSOC) * 0.30f;

    float pct = constrain(dispRange / GAUGE_MAX_RANGE * 100.0f, 0, 100);

    // Arc background + gradient
    drawArc(cx, cy, oR, iR, sD, tS, C_DGRAY);
    drawArcGradient(cx, cy, oR, iR, sD, tS, pct);

    // Centre glow line
    tft.drawFastHLine(cx - 45, cy + 28, 90, rangeColor(dispRange));

    // Range number (font 7 = 7-segment, large)
    char buf[8];
    if (dispRange >= 10)
        dtostrf(dispRange, 4, 1, buf);
    else
        dtostrf(dispRange, 3, 1, buf);
    tft.setTextDatum(MC_DATUM);
    tft.setTextPadding(
        100);  // Reduced from 130 to fit inside inner circle (r=60, dia=120)
    tft.setTextColor(rangeColor(dispRange), C_BG);
    tft.drawString(buf, cx, cy - 2, 7);

    // Unit
    tft.setTextPadding(30);  // Reduced from 40 for better fit
    tft.setTextColor(C_GRAY, C_BG);
    tft.drawString("km", cx, cy + 32, 2);

    // SOC bar
    drawSOCBar(18, 175, 204, 16, dispSOC);

    // Info cards — Speed & Power
    char sBuf[10], pBuf[10];
    dtostrf(S.speed, 3, 0, sBuf);
    if (S.power >= 1.0f) {
        dtostrf(S.power, 3, 1, pBuf);
        strcat(pBuf, " kW");
    } else {
        dtostrf(S.power * 1000, 4, 0, pBuf);
        strcat(pBuf, " W");
    }
    drawCard(6, 198, 112, 50, sBuf, "SPEED  km/h", C_WHITE);
    drawCard(122, 198, 112, 50, pBuf, "POWER", C_CYAN);

    // Quick environment strip
    char t1[8], t2[10], t3[8];
    dtostrf(hasHDC ? S.hdcTemp : S.bmpTemp, 3, 0, t1);
    strcat(t1,
           "\xF7"
           "C");
    dtostrf(S.windKmh, 3, 0, t2);
    strcat(t2, " km/h");
    dtostrf(S.altitude, 4, 0, t3);
    strcat(t3, "m");

    drawMini(6, 255, 74, 34, t1, "TEMP", C_WHITE);
    drawMini(83, 255, 74, 34, t2, "WIND", C_BLUE);
    drawMini(160, 255, 74, 34, t3, "ALT", C_GREEN);

    drawPageDots(0, NUM_SCREENS);
}

// -------------------- 2: ENVIRONMENT ---------------------------
void drawScreen2(bool full) {
    if (full) {
        tft.fillScreen(C_BG);
        drawStatusBar();
        drawTitle("ENVIRONMENT");
    }
    int yy = BAR_H + 30;

    // Weather card
    {
        int ch = 68;
        tft.fillRoundRect(6, yy, 228, ch, 6, C_CARD);
        tft.drawRoundRect(6, yy, 228, ch, 6, C_BORDER);

        // Weather status
        const char* wx;
        uint16_t wc;
        if (S.rain < 1000) {
            wx = "HEAVY RAIN";
            wc = C_BLUE;
        } else if (S.rain < 2000) {
            wx = "RAIN";
            wc = C_BLUE;
        } else if (S.ldr > 2500) {
            wx = "SUNNY";
            wc = C_AMBER;
        } else if (S.ldr > 1000) {
            wx = "CLOUDY";
            wc = C_LGRAY;
        } else {
            wx = "NIGHT";
            wc = C_DGRAY;
        }

        tft.setTextDatum(TL_DATUM);
        tft.setTextPadding(100);
        tft.setTextColor(wc, C_CARD);
        tft.drawString(wx, 14, yy + 8, 4);

        char tb[8];
        dtostrf(hasHDC ? S.hdcTemp : S.bmpTemp, 3, 1, tb);
        strcat(tb,
               "\xF7"
               "C");
        tft.setTextDatum(TR_DATUM);
        tft.setTextColor(C_WHITE, C_CARD);
        tft.setTextPadding(70);
        tft.drawString(tb, 228, yy + 8, 4);

        // Second row: humidity, LDR, rain
        char hb[10];
        sprintf(hb, "Hum: %d%%", (int)S.humidity);
        char lb[12];
        sprintf(lb, "LDR: %d", S.ldr);
        char rb[14];
        sprintf(rb, "Rain: %d", S.rain);
        tft.setTextDatum(TL_DATUM);
        tft.setTextPadding(70);
        tft.setTextColor(C_LGRAY, C_CARD);
        tft.drawString(hb, 14, yy + 42, 2);
        tft.drawString(lb, 95, yy + 42, 2);
        tft.setTextDatum(TR_DATUM);
        tft.drawString(rb, 228, yy + 42, 2);
        yy += ch + 6;
    }

    // Atmosphere card
    {
        int ch = 58;
        tft.fillRoundRect(6, yy, 228, ch, 6, C_CARD);
        tft.drawRoundRect(6, yy, 228, ch, 6, C_BORDER);
        tft.setTextDatum(TL_DATUM);
        tft.setTextColor(C_GRAY, C_CARD);
        tft.setTextPadding(0);
        tft.drawString("ATMOSPHERE", 14, yy + 4, 1);

        char ab[12];
        dtostrf(S.altitude, 5, 1, ab);
        strcat(ab, " m");
        char pb[14];
        dtostrf(S.pressure, 6, 1, pb);
        strcat(pb, " hPa");
        char db[14];
        dtostrf(S.altChange30s, 4, 2, db);
        strcat(db, " m/30s");

        tft.setTextColor(C_WHITE, C_CARD);
        tft.setTextPadding(90);
        tft.drawString(ab, 14, yy + 20, 2);
        tft.drawString(pb, 120, yy + 20, 2);
        tft.setTextColor(S.altChange30s > 1    ? C_AMBER
                         : S.altChange30s < -1 ? C_GREEN
                                               : C_LGRAY,
                         C_CARD);
        tft.setTextPadding(100);
        tft.drawString(db, 14, yy + 38, 2);
        yy += ch + 6;
    }

    // Wind & Load card
    {
        int ch = 58;
        tft.fillRoundRect(6, yy, 228, ch, 6, C_CARD);
        tft.drawRoundRect(6, yy, 228, ch, 6, C_BORDER);
        tft.setTextDatum(TL_DATUM);
        tft.setTextColor(C_GRAY, C_CARD);
        tft.setTextPadding(0);
        tft.drawString("WIND & LOAD", 14, yy + 4, 1);

        char wb[14];
        dtostrf(S.windKmh, 4, 1, wb);
        strcat(wb, " km/h");
        char wl[16];
        dtostrf(S.weight, 4, 1, wl);
        strcat(wl, " kg");

        tft.setTextColor(C_BLUE, C_CARD);
        tft.setTextPadding(90);
        tft.drawString(wb, 14, yy + 22, 2);
        tft.setTextColor(C_WHITE, C_CARD);
        tft.setTextPadding(90);
        tft.drawString(wl, 120, yy + 22, 2);

        const char* windDir = S.windDirection == -1  ? "HEADWIND"
                              : S.windDirection == 1 ? "TAILWIND"
                                                     : "CROSSWIND";
        tft.setTextColor(C_LGRAY, C_CARD);
        tft.setTextPadding(100);
        tft.drawString(windDir, 14, yy + 40, 2);
        yy += ch + 6;
    }

    // Rolling resistance & precipitation
    {
        int ch = 42;
        tft.fillRoundRect(6, yy, 228, ch, 6, C_CARD);
        tft.drawRoundRect(6, yy, 228, ch, 6, C_BORDER);
        char rr[16];
        dtostrf(S.rollRes, 5, 4, rr);
        char pp[14];
        dtostrf(S.precipMm, 3, 1, pp);
        strcat(pp, " mm/h");
        tft.setTextDatum(TL_DATUM);
        tft.setTextColor(C_LGRAY, C_CARD);
        tft.setTextPadding(100);
        tft.drawString("Crr:", 14, yy + 6, 1);
        tft.setTextColor(C_WHITE, C_CARD);
        tft.drawString(rr, 40, yy + 6, 2);
        tft.setTextColor(C_LGRAY, C_CARD);
        tft.drawString("Precip:", 120, yy + 6, 1);
        tft.setTextColor(C_BLUE, C_CARD);
        tft.drawString(pp, 165, yy + 6, 2);
    }

    drawPageDots(1, NUM_SCREENS);
}

// -------------------- 3: BATTERY STATUS ------------------------
void drawScreen3(bool full) {
    if (full) {
        tft.fillScreen(C_BG);
        drawStatusBar();
        drawTitle("BATTERY");
    }
    int yy = BAR_H + 30;

    // SOC ring gauge
    {
        int cx = SCR_W / 2, cy = yy + 58;
        int oR = 52, iR = 40;
        drawArc(cx, cy, oR, iR, 225, 270, C_DGRAY);
        float sp = constrain(dispSOC, 0, 100);
        float vs = 270.0f * sp / 100.0f;
        uint16_t sc = socColor(dispSOC);
        // fill arc
        for (float i = 0; i <= vs; i += 0.8f) {
            float a = (225.0f - i) * DEG_TO_RAD;
            float ca = cosf(a), sa = sinf(a);
            tft.drawLine(cx + (int)(oR * ca), cy - (int)(oR * sa),
                         cx + (int)(iR * ca), cy - (int)(iR * sa), sc);
        }
        // centre text
        char sb[6];
        sprintf(sb, "%d%%", (int)dispSOC);
        tft.setTextDatum(MC_DATUM);
        tft.setTextPadding(60);
        tft.setTextColor(C_WHITE, C_BG);
        tft.drawString(sb, cx, cy - 4, 4);
        tft.setTextColor(C_GRAY, C_BG);
        tft.setTextPadding(40);
        tft.drawString("SOC", cx, cy + 18, 1);
        yy = cy + oR + 10;
    }

    // 2×2 stat grid
    {
        char vb[10];
        dtostrf(S.busV, 4, 1, vb);
        strcat(vb, " V");
        char cb[10];
        dtostrf(S.current, 4, 2, cb);
        strcat(cb, " A");
        char pb[10];
        if (S.power >= 1.0f) {
            dtostrf(S.power, 4, 2, pb);
            strcat(pb, " kW");
        } else {
            dtostrf(S.power * 1000, 4, 0, pb);
            strcat(pb, " W");
        }
        char tb[10];
        if (!isnan(S.battTemp)) {
            dtostrf(S.battTemp, 3, 1, tb);
            strcat(tb,
                   "\xF7"
                   "C");
        } else
            strcpy(tb, "--");

        drawCard(6, yy, 112, 48, vb, "VOLTAGE", C_WHITE);
        drawCard(122, yy, 112, 48, cb, "CURRENT", C_CYAN);
        drawCard(6, yy + 54, 112, 48, pb, "POWER", C_AMBER);
        drawCard(122, yy + 54, 112, 48, tb, "BATT TEMP",
                 (S.battTemp > 40 || S.battTemp < 5) ? C_RED : C_GREEN);
        yy += 110;
    }

    // Energy bar
    {
        tft.fillRoundRect(6, yy, 228, 38, 6, C_CARD);
        tft.drawRoundRect(6, yy, 228, 38, 6, C_BORDER);
        char eb[20];
        sprintf(eb, "%.0f / %.0f Wh", S.energyWh, BATT_CAPACITY_WH);
        tft.setTextDatum(MC_DATUM);
        tft.setTextColor(C_WHITE, C_CARD);
        tft.setTextPadding(200);
        tft.drawString(eb, SCR_W / 2, yy + 10, 2);
        // mini bar
        float ep = constrain(S.energyWh / BATT_CAPACITY_WH * 100, 0, 100);
        int bx = 14, by = yy + 25, bw = 212, bh = 6;
        tft.fillRoundRect(bx, by, bw, bh, 3, C_DGRAY);
        int fw = max(bh, (int)(ep / 100.0f * bw));
        tft.fillRoundRect(bx, by, fw, bh, 3, C_AMBER);
    }

    drawPageDots(2, NUM_SCREENS);
}

// -------------------- 4: DYNAMICS ------------------------------
void drawScreen4(bool full) {
    if (full) {
        tft.fillScreen(C_BG);
        drawStatusBar();
        drawTitle("DYNAMICS");
    }
    int yy = BAR_H + 30;

    // Throttle
    tft.setTextDatum(TL_DATUM);
    tft.setTextColor(C_GRAY, C_BG);
    tft.setTextPadding(0);
    tft.drawString("THROTTLE", 8, yy, 1);
    char tp[8];
    sprintf(tp, "%d%%", (int)S.throttle);
    tft.setTextDatum(TR_DATUM);
    tft.setTextColor(C_WHITE, C_BG);
    tft.setTextPadding(40);
    tft.drawString(tp, 232, yy, 2);
    drawThrottleBar(8, yy + 18, 224, 12, S.throttle);
    yy += 38;

    // Accelerometer bubble
    tft.setTextDatum(TL_DATUM);
    tft.setTextColor(C_GRAY, C_BG);
    tft.setTextPadding(0);
    tft.drawString("ACCELERATION", 8, yy, 1);
    yy += 14;
    drawBubble(SCR_W / 2, yy + 55, 55, S.ax, S.ay);

    // Accel numbers beside the bubble
    char axb[10];
    dtostrf(S.ax / 9.81f, 4, 2, axb);
    strcat(axb, " g");
    char ayb[10];
    dtostrf(S.ay / 9.81f, 4, 2, ayb);
    strcat(ayb, " g");
    char azb[10];
    dtostrf(S.az / 9.81f, 4, 2, azb);
    strcat(azb, " g");
    tft.setTextDatum(TL_DATUM);
    tft.setTextColor(C_LGRAY, C_BG);
    tft.setTextPadding(50);
    tft.drawString("X:", 6, yy + 20, 1);
    tft.setTextColor(C_WHITE, C_BG);
    tft.drawString(axb, 18, yy + 20, 1);
    tft.setTextColor(C_LGRAY, C_BG);
    tft.drawString("Y:", 6, yy + 32, 1);
    tft.setTextColor(C_WHITE, C_BG);
    tft.drawString(ayb, 18, yy + 32, 1);
    tft.setTextColor(C_LGRAY, C_BG);
    tft.drawString("Z:", 6, yy + 44, 1);
    tft.setTextColor(C_WHITE, C_BG);
    tft.drawString(azb, 18, yy + 44, 1);

    yy += 118;

    // Tilt
    char pitchB[12];
    dtostrf(S.tiltPitch, 4, 1, pitchB);
    strcat(pitchB, "\xF7");
    char rollB[12];
    dtostrf(S.tiltRoll, 4, 1, rollB);
    strcat(rollB, "\xF7");
    tft.setTextDatum(TL_DATUM);
    tft.setTextColor(C_GRAY, C_BG);
    tft.setTextPadding(0);
    tft.drawString("TILT", 8, yy, 1);
    tft.setTextColor(C_WHITE, C_BG);
    tft.setTextPadding(70);
    tft.drawString(pitchB, 8, yy + 12, 2);
    tft.drawString(rollB, 110, yy + 12, 2);
    tft.setTextColor(C_DGRAY, C_BG);
    tft.setTextPadding(50);
    tft.drawString("Pitch", 8, yy + 30, 1);
    tft.drawString("Roll", 110, yy + 30, 1);
    yy += 44;

    // Speed & Distance cards
    char sb[10];
    dtostrf(S.speed, 3, 0, sb);
    char db[10];
    dtostrf(S.distM / 1000.0f, 4, 2, db);
    drawCard(6, yy, 112, 48, sb, "SPEED km/h", C_WHITE);
    drawCard(122, yy, 112, 48, db, "DIST km", C_CYAN);

    drawPageDots(3, NUM_SCREENS);
}

// -------------------- 5: TRIP HISTORY --------------------------
void drawScreen5(bool full) {
    if (full) {
        tft.fillScreen(C_BG);
        drawStatusBar();
        drawTitle("TRIP");
    }
    int yy = BAR_H + 30;

    // Range history graph
    tft.setTextDatum(TL_DATUM);
    tft.setTextColor(C_GRAY, C_BG);
    tft.setTextPadding(0);
    tft.drawString("RANGE HISTORY (5 min)", 8, yy, 1);
    yy += 14;
    drawGraph(6, yy, 228, 110, rangeHist, histCount, histIdx, C_CYAN);

    // Axis labels
    tft.setTextDatum(TL_DATUM);
    tft.setTextColor(C_DGRAY, C_CARD);
    tft.drawString("-5m", 10, yy + 98, 1);
    tft.setTextDatum(TR_DATUM);
    tft.drawString("now", 230, yy + 98, 1);
    yy += 118;

    // Stat cards (2×2)
    char d1[10];
    dtostrf(S.distM / 1000.0f, 4, 2, d1);
    unsigned long elapsed = (millis() - tripStartMs) / 1000;
    char d2[10];
    sprintf(d2, "%lu:%02lu", elapsed / 60, elapsed % 60);
    char d3[10];
    dtostrf(S.energyEff, 4, 1, d3);
    char d4[10];
    sprintf(d4, "%lu us", S.inferUs);

    drawCard(6, yy, 112, 46, d1, "DISTANCE km", C_WHITE);
    drawCard(122, yy, 112, 46, d2, "ELAPSED", C_LGRAY);
    drawCard(6, yy + 52, 112, 46, d3, "Wh/km", C_AMBER);
    drawCard(122, yy + 52, 112, 46, d4, "INFERENCE", C_GREEN);

    drawPageDots(4, NUM_SCREENS);
}

// ================================================================
//  ALERTS
// ================================================================

void checkAlerts() {
    // Low battery
    if (S.soc < 5) {
        beep(1000, 200);
        delay(100);
        beep(1000, 200);
        delay(100);
        beep(1000, 200);
    } else if (S.soc < 15 && (millis() % 30000 < 500)) {
        beep(1500, 100);
    }
    // Hot battery
    if (!isnan(S.battTemp) && S.battTemp > 50 && (millis() % 20000 < 500)) {
        beep(2000, 150);
    }
}

// ================================================================
//  INPUT  HANDLING
// ================================================================

void handleButton() {
    static bool wasPressed = false;
    static unsigned long pStart = 0;
    bool isP = (digitalRead(BUTTON_PIN) == LOW);

    if (isP && !wasPressed) pStart = millis();

    if (!isP && wasPressed) {
        unsigned long dur = millis() - pStart;
        if (dur > 1200) {
            // long press → home
            currentScreen = 0;
            screenDirty = true;
            beep(1500, 80);
        } else if (dur > 40) {
            // short press → next
            currentScreen = (currentScreen + 1) % NUM_SCREENS;
            screenDirty = true;
            beep(2500, 25);
        }
        lastInteractMs = millis();
    }
    wasPressed = isP;

    // Auto-return to home after 45 s
    if (currentScreen != 0 && (millis() - lastInteractMs > 45000)) {
        currentScreen = 0;
        screenDirty = true;
    }
}

// ================================================================
//  SETUP
// ================================================================

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println("\n[EV Dashboard] booting...");

    // Pin modes
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(WIND_SENSOR_1, INPUT_PULLUP);
    pinMode(WIND_SENSOR_2, INPUT_PULLUP);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    // Buzzer
    buzzerSetup();
    beep(2000, 60);

    // Wind ISRs (dual sensor)
    attachInterrupt(digitalPinToInterrupt(WIND_SENSOR_1), wind1ISR, FALLING);
    attachInterrupt(digitalPinToInterrupt(WIND_SENSOR_2), wind2ISR, FALLING);

    // TFT init
    tft.init();
    tft.setRotation(0);  // portrait 240×320
    tft.fillScreen(C_BG);
    tft.setSwapBytes(true);

    // Sensors
    initSensors();

    // Boot splash
    drawBootScreen();

    // Timers
    tripStartMs = millis();
    lastStopMs = millis();
    lastWindMs = millis();
    lastInteractMs = millis();

    // Initial reads
    readSensors();

    // Initialize altitude buffer with current reading
    for (int i = 0; i < 6; i++) {
        altBuf[i] = S.altitude;
    }

    // Set reasonable initial values for derived quantities
    S.energyEff = 8.1f;  // median from model
    S.distM = 0.0f;
    S.energyWh = 0.0f;
    S.rangeConsumed = 0.0f;
    S.altChange30s = 0.0f;
    S.timeSinceStop = 0.0f;

    runPrediction();
    dispRange = S.predRange;
    dispSOC = S.soc;

    // Debug: Print initial sensor values and model inputs
    Serial.println("\n=== INITIAL SENSOR CHECK ===");
    Serial.printf("SOC: %.1f%% | Voltage: %.2fV\n", S.soc, S.busV);
    Serial.printf("Altitude: %.1fm | Pressure: %.1f hPa\n", S.altitude,
                  S.pressure);
    Serial.printf("LDR: %d | Rain: %d\n", S.ldr, S.rain);
    Serial.printf("Temperature: %.1f°C | Humidity: %.1f%%\n",
                  hasHDC ? S.hdcTemp : S.bmpTemp, S.humidity);
    Serial.printf("Energy Eff: %.2f Wh/km | Roll Res: %.4f\n", S.energyEff,
                  S.rollRes);
    Serial.printf("Predicted Range: %.1f km | Fallbacks: %d\n", S.predRange,
                  S.replaced);
    Serial.println("============================\n");

    screenDirty = true;
}

// ================================================================
//  LOOP
// ================================================================

void loop() {
    unsigned long now = millis();

    // --- Button ---
    handleButton();

    // --- Sensor read ---
    if (now - lastSensorMs >= SENSOR_INTERVAL) {
        float dt = (now - lastSensorMs) / 1000.0f;
        lastSensorMs = now;
        readSensors();
        updateCumulatives(dt);
        valuesDirty = true;
    }

    // --- ML prediction ---
    if (now - lastPredictMs >= PREDICT_INTERVAL) {
        lastPredictMs = now;
        runPrediction();
        Serial.printf("[ML] Range: %.1f km  Infer: %lu us  Fb: %d\n",
                      S.predRange, S.inferUs, S.replaced);
    }

    // --- Alerts ---
    checkAlerts();

    // --- Screen update ---
    if (now - lastScreenMs >= SCREEN_INTERVAL) {
        lastScreenMs = now;

        bool full = screenDirty;
        screenDirty = false;

        switch (currentScreen) {
            case 0:
                drawScreen1(full);
                break;
            case 1:
                drawScreen2(full);
                break;
            case 2:
                drawScreen3(full);
                break;
            case 3:
                drawScreen4(full);
                break;
            case 4:
                drawScreen5(full);
                break;
        }
    }
}
