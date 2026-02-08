# 🚴‍♂️ TI Smart Range Estimator

**AI-Powered EV Range Prediction Dashboard for Electric Vehicles**

A professional TFT dashboard system for electric scooters and e-bikes that uses Machine Learning (Random Forest) to predict remaining range based on real-time sensor data - battery state, environmental conditions, terrain, and riding dynamics.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![ESP32](https://img.shields.io/badge/Platform-ESP32-blue) ![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)

---

## 🎯 Features

### 📊 Smart ML Prediction

- **Random Forest Regressor** (50 trees, 11 depth, 12 features)
- Real-time range estimation with sensor validation
- Handles sensor failures with median fallbacks
- Inference time: ~200-400µs per prediction

### 📱 Professional Dashboard (240×320 TFT)

- **5 Interactive Screens:**
  1. **Main Range** - Gradient arc gauge with SOC bar
  2. **Environment** - Weather, atmosphere, wind, load
  3. **Battery** - Voltage, current, power, temperature
  4. **Dynamics** - Throttle, acceleration, tilt, speed
  5. **Trip History** - Energy efficiency & range consumption

- **Dark Theme UI** with electric cyan accents
- Smooth animations and color-coded alerts
- Single-button navigation (short press = next, long press = home)

### 🔌 Multi-Sensor Integration

- **Power Monitoring:** INA219 (voltage, current, power)
- **Environmental:** BMP280 (altitude, pressure), HDC1080 (temp, humidity)
- **Motion:** MPU6050 (3-axis accel/gyro, tilt)
- **Analog:** LDR (light), rain sensor, NTC (battery temp), throttle hall
- **Advanced:** ADS1232 24-bit ADC (load cell), dual hall wind sensors

---

## 🛠️ Hardware Requirements

### Core Components

| Component           | Model                  | Purpose                 |
| ------------------- | ---------------------- | ----------------------- |
| **Microcontroller** | ESP32 DevKit           | Main processor          |
| **Display**         | ILI9341/ST7789 240×320 | TFT dashboard           |
| **Power Monitor**   | INA219 (0x41)          | Voltage/current sensing |
| **Altitude Sensor** | BMP280 (0x76)          | Barometric altitude     |
| **IMU**             | MPU6050 (0x68)         | Acceleration & tilt     |
| **Climate Sensor**  | HDC1080 (0x40)         | Temperature & humidity  |
| **Load Cell ADC**   | ADS1232 24-bit         | Weight measurement      |

### Pin Configuration

```cpp
// I²C Bus (SDA=21, SCL=22)
INA219:  0x41  | BMP280:  0x76  | MPU6050: 0x68  | HDC1080: 0x40

// Analog Inputs
LDR:         GPIO39 | Rain Sensor: GPIO35
NTC 10K:     GPIO34 | Throttle:    GPIO36

// Digital Sensors
Wind Hall 1: GPIO26 | Wind Hall 2: GPIO25 (90° separation)
Button:      GPIO12 | Buzzer:      GPIO5

// ADS1232 (Load Cell)
DOUT: GPIO14 | SCLK: GPIO13 | PDWN: GPIO27 | SPEED → GND
```

---

## 🚀 Quick Start

### 1. Hardware Setup

1. Connect all sensors according to pin configuration
2. Wire TFT display via SPI (configure in `User_Setup.h` of TFT_eSPI library)
3. Calibrate load cell (see instructions below)
4. Connect to 13S Li-ion battery pack (39V-54.6V)

### 2. Software Installation

#### Arduino Libraries Required:

```bash
# Install via Arduino Library Manager:
- TFT_eSPI
- Adafruit_BMP280
- Adafruit_MPU6050
- Adafruit_INA219
- ClosedCube_HDC1080
- Wire
```

#### TFT_eSPI Configuration:

Edit `User_Setup.h` in TFT_eSPI library folder:

```cpp
#define ILI9341_DRIVER  // or ST7789_DRIVER
#define TFT_WIDTH  240
#define TFT_HEIGHT 320
#define TFT_MOSI 23
#define TFT_SCLK 18
#define TFT_CS   15
#define TFT_DC    2
#define TFT_RST   4
#define SPI_FREQUENCY 40000000
```

### 3. Upload Firmware

1. Open `ev_range_dashboard/ev_range_dashboard.ino` in Arduino IDE
2. Select **Board:** ESP32 Dev Module
3. Select **Port:** Your ESP32 COM port
4. Click **Upload** ⬆️
5. Open Serial Monitor (115200 baud) for debug output

---

## 🧠 ML Model Details

### Feature Inputs (12 Features)

| #   | Feature                    | Source                   | Range         |
| --- | -------------------------- | ------------------------ | ------------- |
| 0   | `soc_percent`              | INA219 voltage mapping   | 0-100%        |
| 1   | `energy_efficiency_whkm`   | INA219 power integration | 0.1-50 Wh/km  |
| 2   | `precipitation_mm`         | Rain sensor lookup table | 0-20 mm/h     |
| 3   | `altitude_bmp280_m`        | BMP280 barometric        | -500 to 3000m |
| 4   | `rain_sensor_value`        | GPIO35 ADC               | 0-4095        |
| 5   | `ldr_value`                | GPIO39 ADC               | 0-4095        |
| 6   | `rolling_resistance_coeff` | Rain-based selection     | 0.004-0.020   |
| 7   | `energy_consumed_wh`       | Cumulative INA219        | 0+ Wh         |
| 8   | `altitude_change_30s_m`    | BMP280 ring buffer       | -200 to +200m |
| 9   | `distance_m`               | Speed integration        | 0+ meters     |
| 10  | `range_consumed_km`        | Distance / 1000          | 0+ km         |
| 11  | `time_since_last_stop_s`   | Timer                    | 0+ seconds    |

### Model Performance

- **Inference Time:** 200-400 microseconds
- **Model Size:** ~115KB (fits in ESP32 flash)
- **Validation:** Built-in sensor bounds checking and fallbacks
- **Safety:** Handles NaN, invalid readings, sensor failures

---

## ⚙️ Calibration

### Load Cell (ADS1232)

1. Place empty platform → record tare value:

   ```cpp
   #define ADS_TARE_VALUE 39172L  // Update this
   ```

2. Place known weight (e.g., 10kg) → calculate scale factor:
   ```cpp
   scale = (reading - tare) / weight_kg
   #define ADS_SCALE_FACTOR 17667.50f  // Update this
   ```

### Battery Pack

Adjust for your battery configuration:

```cpp
#define BATT_FULL_V 54.6f     // 13S at 4.2V/cell
#define BATT_EMPTY_V 39.0f    // 13S at 3.0V/cell
#define BATT_CAPACITY_WH 480.0f  // Total capacity
```

### Throttle ADC Range

Calibrate min/max throttle values:

```cpp
#define THROT_ADC_MIN 490    // No throttle
#define THROT_ADC_MAX 3500   // Full throttle
```

---

## 📁 Project Structure

```
TI-Smart-Range-Estimator/
├── ev_range_dashboard/
│   ├── ev_range_dashboard.ino      # Main firmware (1490 lines)
│   └── ev_range_model-new-v4.h     # ML model (50 trees, 115KB)
│
├── scripts/
│   ├── Data-gen.py                 # Training dataset generator
│   └── model-gen.py                # ML model training & export
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🎨 Display Screens

### Screen 1: Main Range

- **Gradient arc gauge** (red → amber → green → cyan)
- Large 7-segment range display
- SOC progress bar with color coding
- Quick info cards: Speed, Power, Temp, Wind, Altitude

### Screen 2: Environment

- Weather status (Sunny/Cloudy/Rainy/Night)
- Humidity, LDR, Rain sensor readings
- Atmospheric pressure & altitude change
- Wind speed/direction (tailwind/headwind/crosswind)
- Load cell weight display

### Screen 3: Battery

- SOC ring gauge (0-100%)
- Voltage, current, power cards
- Battery temperature with thermal alerts
- Energy consumption progress bar

### Screen 4: Dynamics

- Throttle progress bar
- 3-axis acceleration bubble level
- G-force readings (X, Y, Z)
- Tilt angles (pitch & roll)
- Speed & distance cards

### Screen 5: Trip History

- Energy efficiency trend graph (30 points)
- Range consumption trend graph (30 points)
- Trip statistics

---

## 🔍 Debugging

### Serial Monitor Output (115200 baud)

```
========================================
  EV RANGE DASHBOARD — BOOT
========================================
[INIT] SDA=21 SCL=22 @ 400 kHz
[INIT] INA219  Pwr  OK
[INIT] BMP280  Alt  OK
[INIT] MPU6050 IMU  OK
[INIT] HDC1080 T/H  OK
[INIT] ADS1232 Wgt  OK

[BOOT] Initial Readings:
  SOC: 75.2%  Voltage: 51.4V
  Altitude: 142.3m  Pressure: 1013.8 hPa
  LDR: 447  Rain: 162
  Temp: 25.3°C  Humidity: 45%
  Energy Eff: 8.1 Wh/km
  Predicted Range: 21.9 km

[ML] Range: 21.9 km  Infer: 287 us  Fb: 0
```

### Common Issues

| Problem         | Solution                                  |
| --------------- | ----------------------------------------- |
| White screen    | Check TFT wiring & `User_Setup.h` config  |
| Red sensor dots | Verify I²C connections & addresses        |
| Range = 0       | Check INA219 connection & battery voltage |
| High `Fb` count | Sensor out of range, check wiring         |

---

## 🧪 ML Model Training

### Dataset Generation (`scripts/Data-gen.py`)

- Physics-based synthetic data generation
- Includes realistic noise and variations
- Covers diverse riding scenarios (urban, highway, hills)

### Model Training (`scripts/model-gen.py`)

```python
# Trains Random Forest on generated dataset
# Exports to ESP32-compatible C header file using emlearn
python scripts/model-gen.py
```

**Model Output:** `ev_range_model-new-v4.h`

---

## 📊 Model Features Explained

### Calculated Features

```cpp
// SOC from voltage
S.soc = (voltage - 39.0V) / (54.6V - 39.0V) × 100%

// Energy efficiency
S.energyEff = energyConsumed (Wh) / distance (km)

// Precipitation estimate
S.precipMm = rainADC_to_mmPerHour(S.rain)

// Rolling resistance coefficient
S.rollRes = (rain < 1500) ? 0.013 : 0.008  // wet : dry

// Altitude change (30s window)
S.altChange30s = altitude_now - altitude_30s_ago

// Speed estimation from throttle
S.speed = throttle% × 0.35 km/h  // max ~35 km/h
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add GPS for real speed/distance measurement
- [ ] Implement CAN bus for direct motor controller integration
- [ ] Add Kalman filter for altitude smoothing
- [ ] Support for more battery chemistries (LiFePO4, etc.)
- [ ] Mobile app for data logging (BLE/WiFi)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 👨‍💻 Author

**Samiul Islam**  
📧 Contact: [GitHub Issues](https://github.com/samiul1221/TI-Smart-Range-Estimator/issues)

---

## 🙏 Acknowledgments

- **TFT_eSPI** by Bodmer - Excellent TFT library
- **emlearn** - ML model conversion to C
- **Adafruit** - Sensor libraries
- Inspired by Tesla's range prediction algorithm

---

## 📸 Screenshots

_Upload dashboard photos to `docs/images/` and link here_

---

## ⚠️ Safety Notice

This is an **experimental project** for educational purposes. Always:

- Monitor battery voltage manually
- Don't rely solely on predicted range for critical decisions
- Ensure proper insulation of high-voltage components
- Follow electrical safety guidelines

---

**Star ⭐ this repo if you find it useful!**
