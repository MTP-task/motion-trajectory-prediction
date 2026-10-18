#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"

MPU6050 mpuA(0x68), mpuB(0x69);

struct MyData {
  int16_t ax, ay, az;
  int16_t gx, gy, gz;
};

MyData A, B;

// --- Gyro range ---
const uint8_t GYRO_RANGE = MPU6050_GYRO_FS_500;  // ±500 dps
float gyroLSBperDPS = 65.5f;                     // for ±500 dps

// --- Accel range (NEW) ---
const uint8_t ACCEL_RANGE = MPU6050_ACCEL_FS_2;  // ±2g
float accelLSBperG = 16384.0f;                   // for ±2g

// gyro bias
float A_gx_bias=0, A_gy_bias=0, A_gz_bias=0;
float B_gx_bias=0, B_gy_bias=0, B_gz_bias=0;

// ---- 100 Hz timing ----
const uint32_t INTERVAL_US = 10000;   // 1/100 s 100 Hz
uint32_t nextTickUs;

void setRanges(MPU6050& mpu, uint8_t gyroRange, uint8_t accelRange){
  mpu.setFullScaleGyroRange(gyroRange);
  mpu.setFullScaleAccelRange(accelRange);

  // DLPF + sample rate
  mpu.setDLPFMode(3);        // ~44 Hz
  mpu.setRate(4);            // 1kHz/(1+4)=200 Hz internal update
}

void calibrateGyro(MPU6050& mpu, float& bx, float& by, float& bz, int samples=500){
  long sx=0, sy=0, sz=0;
  int16_t ax,ay,az,gx,gy,gz;
  delay(200);
  for(int i=0;i<samples;i++){
    mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);
    sx += gx; sy += gy; sz += gz;
    delay(2);
  }
  bx = (float)sx / samples;
  by = (float)sy / samples;
  bz = (float)sz / samples;
}

void setup(){
  Serial.begin(115200);
  Wire.begin();

  mpuA.initialize();
  mpuB.initialize();

  setRanges(mpuA, GYRO_RANGE, ACCEL_RANGE);
  setRanges(mpuB, GYRO_RANGE, ACCEL_RANGE);

  // Header
  Serial.println("Aax_g,Aay_g,Aaz_g,Bax_g,Bay_g,Baz_g,Agx_dps,Agy_dps,Agz_dps,Bgx_dps,Bgy_dps,Bgz_dps");

  // Calibration (keep still)
  Serial.println("Start Calibration");
  calibrateGyro(mpuA, A_gx_bias, A_gy_bias, A_gz_bias);
  calibrateGyro(mpuB, B_gx_bias, B_gy_bias, B_gz_bias);
  Serial.println("Finish Calibration");

  nextTickUs = micros() + INTERVAL_US;
}

void loop(){
  uint32_t now = micros();
  if ((int32_t)(now - nextTickUs) >= 0) {

    nextTickUs += INTERVAL_US;

    // --- Read ---
    mpuA.getMotion6(&A.ax,&A.ay,&A.az,&A.gx,&A.gy,&A.gz);
    mpuB.getMotion6(&B.ax,&B.ay,&B.az,&B.gx,&B.gy,&B.gz);

    // --- Accel raw -> g (NEW) ---
    float Aax_g = (float)A.ax / accelLSBperG;
    float Aay_g = -(float)A.ay / accelLSBperG;
    float Aaz_g = -(float)A.az / accelLSBperG;

    float Bax_g = (float)B.ax / accelLSBperG;
    float Bay_g = -(float)B.ay / accelLSBperG;
    float Baz_g = (float)B.az / accelLSBperG;

    // --- Gyro raw -> dps (remove bias) ---
    float Agx_dps = ((float)A.gx - A_gx_bias) / gyroLSBperDPS;
    float Agy_dps = ((float)A.gy - A_gy_bias) / gyroLSBperDPS;
    float Agz_dps = ((float)A.gz - A_gz_bias) / gyroLSBperDPS;

    float Bgx_dps = ((float)B.gx - B_gx_bias) / gyroLSBperDPS;
    float Bgy_dps = ((float)B.gy - B_gy_bias) / gyroLSBperDPS;
    float Bgz_dps = ((float)B.gz - B_gz_bias) / gyroLSBperDPS;

    // --- Output ---
    // If you want axis sign flips, do it here.
    // (I kept your gyro flips + left accel as raw g for now.)

    Serial.print(Aax_g, 4); Serial.print(",");
    Serial.print(Aay_g, 4); Serial.print(",");
    Serial.print(Aaz_g, 4); Serial.print(",");

    Serial.print(Bax_g, 4); Serial.print(",");
    Serial.print(Bay_g, 4); Serial.print(",");
    Serial.print(Baz_g, 4); Serial.print(",");

    Serial.print(Agx_dps, 3); Serial.print(",");
    Serial.print(-Agy_dps, 3); Serial.print(",");
    Serial.print(-Agz_dps, 3); Serial.print(",");

    Serial.print(Bgx_dps, 3); Serial.print(",");
    Serial.print(-Bgz_dps, 3); Serial.print(",");
    Serial.println(Bgy_dps, 3);

    // Overrun guard
    now = micros();
    if ((int32_t)(now - nextTickUs) > (int32_t)(INTERVAL_US * 4)) {
      nextTickUs = now + INTERVAL_US;
    }
  }
}
