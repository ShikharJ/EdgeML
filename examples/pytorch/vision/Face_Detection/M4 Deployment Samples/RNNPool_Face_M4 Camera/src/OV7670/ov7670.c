#include <stdio.h>
#include "main.h"
#include "stm32f4xx_hal.h"
#include "ov7670.h"
#include "reg.h"
#include "diag/Trace.h"

/*** Internal Static Variables ***/
static DCMI_HandleTypeDef *sp_hdcmi;
static DMA_HandleTypeDef *sp_hdma_dcmi;
static I2C_HandleTypeDef *sp_hi2c;
static void (*s_cbHsync)(uint32_t h);
static void (*s_cbVsync)(uint32_t v);
static uint32_t s_currentH;
static uint32_t s_currentV;

/*** Internal Function Declarations ***/
static RET camera_write(uint8_t register_address, uint8_t data);
static RET camera_read(uint8_t register_address, uint8_t *data);

/*** External Function Defines ***/
RET camera_init(DCMI_HandleTypeDef *p_hdcmi, DMA_HandleTypeDef *p_hdma_dcmi,
                I2C_HandleTypeDef *p_hi2c) {
  sp_hdcmi = p_hdcmi;
  sp_hdma_dcmi = p_hdma_dcmi;
  sp_hi2c = p_hi2c;

  HAL_GPIO_WritePin(DCMI_Reset_GPIO_Port, DCMI_Reset_Pin, GPIO_PIN_RESET);
  HAL_Delay(50);
  HAL_GPIO_WritePin(DCMI_Reset_GPIO_Port, DCMI_Reset_Pin, GPIO_PIN_SET);
  HAL_Delay(50);

  // SCCB Register Reset
  if (camera_write(0x12, 0x80) != RET_OK) {
    return RET_ERR;
  }
  HAL_Delay(30);

  uint8_t buffer[4];
  // VER Register (Product ID LSB) Read
  if (camera_read(0x0b, buffer) != RET_OK) {
    return RET_ERR;
  }
  trace_printf("[OV7670] Dev ID = %02X\n", buffer[0]);

  return RET_OK;
}

RET camera_config() {
  // Stop Frame Capture
  if (camera_stop_capture() != RET_OK) {
    return RET_ERR;
  }
  // SCCB Register Reset
  if (camera_write(0x12, 0x80) != RET_OK) {
    return RET_ERR;
  }
  HAL_Delay(30);

  // Update Camera Register Configurations
  for (unsigned i = 0; OV7670REG[i][0] != REG_BATT; i++) {
    /*uint8_t buffer;
    if (camera_read(OV7670REG[i][0], &buffer) != RET_OK) {
      HAL_Delay(1);
      i--;
      continue;
    }*/
    if (camera_write(OV7670REG[i][0], OV7670REG[i][1]) != RET_OK) {
      HAL_Delay(1);
      i--;
      continue;
    }
    /*if (camera_read(OV7670REG[i][0], &buffer) != RET_OK) {
      HAL_Delay(1);
      i--;
      continue;
    }
    trace_printf("Read Register: %02X, Value: %02X\n", OV7670REG[i][0], buffer);*/
    HAL_Delay(1);
  }

  return RET_OK;
}

RET camera_start_capture(uint32_t destination_address) {
  if (camera_stop_capture() != RET_OK) {
    return RET_ERR;
  }

  if (HAL_DCMI_Start_DMA(sp_hdcmi, DCMI_MODE_SNAPSHOT, destination_address, INPUT_IMG_WIDTH * INPUT_IMG_HEIGHT / 2) != HAL_OK) {
    return RET_ERR;
  }

  return RET_OK;
}

RET camera_stop_capture() {
  if (HAL_DCMI_Stop(sp_hdcmi) != HAL_OK) {
    return RET_ERR;    
  }

  return RET_OK;
}

void camera_register_callback(void (*cbHsync)(uint32_t h),
                              void (*cbVsync)(uint32_t v)) {
  s_cbHsync = cbHsync;
  s_cbVsync = cbVsync;
}

void HAL_DCMI_FrameEventCallback(DCMI_HandleTypeDef *hdcmi) {
  if (s_cbVsync) {
    s_cbVsync(s_currentV);
  }

  s_currentV++;
  s_currentH = 0;
}

/*** Internal Function Defines ***/
static RET camera_write(uint8_t register_address, uint8_t data) {
  HAL_StatusTypeDef ret;

  do {
    ret = HAL_I2C_Mem_Write(sp_hi2c, SLAVE_ADDR, register_address,
                            I2C_MEMADD_SIZE_8BIT, &data, 1, 100);
  } while (ret != HAL_OK && 0);

  if (ret == HAL_OK) {
	  return RET_OK;
  }
  return RET_ERR;
}

static RET camera_read(uint8_t register_address, uint8_t *data) {
  HAL_StatusTypeDef ret;

  do {
    ret = HAL_I2C_Master_Transmit(sp_hi2c, SLAVE_ADDR, &register_address, 1, 100);
    ret |= HAL_I2C_Master_Receive(sp_hi2c, SLAVE_ADDR, data, 1, 100);
  } while (ret != HAL_OK && 0);

  if (ret == HAL_OK) {
  	return RET_OK;
  }
  return RET_ERR;
}
