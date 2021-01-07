#ifndef OV7670_OV7670_H_
#define OV7670_OV7670_H_

#include "common.h"

#define SLAVE_ADDR 0x42

RET camera_init(DCMI_HandleTypeDef *p_hdcmi, DMA_HandleTypeDef *p_hdma_dcmi,
                I2C_HandleTypeDef *p_hi2c);
RET camera_config();
RET camera_start_capture(uint32_t destAddress);
RET camera_stop_capture();
void camera_register_callback(void (*cbHsync)(uint32_t h),
                              void (*cbVsync)(uint32_t v));
#endif /* OV7670_OV7670_H_ */
