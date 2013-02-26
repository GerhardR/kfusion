#ifndef INTERFACE_H
#define INTERFACE_H

int InitKinect( uint16_t * depth_buffer[2], unsigned char * rgb_buffer );
bool KinectFrameAvailable();
int GetKinectFrame();
void CloseKinect();

#endif
