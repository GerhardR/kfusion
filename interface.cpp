#if 1

#include <Windows.h>
#include <NuiApi.h>

#include <iostream>
#include <stdint.h>

using namespace std;

HANDLE        m_hNextDepthFrameEvent;
HANDLE        m_hNextVideoFrameEvent;
HANDLE        m_pDepthStreamHandle;
HANDLE        m_pVideoStreamHandle;

INuiSensor * m_pSensor;
INuiCoordinateMapper * m_pMapping;

// thread handling
HANDLE        m_hThNuiProcess;
HANDLE        m_hEvNuiProcessStop;

bool gotDepth;
int depth_index;

uint16_t * buffers[2];
unsigned char * rgb;

DWORD WINAPI run(LPVOID pParam)
{
    HANDLE hEvents[3];
    int	nEventIdx;

    // Configure events to be listened on
    hEvents[0]=m_hEvNuiProcessStop;
    hEvents[1]=m_hNextDepthFrameEvent;
    hEvents[2]=m_hNextVideoFrameEvent;
	
	NUI_IMAGE_FRAME pImageFrame;
	NUI_LOCKED_RECT LockedRect;

    // Main thread loop
    while(1)
    {
        // Wait for an event to be signalled
        nEventIdx=WaitForMultipleObjects(sizeof(hEvents)/sizeof(hEvents[0]),hEvents,FALSE,100);

        // If the stop event, stop looping and exit
        if(nEventIdx==0)
            break;            

        // Process signal events
        switch(nEventIdx)
        {
            case 1: {
				depth_index = (depth_index+1) % 2;
				HRESULT hr =  m_pSensor->NuiImageStreamGetNextFrame(m_pDepthStreamHandle, 0, &pImageFrame );

				if( S_OK == hr ){
					pImageFrame.pFrameTexture->LockRect( 0, &LockedRect, NULL, 0 );
					if( LockedRect.Pitch != 0 ) {
						uint16_t * pBuffer = (uint16_t*) LockedRect.pBits;
						std::copy(pBuffer, pBuffer + 640*480, buffers[depth_index]);
					} else {
						cout << "Buffer length of received texture is bogus\r\n" << endl;
					}
					// cout << "Depthframe \t" << pImageFrame->dwFrameNumber << endl;
					m_pSensor->NuiImageStreamReleaseFrame( m_pDepthStreamHandle, &pImageFrame );	
					gotDepth = true;
				}
			} break;

            case 2: {
				HRESULT hr =  m_pSensor->NuiImageStreamGetNextFrame( m_pVideoStreamHandle, 0, &pImageFrame );
				if( S_OK == hr ){
					pImageFrame.pFrameTexture->LockRect( 0, &LockedRect, NULL, 0 );
					if( LockedRect.Pitch != 0 ) {
						unsigned char * dest = rgb;
						unsigned char * pBuffer = (unsigned char *) LockedRect.pBits;
						for(int i = 0; i < 640*480; ++i, dest+=3, pBuffer +=4){
							dest[0] = pBuffer[0];
							dest[1] = pBuffer[1];
							dest[2] = pBuffer[2];
						}
					} else {
						cout << "Buffer length of received texture is bogus\r\n" << endl;
					}
					// cout << "Rgbframe \t" << pImageFrame->dwFrameNumber << endl;
					m_pSensor->NuiImageStreamReleaseFrame( m_pVideoStreamHandle, &pImageFrame );
				} 
			} break;
        }
    }

    return (0);
}

int InitKinect( uint16_t * depth_buffer[2], unsigned char * rgb_buffer ){
	buffers[0] = depth_buffer[0];
	buffers[1] = depth_buffer[1];
	rgb = rgb_buffer;
	depth_index = 0;
	gotDepth = false;

	HRESULT hr;

    m_hNextDepthFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );
    m_hNextVideoFrameEvent = CreateEvent( NULL, TRUE, FALSE, NULL );

	hr = NuiCreateSensorByIndex( 0, &m_pSensor );
    if( FAILED( hr ) ){
        cout << "Kinect3DDevice: Could not open Kinect Device" << endl;
        return 1;
    }

    hr =  m_pSensor->NuiInitialize( NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH );
    
	hr =  m_pSensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_COLOR,
        NUI_IMAGE_RESOLUTION_640x480,
        0,
        2,
        m_hNextVideoFrameEvent,
        &m_pVideoStreamHandle );

     hr =  m_pSensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_DEPTH,
        NUI_IMAGE_RESOLUTION_640x480,
        0,
        2,
        m_hNextDepthFrameEvent,
        &m_pDepthStreamHandle );

	hr =  m_pSensor->NuiGetCoordinateMapper(&m_pMapping);

    // Start the Nui processing thread
    m_hEvNuiProcessStop=CreateEvent(NULL,FALSE,FALSE,NULL);
    m_hThNuiProcess=CreateThread(NULL,0,run,NULL,0,NULL); 

	return 0;
}

bool KinectFrameAvailable(){
	bool result = gotDepth;
	gotDepth = false;
	return result;
}

int GetKinectFrame(){
	return depth_index;
}

void CloseKinect(){
	// Stop the Nui processing thread
    if(m_hEvNuiProcessStop!=INVALID_HANDLE_VALUE)
    {
        // Signal the thread
        SetEvent(m_hEvNuiProcessStop);

        // Wait for thread to stop
        if(m_hThNuiProcess!=INVALID_HANDLE_VALUE)
        {
            WaitForSingleObject(m_hThNuiProcess,INFINITE);
            CloseHandle(m_hThNuiProcess);
        }
        CloseHandle(m_hEvNuiProcessStop);
    }

     m_pSensor->NuiShutdown( );
    if( m_hNextDepthFrameEvent && ( m_hNextDepthFrameEvent != INVALID_HANDLE_VALUE ) )
    {
        CloseHandle( m_hNextDepthFrameEvent );
        m_hNextDepthFrameEvent = NULL;
    }
    if( m_hNextVideoFrameEvent && ( m_hNextVideoFrameEvent != INVALID_HANDLE_VALUE ) )
    {
        CloseHandle( m_hNextVideoFrameEvent );
        m_hNextVideoFrameEvent = NULL;
    } 
}

#else

#include <libfreenect.h>

#include <pthread.h>

freenect_context *f_ctx;
freenect_device *f_dev;
bool gotDepth;
int depth_index;

pthread_t freenect_thread;
volatile bool die = false;

uint16_t * buffers[2];

void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
    gotDepth = true;
    depth_index = (depth_index+1) % 2;
    freenect_set_depth_buffer(dev, buffers[depth_index]);
}

void *freenect_threadfunc(void *arg)
{
    while(!die){
        int res = freenect_process_events(f_ctx);
        if (res < 0 && res != -10) {
            cout << "\nError "<< res << " received from libusb - aborting.\n";
            break;
        }
    }
    freenect_stop_depth(f_dev);
    freenect_stop_video(f_dev);
    freenect_close_device(f_dev);
    freenect_shutdown(f_ctx);
}

int InitKinect( uint16_t * depth_buffer[2], void * rgb_buffer ){
    if (freenect_init(&f_ctx, NULL) < 0) {
        cout << "freenect_init() failed" << endl;
        return 1;
    }

    freenect_set_log_level(f_ctx, FREENECT_LOG_WARNING);
    freenect_select_subdevices(f_ctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int nr_devices = freenect_num_devices (f_ctx);
    cout << "Number of devices found: " << nr_devices << endl;

    if (nr_devices < 1)
        return 1;

    if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
        cout << "Could not open device" << endl;
        return 1;
    }

    depth_index = 0;
    buffers[0] = depth_buffer[0];
    buffers[1] = depth_buffer[1];
    freenect_set_depth_callback(f_dev, depth_cb);
    freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED));
    freenect_set_depth_buffer(f_dev, buffers[depth_index]);

    freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
    freenect_set_video_buffer(f_dev, rgb_buffer);

    freenect_start_depth(f_dev);
    freenect_start_video(f_dev);

    gotDepth = false;

    int res = pthread_create(&freenect_thread, NULL, freenect_threadfunc, NULL);
    if(res){
        cout << "error starting kinect thread " << res << endl;
        return 1;
    }

    return 0;
}

void CloseKinect(){
    die = true;
    pthread_join(freenect_thread, NULL);
}

#endif
