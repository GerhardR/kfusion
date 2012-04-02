#ifndef GLPROXY_H
#define GLPROXY_H

#include <cstdio>

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN 1
#include <Windows.h>
#include <GL/GL.h>

// well, Win32 still just come with GL 1.1 headers
// so either we throw in GLEW or quickly do this...

typedef signed long  int khronos_ssize_t;
typedef khronos_ssize_t  GLsizeiptr;

typedef void (APIENTRY *glDeleteBuffers_f)(GLsizei,const GLuint*);
typedef void (APIENTRY *glGenBuffers_f)(GLsizei,GLuint*);
typedef void (APIENTRY *glBindBuffer_f)(GLenum,GLuint);
typedef void (APIENTRY *glBufferData_f)(GLenum , GLsizeiptr , const GLvoid* , GLenum );

#define GL_PIXEL_PACK_BUFFER 0x88EB
#define GL_PIXEL_UNPACK_BUFFER 0x88EC

#define GL_STREAM_DRAW                    0x88E0
#define GL_STATIC_DRAW                    0x88E4
#define GL_DYNAMIC_DRAW                   0x88E8

static glDeleteBuffers_f glDeleteBuffers;
static glGenBuffers_f glGenBuffers;
static glBindBuffer_f glBindBuffer;
static glBufferData_f glBufferData;

class GLProxy {
public:
	GLProxy()
	{
	    glDeleteBuffers = (glDeleteBuffers_f)wglGetProcAddress("glDeleteBuffersARB");
	    glGenBuffers = (glGenBuffers_f)wglGetProcAddress("glGenBuffersARB");
	    glBindBuffer = (glBindBuffer_f)wglGetProcAddress("glBindBufferARB");
	    glBufferData = (glBufferData_f)wglGetProcAddress("glBufferDataARB");
	}
}

// initialize 
static GLProxy gs_glproxy = GLProxy();

#endif


#endif
