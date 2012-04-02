#ifndef GLPROXY_H
#define GLPROXY_H

#if !defined(__APPLE__)
// we are happy campers
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN 1
#include <Windows.h>
#include <GL/GL.h>
#elif defined(__linux)
    #include <GL/gl.h>
    #include <GL/glx.h>
#endif

// well, Win32 still just come with GL 1.1 headers
// so either we throw in GLEW or quickly do this...

typedef signed   long  int     khronos_ssize_t;
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


inline void*
getGLProcAddress(const char* name)
{
    return
#if defined(_WIN32)
    wglGetProcAddress(name);
#else
    (void*)glXGetProcAddress((const GLubyte*)name);
#endif
}


inline static
bool OpenGLRuntimeLinking()
{
    glDeleteBuffers = (glDeleteBuffers_f)getGLProcAddress("glDeleteBuffersARB");
    glGenBuffers = (glGenBuffers_f)getGLProcAddress("glGenBuffersARB");
    glBindBuffer = (glBindBuffer_f)getGLProcAddress("glBindBufferARB");
    glBufferData = (glBufferData_f)getGLProcAddress("glBufferDataARB");

    return (glDeleteBuffers && glGenBuffers && glBindBuffer && glBufferData);
}


#else

inline static
bool OpenGLRuntimeLinking() { return true; }

#endif


#endif
