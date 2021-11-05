/*
Copyright 2012-2021 Fastvideo, Ltd. All rights reserved.

Please refer to the Fastvideo Standard License Agreement (SLA), associated
 with this source code for terms and conditions that govern your use of
 this software. Any use, reproduction, disclosure, or distribution of
 this software and related documentation outside the terms of the SLA is strictly prohibited.

IN NO EVENT SHALL FASTVIDEO BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF FASTVIDEO HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FASTVIDEO SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". FASTVIDEO HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/
#ifndef __RUN_CAMERA__
#define __RUN_CAMERA__

#include <thread>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "fastvideo_sdk.h"

#include "supported_files.hpp"
#include "EnumToStringSdk.h"
#include "helper_pfm.hpp"
#include "helper_lut.hpp"
#include "PreloadUncompressedImage.hpp"

#include "Image.h"

#include "DebayerFfmpeg.hpp"
#include "CameraSampleOptions.h"

#include "FastAllocator.h"

#include "DecodeError.hpp"

static GLuint pbo_buffer = 0;
static GLuint texid = 0;
static struct cudaGraphicsResource *cuda_pbo_resource;

static unsigned imageWidth;
static unsigned imageHeight;

static volatile bool closeWorkerThread;
static volatile bool workerThreadClosed;
static volatile bool renderAction;
static volatile bool applicationClosing;

static DebayerFfmpeg *hDebayerFfmpeg;

static void timerEvent(int value) {
	if (!workerThreadClosed) {
		glutPostRedisplay();
	} else if (!applicationClosing) {
		while (renderAction);
		cudaError_t error = cudaGraphicsUnregisterResource(cuda_pbo_resource);
		if (error != cudaSuccess) {
			fprintf(stderr, "Cannot unregister CUDA Graphic Resource: %s\n", cudaGetErrorString(error));
		}

		glutLeaveMainLoop();
	}
}

static void display(void) {
	unsigned char *data = NULL;
	size_t pboBufferSize = 0;

	cudaError_t error = cudaSuccess;

	renderAction = true;

	if ((error = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0)) != cudaSuccess ||
		(error = cudaGraphicsResourceGetMappedPointer((void **)&data, &pboBufferSize, cuda_pbo_resource)) != cudaSuccess ||
		pboBufferSize < (imageWidth * imageHeight * 3 * sizeof(unsigned char)) ||
		(error = cudaMemcpy(data, hDebayerFfmpeg->GetDevicePtr(), imageWidth * imageHeight * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice)) != cudaSuccess ||
		(error = cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0)) != cudaSuccess) {
		fprintf(stderr, "CUDA Graphics resource failed: %s\n", cudaGetErrorString(error));
		glutLeaveMainLoop();
		return;
	}

	{
		glClear(GL_COLOR_BUFFER_BIT);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
		glBindTexture(GL_TEXTURE_2D, texid);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);

		glBegin(GL_QUADS);
		{
			// NOTE: glTexCoord2f determine state for next vertex = must be BEFORE the Vertex
			//		 Y goes down (Ymax=0, Ymin =1) for the texture
			glTexCoord2f(0, 1);
			glVertex2f(0, 0);
			glTexCoord2f(1, 1);
			glVertex2f(1, 0);
			glTexCoord2f(1, 0);
			glVertex2f(1, 1);
			glTexCoord2f(0, 0);
			glVertex2f(0, 1);
		}
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	glutSwapBuffers();

	renderAction = false;

	glutTimerFunc(40, timerEvent, 0);
}

static void keyboard(unsigned char key, int /*x*/, int /*y*/) {
	switch (key) {
		case 27:
		case 'q':
		case 'Q':
			applicationClosing = true;
			cudaError_t error;

			closeWorkerThread = true;
			while (!workerThreadClosed);
			while (renderAction); //?


			/*
			FROM SPECIFICATION:
			Be careful when using this function. In my tests freeglut continued to process the events that were already
			queued so performing any cleanup before actually getting out of the event loop could crash the application.
			*/
			error = cudaGraphicsUnregisterResource(cuda_pbo_resource);
			if (error != cudaSuccess) {
				fprintf(stderr, "Cannot unregister CUDA Graphic Resource: %s\n", cudaGetErrorString(error));
			}

			glutLeaveMainLoop();
			break;

		default:
			break;
	}
}

static void reshape(int x, int y) {
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

static void cudaThreadSample(CameraSampleOptions options, std::list< Image<FastAllocator> > inputImg) {
	
	for (auto fileit = inputImg.begin(); fileit != inputImg.end(); fileit++) {
		if (!DecodeError(hDebayerFfmpeg->StoreFrame(*fileit))) {
			return;
		}
	}
	workerThreadClosed = true;
	glutLeaveMainLoop();
}

static fastStatus_t RunCamera(int argc, char *argv[], CameraSampleOptions &options) {
	fastSdkParametersHandle_t sdkParameters;
	fastGetSdkParametersHandle(&sdkParameters);
	fastMJpegLibraryInit(sdkParameters);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	unsigned width, height, numberOfChannels, bitsPerChannel;
	CHECK_FAST(PreloadImageFromFolder(options.InputPath, width, height, numberOfChannels, bitsPerChannel));

	glutInitWindowSize(width / 2, height / 2);
	glutCreateWindow("Fastvideo Debayer Sample");

	glewInit();

	if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		fprintf(stderr, "This sample requires:\n");
		fprintf(stderr, "  OpenGL version 1.5\n");
		fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
		fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
		exit(EXIT_FAILURE);
	}

	hDebayerFfmpeg = new DebayerFfmpeg(false);

	std::list< Image<FastAllocator> > inputImg;
	CHECK_FAST(fvLoadImages(options.InputPath, options.OutputPath, inputImg, options.RawWidth, options.RawHeight, options.BitsPerChannel, false));
	options.SurfaceFmt = FAST_I8;

	for (auto i = inputImg.begin(); i != inputImg.end(); i++) {
		if ((*i).surfaceFmt != FAST_I8) {
			// validate input images
			fprintf(stderr, "Input file must not be color\n");
			return FAST_IO_ERROR;
		}
	}

	printf("Input surface format: grayscale\n");
	printf("Pattern: %s\n", EnumToString(options.Debayer.BayerFormat));
	printf("Output surface format: %s\n", EnumToString(FAST_RGB8));
	printf("Output sampling format: %s\n", EnumToString(options.JpegEncoder.SamplingFmt));
	printf("Debayer algorithm: %s\n", EnumToString(options.Debayer.BayerType));
	printf("JPEG quality: %d%%\n", options.JpegEncoder.Quality);
	printf("Restart interval: %d\n", options.JpegEncoder.RestartInterval);

	if (options.BaseColorCorrection.BaseColorCorrectionEnabled) {
		printf("Correction matrix:\n");
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n", options.BaseColorCorrection.BaseColorCorrection[0], options.BaseColorCorrection.BaseColorCorrection[1], options.BaseColorCorrection.BaseColorCorrection[2], options.BaseColorCorrection.BaseColorCorrection[3]);
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n", options.BaseColorCorrection.BaseColorCorrection[4], options.BaseColorCorrection.BaseColorCorrection[5], options.BaseColorCorrection.BaseColorCorrection[6], options.BaseColorCorrection.BaseColorCorrection[7]);
		printf("\t%.3f\t%.3f\t%.3f\t%.3f\n", options.BaseColorCorrection.BaseColorCorrection[8], options.BaseColorCorrection.BaseColorCorrection[9], options.BaseColorCorrection.BaseColorCorrection[10], options.BaseColorCorrection.BaseColorCorrection[11]);
	}

	ImageT< float, FastAllocator > matrixA;
	if (options.GrayscaleCorrection.MatrixA != NULL) {
		unsigned channels;
		bool failed = false;

		printf("\nMatrix A: %s\n", options.GrayscaleCorrection.MatrixA);
		CHECK_FAST(fvLoadPFM(options.GrayscaleCorrection.MatrixA, matrixA.data, matrixA.w, matrixA.wPitch, FAST_ALIGNMENT, matrixA.h, channels));

		if (channels != 1) {
			fprintf(stderr, "Matrix A file must not be color\n");
			failed = true;
		}

		if (options.MaxHeight != matrixA.h || options.MaxWidth != matrixA.w) {
			fprintf(stderr, "Input and matrix A file parameters mismatch\n");
			failed = true;
		}

		if (failed) {
			options.GrayscaleCorrection.MatrixA = NULL;
			fprintf(stderr, "Matrix A file reading error. Ignore parameters\n");
			failed = false;
		}
	}

	ImageT<char, FastAllocator > matrixB;
	if (options.GrayscaleCorrection.MatrixB != NULL) {
		bool failed = false;

		printf("\nMatrix B: %s\n", options.GrayscaleCorrection.MatrixB);
		CHECK_FAST(fvLoadImage(
			std::string(options.GrayscaleCorrection.MatrixB),
			std::string(""),
			matrixB,
			options.MaxHeight,
			options.MaxWidth,
			8, false
			));

		if (matrixB.surfaceFmt != FAST_I8) {
			fprintf(stderr, "Matrix B file must not be color\n");
			failed = true;
		}

		if (options.MaxHeight != matrixB.h || options.MaxWidth != matrixB.w) {
			fprintf(stderr, "Input and matrix B file parameters mismatch\n");
			failed = true;
		}

		if (failed) {
			options.GrayscaleCorrection.MatrixB = NULL;
			fprintf(stderr, "Matrix B file reading error. Ignore parameters\n");
			failed = false;
		}
	}

	std::unique_ptr<unsigned char, FastAllocator> lutData;
	if (options.Lut != NULL) {
		printf("\nLUT file: %s\n", options.Lut);
		CHECK_FAST(fvLoadLut(options.Lut, lutData, 256));
	}

	imageWidth = (*(inputImg.begin())).w;
	imageHeight = (*(inputImg.begin())).h;

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);

	glGenBuffers(1, &pbo_buffer);

	GLint bsize;
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * sizeof(unsigned char) * options.MaxWidth * options.MaxHeight, NULL, GL_STREAM_COPY);
	glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

	if (bsize != (3 * sizeof(unsigned char) * options.MaxWidth * options.MaxHeight)) {
		printf("Buffer object (%d) has incorrect size (%d).\n", (unsigned)pbo_buffer, (unsigned)bsize);
		return FAST_EXECUTION_FAILURE;
	}

	glGenTextures(1, &texid);
	cudaError_t error = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard);
	if (error != cudaSuccess) {
		fprintf(stderr, "Cannot register CUDA Graphic Resource: %s\n", cudaGetErrorString(error));
		return FAST_EXECUTION_FAILURE;
	}

	CHECK_FAST(hDebayerFfmpeg->Init(
		options,
		lutData,
		options.GrayscaleCorrection.MatrixA != NULL ? matrixA.data.get() : NULL,
		options.GrayscaleCorrection.MatrixB != NULL ? matrixB.data.get() : NULL
		));

	closeWorkerThread = false;
	workerThreadClosed = false;
	applicationClosing = false;
	std::thread cudaThread(cudaThreadSample, options, inputImg);

	printf("N: next image\n");
	printf("P: previous image\n");
	printf("Esc or Q: exit\n");

	// return to the function when close app (!)
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutMainLoop();
	printf("Close Main Loop\n");
	cudaThread.join();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &pbo_buffer);
	glDeleteTextures(1, &texid);

	inputImg.clear();
	CHECK_FAST(hDebayerFfmpeg->Close());

	delete hDebayerFfmpeg;

	return FAST_OK;
}

#endif // __RUN_CAMERA__