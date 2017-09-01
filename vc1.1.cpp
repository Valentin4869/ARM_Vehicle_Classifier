
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "Utils.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <stdarg.h>
#include <iostream>


using namespace arm_compute;
using namespace test_helpers;

#define FRAMES 100

//This timer is from the NVIDIA CUDA samples
double uclDeltaT(int iCounterID)
{
	// local var for computation of microseconds since last call
	double DeltaT;

#ifdef _WIN32 // Windows version of precision host timer

	// Variables that need to retain state between calls
	static LARGE_INTEGER liOldCount[3] = { { 0, 0 },{ 0, 0 },{ 0, 0 } };

	// locals for new count, new freq and new time delta 
	LARGE_INTEGER liNewCount, liFreq;
	if (QueryPerformanceFrequency(&liFreq))
	{
		// Get new counter reading
		QueryPerformanceCounter(&liNewCount);

		if (iCounterID >= 0 && iCounterID <= 2)
		{
			// Calculate time difference for timer 0.  (zero when called the first time) 
			DeltaT = liOldCount[iCounterID].LowPart ? (((double)liNewCount.QuadPart - (double)liOldCount[iCounterID].QuadPart) / (double)liFreq.QuadPart) : 0.0;
			// Reset old count to new
			liOldCount[iCounterID] = liNewCount;
		}
		else
		{
			// Requested counter ID out of range
			DeltaT = -9999.0;
		}

		// Returns time difference in seconds since last call
		return DeltaT;
	}
	else
	{
		// No high resolution performance counter
		return -9999.0;
	}
#else // Linux version of precision host timer. See http://www.informit.com/articles/article.aspx?p=23618&seqNum=8
	static struct timeval _NewTime;  // new wall clock time (struct representation in seconds and microseconds)
	static struct timeval _OldTime[3]; // old wall clock timers 0, 1, 2 (struct representation in seconds and microseconds)

									   // Get new counter reading
	gettimeofday(&_NewTime, NULL);

	if (iCounterID >= 0 && iCounterID <= 2)
	{
		// Calculate time difference for timer (iCounterID).  (zero when called the first time) 
		DeltaT = ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime[iCounterID].tv_sec + 1.0e-6 * (double)_OldTime[iCounterID].tv_usec);
		// Reset old timer (iCounterID) to new timer
		_OldTime[iCounterID].tv_sec = _NewTime.tv_sec;
		_OldTime[iCounterID].tv_usec = _NewTime.tv_usec;
	}
	else
	{
		// Requested counterID is out of rangewith respect to available counters
		DeltaT = -9999.0;
	}

	// Returns time difference in seconds sunce the last call
	return DeltaT;

#endif

}  


void LoadData(const char *cFileName, float *wgt, int len) {
	

	FILE *pFileStream = fopen(cFileName, "rb");
		if (pFileStream == NULL) 
		printf("Could not open %s\n", cFileName);

	int count = fread(wgt, sizeof(float), len, pFileStream);

	if (count != len) {
		printf("file read error: %i of %i samples acquired\n", count, len);
	}

	fclose(pFileStream);
}

void DisplayMatrix(float* data, int m, int n, int offset) {

	int i, j;

	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j)
			printf("(%f) ", data[offset + i * n + j]);

		printf("\n");
	}
		

}


Tensor* CreateTensor1d(unsigned int x){

Tensor *X=new Tensor;

const TensorShape x_shape(x);

X->allocator()->init(TensorInfo(x_shape, 1, DataType::F32));


return X;
}

Tensor* CreateTensor2d(unsigned int x, unsigned int y){

Tensor *X=new Tensor;

const TensorShape x_shape(x,y);

X->allocator()->init(TensorInfo(x_shape, 1, DataType::F32));

return X;
}

Tensor* CreateTensor3d(unsigned int x, unsigned int y, unsigned int z){

Tensor *X=new Tensor;

const TensorShape x_shape(x,y,z);

X->allocator()->init(TensorInfo(x_shape, 1, DataType::F32));


return X;
}

Tensor* CreateTensor4d(unsigned int x, unsigned int y, unsigned int z, unsigned int w){

Tensor *X=new Tensor;

const TensorShape x_shape(x,y,z,w);

X->allocator()->init(TensorInfo(x_shape, 1, DataType::F32));


return X;
//*reinterpret_cast<float*>( X.buffer() + X.info()->offset_element_in_bytes(Coordinates(0,0,0)))=100;

}
//fill tensor with row-major-ordered data

void FillTensor(Tensor* X, float* src, int sz){

  X->allocator()->allocate();


for(int i=0; i<sz; i++)
reinterpret_cast<float*>( X->buffer())[i]=src[i];

}


void DisplayTensor(Tensor* X, int m, int n, int offset){

  int i, j;

	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j)
			printf("(%f) ",  reinterpret_cast<float*>(X->buffer())[offset + i * n + j]);

		printf("\n");
	}
 
  
}


void prtDims(Tensor* X, const char* name){
  printf("%s(%i,%i,%i,%i):\n",name,X->info()->dimension(0),X->info()->dimension(1),X->info()->dimension(2),X->info()->dimension(3));
}

int main(int argc, const char **argv)
{
printf("Begin\n");


unsigned int dimx= 5, dimy= 5, dimz= 3, dimw= 8;

float wgtConv1[5*5*3*32];
float wgtConv2[5*5*32*32];
float im[96*96*3];
float wgtDense1[24*24*32*100];
float wgtDense2[100*100];
float wgtDense3[100*4];


LoadData("parameter/c.bin",im,96*96*3);
LoadData("parameter/conv1_flipped.bin",wgtConv1,5*5*3*32);
LoadData("parameter/conv2_flipped.bin",wgtConv2,5*5*32*32);
LoadData("parameter/ip3.bin",wgtDense1,24*24*32*100);
LoadData("parameter/ip4.bin",wgtDense2,100*100);
LoadData("parameter/ip_last.bin",wgtDense3,100*4);


int szX=dimx*dimy*dimz, sz=dimx*dimy*dimz*dimw;

Tensor* X;
Tensor* weights1;
Tensor* weights2;
Tensor* weights3;
Tensor* weights4;
Tensor* weights5;
Tensor* reshaped_x;

Tensor* out_conv1;
Tensor* out_relu1;
Tensor* out_mxpool1;
Tensor* out_conv2;
Tensor* out_relu2;
Tensor* out_mxpool2;
Tensor* out_relu3;
Tensor* out_relu4;
Tensor* out_dense1;
Tensor* out_dense2;
Tensor* out_dense3;
Tensor* out_softmax;


Tensor* biases;
Tensor* biases3;
Tensor* biases4;
Tensor* biases5;
Tensor* reshape_x;
NEConvolutionLayer conv1;
NEConvolutionLayer conv2;
NEPoolingLayer        mxpool1;
NEPoolingLayer        mxpool2;
NEActivationLayer     relu1;
NEActivationLayer     relu2;
NEActivationLayer     relu3;
NEActivationLayer     relu4;
NEFullyConnectedLayer dense1;
NEFullyConnectedLayer dense2;
NEFullyConnectedLayer dense3;
NESoftmaxLayer softmax;
NEGEMM mult1;
NEGEMM mult2;
NEGEMM mult3;

 
X=CreateTensor3d(96,96,3);
weights1=CreateTensor4d(5,5,3,32);
weights2=CreateTensor4d(5,5,32,32);
weights3=CreateTensor2d(32*24*24,100);
weights4=CreateTensor2d(100,100);
weights5=CreateTensor2d(100,4);
out_conv1=CreateTensor3d(96,96,32);
out_relu1=CreateTensor3d(96,96,32);
out_mxpool1=CreateTensor3d(48,48,32);
out_conv2=CreateTensor3d(48,48,32);
out_relu2=CreateTensor3d(48,48,32);
reshape_x=CreateTensor2d(24*24*32,1);
out_relu3=CreateTensor2d(100,1);
out_relu4=CreateTensor2d(100,1);
out_mxpool2=CreateTensor3d(24,24,32);
out_dense1=CreateTensor2d(100,1);
out_dense2=CreateTensor2d(100,1);
out_dense3=CreateTensor2d(4,1);
biases=CreateTensor1d(32);
biases3=CreateTensor1d(100);
biases4=CreateTensor1d(100);
biases5=CreateTensor1d(4);

out_softmax=CreateTensor2d(4,1);

conv1.configure(X, weights1, NULL, out_conv1, PadStrideInfo(1,1,2,2));
relu1.configure(out_conv1, out_relu1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
mxpool1.configure(out_relu1, out_mxpool1, PoolingLayerInfo(PoolingType::MAX, 2,PadStrideInfo(2,2,0,0)));

conv2.configure(out_mxpool1, weights2, NULL, out_conv2, PadStrideInfo(1,1,2,2));
relu2.configure(out_conv2, out_relu2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
mxpool2.configure(out_relu2, out_mxpool2, PoolingLayerInfo(PoolingType::MAX, 2,PadStrideInfo(2,2,0,0)));
//mxpool2.configure(out_relu2, reshape_x, PoolingLayerInfo(PoolingType::MAX, 2,PadStrideInfo(2,2,0,0)));

//mult1.configure(reshape_x,weights3,NULL, out_dense1,1.0,1.0);

printf("configuring dense1\n");
dense1.configure(reshape_x,weights3,NULL,out_dense1);
relu3.configure(out_dense1, out_relu3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
printf("configuring dense2\n");
dense2.configure(out_relu3,weights4,NULL,out_dense2);
//mult2.configure(out_relu3,weights4,NULL, out_dense2,1.0,1.0);
relu4.configure(out_dense2, out_relu4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
printf("configuring dense3\n");
dense3.configure(out_relu4,weights5,NULL,out_dense3);
//mult3.configure(out_relu4,weights5,NULL, out_dense3,1.0,1.0);
//conv2.configure(out_conv1, weights2, biases, out_conv2, PadStrideInfo(1,1,2,2));
//softmax.configure(out_dense3, out_softmax);
 




//FillTensor(reshape_x,NULL,0);
FillTensor(biases,NULL,0);
FillTensor(biases3,NULL,0);
FillTensor(biases4,NULL,0);
FillTensor(biases5,NULL,0);


//prtDims(biases,"b");


//DisplayTensor(biases,1,dimw,0);

FillTensor(weights1,wgtConv1,5*5*3*32);
prtDims(weights1,"W1");
DisplayTensor(weights1,5,5,0);

FillTensor(weights2,wgtConv2,5*5*32*32);
prtDims(weights2,"W2");
DisplayTensor(weights2,5,5,25*3);


FillTensor(weights3,wgtDense1,100*24*24*32);
FillTensor(weights4,wgtDense2,100*100);
FillTensor(weights5,wgtDense3,100*4);

//FillTensor(reshape_x,rshX,24*24*32);

FillTensor(out_dense1,NULL,0);
FillTensor(out_conv1,NULL,0);
FillTensor(out_relu1,NULL,0);
FillTensor(out_mxpool1,NULL,0);
FillTensor(out_conv2,NULL,0);
FillTensor(out_relu2,NULL,0);
FillTensor(out_mxpool2,NULL,0);

FillTensor(out_relu3,NULL,0);
FillTensor(out_relu4,NULL,0);
FillTensor(out_dense2,NULL,0);
FillTensor(out_dense3,NULL,0);
FillTensor(out_softmax,NULL,0);

prtDims(weights3,"W3");
DisplayTensor(weights3,1,8,24*24*32);


FillTensor(reshape_x,NULL,0);
double time_elapsed;
float* buff_mxpool2=reinterpret_cast<float*>(out_mxpool2->buffer());
float* buff_reshape_x=reinterpret_cast<float*>( reshape_x->buffer());

//Waits for preparations before running tests (manually assigning desired cpu cores to process).
printf("Waiting...\n");
getchar();
printf("Running now.\n");


uclDeltaT(0);

FillTensor(X,im,96*96*3);

for(int i=0; i<FRAMES; i++){
		conv1.run();
		relu1.run();
		mxpool1.run();
		conv2.run();
		relu2.run();
		mxpool2.run();
		
//		FillTensor(reshape_x,reinterpret_cast<float*>(out_mxpool2->buffer()),24*24*32);

//Reshaping input is not handeled in NEFullyConnectedLayer_v17.03.1, so have to "reshape" manually. No significant cost in performance though.
for(int j=0; j<24*24*32;j++)
//reinterpret_cast<float*>( reshape_x->buffer())[j]=reinterpret_cast<float*>(out_mxpool2->buffer())[j];
buff_reshape_x[j]=buff_mxpool2[j];
		
		dense1.run();
		relu3.run();
		
		dense2.run();
		relu4.run();
		
		dense3.run();
	softmax.run();
}


time_elapsed=uclDeltaT(0);


printf("--------flat maxpool2--------\n");
for(int i=0; i<1; i++){
DisplayTensor(reshape_x,4,24,0);
printf("\n");
}

printf("--------Output--------\n");
for(int i=0; i<1; i++){
DisplayTensor(out_dense1,1,100,0);
printf("\n");
}

/*for(int i=0; i<1; i++){
DisplayTensor(out_mxpool2,96,96,0);
printf("\n");
}
*/


printf("--------relu(dense2) out--------\n");
for(int i=0; i<1; i++){
DisplayTensor(out_relu4,1,100,0);
printf("\n");
}


printf("--------dense3 out--------\n");
for(int i=0; i<1; i++){
DisplayTensor(out_dense3,4,1,0);
printf("\n");
}

//Buggy softmax in ARM Compute Library v17.03.1 
printf("--------Softmax out--------\n");
for(int i=0; i<1; i++){
DisplayTensor(out_softmax,4,1,0);
printf("\n");
}

printf("t_conv0: %f\n",time_elapsed);
return 0;

}
