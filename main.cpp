/* ----------------------------------------------------------------------
//on MacOS g++ aliases Clang
g++ -O3 -o tst main.cpp -march=native; ./tst 10 4
 
gcc-9 -O3 -o tst main.cpp -march=native; ./tst

g++-9 -O3 -fopenmp -o tst main.cpp -march=native; ./tst

g++-9 -O3 -fopenmp -o tst main.cpp  -mavx2 -march=core-avx2 -mtune=core-avx2; ./tst

g++-9 -O3 -o tst main.cpp  -march=core-avx2; ./tst

g++-9 -O3 -fopenmp -o tst main.cpp; ./tst

 */
#include <cmath> //sqrt()
#include <stdio.h>
#ifdef __x86_64__    
	#include <immintrin.h>
	#define myUseAVX
    // do x64 stuff   
#else  
	#include "sse2neon.h"
	#undef myUseAVX
    // do arm stuff
#endif 
#include <time.h>
#include <climits>
#include <cstring>
#if !defined (HAVE_POSIX_MEMALIGN) && !defined (_WIN32) && !defined (__APPLE__)
 #include <malloc.h>
#endif 
#if defined(_OPENMP) 
 #include <omp.h>
#endif 

#ifndef MAX
#define MAX(A,B) ((A) > (B) ? (A) : (B))
#endif

#ifndef MIN
#define MIN(A,B) ((A) > (B) ? (B) : (A))
#endif

#define kSSE 4 //128-bit SSE handles 4 32-bit floats per instruction
#define kSSE64 2 //128-bit SSE handles 2 64-bit floats per instruction
#define kAVX 8 //256-bit AVX handles 8 32-bit floats per instruction
#define kAVX64 8 //256-bit AVX handles 4 64-bit floats per instruction

//number of voxels for test, based on HCP resting state  https://protocols.humanconnectome.org/HCP/3T/imaging-protocols.html
#define kNVox 808704000 //= 104*90*72*1200;


void fma(float *v, int64_t n, float slope1, float intercept1) {
//fused multiply+add, out = in * slope + intercept
	if ((slope1 == 1.0f) && (intercept1 == 0.0f)) return;
	#pragma omp parallel for
	for (int64_t i = 0; i < n; i+=1) {
		v[i] = (v[i] * slope1) + intercept1;
	}
}//fma()

#ifdef myUseAVX 
void fmaAVX(float *v, int64_t n, float slope1, float intercept1) {
//fused multiply+add, out = in * slope + intercept
	if ((slope1 == 1.0f) && (intercept1 == 0.0f)) return;
	float * vin = v;
	__m256 intercept = _mm256_set1_ps(intercept1);
    __m256 slope = _mm256_set1_ps(slope1);
    #pragma omp parallel for
    for (int64_t i = 0; i <= (n-kAVX); i+=kAVX) {
		__m256 v8 = _mm256_loadu_ps(vin);
		__m256 result = _mm256_fmadd_ps(v8, slope, intercept);
		_mm256_storeu_ps(vin, result);
		vin += kAVX;
	}
	int tail = (n % kAVX);
	while (tail > 0) {
		v[n-tail] = (v[n-tail] * slope1) + intercept1;
		tail --;	
	}
	_mm256_zeroupper();
} //fmaAVX()
#endif

/*void fma8(float *v, int64_t n, float slope1, float intercept1) {
//unrolled loop, hoping this would aid vectorization
	if ((slope1 == 1.0f) && (intercept1 == 0.0f)) return;
	float slope = slope1;
	float intercept = intercept1;
	for (int64_t i = 0; i <= (n-kAVX); i+=kAVX) {
		//printf("%d\n", i);
		for (int64_t j = 0; j < kAVX; j++) {
			//v[i+j] = (v[i+j] * slope) + intercept;
			v[i+j] *= slope1;
			v[i+j] += intercept1;
		}
	}
	int tail = (n % kAVX);
	while (tail > 0) {
		v[n-tail] = (v[n-tail] * slope1) + intercept1;
		tail --;	
	}
} //fma8*/

long timediff(clock_t t1, clock_t t2) {
    long elapsed;
    elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
    return elapsed;
}

void fmaSSE(float *v, int64_t n, float slope1, float intercept1) {
//multiply+add, out = in * slope + intercept
	if ((slope1 == 1.0f) && (intercept1 == 0.0f)) return;
	float * vin = v;
	__m128 intercept = _mm_set1_ps(intercept1);
	__m128 slope = _mm_set1_ps(slope1);
	#pragma omp parallel for
	for (int64_t i = 0; i <= (n-kSSE); i+=kSSE) {
		__m128 v4 = _mm_loadu_ps(vin);
		__m128 m = _mm_mul_ps(v4, slope);
		__m128 ma = _mm_add_ps(m, intercept);
		_mm_storeu_ps(vin, ma); 
		vin += kSSE;
	}
	int tail = (n % kSSE);
	while (tail > 0) {
		v[n-tail] = (v[n-tail] * slope1) + intercept1;
		tail --;	
	}
} //fmaSSE

void i16_f32(int16_t *in16, float *out32, int64_t n, float slope1, float intercept1) {
	for (int64_t i = 0; i < n; i++)
		out32[i] = (in16[i] * slope1) + intercept1;
}

void i16_f32sse(int16_t *in16, float *out32, int64_t n, float slope1, float intercept1) {
	short int * vin = (short int *)in16;
	float * vout = out32;
	__m128 intercept = _mm_set1_ps(intercept1);
	__m128 slope = _mm_set1_ps(slope1);
	#define kStep 8
	for (int64_t i = 0; i <= (n-kStep); i+=kStep) {
		__m128i x = _mm_loadu_si128((__m128i*) vin); //read 8 16-bit ints
		vin += kStep;
		//convert 4 low
		__m128i i4 = _mm_unpacklo_epi16(x, _mm_set1_epi16(0));
		__m128 f4 = _mm_cvtepi32_ps(i4);
		__m128 m = _mm_mul_ps(f4, slope);
		__m128 ma = _mm_add_ps(m, intercept);
		_mm_storeu_ps(vout, ma); 
		vout += kSSE;
		//read 4 high
		i4 = _mm_unpackhi_epi16(x, _mm_set1_epi16(0));
		f4 = _mm_cvtepi32_ps(i4);
		m = _mm_mul_ps(f4, slope);
		ma = _mm_add_ps(m, intercept);
		_mm_storeu_ps(vout, ma); 
		vout += kSSE;
	}
	int tail = (n % kStep);
	while (tail > 0) {
		out32[n-tail] = (in16[n-tail] * slope1) + intercept1;
		tail --;	
	}		
}

void sqrtSSE(float *v, int64_t n) {
//square root using SSE
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sqrt_ps&expand=5364
// https://stackoverflow.com/questions/1528727/why-is-sse-scalar-sqrtx-slower-than-rsqrtx-x
	float * vin = v;
	#pragma omp parallel for
	for (int64_t i = 0; i <= (n-kSSE); i+=kSSE) {
		__m128 v4 = _mm_loadu_ps(vin);
		__m128 ma = _mm_sqrt_ps(v4);
		_mm_storeu_ps(vin, ma); 
		//a compiler will reduce this to...
		//_mm_storeu_ps(vin, _mm_sqrt_ps(_mm_loadu_ps(vin)));
		vin += kSSE;
	}
	int tail = (n % kSSE);
	while (tail > 0) {
		v[n-tail] = sqrt(v[n-tail]);
		tail --;	
	}
} // sqrtSSE()

#ifdef myUseAVX 
void sqrtAVX(float *v, int64_t n) {
//https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sqrt_ps&expand=5364,5367
	float * vin = v;
	#pragma omp parallel for
	for (int64_t i = 0; i <= (n-kAVX); i+=kAVX) {
		__m256 v8 = _mm256_loadu_ps(vin);
		__m256 result = _mm256_sqrt_ps(v8);
		_mm256_storeu_ps(vin, result);
		vin += kAVX;
	}
	int tail = (n % kAVX);
	while (tail > 0) {
		v[n-tail] = sqrt(v[n-tail]);
		tail --;	
	}
	_mm256_zeroupper();
} // sqrtAVX()
#endif
	
void sqrtV(float *v, int64_t n) {
//vectorize square-root, fslmaths "-sqrt"
	#pragma omp parallel for
	for (int64_t i = 0; i < n; i++ )
      		v[i] = sqrt(v[i]);
}//sqrtV()

void conv_tst_i16(int reps) {
	int64_t n = kNVox;
	float* f = (float *)_mm_malloc(n*sizeof(float), 64);
	int16_t* v16 = (int16_t *)_mm_malloc(n*sizeof(int16_t), 64);
	for (int64_t i = 0; i < n; i++)
		v16[i] = i % 255;
	i16_f32(v16,f, n, 0.1f, 10.0f); //warm up
	//SISD
	long mn = INT_MAX;
	long sum = 0.0;
	int reps1 = MAX(reps, 1);
	for (int64_t i = 0; i < reps1; i++) {
		clock_t startTime = clock();
		i16_f32(v16,f, n, 0.1f, 10.0f);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	if (reps < 1) { //reps=0: warmup, do not report values 
		_mm_free (v16);
		_mm_free (f);
		return;
	}
	printf("i16_f32: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	//SSE
	mn = INT_MAX;
	sum = 0.0;
	for (int64_t i = 0; i < reps; i++) {
		clock_t startTime = clock();
		i16_f32sse(v16,f, n, 0.1f, 10.0f);
		mn = MIN(mn, timediff(startTime, clock()));	
		sum += timediff(startTime, clock());
	}
	printf("i16_f32sse: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	/*if (n < 32) {
		for (int64_t i = 0; i < n; i++)
			printf("%g ", f[i]);
		printf("\n");
	}*/
	_mm_free (v16);
	_mm_free (f);
}

void fma_tst(int reps) {
	int64_t n = kNVox;
	float* vin = (float *)_mm_malloc(n*sizeof(float), 64);
	for (int64_t i = 0; i < n; i++)
		vin[i] = i;
	float* v = (float *)_mm_malloc(n*sizeof(float), 64);
	memcpy(v, vin, n*sizeof(float));
	fma(v, n, 1.0, 10.0); //for timing, ignore first run - get CPU in floating point mode
	//SISD
	long mn = INT_MAX;
	long sum = 0.0;
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		fma(v, n, 1.0, 10.0);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("fma: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	//SSE
	mn = INT_MAX;
	sum = 0.0;
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		fmaSSE(v, n, 1.0, 10.0);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("fmaSSE: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	//AVX
	mn = INT_MAX;
	sum = 0.0;
	#ifdef myUseAVX 	
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		fmaAVX(v, n, 1.0, 10.0);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("fmaAVX: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	#endif
	//SISD, do not force alignment
	mn = INT_MAX;
	sum = 0.0;
	for (int64_t i = 0; i < reps; i++) {
		//request new memory on each loop: some calls might be aligned by random chance
		void * data = (void *)calloc(1,n*sizeof(float)) ;
		float * vu = (float *)data;
		for (int64_t i = 0; i < n; i++)
			vu[i] = vin[i];
		clock_t startTime = clock();
		fma(vu, n, 1.0, 10.0);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
		free(data);
	}
	printf("fma (memory alignment not forced): min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	_mm_free (v);
	_mm_free (vin);
}

void sqrt_tst(int reps) {
	int64_t n = kNVox;
	float* vin = (float *)_mm_malloc(n*sizeof(float), 64);
	for (int64_t i = 0; i < n; i++)
		vin[i] = i+1;
	float* v = (float *)_mm_malloc(n*sizeof(float), 64);
	memcpy(v, vin, n*sizeof(float));
	sqrtV(v, n); //for timing, ignore first run - get CPU in floating point mode
	//SISD
	long mn = INT_MAX;
	long sum = 0.0;
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		sqrtV(v, n);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("sqrt: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	//SSE
	mn = INT_MAX;
	sum = 0.0;
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		sqrtSSE(v, n);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("sqrtSSE: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	//AVX
	#ifdef myUseAVX 
	mn = INT_MAX;
	sum = 0.0;	
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		sqrtAVX(v, n);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("sqrtAVX: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	#endif
	_mm_free (v);
	_mm_free (vin);
}


void sqrt64V(double *v, int64_t n) {
//vectorize square-root, fslmaths "-sqrt"
	#pragma omp parallel for
	for (int64_t i = 0; i < n; i++ )
      		v[i] = sqrt(v[i]);
}//sqrtV()

#ifdef __x86_64__ 
void sqrt64SSE(double *v, int64_t n) {
//square root using SSE
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_sqrt_ps&expand=5364
// https://stackoverflow.com/questions/1528727/why-is-sse-scalar-sqrtx-slower-than-rsqrtx-x
	double * vin = v;
	#pragma omp parallel for
	for (int64_t i = 0; i <= (n-kSSE64); i+=kSSE64) {
		__m128d v2 = _mm_loadu_pd(vin);
		__m128d ma = _mm_sqrt_pd(v2);
		_mm_storeu_pd(vin, ma); 
		//a compiler will reduce this to...
		//_mm_storeu_ps(vin, _mm_sqrt_ps(_mm_loadu_ps(vin)));
		vin += kSSE64;
	}
	int tail = (n % kSSE64);
	while (tail > 0) {
		v[n-tail] = sqrt(v[n-tail]);
		tail --;	
	}
} // sqrt64SSE()
#endif

#ifdef myUseAVX 
void sqrt64AVX(double *v, int64_t n) {
//https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_sqrt_ps&expand=5364,5367
	double * vin = v;
	#pragma omp parallel for
	for (int64_t i = 0; i <= (n-kAVX64); i+=kAVX64) {
		__m256d v4 = _mm256_loadu_pd(vin);
		__m256d result = _mm256_sqrt_pd(v4);
		_mm256_storeu_pd(vin, result);
		vin += kAVX64;
	}
	int tail = (n % kAVX64);
	while (tail > 0) {
		v[n-tail] = sqrt(v[n-tail]);
		tail --;	
	}
	_mm256_zeroupper();
} // sqrtAVX()
#endif

void sqrt64_tst(int reps) {
	int64_t n = kNVox;
	double* vin = (double *)_mm_malloc(n*sizeof(double), 64);
	for (int64_t i = 0; i < n; i++)
		vin[i] = i+1;
	double* v = (double *)_mm_malloc(n*sizeof(double), 64);
	memcpy(v, vin, n*sizeof(float));
	sqrt64V(v, n); //for timing, ignore first run - get CPU in floating point mode
	//SISD
	long mn = INT_MAX;
	long sum = 0.0;
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		sqrt64V(v, n);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("sqrt64: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	//SSE
	#ifdef __x86_64__ 
	mn = INT_MAX;
	sum = 0.0;
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		sqrt64SSE(v, n);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("sqrt64SSE: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	#endif
	//AVX
	#ifdef myUseAVX 
	mn = INT_MAX;
	sum = 0.0;	
	for (int64_t i = 0; i < reps; i++) {
		memcpy(v, vin, n*sizeof(float));
		clock_t startTime = clock();
		sqrt64AVX(v, n);
		mn = MIN(mn, timediff(startTime, clock()));
		sum += timediff(startTime, clock());
	}
	printf("sqrt64AVX: min/mean\t%ld\t%ld\tms\n", mn, sum/reps);
	#endif
	_mm_free (v);
	_mm_free (vin);
}

int main(int argc, char * argv[]) {
	int reps = 1; //how many times to repeat each test
	int nThread = 1;
	#if defined(_OPENMP)
	nThread = omp_get_max_threads(); 
	#endif
	#ifdef __aarch64__
		#ifdef __ARM_FEATURE_SVE
		printf("SVE supported: please upgrade functions!\n");
		#endif
	#endif
	if ( argc > 1 ) {
		reps = atoi(argv[1]);
		if ( argc > 2 ) {
			nThread = atoi(argv[2]);
			#ifndef _OPENMP
			if (nThread > 1) 
				printf("Only using 1 thread (not compiled with OpenMP support)\n");
			#endif
		}
	} else {
		#if defined(_OPENMP) 
		printf("Use '%s 3 2' to run each test three times with two threads\n", argv[0]);
		#else
		printf("Use '%s 3' to run each test three times\n", argv[0]);
		#endif
	}
	printf("Reporting minimum time for %d tests\n", reps);
	#if defined(_OPENMP) 
	printf("Using %d threads...\n", nThread);
	omp_set_num_threads(nThread);
	#endif	
	conv_tst_i16(0);
	conv_tst_i16(reps);
	sqrt_tst(reps);
	sqrt64_tst(reps);
	fma_tst(reps);
	#if defined(_OPENMP) 
	if (nThread == 1) return 0;
	printf("Using 1 thread...\n");
	omp_set_num_threads(1);
	conv_tst_i16(reps);
	sqrt_tst(reps);
	fma_tst(reps);	
	#endif	
	return 0;
}