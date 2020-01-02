## About

In theory, [Single instruction, multiple data (SIMD)](https://en.wikipedia.org/wiki/SIMD) vectorization methods can dramatically accelerate data processing. In particular, in brain imaging we often want to analyze the data from millions of voxels. This project explores how processing of 32-bit floats is influenced by 128-bit [SSE](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) (4 voxels per instruction) and 256-bit [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) (8 voxels per instruction) benefit analysis. 

There are three ways that one could use SIMD methods. First, one could hand code assembly, which is very tedious. Second, one could leverage higher-level [intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/). Finally, one could hope a modern compiler would be smart enough to use clever instructions. 

The primary goal of this project was to teach myself about intrinsics and see if they provide any benefit. The brief take-away is that modern compiler optimazation (in particular, [-O3 with its ability to loop vectorize ](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) negates any benefit of explicitly coded SIMD. It may be that these very simple operations are simply constrained by memory latency and bandwidth. Indeed, modern computers face a [memory wall](https://en.wikipedia.org/wiki/Random-access_memory#Memory_wall) where CPUs spend most of their time idle while [waiting for data](https://www.blosc.org/docs/StarvingCPUs-CISE-2010.pdf). 

This project examines a typical sized dataset (173 million voxels) using two instructions. AVX introduced the [FMA](https://en.wikipedia.org/wiki/FMA_instruction_set) for [fused_multiply–add](https://en.wikipedia.org/wiki/Multiply–accumulate_operation#Fused_multiply–add). This computes a multiplication and addition in a single instruction, whereas this is traditionally two separate instructions with early Intel interfaces. Therefore, one might hope that the ability of AVX2 to compute 8 items in parallel could provide 16 times the performance of traditional methods (and four times the performance of 128-bit SSE). The second method explored is the loading and scaling of 16-bit integer data to 32-bit float data. MRI scanners tend to store data with 16-bits of precision, providing a scaling and intercept factor to convert these integers into real numbers. The [SSE _mm_cvtepi32_ps](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_cvtepi32_ps&expand=1449) instruction can convert four of these at once, again offering the promise of higher performance.

Here the program is compiled without optimization. Note that SSE and AVX dramatically speed up the square-root (sqrt) and fused multiply-add (fma) operations. 

```
>g++ -o tst main.cpp -march=native; ./tst 10 4
Only using 1 thread (not compiled with OpenMP support)
Reporting minimum time for 10 tests
i16_f32: min/mean	561	563	ms
i16_f32sse: min/mean	494	499	ms
sqrt: min/mean	420	424	ms
sqrtSSE: min/mean	172	175	ms
sqrtAVX: min/mean	97	98	ms
fma: min/mean	341	343	ms
fmaSSE: min/mean	253	257	ms
fmaAVX: min/mean	107	111	ms
fma (memory alignment not forced): min/mean	342	346	ms
```

Next, we optimize the compiler with `-O3`, and illustrate that the benefits for hand-coded SIMD disappear. Here we also compile with Clang (which does not support OpenMP on this MacOS computer) and gcc. 

```
>g++ -v
Configured with: --prefix=/Applications/Xcode.app/Contents/Developer/usr --with-gxx-include-dir=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/4.2.1
Apple clang version 11.0.0 (clang-1100.0.33.8)
Target: x86_64-apple-darwin18.7.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
>g++ -O3 -o tst main.cpp -march=native; ./tst 10 4
Only using 1 thread (not compiled with OpenMP support)
Reporting minimum time for 10 tests
i16_f32: min/mean	70	71	ms
i16_f32sse: min/mean	70	76	ms
sqrt: min/mean	71	72	ms
sqrtSSE: min/mean	73	74	ms
sqrtAVX: min/mean	71	71	ms
fma: min/mean	72	73	ms
fmaSSE: min/mean	72	73	ms
fmaAVX: min/mean	72	72	ms
fma (memory alignment not forced): min/mean	82	87	ms
>g++-9 -v
Using built-in specs.
COLLECT_GCC=g++-9
COLLECT_LTO_WRAPPER=/usr/local/Cellar/gcc/9.2.0_1/libexec/gcc/x86_64-apple-darwin18/9.2.0/lto-wrapper
Target: x86_64-apple-darwin18
Configured with: ../configure --build=x86_64-apple-darwin18 --prefix=/usr/local/Cellar/gcc/9.2.0_1 --libdir=/usr/local/Cellar/gcc/9.2.0_1/lib/gcc/9 --disable-nls --enable-checking=release --enable-languages=c,c++,objc,obj-c++,fortran --program-suffix=-9 --with-gmp=/usr/local/opt/gmp --with-mpfr=/usr/local/opt/mpfr --with-mpc=/usr/local/opt/libmpc --with-isl=/usr/local/opt/isl --with-system-zlib --with-pkgversion='Homebrew GCC 9.2.0_1' --with-bugurl=https://github.com/Homebrew/homebrew-core/issues --disable-multilib --with-native-system-header-dir=/usr/include --with-sysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
Thread model: posix
gcc version 9.2.0 (Homebrew GCC 9.2.0_1) 
>g++-9 -O3 -fopenmp -o tst main.cpp -march=native; ./tst 10 4
Reporting minimum time for 10 tests
Using 4 threads...
i16_f32: min/mean	69	75	ms
i16_f32sse: min/mean	79	80	ms
sqrt: min/mean	234	238	ms
sqrtSSE: min/mean	72	73	ms
sqrtAVX: min/mean	67	70	ms
fma: min/mean	228	230	ms
fmaSSE: min/mean	72	74	ms
fmaAVX: min/mean	69	69	ms
fma (memory alignment not forced): min/mean	223	225	ms
Using 1 thread...
i16_f32: min/mean	78	78	ms
i16_f32sse: min/mean	79	80	ms
sqrt: min/mean	156	157	ms
sqrtSSE: min/mean	73	75	ms
sqrtAVX: min/mean	70	70	ms
fma: min/mean	72	73	ms
fmaSSE: min/mean	75	75	ms
fmaAVX: min/mean	71	72	ms
fma (memory alignment not forced): min/mean	88	88	ms
```

Therefore, the takeaway is that modern compilers (at least Clang) can allow us to write classic C that is easy to read, maintain and port to other systems while still offering excellent performance. While more complicated routines may benefit from [SIMD](https://software.intel.com/en-us/articles/iir-gaussian-blur-filter-implementation-using-intel-advanced-vector-extensions), explicit coding is not required for simpler operations.

Likewise, OpenMP is an easy way to set up parallel threading. Parallel threads can have dramatic benefits for situations that are not [constrained by memory bandwidth](https://github.com/neurolabusc/niiSmooth). However, most of the code the results above are memory constrained.

Historically, gcc generated faster code than clang. However, this is [no longer the case](https://www.phoronix.com/scan.php?page=article&item=gcc-clang-3960x&num=7). Here we see that Clang does a better job optimizing the sqrt and fma functions. Hopefully, future releases of Clang for MacOS will provide better support OpenMP. By default, Clang on MacOS does not support OpenMP. While it can be [https://iscinumpy.gitlab.io/post/omp-on-high-sierra/](done), the example below shows it the current implementation can be deleterious for these memory-constrained tasks: 

```
>g++ -Xpreprocessor -fopenmp -lomp  -O3 -o tst main.cpp -march=native; ./tst 10 4
Reporting minimum time for 10 tests
Using 4 threads...
i16_f32: min/mean	76	78	ms
i16_f32sse: min/mean	76	78	ms
sqrt: min/mean	224	228	ms
sqrtSSE: min/mean	1265	1421	ms
sqrtAVX: min/mean	733	792	ms
fma: min/mean	215	225	ms
fmaSSE: min/mean	1461	1551	ms
fmaAVX: min/mean	1064	1084	ms
fma (memory alignment not forced): min/mean	221	300	ms
Using 1 thread...
i16_f32: min/mean	79	80	ms
i16_f32sse: min/mean	78	80	ms
sqrt: min/mean	70	71	ms
sqrtSSE: min/mean	91	92	ms
sqrtAVX: min/mean	76	78	ms
fma: min/mean	71	71	ms
fmaSSE: min/mean	90	93	ms
fmaAVX: min/mean	74	75	ms
fma (memory alignment not forced): min/mean	85	93	ms
```

Therefore, for these tests (and my experience) and for the current generation of compilers (early 2020), Clang does a better job of optimizing single-threaded code, but gcc handles multi-threaded code better.

## Links

- [FastMath is an elegant vectorized library for Delphi](https://neslib.github.io/FastMath/) accelerates x86_64 and ARM and CPUs.
-  The[Simd Library](http://ermig1979.github.io/Simd/) for C and C++ accelerates x86_64, PowerPC, ARM and CPUs.

