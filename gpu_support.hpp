/****************************************************************************************
 *
 *    BEGIN: NAG COPYRIGHT NOTICE 
 * 
 * Copyright the Numerical Algorithms Group Ltd, 2015
 *
 * The portion of this file between 
 *    BEGIN: NAG COPYRIGHT NOTICE
 * and 
 *    END: NAG COPYRIGHT NOTICE
 * are copyrighted to the Numerical Algorithms Group Ltd, 2015.
 * The whole file was created by the Numerical Algorithms Group Ltd, 2015.
 *
 **************************************************************************************/

#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <assert.h>



//=========    MACRO Definitions    ======================================================

/**
 * Macro to test for CUDA errors and bail
 * if we find one
 */
#define CHECK(a)   ErrorHandler::checkError( (a), __FILE__, __LINE__ )

/**
 * Macro to test whether warp shuffles are available
 */
#define HAS_SHUFFLE    0
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
#undef HAS_SHUFFLE
#define HAS_SHUFFLE   1
#endif


/**
 * Macros to insert timing code in the source files depending on a compiler -D define
 */
#define TIMING_SETUP(evt,n)   
#if defined(TIMING)   
   #undef TIMING_SETUP
   #define TIMING_SETUP(evt,n)   { for(int i=0; i<n; i++) CHECK( cudaEventCreate( &evt[i],0 ) ); } 
#endif

#define TIMING_RECORD(ev) 
#if defined(TIMING) 
   #undef TIMING_RECORD
   #define TIMING_RECORD(ev)   CHECK( cudaEventRecord((ev),0) );
#endif

#define TIMING_PRINT 0
#if defined(TIMING)
   #undef TIMING_PRINT
   #define TIMING_PRINT 1
#endif

#define TIMING_CLEANUP(evt,n)   
#if defined(TIMING)  
   #undef TIMING_CLEANUP
   #define TIMING_CLEANUP(evt,n)   { for(int i=0; i<n; i++) CHECK( cudaEventDestroy( evt[i] ) ); } 
#endif





//=========    Error Handler class   ==================================================


/**
 * Utility class to help error handling.
 * Users can add their own error handlers
 * to the static function pointers
 *      cudaErrorHandler
 * and
 *      cublasErrorHandler
 * If these are non-NULL then they will be called in
 * preference to the default behaviour which is to
 * print a message to the console and call abort()
 */
struct ErrorHandler {

   static const char * cublasGetErrorEnum(cublasStatus_t error) {
      switch (error) {
      case CUBLAS_STATUS_SUCCESS:
         return "CUBLAS_STATUS_SUCCESS";

      case CUBLAS_STATUS_NOT_INITIALIZED:
         return "CUBLAS_STATUS_NOT_INITIALIZED";

      case CUBLAS_STATUS_ALLOC_FAILED:
         return "CUBLAS_STATUS_ALLOC_FAILED";

      case CUBLAS_STATUS_INVALID_VALUE:
         return "CUBLAS_STATUS_INVALID_VALUE";

      case CUBLAS_STATUS_ARCH_MISMATCH:
         return "CUBLAS_STATUS_ARCH_MISMATCH";

      case CUBLAS_STATUS_MAPPING_ERROR:
         return "CUBLAS_STATUS_MAPPING_ERROR";

      case CUBLAS_STATUS_EXECUTION_FAILED:
         return "CUBLAS_STATUS_EXECUTION_FAILED";

      case CUBLAS_STATUS_INTERNAL_ERROR:
         return "CUBLAS_STATUS_INTERNAL_ERROR";

      case CUBLAS_STATUS_NOT_SUPPORTED:
         return "CUBLAS_STATUS_NOT_SUPPORTED";

      case CUBLAS_STATUS_LICENSE_ERROR:
         return "CUBLAS_STATUS_LICENSE_ERROR";
      }

      return "<unknown>";
   }


   static void (*cudaErrorHandler)(cudaError_t er, const char *file, int line);
   static void (*cublasErrorHandler)(cublasStatus_t er, const char *file, int line);
   static void (*otherErrorHandler)(const char *msg, const char *file, int line);

   inline static void checkError(cudaError_t er, const char * file, int line) {
#ifdef _DEBUG
      // If we don't have an error and we're in
      // DEBUG mode, sync with device and get last
      // error
      if(er==cudaSuccess) {
         cudaDeviceSynchronize();
         er = cudaGetLastError();
      }
#endif

      if(er!=cudaSuccess) {
         if(cudaErrorHandler) {
            cudaErrorHandler(er, file, line);
         } else {
            std::cerr << "CUDA error in " << file << " at line " << line << ":" << std::endl;
            std::cerr << "\t" << cudaGetErrorString(er) << std::endl;
            assert(false);
         }
      }
   }


   inline static void checkError(cublasStatus_t er, const char * file, int line) {
      if(er!=CUBLAS_STATUS_SUCCESS) {
         if(cublasErrorHandler) {
            cublasErrorHandler(er, file, line);
         } else {
            std::cerr << "CUBLAS error in " << file << " at line " << line << ":" << std::endl;
            std::cerr << "\tCUBLAS returned error " << cublasGetErrorEnum(er) << std::endl;
            assert(false);
         }
      }
   }

   inline static void checkError(const char *msg, const char * file, int line) {
      if(otherErrorHandler) {
         otherErrorHandler(msg, file, line);
      } else {
         std::cerr << "General error in " << file << " at line " << line << ":" << std::endl;
         std::cerr << "\t" << msg  << std::endl;
         assert(false);
      }
   }
};


//===========================================================================================

/**
 * Utility functions for doing things on GPU.
 * These must not be available on the host
 */
#ifdef __CUDACC__



// These functions were added at CUDA 6.5
#if 0
static __device__ __inline__ double __shfl_down(double var, unsigned int delta, int width=warpSize)
{
   float lo, hi;
   asm volatile("mov.b64 {%0,%1}, %2;" : "=f"(lo), "=f"(hi) : "d"(var));
   hi = __shfl_down(hi, delta, width);
   lo = __shfl_down(lo, delta, width);
   asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "f"(lo), "f"(hi));
   return var;
}

static __device__ __inline__ double __shfl_xor(double var, int laneMask, int width=warpSize)
{
   float lo, hi;
   asm volatile("mov.b64 {%0,%1}, %2;" : "=f"(lo), "=f"(hi) : "d"(var));
   hi = __shfl_xor(hi, laneMask, width);
   lo = __shfl_xor(lo, laneMask, width);
   asm volatile("mov.b64 %0, {%1,%2};" : "=d"(var) : "f"(lo), "f"(hi));
   return var;
}
#endif

//===========================================================================================




//    Reduction functions: either warp reductions, or block reductions



//////////////////////////////////
//
//       Kepler - shuffles
//
/////////////////////////////////


/**
 * Performs a warp reduction where the warp can be split
 * into teams.  Reduction is within a team, not between teams.
 * Teams all have the same size, which must
 * be a power of 2.  This function is intended for Kepler cards
 *
 * Parameters:
 *   x :
 *      On Input, a value to reduce
 *      On Output, the value of the reduction across all threads
 *                 in the team
 *  teamSz :
 *      On Input: the number of threads in each team.
 *                Must be a 2,4,8,16 or 32
 */
template<typename FP>
static __inline__ __device__
void warp_allreduceplus(FP &x, int teamSz=32)
{
   for(int i=1; i<teamSz; i*=2) {
      FP z = __shfl_xor(x, i);
      x += z;
   }
}
template<typename FP>
static __inline__ __device__
void warp_allreducemax(FP &x, int teamSz=32)
{
   for(int i=1; i<teamSz; i*=2) {
      FP z = __shfl_xor(x, i);
      x = max(x,z);
   }
}


/**
 * Warp reduction function as above, but where each
 * thread now has C independent elements to reduce
 *
 * Parameters:
 *    x[C] :
 *       On Input, the values to reduce
 *       On Output, x[i] contains the reduction of all x[i]
 *                  values across the team (0 <= i < C)
 *   teamSz :
 *       On Input, the number of threads in each team.
 *                Must be a 2,4,8,16 or 32
 */
template<typename FP, int C>
static __inline__ __device__
void warp_allreduceplus(FP x[], int teamSz=32)
{
   FP z[C];
   for(int i=1; i<teamSz; i*=2) {
#pragma unroll
      for(int cc=0; cc<C; cc++) {
         z[cc] = __shfl_xor(x[cc], i);
      }

#pragma unroll
      for(int cc=0; cc<C; cc++) {
         x[cc] += z[cc];
      }

   }
}
template<typename FP, int C>
static __inline__ __device__
void warp_allreducemax(FP x[], int teamSz=32)
{
   FP z[C];
   for(int i=1; i<teamSz; i*=2) {
#pragma unroll
      for(int cc=0; cc<C; cc++) {
         z[cc] = __shfl_xor(x[cc], i);
      }

#pragma unroll
      for(int cc=0; cc<C; cc++) {
         x[cc] = max(x[cc],z[cc]);
      }
   }
}


/**
 * Warp reduction function where each
 * thread has C independent elements to reduce.
 * Only "thread 0" in each warp receives the final
 * answer.  This may be faster than the allreduce
 * above.
 *
 * Parameters:
 *    x[C] :
 *       On Input, the values to reduce
 *       On Output, x[i] for thread 0 in the warp
 *                  contains the reduction of all x[i]
 *                  values across the team (0 <= i < C).
 *                  The other threads in the warp
 *                  have undefined values in x[i]
 */
template<typename FP, int C>
static __inline__ __device__
void warp_reduceplus(FP x[])
{
   FP z[C];
   for(int i=1; i<warpSize; i*=2) {
#pragma unroll
      for(int cc=0; cc<C; cc++) {
         z[cc] = __shfl_down(x[cc], i);
      }
#pragma unroll
      for(int cc=0; cc<C; cc++) {
         x[cc] += z[cc];
      }
   }
}
template<typename FP, int C>
static __inline__ __device__
void warp_reducemax(FP x[])
{
   FP z[C];
   for(int i=1; i<warpSize; i*=2) {
#pragma unroll
      for(int cc=0; cc<C; cc++) {
         z[cc] = __shfl_down(x[cc], i);
      }
#pragma unroll
      for(int cc=0; cc<C; cc++) {
         x[cc] = max(x[cc],z[cc]);
      }
   }
}
template<typename FP, int C>
static __inline__ __device__
void warp_reducemin(FP x[])
{
   FP z[C];
   for(int i=1; i<warpSize; i*=2) {
#pragma unroll
      for(int cc=0; cc<C; cc++) {
         z[cc] = __shfl_down(x[cc], i);
      }
#pragma unroll
      for(int cc=0; cc<C; cc++) {
         x[cc] = min(x[cc],z[cc]);
      }
   }
}


/**
 * Warp reduction function.
 * Only "thread 0" in each warp receives the final
 * answer.  This may be faster than the allreduce
 * above.
 *
 * Parameters:
 *    x :
 *       On Input, the value to reduce
 *       On Output, x for thread 0 in the warp
 *                  contains the reduction of all x's
 *                  across the team
 *                  The other threads in the warp
 *                  have undefined values in x
 */
template<typename FP>
static __inline__ __device__
void warp_reduceplus(FP &x)
{
   FP z;
#pragma unroll
   for(int i=1; i<warpSize; i*=2) {
      z = __shfl_down(x, i);
      x += z;
   }
}
template<typename FP>
static __inline__ __device__
void warp_reducemax(FP &x)
{
   FP z;
#pragma unroll
   for(int i=1; i<warpSize; i*=2) {
      z = __shfl_down(x, i);
      x = max(x,z);
   }
}
template<typename FP>
static __inline__ __device__
void warp_reducemin(FP &x)
{
   FP z;
#pragma unroll
   for(int i=1; i<warpSize; i*=2) {
      z = __shfl_down(x, i);
      x = min(x,z);
   }
}

/**
 * Performs an exclusive scan up the warp
 * and returns the scan value.  Consider the 
 * i-th thread in a warp with i=0, 1, ..., 31
 * and suppose that 
 *    x = d_x[i]
 * Then this routine computes
 *
 *   int cusum = 0;
 *   for(int j=0; j<i; j++)
 *      cusum += d_x[j];
 *
 *   return cusum;
 *
 * In other words the routine returns the sum of
 * all x values for all preceding threads in the warp.
 * The sum does not include the x value for this thread.
 */
template<typename FP>
   static __inline__ __device__
FP warp_exclusive_scanplus(const FP &x)
{
   const int tid = threadIdx.x;
   const int laneId = tid & 0x1F;
   FP cusum = __shfl_up(x, 1);
   // Lane 0 must not initialise cusum
   if(laneId == 0) cusum = 0;

   for (int i=1; i<=32; i*=2) {
      int n = __shfl_up(cusum, i);
      if (laneId >= i) cusum += n;
   }
   return cusum;
}





/**
 * Performs an exclusive scan of the true/false predicate
 * up the warp and returns the scan value.  Consider the 
 * i-th thread in a warp with i=0, 1, ..., 31
 * and suppose that 
 *    predicate = d_predicate[i]
 * Then this routine computes
 *
 *   int cusum = 0;
 *   for(int j=0; j<i; j++)
 *      cusum += (d_predicate[j] ? 1 : 0);
 *
 *   return cusum;
 *
 * In other words the routine returns the sum of
 * all true predicate values for all preceding threads in the warp.
 * The sum does not include the predicate value for this thread.
 */
static __inline__ __device__
int warp_exclusive_scanplus(const bool &predicate)
{
   const int tid = threadIdx.x;
   const int laneId = tid & 0x1F;

   unsigned int ballot = __ballot(predicate);
   // A 1 for all preceding threads in the warp
   unsigned int mask = (1 << laneId) - 1;
   int cusum = __popc( ballot & mask );

   return cusum;
}




/**
 * Performs an exclusive scan up the block
 * and returns the scan value.  Consider the 
 * i-th thread in a block with i=0, 1, ..., blockDim.x-1
 * and suppose that 
 *    x = d_x[i]
 * Then this routine computes
 *
 *   cusum = 0;
 *   for(int j=0; j<i; j++)
 *      cusum += d_x[j];
 *  
 *   total = cusum;
 *   for(int j=i; j<blockDim.x; j++)
 *      total += d_x[j];
 *
 * In other words the routine returns the sum of
 * all x values for all preceding threads in the block.
 * The sum does not include the x value for this thread.
 * The value 'total' is the sum over all threads in the block
 *
 * NB: ALL THREADS MUST CALL THIS SINCE IT HAS __syncthreads 
 */
template<typename FP>
   static __inline__ __device__
void block_exclusive_scanplus(const FP &x, FP & cusum, FP & total, volatile FP *shmem)
{  
   cusum = warp_exclusive_scanplus(x);   
   const int tid = threadIdx.x;
   const int nwarps = blockDim.x / warpSize;
   const int wid = tid / warpSize;

   // This is necessary 'cause we have to wait until all threads
   // are finished with the shared mem array
   __syncthreads();

   // Elect the leader
   //int leader = __ffs( __ballot(1) ) - 1;
   if(tid % warpSize == warpSize-1)
      shmem[wid] = cusum + x;

   __syncthreads();

   if(tid==0) {
      for(int i=1; i<nwarps; i++) {
         shmem[i] += shmem[i-1];
      }
   }
   __syncthreads();

   FP y = (wid>0 ? shmem[wid-1] : FP(0));
   total = shmem[nwarps-1];
   cusum += y;
}

/**
 * Block reduction function where each
 * thread in the block has C independent elements to reduce.
 * Only thread 0 in the block receives the final
 * answer.
 *
 * Parameters:
 *    x[C] :
 *       On Input, the values to reduce across the thread block
 *       On Output, threadIdx.x=0 contains the reduced values.
 *       All other thread in the block have undefined values
 *    shmem[nwarps*C] :
 *       Workspace, shared memory with C elements for
 *       every warp
 */
template<typename FP, int C>
static __inline__ __device__
void block_reduceplus(FP x[], volatile FP * shmem)
{
   warp_reduceplus<FP,C>(x);
   if(threadIdx.x % warpSize ==0) {
      for(int cc=0; cc<C; cc++) {
         shmem[C * threadIdx.x/warpSize + cc] = x[cc];
      }
   }
   __syncthreads();

   if(threadIdx.x==0) {
      // Start reading from 1 so that we don't have to zero x[]
      for(int w=1; w<blockDim.x / warpSize; w++) {
         for(int cc=0; cc<C; cc++) {
            x[cc] += shmem[C*w + cc];
         }
      }
   }

   __syncthreads();

}
template<typename FP, int C>
static __inline__ __device__
void block_reducemax(FP x[], volatile FP * shmem)
{
   warp_reducemax<FP,C>(x);
   if(threadIdx.x % warpSize ==0) {
      for(int cc=0; cc<C; cc++) {
         shmem[C * threadIdx.x/warpSize + cc] = x[cc];
      }
   }
   __syncthreads();

   if(threadIdx.x==0) {
      // Start reading from 1 so that we don't have to zero x[]
      for(int w=1; w<blockDim.x / warpSize; w++) {
         for(int cc=0; cc<C; cc++) {
            x[cc] = max(x[cc], shmem[C*w + cc] );
         }
      }
   }

   __syncthreads();

}
template<typename FP, int C>
static __inline__ __device__
void block_reducemin(FP x[], volatile FP * shmem)
{
   warp_reducemin<FP,C>(x);
   if(threadIdx.x % warpSize ==0) {
      for(int cc=0; cc<C; cc++) {
         shmem[C * threadIdx.x/warpSize + cc] = x[cc];
      }
   }
   __syncthreads();

   if(threadIdx.x==0) {
      // Start reading from 1 so that we don't have to zero x[]
      for(int w=1; w<blockDim.x / warpSize; w++) {
         for(int cc=0; cc<C; cc++) {
            x[cc] = min(x[cc], shmem[C*w + cc] );
         }
      }
   }

   __syncthreads();

}


/**
 * Block reduction function.  Only thread 0 in
 * the block receives the final answer.
 *
 * Parameters:
 *    x :
 *       On Input, the values to reduce across the thread block
 *       On Output, threadIdx.x=0 contains the reduced values.
 *       All other thread in the block have undefined values
 *    shmem[nwarps] :
 *       Workspace, shared memory with one element for
 *       every warp
 */
template<typename FP>
static __inline__ __device__
void block_reduceplus(FP &x, volatile FP * shmem)
{
   warp_reduceplus<FP,1>(&x);

   const int warpsize = 32;
   const int tid = threadIdx.x + threadIdx.y*blockDim.x;
   const int nthds = blockDim.x*blockDim.y;

   if(tid % warpsize ==0) {
      shmem[tid/warpsize] = x;
   }
   __syncthreads();

   if(tid==0) {
      // Start reading from 1 so that we don't have to zero x[]
      for(int w=1; w<nthds / warpsize; w++) {
         x += shmem[w];
      }
   }

   __syncthreads();

}
template<typename FP>
static __inline__ __device__
void block_reducemax(FP &x, volatile FP * shmem)
{
   warp_reducemax<FP,1>(&x);

   const int warpsize = 32;
   if(threadIdx.x % warpsize ==0) {
      shmem[threadIdx.x/warpsize] = x;
   }
   __syncthreads();

   if(threadIdx.x==0) {
      // Start reading from 1 so that we don't have to zero x[]
      for(int w=1; w<blockDim.x / warpsize; w++) {
         x = max(x, shmem[w]);
      }
   }

   __syncthreads();

}
template<typename FP>
static __inline__ __device__
void block_reducemin(FP &x, volatile FP * shmem)
{
   warp_reducemin<FP,1>(&x);

   const int warpsize = 32;
   if(threadIdx.x % warpsize ==0) {
      shmem[threadIdx.x/warpsize] = x;
   }
   __syncthreads();

   if(threadIdx.x==0) {
      // Start reading from 1 so that we don't have to zero x[]
      for(int w=1; w<blockDim.x / warpsize; w++) {
         x = min(x, shmem[w]);
      }
   }

   __syncthreads();

}











/***************************   Some specialised reducemax functions   *****************************/


/**
 * Does a max reduction on the value mx
 * while also keeping track of the maximal
 * idx.
 *
 * Parameters :
 *   mx :
 *      On Input, the value to reduce
 *      On Output, the maximal value across all threads in the warp
 *   idx :
 *      On Input, an integer associated with mx
 *      On Output, the integer associated with the maximal mx value
 *                across all threads in the warp
 */
template<typename FP>
static __inline__ __device__
void warp_allreducemax(FP &mx, int &idx)
{
   for(int i=1; i<warpSize; i*=2) {
      FP z = __shfl_xor(mx, i);
      int zidx = __shfl_xor(idx,i);
      if(z > mx) {
         mx = z;
         idx = zidx;
      }
   }
}



/**
 * Does a max reduction on the value mx
 * while also keeping track of the maximal
 * idx. Only "thread 0" in the warp gets the
 * maximal value and index.  This is potentially
 * faster than an allreduce
 *
 * Parameters :
 *   mx :
 *      On Input, the value to reduce
 *      On Output, "thread 0" in the warp has the maximal
 *                 value across all threads in the warp.
 *                 All other threads in the warp have undefined
 *                 values
 *   idx :
 *      On Input, an integer associated with mx
 *      On Output, "thread 0" in the warp has the integer
 *                associated with the maximal mx value
 *                across all threads in the warp. All
 *                other threads in the warp have undefined
 *                values
 */
template<typename FP>
static __inline__ __device__
void warp_reducemax(FP &mx, int &idx)
{
   for(int i=1; i<warpSize; i*=2) {
      FP z = __shfl_down(mx, i);
      int zidx = __shfl_down(idx,i);
      if(z > mx) {
         mx = z;
         idx = zidx;
      }
   }
}

/**
 * Does a max reduction across all threads in a block on the value mx
 * while also keeping track of the maximal
 * idx.
 *
 * Parameters :
 *   mx :
 *      On Input, the value to reduce
 *      On Output, the maximal value across all threads in the block.
 *   idx :
 *      On Input, an integer associated with mx
 *      On Output, the integer associated with the maximal mx value
 *                across all threads in the block.
 *   shmem[nwarps] :
 *      Workspace in shared memory
 *   ishmem[nwarps] :
 *      Workspace in shared memory
 */
template<typename FP>
static __inline__ __device__
void block_allreducemax(FP &mx, int &idx, volatile FP * shmem, volatile int * ishmem)
{
   warp_reducemax(mx, idx);
   const int warpsize = 32;
   const int wid = threadIdx.x / warpsize;
   const int nwarps = blockDim.x / warpsize;

   if(threadIdx.x % warpsize == 0) {
      shmem[wid] = mx;
      ishmem[wid] = idx;
   }

   __syncthreads();

   if(threadIdx.x == 0) {
      for(int i=1; i<nwarps; i++) {
         FP x = shmem[i];
         if(x > mx) {
            mx = x;
            idx = ishmem[i];
         }
      }
      shmem[0] = mx;
      ishmem[0] = idx;
   }
   __syncthreads();
   mx = shmem[0];
   idx = ishmem[0];
   __syncthreads();
}


//////////////////////////////////
//
//       Fermi - use shared mem
//
/////////////////////////////////


/**
 * Performes a warp reduction where the warp can be split
 * into teams.  Reduction is within a team, not between teams.
 * Teams all have the same size, which must
 * be a power of 2.  This function is indended for Fermi
 * cards and requires in input array of shared memory.
 *
 * Parameters :
 *   x :
 *      On Input, a value to reduce
 *      On Output, the value of the reduction across the team
 *   shared[teamSz] :
 *      Workspace, an array of shared memory for this team.
 *   tid :
 *      On Input, the id of this thread in the team
 *   teamSz :
 *      On Input, the number of threads in each team. Must be
 *                either 2,4,8,16, or 32
 */
template<typename FP>
static __inline__ __device__
void warp_allreduceplusFermi(FP &x, volatile FP * shared, const int tid=threadIdx.x, int teamSz=32)
{
   shared[tid] = x;
   if(tid < teamSz/2) {
      if(teamSz > 16) shared[tid] += shared[tid + 16];
      if(teamSz > 8)  shared[tid] += shared[tid + 8];
      if(teamSz > 4)  shared[tid] += shared[tid + 4];
      if(teamSz > 2)  shared[tid] += shared[tid + 2];
      if(teamSz > 1)  shared[tid] += shared[tid + 1];
   }
   x = shared[0];
}



/**
 * As above, but each thread now has C independent elements
 * to reduce.  This routine is intended for Fermi cards.
 *
 * Parameters :
 *   x[C] :
 *      On Input, the values to reduce
 *      On Output, x[i] contains the reduction of all x[i] values across
 *                 the team (0 <= i < C)
 *   shared[teamSz] :
 *      Workspace, an array of shared memory for this team.
 *   tid :
 *      On Input, the id of this thread in the team
 *   teamSz :
 *      On Input, the number of threads in each team. Must be
 *                either 2,4,8,16, or 32
 */
template<typename FP, int C>
static __inline__ __device__
void warp_allreduceplusFermi(FP x[], volatile FP * shared, const int tid = threadIdx.x, int teamSz=32)
{
   for(int cc=0; cc<C; cc++) {
      warp_allreduceplusFermi<FP>(x[cc], shared, tid, teamSz);
   }
}




/**
 * Block reduction function where each
 * thread in the block has C independent elements to reduce.
 * Only thread 0 in the block receives the final
 * answer. This routine is for Fermi cards.
 *
 * NOTE: Must have C <= 32
 *
 * Parameters:
 *    x[C] :
 *       On Input, the values to reduce across the thread block
 *       On Output, threadIdx.x=0 contains the reduced values.
 *       All other thread in the block have undefined values
 *    shmem[nwarps*32] :
 *       Workspace, shared memory with one element for
 *       every thread
 */
template<typename FP, int C>
static __inline__ __device__
void block_reduceplusFermi(FP x[], volatile FP * shmem)
{
   const int wid = threadIdx.x / warpSize;
   const int tid = threadIdx.x % warpSize;

   warp_allreduceplusFermi<FP,C>(x, shmem + wid*warpSize, tid);
   __syncthreads();
   if(tid == 0) {
      for(int cc=0; cc<C; cc++) {
         shmem[C * wid + cc] = x[cc];
      }
   }
   __syncthreads();

   if(threadIdx.x==0) {
      // Start reading from 1 so that we don't have to zero x[]
      for(int w=1; w<blockDim.x / warpSize; w++) {
         for(int cc=0; cc<C; cc++) {
            x[cc] += shmem[C*w + cc];
         }
      }
   }

   __syncthreads();

}


/**
 * Block reduction function.
 * Only thread 0 in the block receives the final
 * answer. This routine is for Fermi cards.
 *
 * Parameters:
 *    x :
 *       On Input, the values to reduce across the thread block
 *       On Output, threadIdx.x=0 contains the reduced value.
 *       All other thread in the block have undefined value
 *    shmem[nwarps*32] :
 *       Workspace, shared memory with one element for
 *       every thread
 */
template<typename FP>
static __inline__ __device__
void block_reduceplusFermi(FP &x, volatile FP * shmem)
{
   const int wid = threadIdx.x / warpSize;
   const int tid = threadIdx.x % warpSize;

   warp_allreduceplusFermi<FP>(x, shmem + wid*warpSize, tid);
   __syncthreads();
   if(tid == 0) {
      shmem[wid] = x;
   }
   __syncthreads();

   if(threadIdx.x==0) {
      // Start reading from 1 so that we don't have to zero x[]
      for(int w=1; w<blockDim.x / warpSize; w++) {
         x += shmem[w];
      }
   }

   __syncthreads();

}


/**
 * Does a max reduction on the value mx
 * while also keeping track of the maximal
 * idx.  This routine is intended for Fermi cards.
 *
 * Parameters :
 *   mx :
 *      On Input, the value to reduce
 *      On Output, the maximal value across all threads in the warp
 *   idx :
 *      On Input, an integer associated with mx
 *      On Output, the integer associated with the maximal mx value
 *                across all threads in the warp
 *   shared[32] :
 *      Workspace, an array of shared memory with one element for each
 *                 thread in the warp.
 *   tid :
 *      On Input, the id of this thread in the warp
 */
template<typename FP>
static __inline__ __device__
void warp_allreducemaxFermi(FP &mx, int &idx, volatile FP * shared, const int tid)
{
   shared[tid] = mx;
   FP z, y=mx;
   if(tid<16) {
      z = shared[tid+16];
      if(z > y) {
         y = z;
      }
      shared[tid] = y;
      z = shared[tid+8];
      if(z > y) {
         y = z;
      }
      shared[tid] = y;
      z = shared[tid+4];
      if(z > y) {
         y = z;
      }
      shared[tid] = y;
      z = shared[tid+2];
      if(z > y) {
         y = z;
      }
      shared[tid] = y;
      z = shared[tid+1];
      if(z > y) {
         y = z;
      }
      shared[tid] = y;
   }
   // All threads read max value from shared[0]
   y = shared[0];
   // Write idx to shared
   volatile int * ish = (volatile int *)shared;
   ish[tid] = idx;
   // Which thread has the maximum value?
   int which = __ballot(y==mx);
   which = __ffs(which)-1;
   // Store maximum value to mx
   mx = y;
   // Get and store max idx
   idx = ish[which];
}

/**
 * Does a max reduction across all threads in a block on the value mx
 * while also keeping track of the maximal
 * idx. All threads in the block get the
 * maximal value and index.
 *
 * Parameters :
 *   mx :
 *      On Input, the value to reduce
 *      On Output, the maximal value across all threads in the block.
 *   idx :
 *      On Input, an integer associated with mx
 *      On Output, the integer associated with the maximal mx value
 *                across all threads in the block.
 *   shmem[nwarps*32] :
 *      Workspace
 */
template<typename FP>
static __inline__ __device__
void block_allreducemaxFermi(FP &mx, int &idx, volatile FP * shmem)
{
   const int wid = threadIdx.x / warpSize;
   const int tid = threadIdx.x % warpSize;

   warp_allreducemaxFermi(mx, idx, shmem + wid*warpSize, tid);
   const int nwarps = blockDim.x / warpSize;

   int * ishmem = (int*)(shmem + nwarps);

   __syncthreads();

   if(tid == 0) {
      shmem[wid] = mx;
      ishmem[wid] = idx;
   }

   __syncthreads();

   if(threadIdx.x == 0) {
      for(int i=1; i<nwarps; i++) {
         if(shmem[i] > mx) {
            mx = shmem[i];
            idx = ishmem[i];
         }
      }
      shmem[0] = mx;
      ishmem[0] = idx;
   }
   __syncthreads();
   mx = shmem[0];
   idx = ishmem[0];
   __syncthreads();
}








//============    Inline PTX intrinsics for decorating loads and stores   ===================
//
//
// These can help to improve use of the cache


#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __DEV_PTR   "l"
#else
#define __DEV_PTR   "r"
#endif

static __device__ __inline__ double ld_cs(const double & ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
   const double * pt = &ptr;
   double ret;
   asm volatile ("ld.global.cs.f64 %0, [%1];"  : "=d"(ret) : __DEV_PTR (pt));
   return ret;
#else
   return ptr;
#endif
}

static __device__ __inline__ float ld_cs(const float & ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
   const float * pt = &ptr;
   float ret;
   asm volatile ("ld.global.cs.f32 %0, [%1];"  : "=f"(ret) : __DEV_PTR (pt));
   return ret;
#else
   return ptr;
#endif
}

static __device__ __inline__ double ld_ca(const double & ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
   const double * pt = &ptr;
   double ret;
   asm volatile ("ld.global.ca.f64 %0, [%1];"  : "=d"(ret) : __DEV_PTR (pt));
   return ret;
#else
   return ptr;
#endif
}

static __device__ __inline__ float ld_ca(const float & ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
   const float * pt = &ptr;
   float ret;
   asm volatile ("ld.global.ca.f32 %0, [%1];"  : "=f"(ret) : __DEV_PTR (pt));
   return ret;
#else
   return ptr;
#endif
}


static __device__ __inline__ void st_cs(double & ptr, const double val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
   const double * pt = &ptr;
   asm volatile ("st.global.cs.f64 [%0], %1;"  : :  __DEV_PTR (pt), "d" (val) );
#else
   ptr = val;
#endif
}

static __device__ __inline__ void st_cs(float & ptr, const float val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
   const float * pt = &ptr;
   asm volatile ("st.global.cs.f32 [%0], %1;"  : :  __DEV_PTR (pt), "f" (val) );
#else
   ptr = val;
#endif
}


// Non-coherent cache functions
template<typename FP>
static __device__ __inline__ FP ld_nc_ca(FP & ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 320
   return __ldg(&ptr);
#else
   return ld_ca(ptr);
#endif
}
static __device__ __inline__ double ld_nc_cs(const double & ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 320
   const double * pt = &ptr;
   double ret;
   asm volatile ("ld.global.cs.nc.f64 %0, [%1];"  : "=d"(ret) : __DEV_PTR (pt));
   return ret;
#else
   return ld_cs(ptr);
#endif
}
static __device__ __inline__ float ld_nc_cs(const float & ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 320
   const float * pt = &ptr;
   float ret;
   asm volatile ("ld.global.cs.nc.f32 %0, [%1];"  : "=f"(ret) : __DEV_PTR (pt));
   return ret;
#else
   return ld_cs(ptr);
#endif
}


template<typename FP>
static __device__ __inline__ void prefetch_L2(const FP & ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
   const FP * pt = &ptr;
   asm volatile ("prefetch.global.L2 [%0];"  : : __DEV_PTR (pt));
#endif
}

#if defined(__DEV_PTR)
#undef __DEV_PTR
#endif




//==================================================================================
#endif  // end #ifdef __CUDACC__






/**
 * So this is more fiddling to work around the very complex NVCC tool chain.
 * 
 * According to NVIDIA:
 *  "The sm_60 architecture, that is newly supported in CUDA 8.0, has native fp64 atomicAdd function. Because of the 
 *   limitations of our toolchain and CUDA language, the declaration of this function needs to be present even when 
 *   the code is not being specifically compiled for sm_60. This causes a problem in your code because you also define 
 *   a fp64 atomicAdd function.
 *
 *   CUDA builtin functions such as atomicAdd are implementation-defined and can be changed between CUDA releases. 
 *   Users should not define functions with the same names as any CUDA builtin functions. We would suggest you to 
 *   rename your atomicAdd function to one that is not the same as any CUDA builtin functions."
 * 
 * Yeah well I'm not going to call the thing "myAtomicAdd" or whatever, so we do a rather hacky workaround which is to
 * declare the function, but not define it.  This issues a warning when compiled with CUDA 7.5, but
 * at least compilation succeeds.  There are no warnings when compiling with CUDA 8.0
 */ 
#if defined(__CUDACC__)
#if !defined(__CUDA_ARCH__)
static __inline__ __device__  double atomicAdd(double* address, double val);
#elif (__CUDA_ARCH__<600) 
static __inline__ __device__  double atomicAdd(double* address, double val)
{
   unsigned long long int* address_as_ull =
      (unsigned long long int*)address;
   unsigned long long int old = *address_as_ull, assumed;
   do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val +
               __longlong_as_double(assumed)));
   } while (assumed != old);
   return __longlong_as_double(old);
}
#endif
#endif

