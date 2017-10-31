#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>

#include <vector>
#include <chrono>
#include <cmath>

#include "bmp_support.hpp"
#include "utils.hpp"
#include "oskar_grid_wproj.hpp"
#include "gpu_support.hpp"

// A stupid experiment to estimate if it was useful to process 
// tiles with a small grid_w separately (so the wkernel data remains in L2).
// That does not seem to be the case probably because most tiles with a small 
// grid_w are already grouped (in the central box) so cache hit is already high 
#define CLIP_GRID_W  false
//#define CLIP_GRID_W  !(grid_w>200)




// The default 8+8 seems to be a good choice but other combinations may be interesting
// 
// Remark: Surprisingly, replacing registers by shmem does not seem to reduce register usage 
//         in my current best version. I suspect that ptxas notices that the shared memory 
//         elements are written and read multiple times without any possibility that 
//         the value was changed by another thread (because there is no barrier between the
//         reads and writes). 
//         I fixed that problem by introducing a fake __syncthread() at the end of the loop.
//         However, using shared memory is costing a few registers so the benefit of using 
//         shm is limited. So far, the best configuration appears to be 0+16 (no shared at all
//         but 72 registers) and 4+12 (a bit of shared and only 64 registers so optimal 50% 
//         occupancy).  
//           
//         Reminder: using less than 64 registers does not bing anything when using blocks
//                   of size 32x1 (on Pascal)
//
//         By default. prefer to keep the 4+12 configuration to always verify that the shm 
//         code is not broken by my changes
//
// As of now: 0+16 is good for VERS==6 but bad for VERS==1 
//            4+12 is good for both (the best?)
//
//         For version 1, the case 4+12 seems to be the best choice by far. 
//         The case 5+11 is 20% sloweron the outer tiles for the same register usage. Why? 
//         There must be a reason. Maybe some shared memory alignment issues 
//
#define GRIDDING_TILED_SHMSZ 8
#define GRIDDING_TILED_REGSZ 8

#define GRIDDING_TILED (GRIDDING_TILED_SHMSZ+ GRIDDING_TILED_REGSZ)

// TO INVESTIGATE:
//    Th profiling metrics shared_load_transactions_per_request and shared_store_transactions_per_request 
//    are significantly different for versions 6 and 1 ( ~2.0 vs ~1.5 ). I cannot figure why since 
//    both versions are supposed to access the shared memory exactly in the same way. 
//    However, shared_load_transactions and shared_store_transactions are reporting the exact same values.
//    This is very strange!!!!   
//    POSSIBLE ANSWER: Could it be that shared_load_transactions and shared_store_transactions are computed globally
//                     while shared_load_transactions_per_request and shared_store_transactions_per_request 
//                     are computed on a single SM or block.  
//


// Current version with manual software pipelining = 6
// Original version = 1 
//
// Initially, the version 6 was the best by far but some of the other
// small optimizations I introduced have a greater effect on version 1 so
// both versions now give similar performances.
// 
// In face, version 1 is now the fastest.
//
// Version 6 has the advantage of reducing the stalls caused by load 
// latencies but the code is slighly more complex and so generate 
// more instructions which is detrimental since the kernel is instruction
// bound. 
// 
#define VERS 1

// During gridding, check the range of j and exit early if outside bounds
// THAT DOES NOT SEEM TO BE GOOD FOR MY DYNAMIC VERSION!!!!!
#define EARLY_J_EXIT false

// Use float instead of double for the inner 'sum'  
// USE WITH CARE: That may introduce some rounding errors. 
#define FLOAT_SUM 1

// An attempt to split the float2 in shared memory into two float managed independently .
// In theory that could avoid a few bank conflits but that does not seem to be beneficial here.
#define SPLIT_SHMEM 0

// Control a small pointer optimization that may or not be beneficial 
// The benefit if any is small (~1%)
// UPDATE: THE BENEFIT IS SIGNIFICANT FOR THE DYNAMIC VERSION (~30ms) BUT ... THAT COULD JUST BE AN UNEXPECTED SIZE EFFECT ...
#define  PTR_OPTIM 1

// Control if some integral comptation must be done in a different way
// This is actually benefical for VERS==1 but not for VERS==6
// It is unclear why. 
// This is probably because the optimization introduces more instructions (so larger code and more instruction cache misses)
#define OFF_U_OPTIM (VERS==1)
//#define OFF_U_OPTIM 1

// Force the use of FMA in the gridding kernel
//     x += a*b + c*d 
//  => 
//     x += a*b 
//     x += c*d 
// That may cause rounding errors so the compiler won't do it by default (even with fast-math)
// The benefit is quite small but measurable (1% to 2%) 
#define FORCE_FMA 1
 
// An attempt to specialize the inner code for the two cases
//  conv_mul==-1.0 and conv_mul==+1.0
// This effectively saves one float operation but globally speaking
// this is not beneficial. CURRENTLY BROKEN!!!
// #define USE_CONV_MUL2 0

// Chose the number of warps per block during gridding (1 or 2)
// Warning: This is also going to affect  the tile size
// The default is to use 2 warps per block which makes sense on Kepler since 
// the number of block per SM is limited to 16 (so max occupancy of (2*32)*16 = 1024 = 50%.
// However, on Pascal, the limit was raised to 32 blocks per SM which mean that 
// 50% occupancy can now be achieved with blocks of size 32. 
// The main advantage of using a single warp is that we avoid the work imbalance between 
// the 2 warps. 
// The disavantage is that the tiles are smaller. 
// Be aware that in the current implementation the central box has a hardcoded size of 9x9 tiles 
// so reducing the tile size will also reduce the the size of the central box.
// 
// At the time I write those lines I get 
//
//   With 32x32 tiles (so NWARPS=2)
//      GPU processing tiles:    outer tiles is 754.954ms
//      GPU processing tiles:    central box is 151.815ms
//      GPU processing non-tiles:         0.008192ms
//      GPU overall gridding tile:        906.801ms
//      Total GPU processing time is 1154.11ms
//  With 32x16 tiles (so NWARPS=1)
//      GPU processing tiles:    outer tiles is 734.518ms
//      GPU processing tiles:    central box is 104.099ms   (REMINDER: not the same central box)
//      GPU processing non-tiles:         0.00816ms
//      GPU overall gridding tile:        838.647ms
//      Total GPU processing time is 1155.7ms
//
// So the overall gridding time is better the TOTAL time is the same because the 'initialisation' 
// is costing more (because of there is more tiles to prepare?)
//
// IDEA: Consider the idea of tiles of size 16x16 processed by a single warp. Each half warp would
//       process either the odd or the even values of 'j'. The main advantage of using tiles with a smaller 
//       width is to improve the warp efficiency (i.e. less divergence at the is_my_k test ) 
//       This is not very difficult to implement: Rename the current BlockDim.y into BlockDim.z to 
//       control the number of warps in each block, and use BlockDim.x*BlockDim.y=16*2=32 to control how the 
//       thread in each warp are distributed. The main advatnage of that approach is of course that
//       each thread only has to process 8 value in the j direction so register usage could be quite low.
//    
//

#define NWARPS 1


// Tile size 

#define TX 32
#define TY (NWARPS*GRIDDING_TILED)

// When that macro is defined, it is used as 'oversample' in the gridding kernel.
// That can lead to a few percents acceleration.
// Comment it out to use the 'unknown' value passed as argument
// In prouction code, the kernel should be specialized for multiple values of oversample (so a template argument?)
#if 1
#define ASSUME_OVERSAMPLE 4
#endif

/**
 * Make the grid hit intensity for a given kernel and set of visibilities.
 * 
 * The kernel must be launched with a 2D thread block with 32 threads in the X direction, 
 * i.e. 
 *        blockDim.x == 32
 * The kernel will process each visibility with a warp, so I need a whole number of warps.
 * We just use a standard 1D grid of any size
 */
__global__ void oskar_make_grid_hit_intensity(
      const int num_w_planes, 
      const int* support,       
      const int num_vis,
      const float* uu, 
      const float* vv,
      const float* ww, 
      const double cell_size_rad,
      const double w_scale, 
      const int grid_size, 
      int * grid_intensity)
{
   const int g_centre = grid_size / 2;
   const double scale = grid_size * cell_size_rad;

#if 1
   /* Compute the grid hit intensity
    */
   const int warpIdKernel = blockIdx.x * blockDim.y + threadIdx.y;
   const int nwarpsKernel = gridDim.x * blockDim.y;


   /* Loop over visibilities. */
   for (int i = warpIdKernel; i < num_vis; i+=nwarpsKernel)
   {
      /* Convert UV coordinates to grid coordinates. */
      const float pos_u = -uu[i] * scale;
      const float pos_v = vv[i] * scale;
      const float ww_i = ww[i];

      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
      if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      /* Catch points that would lie outside the grid. */
      const int wsupport = support[grid_w];
      if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
            grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
      {
         continue;
      }

      for (int j = -wsupport; j <= wsupport; ++j)
      {
         for (int k = -wsupport+threadIdx.x; k <= wsupport; k+=warpSize)
         {
            int p = (((grid_v + j) * grid_size) + grid_u + k);
            atomicAdd(grid_intensity+p, 1);
         }
      }
   }

#else
   /* Plot the visibility intensity, i.e. a intensity map of just the visibility points
    */
   const int tidKernel = blockIdx.x * blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x;
   const int nthdsKernel = gridDim.x * blockDim.y*blockDim.x;


   /* Loop over visibilities. */
   for (int i = tidKernel; i < num_vis; i+=nthdsKernel)
   {
      /* Convert UV coordinates to grid coordinates. */
      const float pos_u = -uu[i] * scale;
      const float pos_v = vv[i] * scale;
     const float ww_i = ww[i];

      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
      if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      /* Catch points that would lie outside the grid. */
      const int wsupport = support[grid_w];
      if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
            grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
      {
         continue;
      }

      int p = (((grid_v + 0) * grid_size) + grid_u + 0);
      atomicAdd(grid_intensity+p, 1);
   }

#endif
}




/**
 * Computing the box which bounds the area of grid that is actually updated.  We call this the 'active grid'.
 *
 * Launch kernel with standard 1D grid of 1D blocks, no restrictions.
 */
__global__ void oskar_get_bounding_box(
      const int num_w_planes, 
      const int* support,
      const int num_vis,
      const float* uu, 
      const float* vv,
      const float* ww, 
      const double cell_size_rad,
      const double w_scale, 
      const int grid_size, 

      Box * d_box,                  // The active grid
      int * num_skipped             // The number of visibilities skipped
      )
{
   const int g_centre = grid_size / 2;
   const double scale = grid_size * cell_size_rad;

   // Our local bounding box
   Point topLeft(2*grid_size, 2*grid_size);
   Point botRight(-1, -1);

   int loc_num_skipped = 0;

   // Need 1D thread block
   const int tid = threadIdx.x + blockIdx.x*blockDim.x;
   // We need 1D grid 
   const int nthds = blockDim.x*blockDim.y * gridDim.x;

   /* Loop over visibilities. */
   for (int i = tid; i < num_vis; i+=nthds)
   {
      /* Convert UV coordinates to grid coordinates. */
      const float pos_u = -uu[i] * scale;
      const float pos_v = vv[i] * scale;
      const float ww_i = ww[i];

      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
      if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      /* Catch points that would lie outside the grid. */
      const int wsupport = support[grid_w];
      if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
            grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
      {
         loc_num_skipped++;
         continue;
      }

      int u = grid_u - wsupport;
      int v = grid_v - wsupport;
      topLeft.u = (u < topLeft.u ? u : topLeft.u);
      topLeft.v = (v < topLeft.v ? v : topLeft.v);

      u = grid_u + wsupport;
      v = grid_v + wsupport;
      botRight.u = (u > botRight.u ? u : botRight.u);
      botRight.v = (v > botRight.v ? v : botRight.v);
   }

   // Now need to reduce these quantities across all the threads in the block
   __shared__ int shmem[2*32];

   int point[2] = {topLeft.u, topLeft.v};
   block_reducemin<int,2>(point, shmem);
   topLeft.u = point[0];  topLeft.v = point[1];

   point[0] = botRight.u;
   point[1] = botRight.v;
   block_reducemax<int,2>(point, shmem);
   botRight.u = point[0];  botRight.v = point[1];

   block_reduceplus<int>(loc_num_skipped, shmem);

   // Thread 0 now atomically updates results in GDDR
   if(threadIdx.x==0 && threadIdx.y==0) {
      atomicMin(&(d_box[0].topLeft.u), topLeft.u);
      atomicMin(&(d_box[0].topLeft.v), topLeft.v);

      atomicMax(&(d_box[0].botRight.u), botRight.u);
      atomicMax(&(d_box[0].botRight.v), botRight.v);

      atomicAdd(num_skipped, loc_num_skipped);
   }
}



/**
 * Runs through all the visibilities and counts how many fall into each tile.
 * This is written to numPointsInTiles.
 * 
 * Launch with a regular 1D grid with 1D blocks.
 */
__global__ void oskar_count_elements_in_tiles(
      const int num_w_planes, 
      const int* support,
      const int oversample, 
      const int num_vis,
      const float* uu, 
      const float* vv,
      const float* ww, 
      const double cell_size_rad,
      const double w_scale, 
      const float grid_size, 

      const Box   tileBox,          // The tile box describing the region to be tiled
      const Point tileSize,         // The size of a tile
      const Point numTiles,         // Number of tiles in the U and V directions
      int * numPointsInTiles        // Array holding the number of points in each tile
      )
{
   const int g_centre = int(grid_size) / 2;
   const double scale = grid_size * cell_size_rad;

   const int tid = blockIdx.x * blockDim.x + threadIdx.x;
   const int nthds = gridDim.x * blockDim.x;

#define NUM_POINTS_IN_TILES(uu, vv)  numPointsInTiles[ (uu) + (vv)*numTiles.u ]
#define NUM_POINTS_OUTSIDE_TILES numPointsInTiles[ numTiles.v*numTiles.u ]

   const float tileWidth = float(tileSize.u);
   const float tileHeight = float(tileSize.v);


   /* Loop over visibilities. */
   for (int i = tid; i < num_vis; i+=nthds)
   {
      /* Convert UV coordinates to grid coordinates. */
      const float pos_u = -uu[i] * scale;
      const float pos_v = vv[i] * scale;
      const float ww_i = ww[i];


      const float grid_u = (int)round(pos_u) + g_centre;
      const float grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
      if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      /* Catch points that would lie outside the grid. */
      const float wsupport = support[grid_w];

      if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
            grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
      {
         continue;
      }

      if (CLIP_GRID_W) continue ; 

      /* This is a fiddly calculation.  We know each visibility will intersect one
       * or more tiles, and we know the set of tiles that are intersected will 
       * form a rectangle (since we have rectangles intersecting a square). So we now
       * try to find the start and end (U,V) tile coordinates which define this 
       * rectangle.  We compute the following in floating point:
       *      tileBox.topLeft.u + pu1*tileWidth <= grid_u - wsupport
       *                                           grid_u + wsupport <= tileBox.topLeft.u + pu2*tileWidth - 1
       */
      float fu1 = float(grid_u - wsupport - tileBox.topLeft.u)/tileWidth;
      float fu2 = float(grid_u + wsupport - tileBox.topLeft.u + 1.0f)/tileWidth;
      // Intersect [fu1, fu2] with [0, numTiles.u)
      float fu_int[] = { (fu1<0.0f ? 0.0f: fu1), (fu2>numTiles.u ? numTiles.u : fu2) };
      int u_int[] = { int( floor(fu_int[0]) ), int( ceil(fu_int[1]) ) };

      float fv1 = float(grid_v - wsupport - tileBox.topLeft.v)/tileHeight;
      float fv2 = float(grid_v + wsupport - tileBox.topLeft.v + 1)/tileHeight;
      // Intersect [fv1, fv2] with [0, numTiles.v)
      float fv_int[] = { (fv1<0.0f ? 0.0f: fv1), (fv2>numTiles.v ? numTiles.v : fv2) };
      int v_int[] = { int( floor(fv_int[0]) ), int( ceil(fv_int[1]) ) };


      for (int pv=v_int[0]; pv < v_int[1]; pv++)
      {
         for (int pu = u_int[0]; pu < u_int[1]; pu++)
         {
            atomicAdd(&NUM_POINTS_IN_TILES(pu, pv), 1);
         }
      }
      // Now need to check whether this grid point would also have hit grid areas
      // not covered by any tiles.
      if(   grid_u-wsupport < tileBox.topLeft.u ||
            grid_u+wsupport >= tileBox.botRight.u ||
            grid_v-wsupport < tileBox.topLeft.v ||
            grid_v+wsupport >= tileBox.botRight.v )
      {
         atomicAdd(&NUM_POINTS_OUTSIDE_TILES, 1);
      }

   } // END of visibilities loop

#undef NUM_POINTS_IN_TILES
#undef NUM_POINTS_OUTSIDE_TILES
}






/**
 * This is just a simple scan up the numPointsInTiles array.  Since the array is small
 * we just use one thread block.  The scan is written to offsetsPointsInTiles as
 * well as wk_offsetsPointsInTiles - this second copy will be used by the bucket sort
 *
 * Launch with 1 thread block with 1024 threads.
 */
__global__ void oskar_accumulate_tile_offsets(
      const Point numTiles,                     // Number of tiles
      const int * numPointsInTiles,             // The number of points in each tile
      int * offsetsPointsInTiles,               // Starting indexes of visibilities in each tile
      int * wk_offsetsPointsInTiles             // Copy of the previous array
      )
{
   __shared__ int intshmem[32];

   const int nTiles = numTiles.u*numTiles.v + 1;
   int runningTotal = 0;
   // Because we have __syncthreads in the loop, we need all the threads
   // to do the same number of iterations.  Call me paranoid, but ...
   int nloops = (nTiles + blockDim.x - 1) / blockDim.x;

   int idx = threadIdx.x;
   for(int i = 0; i<=nloops; i++)
   {
      int n = 0;
      if(idx < nTiles)
         n = numPointsInTiles[idx];

      int cusum, total;
      block_exclusive_scanplus(n, cusum, total, intshmem);

      cusum += runningTotal;
      if(idx < nTiles + 1) {
         offsetsPointsInTiles[idx] = cusum;
         wk_offsetsPointsInTiles[idx] = cusum;
      }

      idx += blockDim.x;
      runningTotal += total;
   }   
}





/**
 * A thoroughly deserving name.
 * Exercise to the reader : this can only go faster.  Make it so ...
 *
 * Does a bucket sort on the input visibilities.  Each tile is a bucket.
 * There is also a final bucket for any visibilities that fall outside
 * the tiled region.  This is to support the use case where only part 
 * of the active grid (or none of the active grid) is tiled.
 *
 * Launch this kernel with a regular 1D grid of 1D blocks, nothing special.
 */
__global__ void oskar_worlds_least_efficient_bucket_sort(
      const int num_w_planes, 
      const int* support,
      const int num_vis,
      const float* uu, 
      const float* vv,
      const float* ww, 
      const float* vis, 
      const float* weight, 
      const double cell_size_rad,
      const double w_scale, 
      const int grid_size, 

      const Box                        tileBox,                   // The grid region that is tiled
      const Point                      tileSize,                  // The tile size
      const Point                      numTiles,                  // Number of tiles in U and V directions
      const int * __restrict const     numPointsInTiles,          // Number of visibilities in each tile and no tiles
      int *                            offsetsPointsInTiles,      // Start of visibility data for each tile.  This will be 
                                                                  // modified by this routine!
      float *                          bucket_uu,                 // Output: bucket sorted values
      float *                          bucket_vv,                 // Output: bucket sorted values
      float *                          bucket_ww,                 // Output: bucket sorted values
      float2 *                         bucket_vis,                // Output: bucket sorted values
      float *                          bucket_weight,             // Output: bucket sorted values
      int2 *                           bucket_tile                // Output: bucket sorted values
      )
{
   const int g_centre = grid_size / 2;
   const double scale = grid_size * cell_size_rad;

#define NUM_POINTS_IN_TILES(uu, vv)  numPointsInTiles[ (uu) + (vv)*numTiles.u]
#define OFFSETS_IN_TILES(uu, vv)     offsetsPointsInTiles[ (uu) + (vv)*numTiles.u ]
#define OFFSETS_OUTSIDE_TILES        offsetsPointsInTiles[ numTiles.u*numTiles.v ]

   const int nthds = gridDim.x * blockDim.x;
   const int tid   = blockIdx.x * blockDim.x + threadIdx.x;


   /* Loop over visibilities. */
   for (int i = tid; i < num_vis; i+=nthds)
   {
      /* Convert UV coordinates to grid coordinates. */
      float pos_u = -uu[i] * scale;
      float pos_v = vv[i] * scale;
      float ww_i = ww[i];

      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
      if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      /* Skip points that would lie outside the grid. */
      const int wsupport = support[grid_w];
      if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
            grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
      {
         continue;
      }

      if (CLIP_GRID_W) continue ; 

      /* This is a fiddly calculation.  We know each visibility will intersect one
       * or more tiles, and we know the set of tiles that are intersected will 
       * form a rectangle (since we have rectangles intersecting a square). So we now
       * try to find the start and end (U,V) tile coordinates which define this 
       * rectangle.  We compute the following in floating point:
       *      tileBox.topLeft.u + pu1*tileWidth <= grid_u - wsupport
       *                                           grid_u + wsupport <= tileBox.topLeft.u + pu2*tileWidth - 1
       */
      float fu1 = float(grid_u - wsupport - tileBox.topLeft.u)/tileSize.u;
      float fu2 = float(grid_u + wsupport - tileBox.topLeft.u + 1)/tileSize.u;
      // Intersect [fu1, fu2] with [0, numTiles.u)
      float fu_int[] = { (fu1<0.0f ? 0.0f: fu1), (fu2>numTiles.u ? numTiles.u : fu2) };
      int u_int[] = { (int)floor(fu_int[0]), (int)ceil(fu_int[1]) };

      float fv1 = float(grid_v - wsupport - tileBox.topLeft.v)/tileSize.v;
      float fv2 = float(grid_v + wsupport - tileBox.topLeft.v + 1)/tileSize.v;
      // Intersect [fv1, fv2] with [0, numTiles.v)
      float fv_int[] = { (fv1<0.0f ? 0.0f: fv1), (fv2>numTiles.v ? numTiles.v : fv2) };
      int v_int[] = { (int)floor(fv_int[0]), (int)ceil(fv_int[1]) };


      for (int pv=v_int[0]; pv < v_int[1]; pv++)
      {
         for (int pu = u_int[0]; pu < u_int[1]; pu++)
         {
            // Get current offset and increment offset by one
            int off = atomicAdd(&OFFSETS_IN_TILES(pu, pv), 1);

            bucket_uu[off] = uu[i]; 
            bucket_vv[off] = vv[i]; 
            bucket_ww[off] = ww[i]; 

            float2 v;
            v.x = vis[2 * i];
            v.y = vis[2 * i + 1];
            bucket_vis[off] = v; 
            bucket_weight[off] = weight[i];
            int2 pu_pv = { pu,pv } ;
            pu_pv.x = pu ;
            pu_pv.y = pv ;
            bucket_tile[off] = pu_pv ;
         }
      }

      // Now need to check whether this grid point would also have hit grid areas
      // not covered by any tiles.
      if(   grid_u-wsupport < tileBox.topLeft.u ||
            grid_u+wsupport >= tileBox.botRight.u ||
            grid_v-wsupport < tileBox.topLeft.v ||
            grid_v+wsupport >= tileBox.botRight.v )
      {
         // Get current offset and increment offset by one
         int off = atomicAdd(&OFFSETS_OUTSIDE_TILES, 1);

         bucket_uu[off] = uu[i]; 
         bucket_vv[off] = vv[i]; 
         bucket_ww[off] = ww[i]; 

         float2 v;
         v.x = vis[2 * i];
         v.y = vis[2 * i + 1];
         bucket_vis[off] = v; 
         bucket_weight[off] = weight[i];
      }

   } // END loop over visibilities

#undef NUM_POINTS_IN_TILES
#undef OFFSETS_IN_TILES
#undef OFFSETS_OUTSIDE_TILES     
}






/**
 * Process all visibilities that fall outside the tiled region.  This supports the use case
 * where part of the active region is tiled, but not all of it.  The extreme case is when
 * the active region is not tiled at all, when all visibilities are processed by this kernel.
 * This is in fact faster on pre-Pascal architectures.
 *
 * Launch this kernel with a 1D grid of 2D blocks.  We must have 
 *
 *          blockDim.x == 32
 *
 * We can set blockDim.y to any value.  This kernel uses a whole warp to 
 * process each visibility.
 */
__global__ void oskar_process_visibilities_outside_tile_box(
      const int num_w_planes, 
      const int* support,

      const int oversample, 
      const int * const __restrict        compacted_wkernel_start_idx,   // Start index of each convolution kernel
      const float2 * const __restrict     compacted_wkernels,            // The compacted convolution stack

      const double                        cell_size_rad,
      const double                        w_scale, 
      const int                           grid_size, 

      const Box                           boundingBox,
      const Box                           tileBox,
      const Point                         numTiles,

      const int * __restrict const numPointsInTiles,
      const int * __restrict const offsetsPointsInTiles,
      const float * __restrict const  bucket_uu,
      const float * __restrict const  bucket_vv,
      const float * __restrict const  bucket_ww,
      const float2 * __restrict const bucket_vis,
      const float * __restrict const  bucket_weight,

      double* norm, 
      float * grid,
      const int ld_grid
      )
{
#define OFFSETS_OUTSIDE_TILES        offsetsPointsInTiles[ numTiles.u*numTiles.v ]
#define NUM_POINTS_OUTSIDE_TILES     numPointsInTiles[ numTiles.u*numTiles.v ]
   const int g_centre = grid_size / 2;
   const double scale = grid_size * cell_size_rad;

   // Index of this warp in the whole kernel
   const int warpIdKernel = blockIdx.x * blockDim.y + threadIdx.y;
   // Number of warps in the kernel
   const int nwarpsKernel = gridDim.x * blockDim.y;

   double loc_norm = 0;
   const int tileOffset = OFFSETS_OUTSIDE_TILES;
   const int num_vis = NUM_POINTS_OUTSIDE_TILES;

   /* Loop over visibilities. */
   for (int i = warpIdKernel; i < num_vis; i+=nwarpsKernel)
   {
      double sum;

      /* Convert UV coordinates to grid coordinates. */
      const float pos_u = -bucket_uu[tileOffset+i] * scale;
      const float pos_v = bucket_vv[tileOffset+i] * scale;
      const float ww_i = bucket_ww[tileOffset+i];
      const float w    = bucket_weight[tileOffset+i];
      float2 val = bucket_vis[tileOffset+i];
      val.x *= w;
      val.y *= w;


      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
      if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      const int wsupport = support[grid_w];
      const int wkernel_start = compacted_wkernel_start_idx[grid_w];
      const int wkernel_size = oversample/2 + wsupport*oversample + 1;


      /* Scaled distance from nearest grid point. */
      const int off_u = (int)round(  (round(pos_u)-pos_u) * oversample);   // \in [-oversample/2, oversample/2]
      const int off_v = (int)round( (round(pos_v)-pos_v) * oversample);    // \in [-oversample/2, oversample/2]

      /* Convolve this point. */
      sum = 0.0;
      float conv_mul = (ww_i > 0 ? -1.0f : 1.0f);

      // Nicer indexing in the wkernel.  We only change indexing in u, not v
      int abs_offu = (off_u < 0 ? -off_u : off_u);
      if(abs_offu == 0) abs_offu = oversample/2;
      int wkernel_row_off = (abs_offu-1)*(wsupport+wsupport+1);
      if(off_u == oversample/2 || off_u==-oversample/2)
         wkernel_row_off += wsupport+1;

      // Now need to clamp iteration range to exclude the tile box
      // We want
      //     grid_u + k <  tileBox.topLeft.u  or
      //     grid_u + k >= tileBox.botRight.u
      // Our grid iteration range is
      //       grid_u + k
      // for -wsupport <= k <= wsupport and 
      //       grid_v + j
      // for -wsupport <= j <= wsupport
      for (int j = -wsupport; j <= wsupport; ++j)
      {
         // Assume we are not intersecting the tile box
         int kstart = -wsupport, kend = wsupport;
         // Check if we intersect
         if(tileBox.topLeft.v<=grid_v + j && grid_v+j<tileBox.botRight.v) {
            // Clamp k iteration interval
            kstart = (grid_u+kstart >= tileBox.topLeft.u   ?  max(tileBox.botRight.u-grid_u, kstart) : kstart);
            kend   = (grid_u+kend    < tileBox.botRight.u  ?  min(tileBox.topLeft.u-grid_u-1, kend)   : kend);
         }

         int iy = abs(off_v + j * oversample);
         // Parallelise this loop over all threads in a warp
         for (int k = kstart+threadIdx.x; k <= kend; k+=warpSize)
         {
            // More indexing calculations for the wkernel
            int myk = k;
            if(off_u > 0) myk = -myk;
            int wkernel_u_idx;
            if(off_u == 0) {
               wkernel_u_idx = wsupport - abs(myk);
            }
            else if(abs_offu == oversample/2) {
               wkernel_u_idx = wsupport - abs(myk) + (myk >= 1 ? 1 : 0);
            }
            else {
               wkernel_u_idx = wsupport + myk;
            }
            float2 c = compacted_wkernels[wkernel_start + iy*wkernel_size + wkernel_row_off + wkernel_u_idx];
            c.y *= conv_mul;

            sum += c.x; /* Real part only. */
            int p = 2 * (((grid_v-boundingBox.topLeft.v + j) * ld_grid) + grid_u-boundingBox.topLeft.u + k);
            atomicAdd(grid+p,     (val.x*c.x - val.y*c.y) );
            atomicAdd(grid+p + 1, (val.y*c.x + val.x*c.y) );
         }
      }
      warp_reduceplus(sum);
      loc_norm += sum * w;
   }

   __shared__ double shmem[32];
   {
      if(threadIdx.x == 0) shmem[threadIdx.y] = loc_norm;
      __syncthreads();
      if(threadIdx.x==0 && threadIdx.y==0) {
         loc_norm = shmem[0];
         for(int i=1; i<blockDim.y; i++) loc_norm += shmem[i];
         atomicAdd(norm, loc_norm);
      }
   }

#undef OFFSETS_OUTSIDE_TILES       
#undef NUM_POINTS_OUTSIDE_TILES    
}







/**
 * Process all the tiles in the tile box, but skip any tiles in the exclusion box.
 * If the exclusion box is empty, then we will process all the tiles.
 *
 * Each thread block processes visibilities in one tile only.  Within a thread block, the parallelism
 * is over the elements in the tile that are updated (so all threads in a thread block process 
 * the same visibilities).  We can assign multiple thread blocks to each tile so that the work load 
 * for a given tile is spread over multiple blocks.
 *
 * Launch with a 3D grid of 2D thread blocks.
 *   gridDim.x  : the number of blocks which will process each tile, i.e. parallelise the visibility
 *                calculation for a given tile across gridDim.x thread blocks
 *   gridDim.y  : the number of tiles in the U direction
 *   gridDim.z  : the number of tiles in the V direction
 *  
 *   blockDim.x : this must be the same as tileSize.u
 *   blockDim.y : this must satisfy the constraint blockDim.y*(REGSZ+SHMSZ) == tileSize.v
 */

template<int REGSZ, int SHMSZ>
 __global__ void oskar_process_tiles_excluding_box(
      const int                           num_w_planes, 
      const int*     const __restrict     support,
      const int                           oversample_, 
      const int*     const __restrict     compact_wkernel_start_idx,    // Start index of each convolution kernel
      const float2 * const __restrict     compact_wkernel,              // The compacted convolution stack
      const double                        cell_size_rad,
      const double                        w_scale, 
      const int                           grid_size, 

      const Box                           boundingBox,                  // Bounding box defining the active grid region
      const Box                           tileBox,                      // Region of the active grid that is tiled.  We actually
                                                                        // .. only need the top left corner
      const Point                         tileSize,                     // The tile size. This can be inferred from blockDim
      const Point                         numTiles,                     // Number of tiles in U and V directions.  The tiled
                                                                        // .. region starts at tileBox.topLeft

      const Point                         offsetInTileIdx,              // An offset, in tile index space, that can be applied when
                                                                        // .. each thread block works out the tile that it should
                                                                        // .. process.  This will either be (0,0) or else will be
                                                                        // .. the top left corner of the exclusion box when this
                                                                        // .. kernel is used to process only the tiles in the 
                                                                        // .. exclusion box, and in that case exclusionBoxInTileIdx
                                                                        // .. will be empty
      const Box                           exclusionBoxInTileIdx,        // The region of tile space to exclude

      const int *    const __restrict     d_numPointsInTiles,           // Number of visibilities in each tile
      const int *    const __restrict     d_offsetsPointsInTiles,       // Start index of buckets, where visibility data in each
                                                                        // .. tile start
      const float *  const __restrict     d_bucket_uu,                  // Bucket sorted visibility data
      const float *  const __restrict     d_bucket_vv,                  // Bucket sorted visibility data
      const float *  const __restrict     d_bucket_ww,                  // Bucket sorted visibility data
      const float2 * const __restrict     d_bucket_vis,                 // Bucket sorted visibility data
      const float *  const __restrict     d_bucket_weight,              // Bucket sorted visibility data

      double*                             d_norm,                       // Output norm 
      float*                              d_grid,                       // The active grid
      const int                           ld_grid                       // The stride separating rows in the active grid. This is 
                                                                        // .. for alignment
      )
{

  // Oversample is probably a value that could be known at compile time by specializing the kernel for all
  // common valiues. The gain can be 5% so not negligible. 
  // The variable oversample_is_2_or_3 indicate if it known at compile time that oversample==2 or oversample==3 
  //    This is potentially interesting because in those cases, oversample/2 == 1 and -1 <= off_u <= +1 
  //    So off_u is one of the 3 cases -oversample/2, 0 , +oversample/2
  // Another potentially interesting case would be oversample==1 for which off_u is always 0.
  //    If that is a possible case, then the code should also be optimized for it. 
#ifdef ASSUME_OVERSAMPLE
  int oversample = ASSUME_OVERSAMPLE ;
  bool oversample_is_2_or_3 = (ASSUME_OVERSAMPLE==2) ;
#else
  // WARNING: THIS VALUE IS FOR THE OSKAR EL30-EL56 DATASET 
  int oversample = oversample_ ; 
  bool oversample_is_2_or_3 = false ;  
#endif
    
   const int g_centre = grid_size / 2;
   const double scale = grid_size * cell_size_rad;

#define NUM_POINTS_IN_TILES(uu, vv)  d_numPointsInTiles[ (uu) + (vv)*numTiles.u]
#define OFFSETS_IN_TILES(uu, vv)     d_offsetsPointsInTiles[ (uu) + (vv)*numTiles.u]

   const int pu = offsetInTileIdx.u + blockIdx.y;  // U index of the tile we will process
   const int pv = offsetInTileIdx.v + blockIdx.z;  // V index of the tile we will process

   // If the tile falls in the exclusion box, return
   if(exclusionBoxInTileIdx.topLeft.u <= pu && pu <= exclusionBoxInTileIdx.botRight.u &&
         exclusionBoxInTileIdx.topLeft.v <= pv && pv <= exclusionBoxInTileIdx.botRight.v)
      return;

   // Yes, I've left these in on purpose!  Consider it your seat belt ...
   // REMOVED BECAUSE THEY HAVE A NOTICEABLE COST ( 900ms -> 876ms on my curtent best version)
#if 0
   assert(tileSize.u==blockDim.x);
   assert(tileSize.v==blockDim.y*(REGSZ+SHMSZ));
#endif
   
   // We now need to account for the fact that multiple blocks are processing this tile
   const int bid = blockIdx.x;
   // Num blocks allocated to this tile
   int nblks = gridDim.x;


   // We now need to fix the indexes of the grid elements in the tile that we're responsible for.
   // Each thread handles a section of one column in (V,U) space [i.e. column index is u]
   // The length of the column segment is REGSZ+SHMSZ
   const int my_grid_u_idx = threadIdx.x                          + tileBox.topLeft.u + pu*tileSize.u;
   const int my_grid_v_idx_start = threadIdx.y * (REGSZ+SHMSZ)    + tileBox.topLeft.v + pv*tileSize.v;


   // Allocate the register array used to store the grid elements for this tile
   float2 mygrid[REGSZ+1];   // +1 so that we can set REGSZ==0 without a compilation error. Should not matter since the additional element is never used
   // Set to zero
#pragma unroll
   for(int r=0; r<REGSZ; r++) {
      mygrid[r].x = 0;
      mygrid[r].y = 0;
   }

#if 1
   // As a rule of thumb, knowing those values at compile time should be better
   // For whatever reason, the kernel sometime use more registers!!!!!
   // 
#define BLOCKDIM_X TX 
#define BLOCKDIM_Y (TY/(REGSZ + SHMSZ))
#else
#define BLOCKDIM_X blockDim.x
#define BLOCKDIM_Y blockDim.y
#endif

   // An attempt to separate the x and y values in shared memory  to limit bank conflits.
   //  => That does not seem to be beneficial. 
#if SPLIT_SHMEM
   extern __shared__ float myshmem_root[] ;
   float * myshmem_base= &myshmem_root[threadIdx.x + 2*BLOCKDIM_X*SHMSZ*threadIdx.y ] ;
#define myshmem_x(s) myshmem_base[(s)*BLOCKDIM_X] 
#define myshmem_y(s) myshmem_base[(s)*BLOCKDIM_X+BLOCKDIM_X*SHMSZ] 
   // Zero out the shared mem2
   for(int s=0; s<SHMSZ; s++) {
     myshmem_x(s) = 0 ;
     myshmem_y(s) = 0;
   }
#else
   extern __shared__ float2 myshmem[];
   // Thread idx in warp wid accesses element i at
   //        shmem[idx + i*BLOCKDIM_X + SHMSZ*BLOCKDIM_X*wid]
   const int shmemoff = threadIdx.x + SHMSZ*BLOCKDIM_X*threadIdx.y;
   // Zero out the shared mem
   for(int s=0; s<SHMSZ; s++) {
      float2 zero;  zero.x = 0;  zero.y = 0;
      myshmem[shmemoff + s*BLOCKDIM_X] = zero;
   }
#endif
   
   const int tileOffset = OFFSETS_IN_TILES(pu,pv);
   const int num_vis = NUM_POINTS_IN_TILES(pu,pv);

#if 1
   int min_vis_per_blk = 64 ;  
   if (num_vis < nblks*min_vis_per_blk ) {
     nblks = num_vis / min_vis_per_blk ; 
     if (nblks<=1) nblks=1 ;
     if (bid>=nblks) return ; 
   }
#endif


   double loc_norm = 0;

#define TSIZE (REGSZ+SHMSZ)

   int imin = tileOffset+bid ;
   int imax = tileOffset+num_vis;
    
   // THIS IS AN ATTEMPT TO PREFETCH ww_i BUT THAT DOES NOT SEEM TO HELP 
   // 
   float ww_i, pf_ww_i ; bool use_pf_ww_i = false ;    
   if (use_pf_ww_i)  pf_ww_i = d_bucket_ww[imin] ;
   
   /* Loop over visibilities. */
   for (int i = imin; i < imax; i+=nblks)
   {
     int inext = i+nblks ;
     
      /* Convert UV coordinates to grid coordinates. */
      float pos_u    = -d_bucket_uu[i] * scale;   // LOAD
      float pos_v    =  d_bucket_vv[i] * scale;   // LOAD

      if ( use_pf_ww_i ) {
        ww_i = pf_ww_i ; 
        if (inext<imax || true) pf_ww_i = d_bucket_ww[inext]; 
      } else {
        ww_i     =  d_bucket_ww[i];
      } 
       const float w  =  d_bucket_weight[i]; // LOAD 
      
      float2 val     =  d_bucket_vis[i]; // LOAD
      val.x *= w;
      val.y *= w;


      // TODO: sqrt(), round() and fabs() are double operation.
      // Using their float version is causing some rounding errors.
      //         
      //  
#if 0
#define ROUND(x) roundf(x)
#else
#define ROUND(x) round(x)
#endif
      
      const int grid_u = (int)ROUND(pos_u) + g_centre;
      const int grid_v = (int)ROUND(pos_v) + g_centre;
      
      // TODO: precompute: grid_w is a function of ww_i and w_scale.
      //       w_scale is an argument so it can probably be considered constant.
      //       could it be possible to precompute grid_w in a array indexed by 'i'.
      //       The size of ww_i is also needed so the actual precomputed value 
      //       would be grid_w*sign(ww_i).
      int grid_w = (int)ROUND(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
      if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      // TODO: support[] and compact_wkernel_start[] are indexed by grid_w so a value that is itself loaded from 
      //       global memory: so double latency cost. 
      // IDEA: They are of size 922 so they may both fit in const memory.
      //       If they do not fit, then it might be possible to pack both values in a single int (e.g. 8bit for 
      //       support and 24bit for compact_wkernel_start. 
      // IDEA2: If grid_w[i] can be precomputed as indicated above, then 
      //        that should also be possible for support and compact_wkernel_start_idx
      //        If so, all the values loaded from global memory would be indexed by [i] 
      const int wsupport = support[grid_w]; // LOAD LEVEL 2

      /* Scaled distance from nearest grid point. */
      int off_u = (int)ROUND( (ROUND(pos_u)-pos_u) * oversample);   // \in [-oversample/2, oversample/2]
      const int off_v = (int)ROUND( (ROUND(pos_v)-pos_v) * oversample);    // \in [-oversample/2, oversample/2]

      /* Convolve this point. */
      
      int wkernel_start = compact_wkernel_start_idx[grid_w]; // LOAD_LEVEL_2: Could that go in const memory? 


      const int wkernel_size = oversample/2 + oversample*wsupport + 1;

      // TODO: So we are consimong  a register to control a sign change. 
      // 
      float conv_mul = (ww_i > 0 ? -1.0f : 1.0f);


      
      // Compute our k index. Our grid iteration range is
      //       grid_u + k
      // for -wsupport <= k <= wsupport and 
      //       grid_v + j
      // for -wsupport <= j <= wsupport
      // 
      //     grid_u + k = my_grid_u_idx
      // means
      //     k = my_grid_u_idx - grid_u
      int k = my_grid_u_idx - grid_u;


      // TODO: I am pretty sure that there is a better way to compute
      //       all that. This is quite expensive in term of instruction count.
      // TODO: I notice that most of that computation depends of 
      //         - oversample (so a small constant)
      //         - off_u in range [-oversample/2,oversample/2] 
      //         - wsupport so a value that depend of grid_w
      //         - 
      
      // Nicer indexing in the wkernel.  We only change indexing in u, not v

      
#if OFF_U_OPTIM==0
      // This is the original code
      int abs_offu = (off_u < 0 ? -off_u : off_u);
      if(abs_offu == 0) abs_offu = oversample/2;
      int wkernel_row_off = (abs_offu-1)*(wsupport+wsupport+1);
      if(off_u == oversample/2 || off_u==-oversample/2) wkernel_row_off += wsupport + 1;
      wkernel_row_off += wsupport;
      if(off_u > 0) k = -k;
      if(off_u == 0) {
         wkernel_row_off += - abs(k);
      }
      else if(abs_offu == oversample/2) {
         wkernel_row_off += - abs(k) + (k >= 1 ? 1 : 0);
      }
      else {
         wkernel_row_off += k;
      }
#elif 0
      // Thre trick here was to separate the two case off_u=0 and off_u!=0 
      int wkernel_row_off = 0 ;
      if (off_u==0) {
        wkernel_row_off += (oversample/2)*(wsupport+wsupport+1) - wsupport - 1;
        wkernel_row_off += - abs(k);        
      } else {
        int abs_offu = abs(off_u) ;
        wkernel_row_off += (abs_offu)*(wsupport+wsupport+1) - (wsupport+1) ;
        if( abs_offu == oversample/2 ) wkernel_row_off += wsupport + 1;
        if(off_u > 0) k = -k;
        if(abs_offu == oversample/2) {
          wkernel_row_off += - abs(k) + (k >= 1 ? 1 : 0);
        }
        else {
          wkernel_row_off += k;
        } 
      }

#elif 0
      // Thre trick here was to separate the two case off_u=0 and off_u!=0 
      int wkernel_row_off = 0 ;
      if (off_u==0) {
        wkernel_row_off += (oversample/2)*(wsupport+wsupport+1) - wsupport - 1;
        wkernel_row_off += - abs(k);        
      } else {
        int abs_offu = abs(off_u) ;
        wkernel_row_off += (abs_offu)*(wsupport+wsupport+1) - (wsupport+1) ;
        if( abs_offu == oversample/2 ) wkernel_row_off += wsupport + 1;
        if(off_u > 0) k = -k;
        if(abs_offu == oversample/2) {
          wkernel_row_off += - abs(k) + (k >= 1 ? 1 : 0);
        }
        else {
          wkernel_row_off += k;
        } 
      }
#elif 0
      // This is a version that actually works gives a bit of speedup for VERS==6
      // 
      int wkernel_row_off = 0 ;
      if (off_u==0) {
        wkernel_row_off += (oversample/2)*(wsupport+wsupport+1) - wsupport - 1;
        wkernel_row_off += - abs(k);        
      } else {
        if (off_u < 0) { 
          k = -k;
          off_u=-off_u;
        } 
        wkernel_row_off += off_u*(wsupport+wsupport+1) ;
        if(off_u == oversample/2) {
          wkernel_row_off += - abs(k) + (k<0);
        }
        else {
          wkernel_row_off -= (wsupport+1) ;
          wkernel_row_off -= k;
        } 
      }

#elif 0
      // Now specialize for positive and negative off_u  
      // Remark: we do not care about inverting the sign of k since we
      //         only need its absolute value
      // Remark: Another way to specialize could be using a switch(off_u) {} .
      //         For oversample==4 there are only 5 possible values
      int wkernel_row_off = 0 ;
      if (off_u==0) {
        wkernel_row_off += (oversample/2)*(2*wsupport+1)  ;
        wkernel_row_off -=  wsupport+1; 
        wkernel_row_off += - abs(k);        
      } else if (off_u>0) {
        // oversample_is_2_or_3 is a trick to optimize event more in 
        // case oversample is equal to 2 or 3 
        if(off_u == oversample/2 || oversample_is_2_or_3 ) {         
          //          wkernel_row_off += off_u*(wsupport+wsupport+1) ;
          wkernel_row_off += (oversample/2)*(wsupport+wsupport+1) ;
          wkernel_row_off += - abs(k) + (k<0);
        }
        else {
#ifdef ASSUME_OVERSAMPLE
          // When oversample==4, -2 <= off_u <= 2 so it can only be equal to +1 here
          if (oversample==4) off_u=+1 ; 
#endif
          wkernel_row_off += off_u*(2*wsupport+1) - (wsupport+1) ;
          wkernel_row_off -= k;
        } 
      } else { // off_u<0
        if(off_u == -oversample/2 || oversample_is_2_or_3 ) {
          wkernel_row_off += (oversample/2)*(wsupport+wsupport+1) ;
          wkernel_row_off += - abs(k) + (k>0);
        }
        else {
 
#ifdef ASSUME_OVERSAMPLE
          // When oversample==4, -2 <= off_u <= 2 so it can only be equal to -1 here
          if (oversample==4) off_u=-1 ; 
#endif
          wkernel_row_off += (-off_u)*(wsupport+wsupport+1) - (wsupport+1) ;
          wkernel_row_off += k;
        } 
      }
#else 
      // *****WARNING ****** 
      // This is a version that was specialized for the case oversample==4
      // so when -2 <= off_u <= 2 
      //
      //   DO NOT USE UNLESS oversample == 4 
      //
      int wkernel_row_off = 0 ;
#if 1
      if (off_u<0) {
        off_u=-off_u ;
        k=-k ;
      }
#endif
      switch(off_u) 
        {
#if 0
        case -2:
          wkernel_row_off += 4*wsupport + 2  ;
          wkernel_row_off += - abs(k) + (k>0);
          break ;
        case -1:
          wkernel_row_off += wsupport + k; 
          break ;
#endif
        case 0:
          wkernel_row_off += 3*wsupport+1  ;
          wkernel_row_off += - abs(k);        
          break ;
        case 1:
          wkernel_row_off += wsupport - k;
          break ;
        case 2:
        default:
          wkernel_row_off += 4*wsupport + 2 ;
          wkernel_row_off += - abs(k) + (k<0);          
          break ;
        }

#endif




      if (EARLY_J_EXIT) {
        // Early exit when our j interval is not relevant.
        // Doing the test earlier is not necessaritly a good idea
        // because of load latencies
        int jmin = my_grid_v_idx_start + 0       - grid_v ;
        if  ( wsupport < jmin ) continue;
        int jmax = my_grid_v_idx_start + TSIZE-1 - grid_v ;
        if ( -wsupport > jmax ) continue;
      }

      // Is the k index covered by the support of this convolution kernel?      
      // A 'continue' is easier to move up and down than a if() 
      // However, it is not necessarily beneficial to do it too early because of load latencies
      bool is_my_k = ( abs(k) <= wsupport ) ;   // using abs() is slightly faster 
      if( !is_my_k ) continue ;  

      //      if (grid_w<10000) continue ; 

#if FLOAT_SUM
      // We are going to sum a few float. Do we really need to use double?
      // The result is ok with the oskar testcase and it is also a bit faster
      // SEE ALSO THE REMARK BELOW AT THE loc_norm accumulation
      float sum = 0.0;
#else
      double sum = 0.0;
#endif


      {

        // An attempt to save a few registers and operations by changing 
        // how compact_wkernel[] is accessed
#if PTR_OPTIM  
        // The new access method
        const float2 * c_ptr = &compact_wkernel[wkernel_start + wkernel_row_off] ; 
#define get_c(iy) c_ptr[ (iy)*wkernel_size ] 
#else
        // The original method
#define get_c(iy) compact_wkernel[wkernel_start + wkernel_row_off+(iy)*wkernel_size]
#endif

        
        
#if VERS==6

        // Same as version 5 but with a lot more micro-optimizations
#undef PREFETCH        
#define PREFETCH 6


        {

          float2 c[TSIZE]  ; 

#pragma unroll
        for(int iter=-PREFETCH; iter<TSIZE; iter++)
         {           
           if ( iter<TSIZE-PREFETCH )
             {
               // Load for step 't'
               int t = iter+PREFETCH ;
               int j = my_grid_v_idx_start + t - grid_v;

               // Is the j index covered by the support of this convolution kernel?
               //               bool is_my_j = (-wsupport<=j && j<=wsupport);

               // Ideally should not be needed but the register allocator
               // does not seem to appreciate when values are optionally assigned.
               //   ==> Could also be done in the 'else' part below but this is slightly less efficient
               // Question: do we have a way to very cheap way tell the compiler & register allocator that
               //           a register shall be discarded?
               //           Ideally, we would need an intrinsic function that would generate
               //           an assignmenr to the register but nothing in SASS (so simply tell
               //           the register allocator that the current content of the register
               //           is trashed). TODO make a bug report with an example based on the
               //           same trick used here.

               c[t].x = 0 ;
               c[t].y = 0 ;

               if ( abs(j) <= wsupport ) {
                 //                 c[t] = compact_wkernel[wkernel_start + iy*wkernel_size + wkernel_row_off];
                 int iy = abs(off_v + j*oversample);
                 c[t] = get_c(iy) ;
               } else {
                 
               }

             }

           if ( iter>=0 )
             {
               // Compute for step 't'
               int t = iter ;
               int j = my_grid_v_idx_start + t - grid_v;

               // Is the j index covered by the support of this convolution kernel?
               //               bool is_my_j = (-wsupport<=j && j<=wsupport);
               //bool is_my_j = abs(j) <= wsupport ;
               if (j>wsupport) {
                 // Nothing more to do so better leave early.
                 // That optimization works quite well.
                 // If you choose to remove it, do not forget to update the test in the following if() 
                 break ;
               }
               //bool is_my_j = (abs(j) <= wsupport )  ;
               if (j>=-wsupport) {
#if USE_CONV_MUL2
                 // GPUs are so unpredictable!!!
                 // Here, the idea was to replace a generic register by a predicate register
                 // but somehow, nvcc was clever enough to detect that conv_mul2 is constant
                 // and the whole code ends up being duplicated. For whatever reason, that is enough
                 // to kill the performances significantly.
                 //   ==> a nasty trick 
                 if (conv_mul2) c[t].y = -c[t].y ; 
#else
                 c[t].y *= conv_mul;
#endif
                 sum += c[t].x; /* Real part only. */
                 
                 if (t<REGSZ) {                  
#if FORCE_FMA
                   mygrid[t].x -= (val.y * c[t].y);
                   mygrid[t].x += (val.x * c[t].x);
                   mygrid[t].y += (val.y * c[t].x);
                   mygrid[t].y += (val.x * c[t].y);
#else
                   mygrid[t].x += (val.x * c[t].x - val.y * c[t].y);
                   mygrid[t].y += (val.y * c[t].x + val.x * c[t].y);
#endif
                 } else {
                   int s = t - REGSZ ;
#if SPLIT_SHMEM
                   float ax = myshmem_x(s) ;
                   float ay = myshmem_y(s) ;
                   myshmem_x(s) = ax + (val.x * c[t].x - val.y * c[t].y); 
                   myshmem_y(s) = ay + (val.y * c[t].x + val.x * c[t].y);
#else

                   float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
#if FORCE_FMA
                   z.x += (val.x * c[t].x);
                   z.x -= (val.y * c[t].y);
                   z.y += (val.y * c[t].x);
                   z.y += (val.x * c[t].y);
#else
                   z.x += (val.x * c[t].x - val.y * c[t].y);
                   z.y += (val.y * c[t].x + val.x * c[t].y);
#endif
                   myshmem[shmemoff + s*BLOCKDIM_X] = z;
#endif
                 }
               }
             }
           
         }
        }
        
#elif VERS==5

        // Same as version 4 but written in a more maintainable way.
        // Also PREFECT indicates how many values must be prefeteched
        // before starting computation. Prefetch will cost registers so
        // use with care
#define PREFETCH 4

#define TSIZE (REGSZ+SHMSZ)
        int j[TSIZE] ;
        bool is_my_j[TSIZE] ;
        float2 c[TSIZE] ; 
       
#pragma unroll
        for(int iter=-PREFETCH; iter<TSIZE; iter++)
         {           
           if ( iter<TSIZE-PREFETCH )
             {
               // Load for step 't'
               int t = iter+PREFETCH ;
               j[t] = my_grid_v_idx_start + t - grid_v;
               int iy = abs(off_v + j[t] * oversample);
               // Is the j index covered by the support of this convolution kernel?
               is_my_j[t] = (-wsupport<=j[t] && j[t]<=wsupport);
               if(is_my_j[t]) {
                 //                 c[t] = compact_wkernel[wkernel_start + iy*wkernel_size + wkernel_row_off];
                 c[t] = get_c(iy) ;
               } else {
                 // Humm... should not be needed 
                 c[t].x = 0 ;
                 c[t].y = 0 ;
               }

             }

           if ( iter>=0 )
             {
               // Compute for step 't'
               int t = iter ;
               
               if(is_my_j[t]) {
                 c[t].y *= conv_mul;
                 sum += c[t].x; /* Real part only. */
                 
                 if (t<REGSZ) {
                   mygrid[t].x += (val.x * c[t].x - val.y * c[t].y);
                   mygrid[t].y += (val.y * c[t].x + val.x * c[t].y);
                 } else {
                   int s = t - REGSZ ;
#if SPLIT_SHMEM
                   float ax = myshmem_x(s) ;
                   float ay = myshmem_y(s) ;
                   myshmem_x(s) = ax + (val.x * c[t].x - val.y * c[t].y); 
                   myshmem_y(s) = ay + (val.y * c[t].x + val.x * c[t].y);
#else

                   float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
                   z.x += (val.x * c[t].x - val.y * c[t].y);
                   z.y += (val.y * c[t].x + val.x * c[t].y);
                   myshmem[shmemoff + s*BLOCKDIM_X] = z;
#endif
                 }
               }
             }
               
         }
        
#elif VERS==4
        // merged Version with SW-pipelining

#define TSIZE (REGSZ+SHMSZ)
        int j[TSIZE] ;
        int iy[TSIZE] ;
        bool is_my_j[TSIZE] ;
        float2 c[TSIZE] ; 

        int t = 0 ;
        j[t] = my_grid_v_idx_start + t - grid_v;
            iy[t] = abs(off_v + j[t] * oversample);
            // Is the j index covered by the support of this convolution kernel?
            is_my_j[t] = (-wsupport<=j[t] && j[t]<=wsupport);
            if(is_my_j[t]) {
              c[t] = compact_wkernel[wkernel_start + iy[t]*wkernel_size + wkernel_row_off];
            } else {
              c[t].x = 0 ;
              c[t].y = 0 ;
            }
#pragma unroll
        for(int iter=0; iter<TSIZE-1; iter++)
         {
            t=iter+1 ;
            j[t] = my_grid_v_idx_start + t - grid_v;
            iy[t] = abs(off_v + j[t] * oversample);
            // Is the j index covered by the support of this convolution kernel?
            is_my_j[t] = (-wsupport<=j[t] && j[t]<=wsupport);
            if(is_my_j[t]) {
               c[t] = compact_wkernel[wkernel_start + iy[t]*wkernel_size + wkernel_row_off];
            } else {
              c[t].x = 0 ;
              c[t].y = 0 ;
            }
            t=iter ;
            is_my_j[t] = (-wsupport<=j[t] && j[t]<=wsupport);
            if(is_my_j[t]) {
               c[t].y *= conv_mul;
               sum += c[t].x; /* Real part only. */

               if (t<REGSZ) {
                 mygrid[t].x += (val.x * c[t].x - val.y * c[t].y);
                 mygrid[t].y += (val.y * c[t].x + val.x * c[t].y);
               } else {
                 int s = t - REGSZ ;
                 float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
                 z.x += (val.x * c[t].x - val.y * c[t].y);
                 z.y += (val.y * c[t].x + val.x * c[t].y);
                 myshmem[shmemoff + s*BLOCKDIM_X] = z;
               }
            }
         }
        t=TSIZE-1 ;
            if(is_my_j[t]) {
               c[t].y *= conv_mul;
               sum += c[t].x; /* Real part only. */

               if (t<REGSZ) {
                 mygrid[t].x += (val.x * c[t].x - val.y * c[t].y);
                 mygrid[t].y += (val.y * c[t].x + val.x * c[t].y);
               } else {
                 int s = t - REGSZ ;
                 float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
                 z.x += (val.x * c[t].x - val.y * c[t].y);
                 z.y += (val.y * c[t].x + val.x * c[t].y);
                 myshmem[shmemoff + s*BLOCKDIM_X] = z;
               }
            }
        
#elif VERS==3
        // Merge the 'r' and 's' loops for future SW-pipelining
        // Also, the j, iy, is_my_j and c are transformed into arrays.
        // That should not matter much since the loop is fully unrolled (so the arrays are registers)
#define TSIZE (REGSZ+SHMSZ)
        int j[TSIZE] ;
        int iy[TSIZE] ;
        bool is_my_j[TSIZE] ;
        float2 c[TSIZE] ; 
#pragma unroll
        for(int t=0; t<REGSZ+SHMSZ; t++)
         {
            j[t] = my_grid_v_idx_start + t - grid_v;
            iy[t] = abs(off_v + j[t] * oversample);
            // Is the j index covered by the support of this convolution kernel?
            is_my_j[t] = (-wsupport<=j[t] && j[t]<=wsupport);
            if(is_my_j[t]) {
               c[t] = compact_wkernel[wkernel_start + iy[t]*wkernel_size + wkernel_row_off];
            } else {
              // I should not need that but not clearing will cause high-register usage
              // probably because the register allocator tries to keep those register alive
              // for multiple iterations of the outer i-loop (i.e. ptxas is unaware of c[].
              // It only sees registers)
              c[t].x = 0 ;
              c[t].y = 0 ;
            }
            if(is_my_j[t]) {
               c[t].y *= conv_mul;
               sum += c[t].x; /* Real part only. */
               // The t loop is unrolled so this IF will be resolved at compile time. 
               if (t<REGSZ) {
                 mygrid[t].x += (val.x * c[t].x - val.y * c[t].y);
                 mygrid[t].y += (val.y * c[t].x + val.x * c[t].y);
               } else {
                 int s = t - REGSZ ;
                 float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
                 z.x += (val.x * c[t].x - val.y * c[t].y);
                 z.y += (val.y * c[t].x + val.x * c[t].y);
                 myshmem[shmemoff + s*BLOCKDIM_X] = z;
               }
            }
         }
#elif VERS==2
#define NRR 4
#pragma unroll
         for(int r=0; r<REGSZ; r+=NRR)
         {
           float2 c[NRR] ;
           int j[NRR] ;
           int iy[NRR] ;
           bool is_my_j[NRR] ;

           #pragma unroll
           for(int rr=0; rr<NRR; rr++) {
            j[rr] = my_grid_v_idx_start + r + rr - grid_v;
            iy[rr] = abs(off_v + j[rr] * oversample);
            // Is the j index covered by the support of this convolution kernel?
            is_my_j[rr] = (-wsupport<=j[rr] && j[rr]<=wsupport);
            if(is_my_j[rr]) {
              c[rr] = compact_wkernel[wkernel_start + iy[rr]*wkernel_size + wkernel_row_off];
            }
           }

           #pragma unroll
           for(int rr=0; rr<NRR; rr++) {
            if(is_my_j[rr]) {
               c[rr].y *= conv_mul;
               sum += c[rr].x; /* Real part only. */
               mygrid[r+rr].x += (val.x * c[rr].x - val.y * c[rr].y);
               mygrid[r+rr].y += (val.y * c[rr].x + val.x * c[rr].y);
            }
           }
           
         }
         for(int s=0; s<SHMSZ; s++)
         {
            const int j = my_grid_v_idx_start + REGSZ+s - grid_v;
            const int iy = abs(off_v + j * oversample);
            // Is the j index covered by the support of this convolution kernel?
            const bool is_my_j = (-wsupport<=j && j<=wsupport);
            if(is_my_j) {
               float2 c = compact_wkernel[wkernel_start + iy*wkernel_size + wkernel_row_off];
               c.y *= conv_mul;
               sum += c.x; /* Real part only. */
               float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
               z.x += (val.x * c.x - val.y * c.y);
               z.y += (val.y * c.x + val.x * c.y);
               myshmem[shmemoff + s*BLOCKDIM_X] = z;
            }
         }
#elif VERS==1
         {
           int j = my_grid_v_idx_start - grid_v; ;
#pragma unroll
         for(int r=0; r<REGSZ; r++)
         {
            const int iy = abs(off_v + j * oversample);
            // Is the j index covered by the support of this convolution kernel?
            //            const bool is_my_j = (-wsupport<=j && j<=wsupport);
            const bool is_my_j = abs(j) <= wsupport;

            //if (j>wsupport) break ;
            // const bool is_my_j = -wsupport<=j ;
            
            if(is_my_j) {
               float2 c = compact_wkernel[wkernel_start + iy*wkernel_size + wkernel_row_off];
#if USE_CONV_MUL2
                 if (conv_mul2) c.y = -c.y ; 
#else
                 c.y *= conv_mul;
#endif
               sum += c.x; /* Real part only. */
#if FORCE_FMA
               mygrid[r].x += (val.x * c.x);
               mygrid[r].x -= (val.y * c.y);
               mygrid[r].y += (val.y * c.x);
               mygrid[r].y += (val.x * c.y);
#else 
               mygrid[r].x += (val.x * c.x - val.y * c.y);
               mygrid[r].y += (val.y * c.x + val.x * c.y);
#endif

            }
            j++ ;
         }
         for(int s=0; s<SHMSZ; s++)
         {
            const int iy = abs(off_v + j * oversample);
            // Is the j index covered by the support of this convolution kernel?
            //            const bool is_my_j = (-wsupport<=j && j<=wsupport);
            const bool is_my_j = abs(j) <= wsupport;
            //if (j>wsupport) break ;
            //            const bool is_my_j = -wsupport<=j ;
            if(is_my_j) {

               float2 c = compact_wkernel[wkernel_start + iy*wkernel_size + wkernel_row_off];
#if USE_CONV_MUL2
                 if (conv_mul2) c.y = -c.y ; 
#else
                 c.y *= conv_mul;
#endif

               sum += c.x; /* Real part only. */
#if SPLIT_SHMEM
               float ax = myshmem_x(s) ;
               float ay = myshmem_y(s) ;
               myshmem_x(s) = ax + (val.x * c.x - val.y * c.y); 
               myshmem_y(s) = ay + (val.y * c.x + val.x * c.y);
#else
               float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
#if FORCE_FMA
               z.x += (val.x * c.x);
               z.x -= (val.y * c.y);
               z.y += (val.y * c.x);
               z.y += (val.x * c.y);
#else
               z.x += (val.x * c.x - val.y * c.y);
               z.y += (val.y * c.x + val.x * c.y);
#endif
               myshmem[shmemoff + s*BLOCKDIM_X] = z;
#endif          
            }
            j++ ;

         }
        
         }
#else
         UNKNOWN_VERS
#endif
      }
#if 1
      // In my optimized version, the compiler is able to figure out that the shared memory
      // is updated multiple tines  and so it keeps the value in registers. Register usage 
      // does not decrease at all. Clever but annoying. 
      // That fake __syncthread() is there to confuse the compiler 
      if (GRIDDING_TILED_SHMSZ>0 && loc_norm==-1.0) __syncthreads();
#endif
      // REMARK: when using float for 'sum' we may want to do this multiplication as double (if precision issues)     
      loc_norm += sum * w;
   }  // END of loop over all visibilities

  // DUMMY2_false and DUMMY2_true are conditions that are false and true but that can
  // not be evaluated at compile time.
  //   - Use DUMMY2_false to estimate the cost of the final reduction phase (should be minimal). 
  //   - in the original code (so VERS=1), I noticed that using DUMMY1_true produces a kernel using a bit less registers (e.g 64 vs 72)
  //     and thus with a slighlty better occupancy but this is not a very robust optimization.
  //     I assume that breaking the control flow with a IF is somehow  helping the register allocator (especially after
  //     a lot of unrolling). 
  //   - The default bevahior can be restored by using 'true' as condition.
#define DUMMY2_false (num_w_planes<0)
#define DUMMY2_true (num_w_planes>=0)

 if ( DUMMY2_true  )   
     {
   // Now have to write our tile back to the grid
#pragma unroll
   for(int r=0; r<REGSZ; r++)
   {
      // Adjust the grid index by the bounding box so that we get coordinates in the active grid, not the full grid
      const int row = my_grid_v_idx_start-boundingBox.topLeft.v + r;
      const int col =  my_grid_u_idx-boundingBox.topLeft.u;
      int p = row * ld_grid + col;
      atomicAdd(d_grid + 2*p,     mygrid[r].x);
      atomicAdd(d_grid + 2*p + 1, mygrid[r].y);
   }
#pragma unroll
   for(int s=0; s<SHMSZ; s++)
   {
      // Adjust the grid index by the bounding box so that we get coordinates in the active grid, not the full grid
      const int row = my_grid_v_idx_start-boundingBox.topLeft.v + REGSZ + s;
      const int col =  my_grid_u_idx-boundingBox.topLeft.u;
      int p = row * ld_grid + col;
#if SPLIT_SHMEM
      atomicAdd(d_grid + 2*p    , myshmem_x(s) );     
      atomicAdd(d_grid + 2*p + 1, myshmem_y(s) ) ;     
#else
      float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
      atomicAdd(d_grid + 2*p, z.x);     
      atomicAdd(d_grid + 2*p + 1, z.y);
#endif
   }


   __shared__ double shmem[32];
   block_reduceplus(loc_norm, shmem);
   if(threadIdx.x==0 && threadIdx.y==0) {
      atomicAdd(d_norm, loc_norm);
   }
   }
#undef NUM_POINTS_IN_TILES
#undef OFFSETS_IN_TILES
}


/**
 * Process all the tiles in the tile box, but skip any tiles in the exclusion box.
 * If the exclusion box is empty, then we will process all the tiles.
 *
 * Each thread block processes visibilities in one tile only.  Within a thread block, the parallelism
 * is over the elements in the tile that are updated (so all threads in a thread block process 
 * the same visibilities).  We can assign multiple thread blocks to each tile so that the work load 
 * for a given tile is spread over multiple blocks.
 *
 * Launch with a 3D grid of 2D thread blocks.
 *   gridDim.x  : the number of blocks which will process each tile, i.e. parallelise the visibility
 *                calculation for a given tile across gridDim.x thread blocks
 *   gridDim.y  : the number of tiles in the U direction
 *   gridDim.z  : the number of tiles in the V direction
 *  
 *   blockDim.x : this must be the same as tileSize.u
 *   blockDim.y : this must satisfy the constraint blockDim.y*(REGSZ+SHMSZ) == tileSize.v
 */

template<int REGSZ, int SHMSZ>
__global__ void oskar_process_tiles_dynamic(
      const int                           num_w_planes, 
      const int*     const __restrict     support,
      const int                           oversample_, 
      const int*     const __restrict     compact_wkernel_start_idx,    // Start index of each convolution kernel
      const float2 * const __restrict     compact_wkernel,              // The compacted convolution stack
      const double                        cell_size_rad,
      const double                        w_scale, 
      const int                           grid_size, 

      const Box                           boundingBox,                  // Bounding box defining the active grid region
      const Box                           tileBox,                      // Region of the active grid that is tiled.  We actually
                                                                        // .. only need the top left corner
      const Point                         tileSize,                     // The tile size. This can be inferred from blockDim
      const Point                         numTiles,                     // Number of tiles in U and V directions.  The tiled
                                                                        // .. region starts at tileBox.topLeft

      const Point                         offsetInTileIdx_OBSOLETE,
      const Box                           exclusionBoxInTileIdx_OBSOLETE,

      const int *    const __restrict     d_numPointsInTiles,           // Number of visibilities in each tile
      const int *    const __restrict     d_offsetsPointsInTiles,       // Start index of buckets, where visibility data in each
                                                                        // .. tile start
      const float *  const __restrict     d_bucket_uu,                  // Bucket sorted visibility data
      const float *  const __restrict     d_bucket_vv,                  // Bucket sorted visibility data
      const float *  const __restrict     d_bucket_ww,                  // Bucket sorted visibility data
      const float2 * const __restrict     d_bucket_vis,                 // Bucket sorted visibility data
      const float *  const __restrict     d_bucket_weight,              // Bucket sorted visibility data

      const int2 *  const __restrict      d_bucket_tile,                // The tile coordinate associated to that bucket

      int  * __restrict                   d_visibility_counter,         // Initially 0. Increased atomically
      int                                 totalVisibilities,            

      double*                             d_norm,                       // Output norm 
      float*                              d_grid,                       // The active grid
      const int                           ld_grid                       // The stride separating rows in the active grid. This is 
                                                                        // .. for alignment
      )
{
  //  assert(TX==32 && TY==16) ;
  const int BCG = 32 ; // Group of buckets loaded together. Ideally = warp size == TX but could be smaller
  const int BCW = 1 ;  // Number of groups to process. 
  const int BCS = BCW*BCG ;  // Number of contiguous buckets to process

  __shared__ float s_pos_u[BCG] ;
  __shared__ float s_pos_v[BCG] ;

  __shared__ float s_bucket_ww[BCG] ;
  __shared__ float2 s_bucket_vis[BCG] ;
  __shared__ float s_bucket_weight[BCG] ;
  
  // s_bucket_pupv contains pu and pv packed together into a single int. 
  // This is to save a bit of shared memory
#define PACK_PUPV 1
#if PACK_PUPV
  __shared__ int s_bucket_pupv[BCG] ;
#else
  __shared__ int s_bucket_pu[BCG] ;
  __shared__ int s_bucket_pv[BCG] ;
#endif

  //  __shared__ int s_wsupport[BCG] ;
  __shared__ int s_wsupport[BCG] ;
  __shared__ int s_wkernel_start[BCG] ;

   // Allocate the register array used to store the grid elements for this tile
   float2 mygrid[REGSZ+1];   // +1 so that we can set REGSZ==0 without a compilation error. Should not matter since the additional element is never used
   extern __shared__ float2 myshmem[];
   // Thread idx in warp wid accesses element i at
   //        shmem[idx + i*BLOCKDIM_X + SHMSZ*BLOCKDIM_X*wid]
   const int shmemoff = threadIdx.x + SHMSZ*BLOCKDIM_X*threadIdx.y;
  
  // Oversample is probably a value that could be known at compile time by specializing the kernel for all
  // common valiues. The gain can be 5% so not negligible. 
  // The variable oversample_is_2_or_3 indicate if it known at compile time that oversample==2 or oversample==3 
  //    This is potentially interesting because in those cases, oversample/2 == 1 and -1 <= off_u <= +1 
  //    So off_u is one of the 3 cases -oversample/2, 0 , +oversample/2
  // Another potentially interesting case would be oversample==1 for which off_u is always 0.
  //    If that is a possible case, then the code should also be optimized for it. 
#ifdef ASSUME_OVERSAMPLE
  int oversample = ASSUME_OVERSAMPLE ;
  bool oversample_is_2_or_3 = (ASSUME_OVERSAMPLE==2) ;
#else
  // WARNING: THIS VALUE IS FOR THE OSKAR EL30-EL56 DATASET 
  int oversample = oversample_ ; 
  bool oversample_is_2_or_3 = false ;  
#endif
  
  const int g_centre = grid_size / 2;
  const double scale = grid_size * cell_size_rad;
   
  // Current tile index
  int pu = -1 ;  // -1 is used as a marker to detect that no tile is currently processed
  int pv = -1 ; 
 
  int my_grid_u_idx = 0 ; 
  int my_grid_v_idx_start = 0 ; 
  double loc_norm = 0 ;

  while (true) {

    // Figure out the next chunk of BCS buckets we have to process
    int imin = 0 ; 
    if (threadIdx.x==0)  {
      imin = atomicAdd(d_visibility_counter,BCS) ;
    }
    imin = __shfl(imin, 0);  
    if ( imin >= totalVisibilities) {
      break ; // We are done.  
    }

    // We are now going to process the BCS buckets by chunks of BCG (or lower for the last chunk)  
    for (int i0=imin; i0<imin+BCS; i0+=BCG) // will iterate exactly BCW times  
    {

      __syncthreads() ; 

      // First, load and copy the BCG buckets to shared memory using BCG threads
      if (threadIdx.x < BCG ) 
      {
        // This part is what I call the "cooperative loading region" in which 
        // BCG thread are executed in lock step. 
        int t = threadIdx.x ;
        int i = i0 + t ;
        if (i<totalVisibilities)  {

  // Each value is loaded exactly once so caches do not matter.
  // Try various cache behaviors
  #define LOAD(x) (x) 
  // #define LOAD(x) __ldg(&(x)) 
  // #define LOAD(x) __ldcg(&(x)) 
  // #define LOAD(x) __ldcs(&(x)) 
  // #define LOAD(x) __ldca(&(x)) 

          float w = LOAD(d_bucket_weight[i]) ;  

          s_bucket_weight[t] = w ; 

          s_pos_u[t]  = -LOAD(d_bucket_uu[i]) * scale;   
          s_pos_v[t]  =  LOAD(d_bucket_vv[i]) * scale;  

          float ww_i = LOAD(d_bucket_ww[i]) ; 
          s_bucket_ww[t]  = ww_i ;

          float2 val = LOAD(d_bucket_vis[i]) ; 
          val.x *= w;
          val.y *= w;
          s_bucket_vis[t] = val ;

          // ...
          // This is the tile coordinate for the bucket.
          // Reminder: this is sorted so that should not change often from one bucket to another
          // Actually, we could probably store that in a single int.
          // Even better, we only need the tiles coordinates for two reasons:
          //   (1) figure out when we change tile
          //   (2) write the output at the right place. 
          // 
          //   IDEA: The output tiles do not need to form a rectangle. We only care
          //         about the tiles with 1 or more buckets so we can describe 
          //         each tile by a single int (its rank in the active tiles).
          //         Also, the tile data can be stored contiguously ( 32x16 float2 each )
          //         with an additional cuda kernel to copy the tiles in the final
          //         2D output. 
          //         The advantage of that approach would a reduction of the
          //         number of instruction needed for the final atomic update of each tile. 
          //        
          //         
          //
          int2 tile = LOAD(d_bucket_tile[i]) ;  
#if PACK_PUPV==1
          // Ugly! Should use a bitfield.
          s_bucket_pupv[t] = tile.y * 4096 + tile.x ;
#else
          s_bucket_pu[t] = tile.x ;
          s_bucket_pv[t] = tile.y ;
#endif
                              
          int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
          if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;
          s_wsupport[t] = support[grid_w]; 
          s_wkernel_start[t] = compact_wkernel_start_idx[grid_w]; 

          // ... 
        }      
      }  

      __syncthreads() ; 

      // Now, process the loaded buckets in sequence
      for (int t=0;t<BCG;t++) 
        {

        int i=i0+t;

        if (i<totalVisibilities ) 
          {
#if PACK_PUPV==1
            int pupv = s_bucket_pupv[t] ;
            int new_pu = pupv & 4095 ;
            int new_pv = pupv >> 12 ;

#else
            int new_pu = s_bucket_pu[t] ;
            int new_pv = s_bucket_pv[t] ;
#endif
          if (new_pu != pu || new_pv!=pv) {

            // We are changing tile. 

            if (pu!=-1) {
              // There is already an active tile. Write it back
              // REMARK: exactly the same code as below
              
#     pragma unroll
              for(int r=0; r<REGSZ; r++)  {
                const int col =  my_grid_u_idx  -boundingBox.topLeft.u  ;
                const int row = my_grid_v_idx_start  -boundingBox.topLeft.v  + r ;
                int p = row * ld_grid + col ;                
                atomicAdd(d_grid + 2*p  , mygrid[r].x);
                atomicAdd(d_grid + 2*p+1, mygrid[r].y);
              }
              
#     pragma unroll
              for(int s=0; s<SHMSZ; s++)
                {
                  const int col =  my_grid_u_idx -boundingBox.topLeft.u; 
                  const int row = my_grid_v_idx_start -boundingBox.topLeft.v + REGSZ +s ;
                  int p = row * ld_grid + col ;                
                  float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
                  atomicAdd(d_grid + 2*p    , z.x);     
                  atomicAdd(d_grid + 2*p + 1, z.y);
                } 
            } // of if(pu!=-1)
            
            // Start working on our new tile.

            pu = new_pu ;
            pv = new_pv ;

            // pu and pv are 
            my_grid_u_idx       = threadIdx.x                    + tileBox.topLeft.u + pu*tileSize.u;
            my_grid_v_idx_start = threadIdx.y * (REGSZ+SHMSZ)    + tileBox.topLeft.v + pv*tileSize.v;

            // Clear gridding data in register and shared mem
            #pragma unroll
            for(int r=0; r<REGSZ; r++) {
              mygrid[r].x = 0;
              mygrid[r].y = 0;
            }

            for(int s=0; s<SHMSZ; s++) 
            {
              float2 zero;  zero.x = 0;  zero.y = 0;
              myshmem[shmemoff + s*BLOCKDIM_X] = zero;
            }
            

          }  // of if (i<totalVisibilities)
          
          // *** IMPORTANT: The computation from here up to the if(is_my_k) statement are
          // *** mostly redundant. Part of it could be done in the 'cooperative' loading region.
     
          /* Convert UV coordinates to grid coordinates. */
          float pos_u    =  s_pos_u[t] ;
          float pos_v    =  s_pos_v[t] ;  
          float ww_i     =  s_bucket_ww[t];
          float w        =  s_bucket_weight[t];
          float2 val     =  s_bucket_vis[t]; 

          
          const int grid_u = (int)round(pos_u) + g_centre;
          const int grid_v = (int)round(pos_v) + g_centre;
     
          const int wsupport = s_wsupport[t] ;
          int wkernel_start  = s_wkernel_start[t] ;

          /* Scaled distance from nearest grid point. */
          int off_u = (int)round( (round(pos_u)-pos_u) * oversample);   // \in [-oversample/2, oversample/2]
          const int off_v = (int)round( (round(pos_v)-pos_v) * oversample);    // \in [-oversample/2, oversample/2]
          
          /* Convolve this point */
          
          const int wkernel_size = oversample/2 + oversample*wsupport + 1;          

            
#define USE_VAL2 1
#if USE_VAL2         
          // val2 is an alternative to conv_mul. 
          // val2 consumes 1 more register but is globally better, See below 
          float2 val2 ;
          if ( ww_i>0) {
            val2.x = - val.x ;   
            val2.y =   val.y ;   
          } else {
            val2.x =   val.x ;   
            val2.y = - val.y ;    
          }
#endif 

          float conv_mul ; 
          if ( false ) {
            // TO BE INVESTIGATED: 
            // Forcing conv_mul to either +1.0f or -1.0f can signicantly change the performances.
            // As of now, -1.0f  is giving 620ms while +1.0f is 789ms.
            // The likely reason is that the compiler is changing some multiply-add into multiply-sub 
            // and those are obviously less efficient. This is ... annoying
            // 
            // IDEA: Could it make sense to compute the negative term in a separate register
            //       and substract it at the very end. That would cost REGSZ registers but 
            //       we would only use proper MAD instruction. TO BE TESTED.
            // WARNING: The results is obviously incorrect.
            conv_mul = +1.0f ;
          } else if ( ww_i > 0 ) {
            conv_mul = -1.0f ;
          } else {
            conv_mul = +1.0f;
           }


          
                
          // Compute our k index. Our grid iteration range is
          //       grid_u + k
          // for -wsupport <= k <= wsupport and 
          //       grid_v + j
          // for -wsupport <= j <= wsupport
          // 
          //     grid_u + k = my_grid_u_idx
          // means
          //     k = my_grid_u_idx - grid_u
          int k = my_grid_u_idx - grid_u;
          

          // TODO: I am pretty sure that there is a better way to compute
          //       all that. This is quite expensive in term of instruction count.
          // TODO: I notice that most of that computation depends of 
          //         - oversample (so a small constant)
          //         - off_u in range [-oversample/2,oversample/2] 
          //         - wsupport so a value that depend of grid_w
          //         - 
          
          // Nicer indexing in the wkernel.  We only change indexing in u, not v
          
      
          int abs_offu = (off_u < 0 ? -off_u : off_u);
          if(abs_offu == 0) abs_offu = oversample/2;
          int wkernel_row_off = (abs_offu-1)*(wsupport+wsupport+1);
          if(off_u == oversample/2 || off_u==-oversample/2) wkernel_row_off += wsupport + 1;
          wkernel_row_off += wsupport;
          if(off_u > 0) k = -k;
          if(off_u == 0) {
            wkernel_row_off += - abs(k);
          }
          else if(abs_offu == oversample/2) {
            wkernel_row_off += - abs(k) + (k >= 1 ? 1 : 0);
          }
          else {
            wkernel_row_off += k;
          }

          // Hummm not beneficial anymore??????
          if (EARLY_J_EXIT ) {
            // Early exit when our j interval is not relevant.
            // Doing the test earlier is not necessaritly a good idea
            // because of load latencies
            // QUESTION: Does that make sense to be able to have an early exit?
            //           By constructon the bucket is supposed to intersect with the tile.           
            //           The optimization has a noticeable effect so something must be going on here. 
            //           So this is either because my jmin:jmax interval is wrong or because
            //           the bucket was incorrectly filled.
            int jmin = my_grid_v_idx_start + 0       - grid_v ;
            if  ( wsupport < jmin ) continue;
            int jmax = my_grid_v_idx_start + TSIZE-1 - grid_v ;
            if ( -wsupport > jmax ) continue;
          }

          // Is the k index covered by the support of this convolution kernel?      
          // A 'continue' is easier to move up and down than a if() 
          // However, it is not necessarily beneficial to do it too early because of load latencies
          bool is_my_k = ( abs(k) <= wsupport ) ;   // using abs() is slightly faster 
//          bool dummy_false = (totalVisibilities<100);
          if( is_my_k /* && dummy_false */ ) {

            
            float sum = 0.0;
            
            // An attempt to save a few registers and operations by changing 
            // how compact_wkernel[] is accessed
#if PTR_OPTIM  
              // The new access method
              const float2 * c_ptr = &compact_wkernel[wkernel_start + wkernel_row_off] ; 
#define get_c(iy) c_ptr[ (iy)*wkernel_size ] 
#else
              // The original method
#define get_c(iy) compact_wkernel[wkernel_start + (iy)*wkernel_size + wkernel_row_off]
#endif
              
              
              // Currently 3 variants of the inner loops:
              //  VERS2==1 is similar to the original code
              //  VERS2==6 introduces some prefetching (see PREF6 below)
              //  VERS2==7 is similar to VERS2==6 but the data is prefected by blocks (see PREF7 below)
              //
              // As of now, the best strategy is to use VERS2 6 or 7  with PREF6 or PREF7 equal to TSIZE (so 
              // prefecth everything before computing everything)
              // I also discovered that it was beneficial to clear the 'c' value when is_my_k is false and 
              // to systematically do the computation (so no if(is_my_k) around the computation)
              // A plausible explanation is that removing the if(is_my_k) gives more scheduling freedom to the 
              // compiler.
              // 
              // Also, the occupancy is not a critical factor. 8shm+8reg, 4shm+12reg, 0shm+16reg all give good
              // good performance with an occupancy below 35%. 
              // 
              // 
             
#define VERS2 7

#if VERS2==7

        // This is a variant of VERS2==6 (see below)
        // Prefetching is done by blocks of size PREF7 followed by PREF7 computations. 
        // There is no IF in the computation to give more scheduling freedom to the compiler.
        // Warning: PREF7 Should be >= 1 
        // Remark: VERS==7 with PREF7==16 is almost equivalent to VERS==6 with PREF6=16
        //         because in both cases, all loads are performed first (assuming that TSIZE==16)
        //
        // 
#undef  PREF7
#define PREF7 16

        {

          float2 c[PREF7]  ; 

#pragma unroll
        for(int iter=0; iter<TSIZE; iter++)
         {           
             
            if ( iter%PREF7==0 )   {
              // Load PREF7 values at once
#pragma unroll              
              for (int t=0 ; t<PREF7 ; t++) {
                 if ( iter+t >= TSIZE ) continue ;
                 int j = my_grid_v_idx_start - grid_v + iter + t ;
                 bool is_my_j = abs(j) <= wsupport; 
                 if(is_my_j) {                 
                   const int iy = abs(off_v + j * oversample);                    
                   c[t] = get_c(iy) ;
                 } else {
                   c[t].x = c[t].y = 0 ;
                 }
              }                   
            }
       

            {
               // Compute for step 'iter' 
               int t = iter % PREF7 ;
               // int j = my_grid_v_idx_start - grid_v + iter ;
               // const bool is_my_j = abs(j) <= wsupport;
               if (true) {                 


                 sum += c[t].x; /* Real part only. */

                 if (iter<REGSZ) {                  
#if USE_VAL2
              // When using val2, we have 4 multiply-add (instead of 3 MAD + 1 multiply-sub)
              // and we also save one mul 
                   mygrid[iter].x += (val.x * c[t].x);
                   mygrid[iter].y += (val.y * c[t].x);
                   mygrid[iter].x += (val2.y * c[t].y);
                   mygrid[iter].y += (val2.x * c[t].y);
#else
                   c[t].y *= conv_mul;
                   mygrid[iter].x -= (val.y * c[t].y);
                   mygrid[iter].x += (val.x * c[t].x);
                   mygrid[iter].y += (val.y * c[t].x);
                   mygrid[iter].y += (val.x * c[t].y);                   
#endif
                 } else {

                   int s = iter - REGSZ ;
                   float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
#if USE_VAL2

                   z.x += (val.x * c[t].x);
                   z.y += (val.y * c[t].x);
                   z.x += (val2.y * c[t].y);
                   z.y += (val2.x * c[t].y);                   
#else
                   c[t].y *= conv_mul; 
                   z.x -= (val.y * c[t].y);
                   z.x += (val.x * c[t].x);
                   z.y += (val.y * c[t].x);
                   z.y += (val.x * c[t].y);
#endif
                   myshmem[shmemoff + s*BLOCKDIM_X] = z;
                 }

               }
             }
           
            }

        }

#elif VERS2==6 

#undef  PREF6
#define PREF6 16


        {

          float2 c[TSIZE]  ; 
          bool is_my_j2[TSIZE]  ; 

#pragma unroll
        for(int iter=-PREF6; iter<TSIZE; iter++)
         {           
           if ( iter<TSIZE-PREF6 )
             {
               // Load for step 't'
               int t = iter+PREF6 ;
               int j = my_grid_v_idx_start - grid_v + t ;
               bool is_my_j = abs(j) <= wsupport; 
               is_my_j2[t] = is_my_j ;

               if(is_my_j) {                 
                 const int iy = abs(off_v + j * oversample);                    
                 c[t] = get_c(iy) ;
               } else {
                 c[t].x = c[t].y = 0 ;
               } 
             }

           if ( iter>=0 )
             {
               // Compute for step 't'
               int t = iter ;
               int j = my_grid_v_idx_start - grid_v + t ;
               const bool is_my_j = abs(j) <= wsupport;
               if(true) {                 

                 c[t].y *= conv_mul;
                 sum += c[t].x; /* Real part only. */
                 
                 if (t<REGSZ) {                  
                   mygrid[t].x -= (val.y * c[t].y);
                   mygrid[t].x += (val.x * c[t].x);
                   mygrid[t].y += (val.y * c[t].x);
                   mygrid[t].y += (val.x * c[t].y);
                 } else {
                   int s = t - REGSZ ;
                   float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
                   z.x -= (val.y * c[t].y);
                   z.x += (val.x * c[t].x);
                   z.y += (val.y * c[t].x);
                   z.y += (val.x * c[t].y);
                   myshmem[shmemoff + s*BLOCKDIM_X] = z;
                 }
               }
             }
           
         }
        }

#elif VERS2==1 
        
              int j = my_grid_v_idx_start - grid_v;                    
#pragma unroll
              for(int r=0; r<REGSZ; r++)
                {

                  const bool is_my_j = abs(j) <= wsupport;                 
                  float2 c = {0,0} ;
                  if(is_my_j) {
                    
                    const int iy = abs(off_v + j * oversample);                    
                    //    float2 c = compact_wkernel[wkernel_start + wkernel_row_off + iy*wkernel_size ];                  
                    c = get_c(iy) ;
                  }
                  {
                    
                    // hummm .., The performance appear to be a bit
                    // sensitive to the order of the following operations.
                    // currently between 615ms and 639ms depending of the
                    // chosen ORDER.
                    // That looks like a scheduler problem. 
                    // It could be that some configurations are more prone to dual-issue than others.
                    // TO BE INVESTIGATED BUT THIS IS LIKELY SOMETHING THAT CANNOT BE CONTROLED. 
#define ORDER 2
#if ORDER==1
                    // That one is an attempt to move the multiplication by conv_mul outside the inner loop. 
                    // This is not better maybe because the GPU cannot reuse registers (so more pressure 
                    // on the RF). To be inverstigated.  
                    // 
                    sum += c.x; /* Real part only. */                 
                    mygrid[r].x -= (val2.y * c.y) ;
                    mygrid[r].x += (val.x * c.x) ;
                    mygrid[r].y += (val.y * c.x) ;
                    mygrid[r].y += (val2.x * c.y);                    
                    

#elif ORDER==2
                    // Currenly the best (615ms) 
                    // Damned! I cannot get 615ms anymore. What else did I change?
                    c.y *= conv_mul; 
                    sum += c.x; /* Real part only. */                 
                    mygrid[r].x -= (val.y * c.y); // B
                    mygrid[r].x += (val.x * c.x); // A
                    mygrid[r].y += (val.y * c.x); // C 
                    mygrid[r].y += (val.x * c.y); // D
#else 
                    // Not a good one (668ms)
                    c.y *= conv_mul; 
                    sum += c.x; /* Real part only. */                 
                    mygrid[r].x += (val.x * c.x);
                    mygrid[r].x -= (val.y * c.y);
                    mygrid[r].y += (val.y * c.x);
                    mygrid[r].y += (val.x * c.y);                    
#endif
                  }
                  j++ ;

                } // end of for(r)
#pragma unroll
              for(int s=0; s<SHMSZ; s++)
                {

                  const bool is_my_j = abs(j) <= wsupport;
                  if(is_my_j) {
                  const int iy = abs(off_v + j * oversample);
                    float2 c = get_c(iy) ;
                    c.y *= conv_mul;
                    sum += c.x; /* Real part only. */                  

                    float2 z = myshmem[shmemoff + s*BLOCKDIM_X]; 
                    z.x += (val.x * c.x); // B
                    z.y += (val.y * c.x); // D               
                    z.y += (val.x * c.y); // C
                    z.x -= (val.y * c.y); // A
                    myshmem[shmemoff + s*BLOCKDIM_X] = z;

                  }
                  j++ ;                  
                }                  
#endif

              // REMARK: when using float for 'sum' we may want to do this multiplication as double (if precision issues)     
              loc_norm += sum * w;

              
          }  // of if(is_my_k)
        } // of if(i<totalVisibilities
      }  // of for(t=...) 
     }  // of for(i0=, ...
    } // of while(true) 

    if (pu!=-1) {
   
           
      // Now have to write our tile back to the grid
#     pragma unroll
      for(int r=0; r<REGSZ; r++)  {
            // Adjust the grid index by the bounding box so that we get coordinates in the active grid, not the full grid
            const int row = my_grid_v_idx_start-boundingBox.topLeft.v + r;
            const int col =  my_grid_u_idx-boundingBox.topLeft.u;
            int p = row * ld_grid + col;
            atomicAdd(d_grid + 2*p,     mygrid[r].x);
            atomicAdd(d_grid + 2*p + 1, mygrid[r].y);
          }
      
#     pragma unroll
      for(int s=0; s<SHMSZ; s++)
        {
            // Adjust the grid index by the bounding box so that we get coordinates in the active grid, not the full grid
            const int row = my_grid_v_idx_start-boundingBox.topLeft.v + REGSZ + s;
            const int col =  my_grid_u_idx-boundingBox.topLeft.u;
            int p = row * ld_grid + col;
            float2 z = myshmem[shmemoff + s*BLOCKDIM_X];
            atomicAdd(d_grid + 2*p, z.x);     
            atomicAdd(d_grid + 2*p + 1, z.y);
          }
      
   } // of if (pu!=-1) 

   // Don't bother to optimize. The threads have a very very long life time
   // and this is occuring exactly once per thread.  

   //   __shared__ double shmem[32];
   // block_reduceplus(loc_norm, shmem);
   // if(threadIdx.x==0 && threadIdx.y==0) {
   atomicAdd(d_norm, loc_norm);
   //  }
    
  return  ; // ==================

}










/*******************************************************************************************************************************
 *
 *
 *
 *                               START OF CPU ROUTINES
 *
 *
 *
 *****************************************************************************************************************************/






/**
 * A wrapper to arrays on the GPU.  We have a few use cases here:
 *  (a) a host array that must be copied to GPU and back again
 *  (b) a GPU array that is zeroed out and copied back to the host
 *  (c) a GPU array that is zeroed out and never copied to the host
 *
 * These use cases are all handled via different constructors
 */
template<typename FP>
struct array_wrap {

   const int size;
   FP * host;
   FP * gpu;

   /**
    * Allocate on GPU and copy host data to GPU
    */
   array_wrap(int sz, const FP *host) : size(sz), host( (FP*)host ), gpu(nullptr) {
      CHECK( cudaMalloc(&gpu, sizeof(FP)*size) );
      assert(gpu);

      CHECK( cudaMemcpy(gpu, host, sizeof(FP)*size, cudaMemcpyHostToDevice) );
   }


   /**
    * Allocate on GPU and copy a std::vector to GPU
    */
   array_wrap(std::vector<FP> &host_vector) : size( host_vector.size() ), host( (FP*)host_vector.data() ), gpu(nullptr) {
      CHECK( cudaMalloc(&gpu, sizeof(FP)*size) );
      assert(gpu);

      CHECK( cudaMemcpy(gpu, host, sizeof(FP)*size, cudaMemcpyHostToDevice) );
   }


   /**
    * Allocate on GPU and do a memset with the specified value.  There is no
    * associated host pointer
    */
   array_wrap(int sz, int byteValue) : size(sz), host(nullptr), gpu(nullptr) {
      CHECK( cudaMalloc(&gpu, sizeof(FP)*size) );
      assert(gpu);

      CHECK( cudaMemset(gpu, byteValue, sizeof(FP)*size) );
   }


   /**
    * Allocate on GPU and do a memset with the specified value.  A host pointer
    * is associated, but no data is copied
    */
   array_wrap(int sz, FP * host, int byteValue) : size(sz), host( (FP*)host ), gpu(nullptr) {
      CHECK( cudaMalloc(&gpu, sizeof(FP)*size) );
      assert(gpu);

      CHECK( cudaMemset(gpu, byteValue, sizeof(FP)*size) );
   }


   /**
    * Allocate on GPU and do a memset with the specified value.  A host std::vector 
    * is associated, but not copied
    */
   array_wrap(std::vector<FP> &host_vector, int byteValue)  : size(host_vector.size()), host( host_vector.data() ), gpu(nullptr) {
      CHECK( cudaMalloc(&gpu, sizeof(FP)*size) );
      assert(gpu);

      CHECK( cudaMemset(gpu, byteValue, sizeof(FP)*size) );
   }


   /**
    * Allocate on GPU only, do nothing else
    */
   array_wrap(int sz) : size(sz), host(nullptr), gpu(nullptr) {
      CHECK( cudaMalloc(&gpu, sizeof(FP)*size) );
      assert(gpu);
   }



   /**
    * Copy data from GPU only if there is an associated host buffer
    */
   void copyFromGpu()
   {
      if(host != nullptr) {
         assert(gpu != nullptr);
         CHECK( cudaMemcpy(host, gpu, sizeof(FP)*size, cudaMemcpyDeviceToHost) );
      }
   }



   /**
    * Copy data to GPU only if there is an associated host buffer.
    * This routine is not actually used
    */
   void copyToGpu() {
      if(host != nullptr) {
         assert(gpu != nullptr);
         CHECK( cudaMemcpy(gpu, host, sizeof(FP)*size, cudaMemcpyHostToDevice) );
      }
   }


   /**
    * Destructor : free the GPU memory
    */
   ~array_wrap() {
      assert(gpu != nullptr);
      CHECK( cudaFree(gpu) );
   }
};






/**
 * Take the original, large input convolution stack and represent it in a more compact memory layout.
 * We also re-arrange the convolution kernel elements in the U direction while we're at it.
 */
void compact_wkernels(
      const int num_w_planes, 
      const int* wkernel_support,
      const int oversample, 
      const int conv_size_half,
      const float* conv_func,                            // The original convolution stack

      std::vector<float2> &compacted_wkernels,           // The new convolution stack
      std::vector<int>    &compacted_wkernel_start_idx   // Start index of each convolution kernel
      )
{
   const int kernel_dim = conv_size_half * conv_size_half;

   // Each slice of wkernel data occupies conv_size_half^2 elements in memory
#define WKERNEL(kidx,iy,ix,r)  conv_func[2*( (kidx)*kernel_dim + (iy)*conv_size_half + (ix) ) + (r)]

   // Inside each kernel, we only access elements at locations
   //
   //    id = abs(off + i*oversample)
   //
   // where 
   //
   //    off \in [-oversample/2, oversample/2] 
   //    
   // and
   //
   //    j = -wsupport, ..., -1, 0, 1, ..., wsupport
   //
   // This means we access locations between
   //
   //   id = 0  and  id = oversample/2 + wsupport*oversample
   //
   // Since we do this in U and Y dimensions the size of each 
   // compacted convolution kernel is
   // 
   //        (oversample/2 + wsupport*oversample + 1)^2
   //
   // float2 values, since we are dealing with complex data

   compacted_wkernel_start_idx.resize(num_w_planes);

   int compacted_size = 0;
   for(int grid_w = 0; grid_w < num_w_planes; grid_w++)
   {
      int wsupport = wkernel_support[grid_w];
      int size = oversample/2 + wsupport*oversample + 1;
      size = size * size;

      compacted_wkernel_start_idx.at(grid_w) = compacted_size;
      compacted_size += size;
   }

   // Allocate memory
   float2 t; t.x = -1000;  t.y = -1000;
   compacted_wkernels.resize(compacted_size, t);

   for(int grid_w = 0; grid_w < num_w_planes; grid_w++) 
   {
      const int wsupport = wkernel_support[grid_w];
      const int start = compacted_wkernel_start_idx.at(grid_w);
      const int size = oversample/2 + wsupport*oversample + 1;

      // We now want to choose a nicer layout for the convolution kernel elements
      // To do this, we need to loop over the off_v and off_u explicitly
      // For clarity, off_u and off_v are the quantities obtained from
      //    const int off_u = (int)round(  (round(pos_u)-pos_u) * oversample);  
      //    const int off_v = (int)round( (round(pos_v)-pos_v) * oversample);  
      // in the gridding kernels.


      for(int off_v=-oversample/2; off_v<=oversample/2; off_v++) {
         for(int j=-wsupport; j<=wsupport; j++) {
            // We need the original layout in V/Y dimension
            const int iy = abs(off_v + j * oversample);

            for(int off_u=-oversample/2; off_u<=oversample/2; off_u++) {
               for(int k=-wsupport; k<=wsupport; k++) {
                  const int ix = abs(off_u + k*oversample);

                  assert(0<=iy && iy<size);
                  assert(0<=ix && ix<size);

                  float2 cc;
                  cc.x = WKERNEL(grid_w,iy,ix,0);
                  cc.y = WKERNEL(grid_w,iy,ix,1);
                  const float2 c = cc;

                  /* So this is where it gets complicated.  We want linear stride in the convolution
                   * kernel, in the U direction.  We do this as follows ( idx(x) = (x ? 1 : 0) below)
                   *
                   * If off_u = -oversample/2
                   *   idx = wsupport - |k| + ind(k >= 1)
                   * If 0 < off_u < -oversample/2
                   *   idx = wsupport - k
                   * If off_u = 0
                   *   idx = wsupport - |k|
                   * If off_u > 0
                   *   idx = idx(-k) - symmetry in k
                   *
                   * We now need to decide how to store these things.
                   *  For 0 > off_u > -oversample/2
                   *     store the row at |off_u-1| * row_length = |off_u-1| * (2wsupport + 1)
                   *  For off_u = 0
                   *     store the row at (oversample/2-1) * prev_row_lengths = (oversample/2-1) * (2wsupport + 1)
                   *  For off_u = -oversample/2
                   *     store the row at the end of the data for off_u
                   *     = (oversample/2 - 1) * (2wsupport + 1) + (wsupport + 1)
                   */
                  int abs_offu = (off_u < 0 ? -off_u : off_u);
                  if(abs_offu == 0) abs_offu = oversample/2;

                  int off = (abs_offu-1)*(wsupport+wsupport+1);
                  if(off_u == oversample/2 || off_u==-oversample/2)
                     off += wsupport+1;

                  int myk = k;
                  if(off_u > 0) myk = -myk;

                  int myidx = -wsupport-100;
                  if(off_u == 0) {
                     myidx = wsupport - abs(myk);
                     assert(0<=myidx && myidx<=wsupport);
                  }
                  else if(abs_offu == oversample/2) {
                     assert(off_u != 0);
                     myidx = wsupport - abs(myk) + (myk >= 1 ? 1 : 0);
                     assert(0<=myidx && myidx<=wsupport);
                  }
                  else {
                     myidx = wsupport + myk;
                     assert(0<=myidx && myidx<=2*wsupport);
                  }

                  assert(0<=off + myidx && off+myidx<size);


                  if(off_u==0) {
                     if(k <= 0)
                        compacted_wkernels.at(start + iy*size + off + myidx) = c;
                     else {
                        float2 t = compacted_wkernels.at(start + iy*size + off + myidx);
                        assert(t.x==c.x && t.y==c.y);
                     }
                  }
                  else if(abs_offu==oversample/2) {
                     assert(off_u != 0);
                     if(off_u == -oversample/2 && k <= 0) {
                        compacted_wkernels.at(start + iy*size + off + myidx) = c;
                     }
                     else {
                        float2 t = compacted_wkernels.at(start + iy*size + off + myidx);
                        assert(t.x==c.x && t.y==c.y);
                     }
                  }
                  else {
                     if(off_u < 0) {
                        compacted_wkernels.at(start + iy*size + off + myidx) = c;
                     }
                     else {
                        float2 t = compacted_wkernels.at(start + iy*size + off + myidx);
                        assert(t.x==c.x && t.y==c.y);
                     }
                  }

               }  // END loop over k
            }  // END loop over off_u

            // and as a final obsessive check ...
            for(int i=0; i<size; i++) {
               float2 t = compacted_wkernels.at(start + iy*size + i);
               assert(t.x != -1000 && t.y != -1000);
            }

         }  // END loop over j
      }  // END loop over off_v

   }  // END loop  over grid_w

   //printf("Compacted Wkernels:\n\toriginal wkernel stack = %gMB\n\tcompacted wkernel stack = %gMB\n", 
   //      2.0*kernel_dim*num_w_planes*4.0/1024.0/1024.0, compacted_wkernels.size()*8.0/1024.0/1024.0);

#undef WKERNEL
}


struct OskarTuningParams {
   float tileFactor;
   Point tileSize;
   int centralBoxRadiusInTileCoordinates;
   int centralBoxNumBlocksPerTile;
   int outerTilesNumBlocksPerTile;

   OskarTuningParams() : tileFactor(1), tileSize(-1,-1), centralBoxRadiusInTileCoordinates(0), centralBoxNumBlocksPerTile(1000),
   outerTilesNumBlocksPerTile(16) { }
};




/**
 * Do the convolution gridding.  The inputs are exactly the same as the original C version
 */
void oskar_grid_wproj_f(
      const int num_w_planes, 
      const int* support,
      const int oversample, 
      const int conv_size_half,
      const float* conv_func, 
      const int num_vis,
      const float* uu, 
      const float* vv,
      const float* ww, 
      const float* vis,
      const float* weight, 
      const double cell_size_rad,
      const double w_scale, 
      const int grid_size, 
      int* num_skipped,
      double* norm, 
      float* input_grid)
{
   // These can be defined outside the function and passed in ...
   OskarTuningParams tuningParams;
   tuningParams.tileFactor = 1.0;
   tuningParams.tileSize = Point(TX,TY);
   tuningParams.centralBoxRadiusInTileCoordinates = 0 ; //4
   tuningParams.centralBoxNumBlocksPerTile = 1000;  //1000
   tuningParams.outerTilesNumBlocksPerTile = 500 ;  //16

   printf("w_scale = %f\n",w_scale) ; 

   printf("num_w_planes = %d\n", num_w_planes) ;


   printf("wscale=w_scale\n") ;
#if 0
   printf("support[:%d] = ",num_w_planes) ;
   for (int w=0;w<num_w_planes;w++) {
     printf("%d ",support[w]) ; 
   }
   printf("\n") ;
#endif
   std::vector<int> compacted_wkernel_start;
   std::vector<float2> compacted_wkernels;
   compact_wkernels(num_w_planes, support, oversample, conv_size_half, conv_func, compacted_wkernels, compacted_wkernel_start);

   printf("compacted_wkernel_start.size() =  %d\n", int(compacted_wkernel_start.size()) ) ; 
   printf("compacted_wkernel[]=  %dMB\n", int(compacted_wkernels.size()/(1024*1024))*sizeof(float2) ) ; 

#if 0
   int *CPT = new int[num_w_planes] ;
   for(int i=0;i<num_w_planes;i++) {
     CPT[i] = 0 ; 
   }
   for (int i=0;i<num_vis;i++) {
     float ww_i = ww[i];
     int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
     if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;
     CPT[grid_w]++ ;      
   }
   int sum_size = 0 ;
    double sum_loaded_gb = 0 ;
   for(int i=0;i<num_w_planes-1;i++) {
     int elems = (compacted_wkernel_start.data()[i+1]-compacted_wkernel_start.data()[i]) ; 
     int size = elems*sizeof(float2) ;
     sum_size += size; 
     //     printf("%d\n",compacted_wkernel_start.data()[i]*sizeof(float2)) ;
     double sum_bytes =  sum_size/(1024.0*1024); 
     // 'loaded' is a crude attempt to estimate how bytes will be loaded.
     double loaded_gb = size*(CPT[i] / (1024*1024*1024.0));
     sum_loaded_gb+=loaded_gb ;
     printf("grid_w[%3d] = %8d (w_support=%d  size_bytes=%d sum_bytes=%4.2fMB loaded=%6.2fGB sum_loaded=%6.2fGB)\n",i,CPT[i],support[i], size , sum_bytes, loaded_gb, sum_loaded_gb) ;
   }
#endif

   cudaEvent_t start, mid[15], stop;
   CHECK( cudaEventCreate(&start) );
   for(int i=0; i<15; i++) CHECK( cudaEventCreate(&mid[i]) );
   CHECK( cudaEventCreate(&stop) );
   float time;
   /*****
    * Persistent workspace - this memory should be kept between calls to this
    * function as it can be re-used.  It will hold the active portion of the 
    * grid on the host.  Of course, we don't know how big that is yet, but we can guess,
    * and then increase the size if necessary.
    ****/
   std::vector<float> aligned_workspace_grid( grid_size*grid_size*2 / 4 );
#define SHOW_BUG
#ifdef SHOW_BUG
   array_wrap<float> d_grid(grid_size*grid_size*2 / 4, 0);
#endif

   CHECK( cudaEventRecord(start, 0) );

   CHECK( cudaEventRecord(mid[7], 0) );
   /****
    *  Copy all input data to GPU - we don't include this in the timing
    ***/
   array_wrap<int>   d_support(num_w_planes, support);
   array_wrap<float> d_uu(num_vis, uu);
   array_wrap<float> d_vv(num_vis, vv);
   array_wrap<float> d_ww(num_vis, ww);
   array_wrap<float> d_vis(num_vis*2, vis);
   array_wrap<float> d_weight(num_vis, weight);
   array_wrap<int>   d_compact_wkernel_start( compacted_wkernel_start.size(), compacted_wkernel_start.data());
   array_wrap<float2>d_compact_wkernels( compacted_wkernels.size(), compacted_wkernels.data() );
   *num_skipped = 0;
   array_wrap<int>   d_num_skipped(1, num_skipped);
   *norm = 0.0;
   array_wrap<double> d_norm(1, norm);
   array_wrap<int>   d_bucket_counter(1, 0); 


   //   printf("==> %d\n",int(compacted_wkernels.size())) ;

   CHECK( cudaEventRecord(mid[8], 0) );



   CHECK( cudaEventRecord(mid[11], 0) );

   // Get the rectangular Bounding Box which contains all grid updates
   Box boundingBox( Point(2*grid_size, 2*grid_size), Point(-1, -1) );
   array_wrap<Box> d_boundingBox(1, &boundingBox);
   {
      int nblks = 1000;
      int nthds = 128;

      oskar_get_bounding_box<<<nblks, nthds>>>(num_w_planes, d_support.gpu, num_vis, d_uu.gpu, d_vv.gpu, d_ww.gpu, cell_size_rad, 
            w_scale, grid_size, d_boundingBox.gpu, d_num_skipped.gpu);

      CHECK( cudaGetLastError() );
   }
   d_boundingBox.copyFromGpu();
   d_num_skipped.copyFromGpu();


   /* Specify the box that we want to break up into tiles (the Tile Box).  This can be a subset of the bounding box,
    * but typically it's faster to just tile the whole bounding box.  The 'tileFactor' below defines how much of the
    * bounding box is tiled: basically the central 'tileFactor' in each u,v direction is tiled with 0 <= tileFactor <= 1
    */
   const float tileFactor = tuningParams.tileFactor;
   Box tileBox;
   {
      // Ensure the grid center is in the tile box
      const Point g_center(grid_size/2, grid_size/2);
      Point len1 = g_center - boundingBox.topLeft;
      Point len2 = boundingBox.botRight - g_center;

      tileBox.topLeft  = g_center - tileFactor * len1;
      tileBox.botRight = g_center + tileFactor * len2;
   }

   /* We now need to fix our Tile Size. A tile consists of SHMSZ elements per thread stored in shared memory
    * and REGSZ elements per thread stored in registers.  Each element is a float2.
    */
#ifdef GRIDDING_TILED_SHMSZ
   constexpr int SHMSZ = GRIDDING_TILED_SHMSZ;
#else
   constexpr int SHMSZ = 8;
#endif
#ifdef GRIDDING_TILED_REGSZ
   constexpr int REGSZ = GRIDDING_TILED_REGSZ;
#else
   constexpr int REGSZ = 8;
#endif

   dim3 tileNthds;
   {
      if(tuningParams.tileSize.v % (SHMSZ+REGSZ) != 0) {
         std::cerr << "\nERROR: tuningParams.tileSize.v = " << tuningParams.tileSize.v << " which is not a multiple of "
            << "REGSZ+SHMSZ = " << REGSZ << " + " << SHMSZ << "\n"
            << "       I cannot proceed ...\n";
         abort();
      }
      if(tuningParams.tileSize.u % 32 != 0) {
         std::cerr << "\nERROR: tuningParams.tileSize.u = " << tuningParams.tileSize.u << " which is not a multiple of 32.\n"
            << "         I use warp-level programming so I need whole numbers of warps ...\n";
         abort();
      }
      if(tuningParams.tileSize.u != tuningParams.tileSize.v) {
         std::cerr << "\nWARNING: non-square input tile " << tuningParams.tileSize.u << "x" << tuningParams.tileSize.v << "\n"
            << "         Are you sure this is what you want to do?\n";
      }
      tileNthds.x = tuningParams.tileSize.u;
      tileNthds.y = tuningParams.tileSize.v / (REGSZ + SHMSZ);
   }
   


   /* We now want to tile the tile box so that
    *   (a) we have a whole number of tiles
    *   (b) the grid center is exactly in the middle of a tile
    *   (c) if 'tileFactor' is 1, then we still tile the entire bounding box
    */
   Point numTiles(0,0);
   const Point tileSize = tuningParams.tileSize;
   if(tileFactor > 0) {
      Point tileExtent = tileBox.botRight - tileBox.topLeft;
      numTiles.u = (tileExtent.u + tileSize.u-1) / tileSize.u;
      numTiles.v = (tileExtent.v + tileSize.v-1) / tileSize.v;
      // We now try to center the grid center in the tile containing it
      Point g_center(grid_size/2, grid_size/2);
      // Which tile contains it?
      int pu = (g_center.u - tileBox.topLeft.u) / tileSize.u;
      int pv = (g_center.v - tileBox.topLeft.v) / tileSize.v;
      // This shouldn't really be necessary ...
      if(tileBox.topLeft.u + (pu+1)*tileSize.u < g_center.u)
         pu++;
      if(tileBox.topLeft.v + (pv+1)*tileSize.v < g_center.v)
         pv++;

      Point centerTileStart(tileBox.topLeft.u + pu*tileSize.u, tileBox.topLeft.v + pv*tileSize.v);
      assert(centerTileStart.u <= g_center.u);
      assert(centerTileStart.v <= g_center.v);
      assert( g_center.u <= (centerTileStart + tileSize).u);
      assert( g_center.v <= (centerTileStart + tileSize).v);

      // Compute difference between center of center tile and grid center
      Point diff = (centerTileStart + 0.5f * tileSize) - g_center;
      tileBox.topLeft -= diff;
      tileBox.botRight = tileBox.topLeft + Point(tileSize.u * numTiles.u, tileSize.v * numTiles.v);

      // If the intention was to tile the entire bounding box, make sure we are still tiling the bounding box
      if(tileFactor == 1.0f) {
         if(tileBox.topLeft.u > boundingBox.topLeft.u) {
            numTiles.u++;
            tileBox.topLeft.u -= tileSize.u;
         }
         if(tileBox.topLeft.v > boundingBox.topLeft.v) {
            numTiles.v++;
            tileBox.topLeft.v -= tileSize.v;
         }
         if(tileBox.botRight.u < boundingBox.botRight.u) {
            numTiles.u++;
            tileBox.botRight.u += tileSize.u;
         }
         if(tileBox.botRight.v < boundingBox.botRight.v) {
            numTiles.v++;
            tileBox.botRight.v += tileSize.v;
         }
         assert(tileBox.topLeft.u <= boundingBox.topLeft.u);
         assert(tileBox.topLeft.v <= boundingBox.topLeft.v);
         assert(tileBox.botRight.u >= boundingBox.botRight.u);
         assert(tileBox.botRight.v >= boundingBox.botRight.v);
      }
   }  

   /* To help the load balancing problem, we define a Central Box which should contain
    * a large proportion of the work to be done.  This central box will be processed 
    * separately.  We define the box symmetrically, and currently there's no robust 
    * way of deciding how big it should be ... 
    * 
    * NOTE: this box is in tile coordinates, not grid coordinates.
    */
   Box centralBox;
   {  
#if 0
      const int W = tuningParams.centralBoxRadiusInTileCoordinates;
      const Point g_center(grid_size/2, grid_size/2);
      int pu = (g_center - tileBox.topLeft).u/tileSize.u;
      int pv = (g_center - tileBox.topLeft).v/tileSize.v;
      centralBox.topLeft = Point(pu-W, pv-W);
      centralBox.botRight = centralBox.topLeft + Point(2*W, 2*W);
#else
      // Double the vertical size of the central box
      // This is to be used when the tile size is 32x16
      const int W = tuningParams.centralBoxRadiusInTileCoordinates;
      const Point g_center(grid_size/2, grid_size/2);
      int pu = (g_center - tileBox.topLeft).u/tileSize.u;
      int pv = (g_center - tileBox.topLeft).v/tileSize.v;
      centralBox.topLeft = Point(pu-W, pv-2*W+1);
      centralBox.botRight = centralBox.topLeft + Point(2*W, 4*W-2);
      
#endif
   }
   printf("Central Box = (%d,%d) --> (%d,%d)\n", centralBox.topLeft.u, centralBox.topLeft.v,centralBox.botRight.u,centralBox.botRight.v) ; 

   if(false) 
   {
      /* We can optionally make the grid hit intensity
       */
      std::vector<int> hit_intensity(grid_size*grid_size);
      hit_intensity.resize(grid_size*grid_size, 0);
      array_wrap<int> d_hit_intensity(hit_intensity);
      // First component must always be warpSize
      dim3 nthds(32, 16);
      oskar_make_grid_hit_intensity<<< 1000,nthds >>>(
            num_w_planes, d_support.gpu, num_vis, d_uu.gpu, d_vv.gpu, d_ww.gpu, cell_size_rad, w_scale, grid_size, 
            d_hit_intensity.gpu);

      d_hit_intensity.copyFromGpu();

      // Write the BMP file
      writeBmpColor("grid.bmp", hit_intensity.data(), boundingBox, grid_size, grid_size);
      writeBmpColorWithTiles("grid_tiles.bmp", hit_intensity.data(), tileBox, numTiles, tileSize, grid_size, grid_size);
   }




   /* We now need to count how many points are in each tile so that we can sort them
    */
   const int nTiles = numTiles.u * numTiles.v;
   array_wrap<int> d_numPointsInTiles( nTiles+1, 0);
   array_wrap<int> d_offsetsPointsInTiles( nTiles+2, 0);
   array_wrap<int> d_wk_offsetsPointsInTiles( nTiles+2, 0);



   /* Count the number of elements in each tile and then compute the offsets for each tile.
    * We do this on the host since we need to allocate GPU memory for the tiles etc
    */
   int totalVisibilities = 0;
   if(tileFactor > 0) {
      int nblks = 1000;
      dim3 nthds(512,1);

      oskar_count_elements_in_tiles<<<nblks, nthds>>>(num_w_planes, d_support.gpu, oversample, num_vis, d_uu.gpu, d_vv.gpu,
            d_ww.gpu, cell_size_rad, w_scale, grid_size, tileBox, tileSize, numTiles, 
            d_numPointsInTiles.gpu);

      CHECK( cudaGetLastError() );

      oskar_accumulate_tile_offsets<<<1, 1024>>>(numTiles, d_numPointsInTiles.gpu, 
            d_offsetsPointsInTiles.gpu, d_wk_offsetsPointsInTiles.gpu);

      CHECK( cudaGetLastError() );
      CHECK( cudaMemcpy(&totalVisibilities,d_offsetsPointsInTiles.gpu+numTiles.u*numTiles.v+1, sizeof(int), cudaMemcpyDeviceToHost));
   }
   else {
      CHECK( cudaMemcpy(d_numPointsInTiles.gpu, &num_vis, sizeof(int), cudaMemcpyHostToDevice) );
      // Just give a non-zero number so the cudaMalloc calls succeed
      totalVisibilities = 1;
   }



   /* Now need to fill the buckets on the GPU.
    * Make a single memory allocation and align it
    */
   int aligned_size = ( (totalVisibilities+63)/64 )*64;
   float * d_bucket_uu;
   CHECK( cudaMalloc( &d_bucket_uu, sizeof(float)*aligned_size * 6 ) );
   float * d_bucket_vv = d_bucket_uu + aligned_size;
   float * d_bucket_ww = d_bucket_vv + aligned_size;
   float * d_bucket_weight = d_bucket_ww + aligned_size;
   float2 * d_bucket_vis = (float2*)(d_bucket_weight + aligned_size);
   int2  * d_bucket_tile;
   CHECK( cudaMalloc( &d_bucket_tile, sizeof(int2)*totalVisibilities  ) );
   if(tileFactor > 0) {
      oskar_worlds_least_efficient_bucket_sort<<<1000, 128>>>
         (num_w_planes, d_support.gpu, num_vis, d_uu.gpu, d_vv.gpu, d_ww.gpu, d_vis.gpu, d_weight.gpu, cell_size_rad,
          w_scale, grid_size, 
          tileBox, tileSize, numTiles, 
          d_numPointsInTiles.gpu, d_wk_offsetsPointsInTiles.gpu,
          d_bucket_uu, d_bucket_vv, d_bucket_ww, d_bucket_vis, d_bucket_weight,
          d_bucket_tile);

      CHECK( cudaGetLastError() );
   }

#if 0
   {
     float * h_bucket_uu = new float[aligned_size * 6] ;
     float * h_bucket_vv = h_bucket_uu + aligned_size;
     float * h_bucket_ww = h_bucket_vv + aligned_size;
     float * h_bucket_weight = h_bucket_ww + aligned_size;
     float2 * h_bucket_vis = (float2*)(d_bucket_weight + aligned_size);     
     CHECK(cudaMemcpy(h_bucket_uu, d_bucket_uu, sizeof(float)*aligned_size * 6 , cudaMemcpyDeviceToHost )) ;
#if 0
     for (int i=0;i<10000;i++) {
       int k = i+39864791+5000000;
       int w_grid = (int)round(sqrt(fabs(h_bucket_ww[k] * w_scale))) ;
       printf(":: %8.0f %8.0f %8.0f (%d)\n",h_bucket_uu[k],h_bucket_ww[k],h_bucket_ww[k] , w_grid) ;
     }
#endif
#if 1 
     {
       int *CPT= new int[num_w_planes] ;
       for (int k=0;k<num_w_planes;k++) {
         CPT[k]=0;
       }
       for (int k=0;k<totalVisibilities;k++) {
         int grid_w = (int)round(sqrt(fabs(h_bucket_ww[k] * w_scale))) ;
         if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;
         CPT[grid_w]++;
         //  printf(":: %8.0f %8.0f %8.0f (%d)\n",h_bucket_uu[k],h_bucket_ww[k],h_bucket_ww[k] , w_grid) ;
       }
       int sum=0;
       for (int k=0;k<num_w_planes;k++) {
         sum+=CPT[k] ; 
         printf("## %d %d (sum=%d %4.2f)\n",k,CPT[k],sum, double(sum)/totalVisibilities);
       }
     }
#endif

     //     exit(1);
   }
#endif
   /* We want to align our grid so that row length is a multiple of 256B = 64 floats
    */
   Point w1 = boundingBox.botRight  - boundingBox.topLeft + Point(1,1);
   Point w2 = tileBox.botRight      - boundingBox.topLeft + Point(1,1);
   const int aligned_workspace_grid_width  = std::max(w1.u, w2.u); 
   const int aligned_workspace_grid_height = std::max(w1.v, w2.v);
   const int ld_grid = ((aligned_workspace_grid_width + 64 - 1)/64 ) * 64;
   // Check if we have enough workspace
   if(aligned_workspace_grid_height * ld_grid * 2 > aligned_workspace_grid.size() ) {
      std::cerr << "\nWARNING: We're enlarging the CPU workspace from " << aligned_workspace_grid.size() << " to " 
                << aligned_workspace_grid_height * ld_grid * 2 << "\n";
      aligned_workspace_grid.resize(aligned_workspace_grid_height * ld_grid * 2);
   }
#ifndef SHOW_BUG
   array_wrap<float> d_grid(aligned_workspace_grid_height * ld_grid * 2 , 0);
#endif

#ifdef ASSUME_OVERSAMPLE
   assert(oversample==ASSUME_OVERSAMPLE) ;
#endif 


   CHECK( cudaEventRecord(mid[12], 0) );

   cudaStream_t streams[3];
   CHECK( cudaStreamCreate( &streams[0] ) );
   CHECK( cudaStreamCreate( &streams[1] ) );
   CHECK( cudaStreamCreate( &streams[2] ) );

   CHECK( cudaEventRecord(mid[0], streams[0]) );

#if 0
   int *h_numPointsInTiles = new int[nTiles] ;
   CHECK(cudaMemcpy(h_numPointsInTiles, d_numPointsInTiles.gpu, sizeof(int)*nTiles , cudaMemcpyDeviceToHost ) ) ;
   int *h_offsetsPointsInTiles = new int[nTiles] ;
   CHECK(cudaMemcpy(h_offsetsPointsInTiles, d_offsetsPointsInTiles.gpu, sizeof(int)*nTiles , cudaMemcpyDeviceToHost ) ) ;

#if 0
   for(int i=0;i<nTiles;i++) {
     if (h_numPointsInTiles[i]>1000000)
     printf("@ %d [%d,%d] %d %d\n",i,i%numTiles.u,i/numTiles.u,h_numPointsInTiles[i], h_offsetsPointsInTiles[i]) ;
   }
#else
   {
     int i=0 ;
     for (int v=0;v<numTiles.v;v++) {
       printf("m@ ") ;
       for (int u=0;u<numTiles.u;u++) {
         int n = h_numPointsInTiles[i++];
         char c ;
         if (n==0) { 
           c='-'  ;
         } else {
           n=log10(n) ;
           if (n>7) n=7 ;
           c='0'+n ;
         }
         printf("\e[0;%dm%c\033[0m",31+n,c) ;
       } 
       printf("\n") ;
     }
   }
#endif
#endif

   printf("totalVisibilities=%d\n",totalVisibilities);

#define MORE_SHM 0

   if(numTiles.u > 0 && numTiles.v > 0) {



     if (false) {
      dim3 grid;
      grid.x = tuningParams.outerTilesNumBlocksPerTile;            
      grid.y = numTiles.u;
      grid.z = numTiles.v;

      const size_t shmemsz = sizeof(float)*2*SHMSZ*tileNthds.x*tileNthds.y*tileNthds.z;

      Point tileGridOffsetInTileCoords(0,0);

      oskar_process_tiles_excluding_box<REGSZ,SHMSZ><<< grid,tileNthds,shmemsz+MORE_SHM, streams[0]>>>(
            num_w_planes, d_support.gpu, oversample, d_compact_wkernel_start.gpu, d_compact_wkernels.gpu,
            cell_size_rad, w_scale, grid_size, boundingBox, tileBox, tileSize, numTiles,
            tileGridOffsetInTileCoords, centralBox, 
            d_numPointsInTiles.gpu, d_offsetsPointsInTiles.gpu, 
            d_bucket_uu, d_bucket_vv, d_bucket_ww, d_bucket_vis, d_bucket_weight,
            d_norm.gpu, d_grid.gpu, ld_grid);

     }  else {

        // Must clear the bucket counter before each call.  
        CHECK( cudaMemset(d_bucket_counter.gpu, 0, sizeof(int)) ) ;

        const size_t shmemsz = sizeof(float)*2*SHMSZ*tileNthds.x*tileNthds.y*tileNthds.z;

        Point tileGridOffsetInTileCoords(0,0);
        int nblk = 10000 ;
//        if (getenv("GRIDSIZE")) nblk=atoi(getenv("GRIDSIZE")) ;
        int moreshm = 0 ;
//        if (getenv("MORESHM")) moreshm=atoi(getenv("MORESHM")) ;

        oskar_process_tiles_dynamic<REGSZ,SHMSZ><<< nblk , 32, shmemsz+MORE_SHM+moreshm, streams[0]>>>(
            num_w_planes, d_support.gpu, oversample, d_compact_wkernel_start.gpu, d_compact_wkernels.gpu,
            cell_size_rad, w_scale, grid_size, boundingBox, tileBox, tileSize, numTiles,
            tileGridOffsetInTileCoords, centralBox, 
            d_numPointsInTiles.gpu, d_offsetsPointsInTiles.gpu, 
            d_bucket_uu, d_bucket_vv, d_bucket_ww, d_bucket_vis, d_bucket_weight,
            d_bucket_tile, 
            d_bucket_counter.gpu, 
            totalVisibilities,
            d_norm.gpu, d_grid.gpu, ld_grid);
      }

      CHECK( cudaGetLastError() );

   }
   CHECK( cudaEventRecord(mid[1], streams[0]) );
   CHECK( cudaEventRecord(mid[2], streams[1]) );
   if(numTiles.u > 0 && numTiles.v > 0) {
      const size_t shmemsz = sizeof(float)*2*SHMSZ*tileNthds.x*tileNthds.y*tileNthds.z;
      Point tileGridOffsetInTileCoords = centralBox.topLeft;
      dim3 grid;
      grid.x = tuningParams.centralBoxNumBlocksPerTile;            
      grid.y = (centralBox.botRight.u - centralBox.topLeft.u) + 1;
      grid.z = (centralBox.botRight.v - centralBox.topLeft.v) + 1;
      printf("=> num_w_planes = %d\n",num_w_planes);
      printf("=> grid_size = %d\n",grid_size);
      printf("=> cell_size_rad = %g\n",cell_size_rad);
      printf("=> w_scale = %g\n",w_scale);
      if( !centralBox.isEmpty() ) {
         oskar_process_tiles_excluding_box<REGSZ,SHMSZ><<< grid,tileNthds,shmemsz+MORE_SHM, streams[1] >>>(
               num_w_planes, d_support.gpu, oversample, d_compact_wkernel_start.gpu, d_compact_wkernels.gpu,
               cell_size_rad, w_scale, grid_size, boundingBox, tileBox, tileSize, numTiles,
               tileGridOffsetInTileCoords, Box(), 
               d_numPointsInTiles.gpu, d_offsetsPointsInTiles.gpu, 
               d_bucket_uu, d_bucket_vv, d_bucket_ww, d_bucket_vis, d_bucket_weight,
               d_norm.gpu, d_grid.gpu, ld_grid);

         CHECK( cudaGetLastError() );
      }
   }
   CHECK( cudaEventRecord(mid[3], streams[1]) );
   CHECK( cudaEventRecord(mid[4], streams[2]) );
   


   if(0 < tileFactor && tileFactor < 1) {
      // First component must always be warpSize
      dim3 nthds(32, 16);
      int nblks = 1000;
      oskar_process_visibilities_outside_tile_box<<< nblks,nthds, 0, streams[2] >>>(
            num_w_planes, d_support.gpu, oversample, d_compact_wkernel_start.gpu, d_compact_wkernels.gpu,
            cell_size_rad, w_scale, grid_size, boundingBox, tileBox, numTiles,
            d_numPointsInTiles.gpu, d_offsetsPointsInTiles.gpu, 
            d_bucket_uu, d_bucket_vv, d_bucket_ww, d_bucket_vis, d_bucket_weight,
            d_norm.gpu, d_grid.gpu, ld_grid);

     CHECK( cudaGetLastError() );
   }
   else if(tileFactor == 0) {
      // First component must always be warpSize
      dim3 nthds(32, 16);
      int nblks = 1000;
      oskar_process_visibilities_outside_tile_box<<< nblks,nthds, 0, streams[2] >>>(
            num_w_planes, d_support.gpu, oversample, d_compact_wkernel_start.gpu, d_compact_wkernels.gpu,
            cell_size_rad, w_scale, grid_size, boundingBox, tileBox, numTiles,
            d_numPointsInTiles.gpu, d_offsetsPointsInTiles.gpu, 
            d_uu.gpu, d_vv.gpu, d_ww.gpu, (float2*)d_vis.gpu, d_weight.gpu,
            d_norm.gpu, d_grid.gpu, ld_grid);

     CHECK( cudaGetLastError() );
   }
   CHECK( cudaEventRecord(mid[5], streams[2]) );

   CHECK( cudaEventRecord(mid[9], 0) );
   d_norm.copyFromGpu();
   CHECK( cudaMemcpy( aligned_workspace_grid.data(), d_grid.gpu, sizeof(float)*d_grid.size, cudaMemcpyDeviceToHost) );
   CHECK( cudaEventRecord(mid[10], 0) );


   CHECK( cudaEventRecord(mid[6], 0) );
   CHECK( cudaEventSynchronize( mid[6] ) );

   // Copy aligned grid back into original grid
   for(int row=0; row < aligned_workspace_grid_height; row++) {
      for(int col=0; col<aligned_workspace_grid_width; col++) {
         int new_p = row * ld_grid + col;
         float x = aligned_workspace_grid.at(2*new_p);
         float y = aligned_workspace_grid.at(2*new_p+1);

         int old_p = (boundingBox.topLeft.v+row)*grid_size + col+boundingBox.topLeft.u;
         assert(boundingBox.topLeft.v+row<grid_size);
         assert(boundingBox.topLeft.u+col<grid_size);
         input_grid[2*old_p] = x;
         input_grid[2*old_p + 1] = y;
      }
   }

   CHECK( cudaEventRecord(stop, 0) );
   CHECK( cudaEventSynchronize( stop ) );

   FILE *timingFile;
   timingFile = fopen("timing.csv", "w");
   fprintf(timingFile, "copy inputs to GPU, copy outputs to CPU, prepare tiles, total gridding, total GPU without data copy, total GPU, total oskargridwprojf\n");

   // print timing to screen and file
   CHECK( cudaEventElapsedTime(&time, mid[7], mid[8]) );
   printf("Time to copy inputs to GPU: \t  %gms\n", time);
   fprintf(timingFile, "%g, ", time);
   CHECK( cudaEventElapsedTime(&time, mid[9], mid[10]) );
   printf("Time to copy outputs to CPU: \t  %gms\n", time);
   fprintf(timingFile, "%g, ", time);
   CHECK( cudaEventElapsedTime(&time, mid[11], mid[12]) );
   printf("GPU preparing tiles: \t  %gms\n", time);
   fprintf(timingFile, "%g, ", time);
   CHECK( cudaEventElapsedTime(&time, mid[0], mid[1]) );
   printf("GPU processing tiles: \t outer tiles is %gms\n", time);
   CHECK( cudaEventElapsedTime(&time, mid[2], mid[3]) );
   printf("GPU processing tiles: \t central box is %gms\n", time);
   CHECK( cudaEventElapsedTime(&time, mid[4], mid[5]) );
   printf("GPU processing non-tiles: \t  %gms\n", time);
   CHECK( cudaEventElapsedTime(&time, mid[0], mid[5]) );
   printf("GPU overall gridding tile: \t  %gms\n", time);
   fprintf(timingFile, "%g, ", time);
   CHECK( cudaEventElapsedTime(&time, mid[11], mid[5]) );
   printf("Total GPU processing time without data copy: \t  %gms\n", time);
   fprintf(timingFile, "%g, ", time);
   CHECK( cudaEventElapsedTime(&time, start, mid[6]) );
   printf("Total GPU processing time is %gms\n", time);
   fprintf(timingFile, "%g, ", time);
   CHECK( cudaEventElapsedTime(&time, start, stop) );
   printf("Total oskar_grid_wproj_f processing time is %gms\n", time);
   fprintf(timingFile, "%g\n", time);
   fclose(timingFile);

   printf("***** THIS IS MY TUNED VERSION ***\n");
   printf("GRIDDING_TILED  = %d+%d\n",GRIDDING_TILED_SHMSZ,GRIDDING_TILED_REGSZ);
   printf("VERS            = %d\n",VERS);
   printf("Tile            = %dx%d\n",TX,TY);


   CHECK( cudaEventDestroy(start) );
   for(int i=0; i<15; i++) CHECK( cudaEventDestroy(mid[i]) );
   CHECK( cudaEventDestroy(stop) );
   CHECK( cudaFree(d_bucket_uu));
   CHECK( cudaStreamDestroy(streams[0]));
   CHECK( cudaStreamDestroy(streams[1]));
   CHECK( cudaStreamDestroy(streams[2]));
}


