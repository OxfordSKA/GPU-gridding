
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
      float *                          bucket_weight              // Output: bucket sorted values
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
      const int                           oversample, 
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
   assert(tileSize.u==blockDim.x);
   assert(tileSize.v==blockDim.y*(REGSZ+SHMSZ));

   // We now need to account for the fact that multiple blocks are processing this tile
   const int bid = blockIdx.x;
   // Num blocks allocated to this tile
   const int nblks = gridDim.x;


   // We now need to fix the indexes of the grid elements in the tile that we're responsible for.
   // Each thread handles a section of one column in (V,U) space [i.e. column index is u]
   // The length of the column segment is REGSZ+SHMSZ
   const int my_grid_u_idx = threadIdx.x                          + tileBox.topLeft.u + pu*tileSize.u;
   const int my_grid_v_idx_start = threadIdx.y * (REGSZ+SHMSZ)    + tileBox.topLeft.v + pv*tileSize.v;


   // Allocate the register array used to store the grid elements for this tile
   float2 mygrid[REGSZ];
   // Set to zero
#pragma unroll
   for(int r=0; r<REGSZ; r++) {
      mygrid[r].x = 0;
      mygrid[r].y = 0;
   }

   extern __shared__ float2 myshmem[];
   // Thread idx in warp wid accesses element i at
   //        shmem[idx + i*blockDim.x + SHMSZ*blockDim.x*wid]
   const int shmemoff = threadIdx.x + SHMSZ*blockDim.x*threadIdx.y;
   // Zero out the shared mem
   for(int s=0; s<SHMSZ; s++) {
      float2 zero;  zero.x = 0;  zero.y = 0;
      myshmem[shmemoff + s*blockDim.x] = zero;
   }


   const int tileOffset = OFFSETS_IN_TILES(pu, pv);
   const int num_vis = NUM_POINTS_IN_TILES(pu,pv);
   double loc_norm = 0;

   /* Loop over visibilities. */
   for (int i = tileOffset+bid; i < tileOffset+num_vis; i+=nblks)
   {
      /* Convert UV coordinates to grid coordinates. */
      float pos_u    = -d_bucket_uu[i] * scale;
      float pos_v    =  d_bucket_vv[i] * scale;
      float ww_i     =  d_bucket_ww[i];
      const float w  =  d_bucket_weight[i];
      float2 val     =  d_bucket_vis[i];
      val.x *= w;
      val.y *= w;

      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); /* w-plane index */
      if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      const int wsupport = support[grid_w];

      /* Scaled distance from nearest grid point. */
      const int off_u = (int)round(  (round(pos_u)-pos_u) * oversample);   // \in [-oversample/2, oversample/2]
      const int off_v = (int)round( (round(pos_v)-pos_v) * oversample);    // \in [-oversample/2, oversample/2]

      /* Convolve this point. */
      double sum = 0.0;

      const int wkernel_start = compact_wkernel_start_idx[grid_w];
      const int wkernel_size = oversample/2 + oversample*wsupport + 1;
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
      // Is the k index covered by the support of this convolution kernel?
      bool is_my_k = (-wsupport<=k && k<=wsupport);

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



      if(is_my_k) {
#pragma unroll
         for(int r=0; r<REGSZ; r++)
         {
            const int j = my_grid_v_idx_start + r - grid_v;
            const int iy = abs(off_v + j * oversample);
            // Is the j index covered by the support of this convolution kernel?
            const bool is_my_j = (-wsupport<=j && j<=wsupport);
            if(is_my_j) {
               float2 c = compact_wkernel[wkernel_start + iy*wkernel_size + wkernel_row_off];
               c.y *= conv_mul;
               sum += c.x; /* Real part only. */
               mygrid[r].x += (val.x * c.x - val.y * c.y);
               mygrid[r].y += (val.y * c.x + val.x * c.y);
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
               float2 z = myshmem[shmemoff + s*blockDim.x];
               z.x += (val.x * c.x - val.y * c.y);
               z.y += (val.y * c.x + val.x * c.y);
               myshmem[shmemoff + s*blockDim.x] = z;
            }
         }
      }
      loc_norm += sum * w;

   }  // END of loop over all visibilities


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
      float2 z = myshmem[shmemoff + s*blockDim.x];
      atomicAdd(d_grid + 2*p, z.x);     
      atomicAdd(d_grid + 2*p + 1, z.y);
   }


   __shared__ double shmem[32];
   block_reduceplus(loc_norm, shmem);
   if(threadIdx.x==0 && threadIdx.y==0) {
      atomicAdd(d_norm, loc_norm);
   }

#undef NUM_POINTS_IN_TILES
#undef OFFSETS_IN_TILES
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
   tuningParams.tileSize = Point(32,32);
   tuningParams.centralBoxRadiusInTileCoordinates = 4;
   tuningParams.centralBoxNumBlocksPerTile = 1000;
   tuningParams.outerTilesNumBlocksPerTile = 16;


   std::vector<int> compacted_wkernel_start;
   std::vector<float2> compacted_wkernels;
   compact_wkernels(num_w_planes, support, oversample, conv_size_half, conv_func, compacted_wkernels, compacted_wkernel_start);

   cudaEvent_t start, mid[10], stop;
   CHECK( cudaEventCreate(&start) );
   for(int i=0; i<10; i++) CHECK( cudaEventCreate(&mid[i]) );
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
      const int W = tuningParams.centralBoxRadiusInTileCoordinates;
      const Point g_center(grid_size/2, grid_size/2);
      int pu = (g_center - tileBox.topLeft).u/tileSize.u;
      int pv = (g_center - tileBox.topLeft).v/tileSize.v;
      centralBox.topLeft = Point(pu-W, pv-W);
      centralBox.botRight = centralBox.topLeft + Point(2*W, 2*W);
   }


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
   if(tileFactor > 0) {
      oskar_worlds_least_efficient_bucket_sort<<<1000, 128>>>
         (num_w_planes, d_support.gpu, num_vis, d_uu.gpu, d_vv.gpu, d_ww.gpu, d_vis.gpu, d_weight.gpu, cell_size_rad,
          w_scale, grid_size, 
          tileBox, tileSize, numTiles, 
          d_numPointsInTiles.gpu, d_wk_offsetsPointsInTiles.gpu,
          d_bucket_uu, d_bucket_vv, d_bucket_ww, d_bucket_vis, d_bucket_weight);

      CHECK( cudaGetLastError() );
   }


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




   cudaStream_t streams[3];
   CHECK( cudaStreamCreate( &streams[0] ) );
   CHECK( cudaStreamCreate( &streams[1] ) );
   CHECK( cudaStreamCreate( &streams[2] ) );

   CHECK( cudaEventRecord(mid[0], streams[0]) );

   if(numTiles.u > 0 && numTiles.v > 0) {
      dim3 grid;
      grid.x = tuningParams.outerTilesNumBlocksPerTile;            
      grid.y = numTiles.u;
      grid.z = numTiles.v;
      const size_t shmemsz = sizeof(float)*2*SHMSZ*tileNthds.x*tileNthds.y*tileNthds.z;
      Point tileGridOffsetInTileCoords(0,0);

      oskar_process_tiles_excluding_box<REGSZ,SHMSZ><<< grid,tileNthds,shmemsz, streams[0]>>>(
            num_w_planes, d_support.gpu, oversample, d_compact_wkernel_start.gpu, d_compact_wkernels.gpu,
            cell_size_rad, w_scale, grid_size, boundingBox, tileBox, tileSize, numTiles,
            tileGridOffsetInTileCoords, centralBox, 
            d_numPointsInTiles.gpu, d_offsetsPointsInTiles.gpu, 
            d_bucket_uu, d_bucket_vv, d_bucket_ww, d_bucket_vis, d_bucket_weight,
            d_norm.gpu, d_grid.gpu, ld_grid);

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

      if( !centralBox.isEmpty() ) {
         oskar_process_tiles_excluding_box<REGSZ,SHMSZ><<< grid,tileNthds,shmemsz, streams[1] >>>(
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

   d_norm.copyFromGpu();
   CHECK( cudaMemcpy( aligned_workspace_grid.data(), d_grid.gpu, sizeof(float)*d_grid.size, cudaMemcpyDeviceToHost) );


   CHECK( cudaEventRecord(stop, 0) );
   CHECK( cudaEventSynchronize( stop ) );

   CHECK( cudaEventElapsedTime(&time, mid[0], mid[1]) );
   printf("GPU processing tiles: \t outer tiles is %gms\n", time);
   CHECK( cudaEventElapsedTime(&time, mid[2], mid[3]) );
   printf("GPU processing tiles: \t central box is %gms\n", time);
   CHECK( cudaEventElapsedTime(&time, mid[4], mid[5]) );
   printf("GPU processing non-tiles: \t  %gms\n", time);
   CHECK( cudaEventElapsedTime(&time, mid[0], mid[5]) );
   printf("GPU overall gridding tile: \t  %gms\n", time);
   CHECK( cudaEventElapsedTime(&time, start, stop) );
   printf("Total GPU processing time is %gms\n", time);



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

   CHECK( cudaEventDestroy(start) );
   for(int i=0; i<10; i++) CHECK( cudaEventDestroy(mid[i]) );
   CHECK( cudaEventDestroy(stop) );
   CHECK( cudaFree(d_bucket_uu));
   CHECK( cudaStreamDestroy(streams[0]));
   CHECK( cudaStreamDestroy(streams[1]));
   CHECK( cudaStreamDestroy(streams[2]));
}

