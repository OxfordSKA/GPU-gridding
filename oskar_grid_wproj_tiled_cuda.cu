/*
 * Copyright (c) 2018, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda_runtime_api.h>

#include "oskar_grid_wproj_tiled_cuda.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/* Templated CUDA atomicAdd() function. */
template <typename T>
__device__ __forceinline__ T oskar_atomic_add(T* address, const T val);

template <>
__device__ __forceinline__ double oskar_atomic_add<double>(
        double* address, const double val)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    /* Implementation of double-precision atomic add for CUDA_ARCH < 6.0. */
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
#endif
}

template <>
__device__ __forceinline__ float oskar_atomic_add<float>(
        float* address, const float val)
{
    return atomicAdd(address, val);
}

/* Templated CUDA round() function. */
template <typename T>
__device__ __forceinline__ int oskar_round(const T val);

template <>
__device__ __forceinline__ int oskar_round<double>(const double val)
{
    return __double2int_rn(val);
}

template <>
__device__ __forceinline__ int oskar_round<float>(const float val)
{
    return __float2int_rn(val);
}

/*
 * Runs through all the visibilities and counts how many fall into each tile.
 * Grid updates for each visibility will intersect one or more tiles.
 * Launch with a regular 1D grid with 1D blocks.
 */
template <typename FP>
__global__ void oskar_grid_wproj_count_elements_in_tiles_cudak(
        const int                   num_w_planes,
        const int*   const restrict support,
        const int                   num_vis,
        const FP*    const restrict uu,
        const FP*    const restrict vv,
        const FP*    const restrict ww,
        const int                   grid_size,
        const int                   grid_centre,
        const FP                    grid_scale,
        const FP                    w_scale,
        const float2                inv_tile_size,
        const int2                  num_tiles,
        const int2                  top_left,
        int* restrict               num_points_in_tiles,
        int* restrict               num_skipped
)
{
    const int num_threads_total = gridDim.x * blockDim.x;

    /* Loop over visibilities. */
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < num_vis; i += num_threads_total)
    {
        /* Convert UV coordinates to grid coordinates. */
        const FP pos_u = -uu[i] * grid_scale;
        const FP pos_v =  vv[i] * grid_scale;
        const float pos_w = (float)(ww[i] * w_scale);
        const int grid_u = oskar_round<FP>(pos_u) + grid_centre;
        const int grid_v = oskar_round<FP>(pos_v) + grid_centre;
        int grid_w = __float2int_rn(sqrtf(fabsf(pos_w)));

        /* Skip points that would lie outside the grid. */
        if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;
        const int w_support = support[grid_w];
        if (grid_u + w_support >= grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= grid_size || grid_v - w_support < 0)
        {
            atomicAdd(num_skipped, 1);
            continue;
        }

        /* Find range of tiles that the convolved visibility affects. */
        const int rel_u = grid_u - top_left.x;
        const int rel_v = grid_v - top_left.y;
        const float u1 = float(rel_u - w_support) * inv_tile_size.x;
        const float u2 = float(rel_u + w_support + 1) * inv_tile_size.x;
        const float v1 = float(rel_v - w_support) * inv_tile_size.y;
        const float v2 = float(rel_v + w_support + 1) * inv_tile_size.y;
        const int u_int[] = { int(floorf(u1)), int(ceilf(u2)) };
        const int v_int[] = { int(floorf(v1)), int(ceilf(v2)) };

        /* Loop over the tiles. */
        for (int pv = v_int[0]; pv < v_int[1]; pv++)
            for (int pu = u_int[0]; pu < u_int[1]; pu++)
                atomicAdd(&num_points_in_tiles[pu + pv * num_tiles.x], 1);
    }
}

/* Launch 1 thread block with the maximum number of threads. */
__global__ void oskar_accumulate_tile_offsets_cudak(const int num_tiles,
        const int* restrict num_points_in_tiles, int* restrict tile_offsets)
{
    extern __shared__ int scratch[];
    const int num_loops = (num_tiles + blockDim.x - 1) / blockDim.x;
    int running_total = 0;
    int idx = threadIdx.x; /* Starting value. */
    for (int i = 0; i <= num_loops; i++)
    {
        int n = 0;
        const int t = threadIdx.x + blockDim.x;

        /* Copy input to local memory. */
        if (idx < num_tiles && idx > 0)
            n = num_points_in_tiles[idx - 1];
        __syncthreads();
        scratch[threadIdx.x] = 0;
        scratch[t] = n;

        /* Prefix sum. */
        for (int j = 1; j < blockDim.x; j <<= 1)
        {
            __syncthreads();
            const int x = scratch[t - j];
            __syncthreads();
            scratch[t] += x;
        }

        /* Store results. Note the very last element is the total number! */
        __syncthreads();
        if (idx < num_tiles + 1)
            tile_offsets[idx] = scratch[t] + running_total;
        idx += blockDim.x;
        running_total += scratch[2 * blockDim.x - 1];
    }
}

/*
 * Does a bucket sort on the input visibilities. Each tile is a bucket.
 * Note that tile_offsets gives the start of visibility data for each tile,
 * and it will be modified by this kernel.
 *
 * Launch with a regular 1D grid of 1D blocks.
 */
template <typename FP, typename FP2>
__global__ void oskar_grid_wproj_bucket_sort_cudak(
        const int                   num_w_planes,
        const int*   restrict const support,
        const int                   num_vis,
        const FP*    restrict const uu,
        const FP*    restrict const vv,
        const FP*    restrict const ww,
        const FP2*   restrict const vis,
        const FP*    restrict const weight,
        const int                   grid_size,
        const int                   grid_centre,
        const FP                    grid_scale,
        const FP                    w_scale,
        const float2                inv_tile_size,
        const int2                  num_tiles,
        const int2                  top_left,
        int*         restrict       tile_offsets,
        FP*          restrict       sorted_uu,
        FP*          restrict       sorted_vv,
        int*         restrict       sorted_grid_w,
        FP2*         restrict       sorted_vis,
        FP*          restrict       sorted_weight,
        int*         restrict       sorted_tile
)
{
    const int num_threads_total = gridDim.x * blockDim.x;

    /* Loop over visibilities. */
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < num_vis; i += num_threads_total)
    {
        /* Convert UV coordinates to grid coordinates. */
        const FP pos_u = -uu[i] * grid_scale;
        const FP pos_v =  vv[i] * grid_scale;
        const float pos_w = (float)(ww[i] * w_scale);
        const int grid_u = oskar_round<FP>(pos_u) + grid_centre;
        const int grid_v = oskar_round<FP>(pos_v) + grid_centre;
        int grid_w = __float2int_rn(sqrtf(fabsf(pos_w)));

        /* Skip points that would lie outside the grid. */
        if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;
        const int w_support = support[grid_w];
        if (grid_u + w_support >= grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= grid_size || grid_v - w_support < 0)
        {
            continue;
        }

        /* Preserve sign of pos_w in grid_w. */
        grid_w *= (pos_w > 0 ? 1 : -1);

        /* Find range of tiles that the convolved visibility affects. */
        const int rel_u = grid_u - top_left.x;
        const int rel_v = grid_v - top_left.y;
        const float u1 = float(rel_u - w_support) * inv_tile_size.x;
        const float u2 = float(rel_u + w_support + 1) * inv_tile_size.x;
        const float v1 = float(rel_v - w_support) * inv_tile_size.y;
        const float v2 = float(rel_v + w_support + 1) * inv_tile_size.y;
        const int u_int[] = { int(floorf(u1)), int(ceilf(u2)) };
        const int v_int[] = { int(floorf(v1)), int(ceilf(v2)) };

        /* Loop over the tiles. */
        for (int pv = v_int[0]; pv < v_int[1]; pv++)
        {
            for (int pu = u_int[0]; pu < u_int[1]; pu++)
            {
                /* Get global output offset as a function of the tile index,
                 * and increment by one for next time.
                 * Pack both tile indices into a single integer. */
                int off = atomicAdd(&tile_offsets[pu + pv*num_tiles.x], 1);
                sorted_uu[off] = pos_u;
                sorted_vv[off] = pos_v;
                sorted_grid_w[off] = grid_w;
                sorted_vis[off] = vis[i];
                sorted_weight[off] = weight[i];
                sorted_tile[off] = pv * 32768 + pu;
            }
        }
    }
}

/*
 * Process all the tiles.
 * Launch with a 1D grid of 1D blocks of size 32 threads.
 */
template <int REGSZ, int SHMSZ, typename FP, typename FP2>
__global__ void oskar_grid_wproj_tiled_cudak(
        const int                   num_w_planes,
        const int*   const restrict support,
        const int                   oversample,
        const int*   const restrict wkernel_start,      // Start index of each convolution kernel.
        const FP2*   const restrict wkernel,            // The rearranged convolution kernels.
        const int                   grid_size,
        const int                   grid_centre,
        const FP                    grid_scale,
        const int2                  tile_size,
        const int2                  top_left,
        const int                   num_visibilities,   // Total number of visibilities to process.
        const FP*    const restrict sorted_uu,
        const FP*    const restrict sorted_vv,
        const int*   const restrict sorted_grid_w,
        const FP2*   const restrict sorted_vis,
        const FP*    const restrict sorted_weight,
        const int*   const restrict sorted_tile,
        int*         restrict       visibility_counter, // Initially 0. Increased atomically.
        double*      restrict       norm,
        FP*          restrict       grid
)
{
    const int BCG = 32; // Group of visibilities loaded together. Ideally warpSize, but could be smaller
    const int BCW = 1;  // Number of groups to process.
    const int BCS = BCW * BCG;

    __shared__ int s_tile_coords[BCG]; /* Packed tile coordinates. */
    __shared__ int s_grid_u[BCG];
    __shared__ int s_grid_v[BCG];
    __shared__ int s_off_v[BCG];
    __shared__ int s_wsupport[BCG];
    __shared__ int s_wkernel_idx[BCG];
    __shared__ FP2 s_vis[BCG];
    __shared__ FP  s_weight[BCG];

    /* The register array used to store the grid elements for this tile. */
    FP2 my_grid[REGSZ + 1]; /* Can set REGSZ = 0 if necessary. */
    extern __shared__ __align__(sizeof(double2)) unsigned char my_smem[];
    FP2* sh_mem = reinterpret_cast<FP2*>(my_smem); /* Allows template. */

    /* Thread idx in warp wid accesses element i at
     *        sh_mem[idx + i * warpSize + SHMSZ * warpSize * wid]       */
    const int sh_mem_off = threadIdx.x + SHMSZ * warpSize * threadIdx.y;

    /* Current tile index. */
    int tile_u = -1; /* -1 means that no tile is currently processed. */
    int tile_v = -1;
    int my_grid_u = 0;
    int my_grid_v_start = 0;
    double loc_norm = 0.0;

#define TSIZE (REGSZ + SHMSZ)

    while (true)
    {
        /* Get the next chunk of (BCS) visibilities we have to process. */
        /* Yes, ugh! But for compatibility... */
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
        int imin = 0;
#else
        __shared__ int imin;
#endif
        if (threadIdx.x == 0)
            imin = atomicAdd(visibility_counter, BCS);
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
    #if CUDART_VERSION >= 9000
        imin = __shfl_sync(0xFFFFFFFF, imin, 0);
    #else
        imin = __shfl(imin, 0);
    #endif
#else
        __syncthreads();
#endif
        if (imin >= num_visibilities) break; /* We have finished. */

        /* Process visibilities in chunks of BCG (lower for last chunk). */
        for (int v0 = imin; v0 < imin + BCS; v0 += BCG) /* Loop BCW times. */
        {
            /* Load BCG visibilities into shared memory using BCG threads. */
            __syncthreads();
            const int t = threadIdx.x;
            const int i = v0 + t;
            if (t < BCG && i < num_visibilities)
            {
                /* Cache the weight and weighted visibility data. */
                const FP weight = sorted_weight[i];
                FP2 vis = sorted_vis[i];
                vis.x *= weight;
                vis.y *= weight;
                s_weight[t] = weight;
                s_vis[t] = vis;

                /* Cache the visibility's tile coordinates.
                 * (This is sorted, so it should not change often
                 * from one visibility to another.)
                 * The tile coordinates are packed into a single int. */
                s_tile_coords[t] = sorted_tile[i];

                /* Store grid_u, grid_v and off_v in shared memory. */
                const FP pos_u = sorted_uu[i];
                const FP pos_v = sorted_vv[i];
                const int r_u = oskar_round<FP>(pos_u);
                const int r_v = oskar_round<FP>(pos_v);
                const int off_u = oskar_round<FP>((r_u - pos_u) * oversample);
                const int off_v = oskar_round<FP>((r_v - pos_v) * oversample);
                s_grid_u[t] = r_u + grid_centre;
                s_grid_v[t] = r_v + grid_centre;
                s_off_v[t] = off_v;

                /* Get the pointer to the convolution kernel using off_u. */
                const int grid_w_signed = sorted_grid_w[i];
                const int grid_w = abs(grid_w_signed);
                const int w_support = support[grid_w];
                const int conv_len = 2 * w_support + 1;
                const int width = (oversample/2 * conv_len + 1) * conv_len;
                const int mid = (abs(off_u) + 1) * width - 1 - w_support;
                s_wkernel_idx[t] = (wkernel_start[grid_w] + mid) *
                        (off_u >= 0 ? 1 : -1);
                s_wsupport[t] = w_support * (grid_w_signed > 0 ? 1 : -1);
            }

            /* Wait for loads to finish. */
            __syncthreads();

            /* Process loaded visibilities in sequence. */
            for (int v = 0; v < BCG; v++)
            {
                if ((v0 + v) >= num_visibilities) continue;
                const int tile_coords = s_tile_coords[v];
                const int new_tile_u = tile_coords & 32767;
                const int new_tile_v = tile_coords >> 15;

                /* Check if we are changing tile. */
                if (new_tile_u != tile_u || new_tile_v != tile_v)
                {
                    /* If there is already an active tile, write it out. */
                    if (tile_u != -1)
                    {
                        #pragma unroll
                        for (int r = 0; r < REGSZ; r++)
                        {
                            const int p = (my_grid_v_start + r) *
                                    grid_size + my_grid_u;
                            oskar_atomic_add<FP>(grid + 2 * p,     my_grid[r].x);
                            oskar_atomic_add<FP>(grid + 2 * p + 1, my_grid[r].y);
                        }
                        #pragma unroll
                        for (int s = 0; s < SHMSZ; s++)
                        {
                            const int p = (my_grid_v_start + s + REGSZ) *
                                    grid_size + my_grid_u;
                            const FP2 z = sh_mem[sh_mem_off + s * warpSize];
                            oskar_atomic_add<FP>(grid + 2 * p,     z.x);
                            oskar_atomic_add<FP>(grid + 2 * p + 1, z.y);
                        }
                    }

                    /* Start working on a new tile. */
                    tile_u = new_tile_u;
                    tile_v = new_tile_v;
                    my_grid_u       = tile_u * tile_size.x + top_left.x +
                            threadIdx.x;
                    my_grid_v_start = tile_v * tile_size.y + top_left.y +
                            threadIdx.y * (REGSZ + SHMSZ);

                    /* Clear local grid values. */
                    #pragma unroll
                    for (int r = 0; r < REGSZ; r++)
                    {
                        my_grid[r].x = 0;
                        my_grid[r].y = 0;
                    }
                    #pragma unroll
                    for (int s = 0; s < SHMSZ; s++)
                    {
                        FP2 zero; zero.x = zero.y = (FP) 0;
                        sh_mem[sh_mem_off + s * warpSize] = zero;
                    }
                }

                /* Compute our k index using grid_u + k = my_grid_u */
                const int k = my_grid_u - s_grid_u[v];

                /* Is the k index covered by the support of this kernel? */
                const int w_support = abs(s_wsupport[v]);
                if (abs(k) <= w_support) /* if (is_my_k) */
                {
                    FP2 c[TSIZE];
                    FP sum = (FP) 0;
                    const int conv_len = 2 * w_support + 1;
                    const int grid_v = s_grid_v[v];
                    const int off_v = s_off_v[v];
                    int wkernel_idx = s_wkernel_idx[v];
                    const int stride = wkernel_idx > 0 ? 1 : -1;
                    wkernel_idx = abs(wkernel_idx) + stride * k;

                    /* val2 is an alternative to using
                     * conv_mul = (ww_i > 0.0) ? -1.0 : 1.0; */
                    FP2 val2;
                    const FP2 val = s_vis[v];
                    if (s_wsupport[v] > 0)
                    {
                        val2.x = -val.x;
                        val2.y =  val.y;
                    }
                    else
                    {
                        val2.x =  val.x;
                        val2.y = -val.y;
                    }

                    /* Prefetch convolution kernel data into "c". */
                    #pragma unroll
                    for (int t = 0; t < TSIZE; t++)
                    {
                        int j = my_grid_v_start - grid_v + t;
                        if (abs(j) <= w_support) /* if (is_my_j) */
                        {
                            const int iy = abs(off_v + j * oversample);
                            c[t] = wkernel[wkernel_idx - iy * conv_len];
                        }
                        else
                        {
                            c[t].x = c[t].y = 0;
                        }
                    }

                    /* The occupancy is not a critical factor.
                     * 8shm + 8reg, 4shm + 12reg, 0shm + 16reg all give
                     * good performance with an occupancy below 35%. */
                    #pragma unroll
                    for (int iter = 0; iter < TSIZE; iter++)
                    {
                        sum += c[iter].x; /* Real part only. */
                        if (iter < REGSZ)
                        {
                            /* When using val2, we have 4 multiply-add
                             * (instead of 3 multiply-add + 1 multiply-sub)
                             * and we also save one multiply. */
                            my_grid[iter].x += (val.x * c[iter].x);
                            my_grid[iter].y += (val.y * c[iter].x);
                            my_grid[iter].x += (val2.y * c[iter].y);
                            my_grid[iter].y += (val2.x * c[iter].y);
                        }
                        else if (SHMSZ > 0)
                        {
                            const int s = iter - REGSZ;
                            FP2 z = sh_mem[sh_mem_off + s * warpSize];
                            z.x += (val.x * c[iter].x);
                            z.y += (val.y * c[iter].x);
                            z.x += (val2.y * c[iter].y);
                            z.y += (val2.x * c[iter].y);
                            sh_mem[sh_mem_off + s * warpSize] = z;
                        }
                    }
                    loc_norm += sum * s_weight[v];
                } /* end if (is_my_k) */
            } /* end for (v) visibilities in shared memory */
        } /* end for (v0) visibility start */
    } /* end while (true) */

    if (tile_u != -1)
    {
        /* Write this tile back to the grid. */
        #pragma unroll
        for (int r = 0; r < REGSZ; r++)
        {
            const int p = (my_grid_v_start + r) * grid_size +
                    my_grid_u;
            oskar_atomic_add<FP>(grid + 2 * p,     my_grid[r].x);
            oskar_atomic_add<FP>(grid + 2 * p + 1, my_grid[r].y);
        }
        #pragma unroll
        for (int s = 0; s < SHMSZ; s++)
        {
            const int p = (my_grid_v_start + s + REGSZ) * grid_size +
                    my_grid_u;
            const FP2 z = sh_mem[sh_mem_off + s * warpSize];
            oskar_atomic_add<FP>(grid + 2 * p,     z.x);
            oskar_atomic_add<FP>(grid + 2 * p + 1, z.y);
        }
    }

    /* Update normalisation value.
     * Don't bother to optimise the atomic add. The threads have a very
     * very long lifetime and this is occurring exactly once per thread. */
    oskar_atomic_add<double>(norm, loc_norm);
}

template <typename FP, typename FP2>
void oskar_grid_wproj_tiled_cuda(
        const int num_w_planes,
        const int* d_support,
        const int oversample,
        const int* d_wkernel_start,
        const FP2* d_wkernel,
        const int num_points,
        const FP* d_uu,
        const FP* d_vv,
        const FP* d_ww,
        const FP2* d_vis,
        const FP* d_weight,
        const FP cell_size_rad,
        const FP w_scale,
        const int grid_size,
        size_t* num_skipped,
        double* norm,
        FP* d_grid)
{
    int count_skipped = 0, num_blocks = 0, total_points = 0;
    int *d_counter = 0, *d_count_skipped = 0, *d_num_points_in_tiles = 0;
    int *d_tile_offsets = 0, *d_sorted_tile = 0, *d_sorted_grid_w = 0;
    double *d_norm = 0, temp_norm = 0.0;
    FP *d_sorted_uu = 0, *d_sorted_vv = 0, *d_sorted_weight = 0;
    FP2 *d_sorted_vis = 0;
    int2 num_tiles, tile_size, top_left;
    size_t sh_mem_size;
    dim3 num_threads;
    float2 inv_tile_size;
    const int grid_centre = grid_size / 2;
    const FP grid_scale = grid_size * cell_size_rad;

    /* Define the tile size and number of tiles in each direction.
     * A tile consists of SHMSZ grid cells per thread in shared memory
     * and REGSZ grid cells per thread in registers. */
    const int SHMSZ = 8;
    const int REGSZ = 8;
    tile_size.x = 32;
    tile_size.y = (SHMSZ + REGSZ);
    num_tiles.x = (grid_size + tile_size.x - 1) / tile_size.x;
    num_tiles.y = (grid_size + tile_size.y - 1) / tile_size.y;

    /* Which tile contains the grid centre? */
    const int c_tile_u = grid_centre / tile_size.x;
    const int c_tile_v = grid_centre / tile_size.y;

    /* Compute difference between centre of centre tile and grid centre
     * to ensure the centre of the grid is in the centre of a tile. */
    top_left.x = grid_centre - c_tile_u * tile_size.x - tile_size.x / 2;
    top_left.y = grid_centre - c_tile_v * tile_size.y - tile_size.y / 2;
    assert(top_left.x <= 0);
    assert(top_left.y <= 0);

    /* Allocate and clear GPU scratch memory. */
    const int total_tiles = num_tiles.x * num_tiles.y;
    cudaMalloc((void**) &d_counter, sizeof(int));
    cudaMalloc((void**) &d_count_skipped, sizeof(int));
    cudaMalloc((void**) &d_norm, sizeof(double));
    cudaMalloc((void**) &d_num_points_in_tiles, total_tiles * sizeof(int));
    cudaMalloc((void**) &d_tile_offsets, (total_tiles + 1) * sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));
    cudaMemset(d_count_skipped, 0, sizeof(int));
    cudaMemset(d_norm, 0, sizeof(double));
    cudaMemset(d_num_points_in_tiles, 0, total_tiles * sizeof(int));
    /* Don't need to clear d_tile_offsets, as it gets overwritten. */

    /* Count the number of elements in each tile,
     * and get the offsets for each. */
    inv_tile_size.x = 1.0f / (float) tile_size.x;
    inv_tile_size.y = 1.0f / (float) tile_size.y;
    num_threads.x = 512;
    num_blocks = (num_points + num_threads.x - 1) / num_threads.x;
    if (num_blocks > 1000) num_blocks = 1000;
    oskar_grid_wproj_count_elements_in_tiles_cudak
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_w_planes, d_support, num_points, d_uu, d_vv, d_ww,
            grid_size, grid_centre, grid_scale, w_scale,
            inv_tile_size, num_tiles, top_left, d_num_points_in_tiles,
            d_count_skipped);
    sh_mem_size = num_threads.x * 2 * sizeof(int);
    oskar_accumulate_tile_offsets_cudak
    OSKAR_CUDAK_CONF(1, num_threads, sh_mem_size) (
            total_tiles, d_num_points_in_tiles, d_tile_offsets);

    /* Get the total number of visibilities to process. */
    cudaMemcpy(&total_points, d_tile_offsets + total_tiles,
            sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count_skipped, d_count_skipped,
            sizeof(int), cudaMemcpyDeviceToHost);
    *num_skipped = (size_t) count_skipped;
    printf("Total points: %d (factor %.3f increase)\n", total_points,
            (float)total_points / (float)num_points);

    /* Bucket sort the data into tiles. */
    cudaMalloc((void**) &d_sorted_uu,     total_points * sizeof(FP));
    cudaMalloc((void**) &d_sorted_vv,     total_points * sizeof(FP));
    cudaMalloc((void**) &d_sorted_grid_w, total_points * sizeof(int));
    cudaMalloc((void**) &d_sorted_weight, total_points * sizeof(FP));
    cudaMalloc((void**) &d_sorted_vis,    total_points * sizeof(FP2));
    cudaMalloc((void**) &d_sorted_tile,   total_points * sizeof(int));
    num_threads.x = 128;
    num_blocks = (num_points + num_threads.x - 1) / num_threads.x;
    if (num_blocks > 1000) num_blocks = 1000;
    oskar_grid_wproj_bucket_sort_cudak
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_w_planes, d_support, num_points, d_uu, d_vv, d_ww, d_vis,
            d_weight, grid_size, grid_centre, grid_scale, w_scale,
            inv_tile_size, num_tiles, top_left, d_tile_offsets,
            d_sorted_uu, d_sorted_vv, d_sorted_grid_w, d_sorted_vis,
            d_sorted_weight, d_sorted_tile);

    /* Update the grid. */
    num_threads.x = tile_size.x;
    num_threads.y = tile_size.y / (REGSZ + SHMSZ);
    num_blocks = (num_points + num_threads.x - 1) / num_threads.x;
    if (num_blocks > 10000) num_blocks = 10000;
    sh_mem_size = sizeof(FP2) * SHMSZ * num_threads.x * num_threads.y;
    oskar_grid_wproj_tiled_cudak<REGSZ, SHMSZ, FP, FP2>
    OSKAR_CUDAK_CONF(num_blocks, num_threads, sh_mem_size) (num_w_planes,
            d_support, oversample, d_wkernel_start, d_wkernel, grid_size,
            grid_centre, grid_scale, tile_size, top_left, total_points,
            d_sorted_uu, d_sorted_vv, d_sorted_grid_w, d_sorted_vis,
            d_sorted_weight, d_sorted_tile, d_counter, d_norm, d_grid);

    /* Update the normalisation value. */
    cudaMemcpy(&temp_norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost);
    *norm += temp_norm;

    /* Free GPU scratch memory. */
    cudaFree(d_counter);
    cudaFree(d_count_skipped);
    cudaFree(d_norm);
    cudaFree(d_num_points_in_tiles);
    cudaFree(d_tile_offsets);
    cudaFree(d_sorted_uu);
    cudaFree(d_sorted_vv);
    cudaFree(d_sorted_grid_w);
    cudaFree(d_sorted_weight);
    cudaFree(d_sorted_vis);
    cudaFree(d_sorted_tile);
}

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_wproj_tiled_cuda_d(
        const int num_w_planes,
        const int* d_support,
        const int oversample,
        const int* d_wkernel_start,
        const double* d_wkernel,
        const int num_points,
        const double* d_uu,
        const double* d_vv,
        const double* d_ww,
        const double* d_vis,
        const double* d_weight,
        const double cell_size_rad,
        const double w_scale,
        const int grid_size,
        size_t* num_skipped,
        double* norm,
        double* d_grid)
{
    oskar_grid_wproj_tiled_cuda(num_w_planes, d_support, oversample,
            d_wkernel_start, (const double2*) d_wkernel, num_points,
            d_uu, d_vv, d_ww, (const double2*) d_vis, d_weight,
            cell_size_rad, w_scale, grid_size, num_skipped, norm, d_grid);
}

void oskar_grid_wproj_tiled_cuda_f(
        const int num_w_planes,
        const int* d_support,
        const int oversample,
        const int* d_wkernel_start,
        const float* d_wkernel,
        const int num_points,
        const float* d_uu,
        const float* d_vv,
        const float* d_ww,
        const float* d_vis,
        const float* d_weight,
        const float cell_size_rad,
        const float w_scale,
        const int grid_size,
        size_t* num_skipped,
        double* norm,
        float* d_grid)
{
    oskar_grid_wproj_tiled_cuda(num_w_planes, d_support, oversample,
            d_wkernel_start, (const float2*) d_wkernel, num_points,
            d_uu, d_vv, d_ww, (const float2*) d_vis, d_weight,
            cell_size_rad, w_scale, grid_size, num_skipped, norm, d_grid);
}

#ifdef __cplusplus
}
#endif
