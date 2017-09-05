/*
 * Copyright (c) 2017, The University of Oxford
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define DBL sizeof(double)
#define FLT sizeof(float)
#define INT sizeof(int)

#include "oskar_grid_wproj.hpp"


static int check_float(float a, float b)
{
#if 1
   // Use these when checking the SKA data - it was created in 'mixed precision'
   const float xover = 0.1;
   const float eps = 9e-4;
#else
   const float xover = 0.05;
   const float eps = 5e-4;
#endif

   const float fa = fabs(a), fb = fabs(b);
   const float smallest = (fa < fb ? fa : fb);
   const float largest  = (fa < fb ? fb : fa);

   // If both values are small, use absolute difference
   if(largest < xover) {
      const float diff = fabs(a-b);
      if( diff < eps ) 
         return 1;
      else {
         printf("check_float sees %.7g, \t%.7g \t: abs diff=%g\n", a, b, diff);
       return 0;
      }
   }
   else {

      // At least one value is big, use relative difference
      const float diff = fabs(a-b) / largest;
      if(diff < eps) {
         return 1;
      }
      else {
         printf("check_float sees %.7g, \t%.7g \t: rel diff=%g\n", a, b, diff);
         return 0;
      }
   }

   /*
    float largest;
    const float diff = fabs(a - b);
    if (diff <= FLT_EPSILON) {
      return 1;
    }

    a = fabs(a);
    b = fabs(b);
    largest = (b > a) ? b : a;
    if (diff <= largest * 5e-5) {
      return 1;
    }
    return 0;
    */
}


int main(int argc, const char** argv)
{
    int i = 0, error = 0;
    FILE* f = 0;

    /* Function inputs. */
    int num_w_planes = 0, oversample = 0, conv_size_half = 0, grid_size = 0;
    int num_vis = 0, w_kernels_size = 0;
    int *w_support = 0;
    float *w_kernels = 0, *uu = 0, *vv = 0, *ww = 0, *vis = 0, *weight = 0;
    double cellsize_rad = 0.0, w_scale = 0.0;

    /* Function outputs. */
    int num_skipped = 0;
    double norm = 0.0;

    /* Function inputs/outputs. */
    float *grid = 0;

    /* Test data for comparison. */
    int test_grid_size = 0, test_num_skipped = 0, test_num_grid_points = 0;
    double test_norm = 0.0;
    float *test_grid = 0;

    /* Read convolution kernel data. */
    if(argc != 3) {
       fprintf(stderr, "I expect exactly two arguments, the input file and reference results file.\n");
       return 1;
    }
    printf("Reading input data file: %s\n", argv[1]);
    f = fopen(argv[1], "rb");
    if (!f)
    {
        fprintf(stderr, "Unable to open input data file.\n");
        return 1;
    }
    fread(&num_w_planes, INT, 1, f);
    w_support = (int*)calloc(num_w_planes, INT);
    fread(w_support, INT, num_w_planes, f);
    fread(&oversample, INT, 1, f);
    fread(&conv_size_half, INT, 1, f);
    w_kernels_size = num_w_planes * conv_size_half * conv_size_half;
    w_kernels = (float*)calloc(w_kernels_size, 2*FLT); /* complex */
    fread(w_kernels, 2*FLT, w_kernels_size, f);

    /* Read input visibility data. */
    fread(&num_vis, INT, 1, f);
    uu = (float*)calloc(num_vis, FLT);
    vv = (float*)calloc(num_vis, FLT);
    ww = (float*)calloc(num_vis, FLT);
    vis = (float*)calloc(num_vis, 2*FLT); /* complex */
    weight = (float*)calloc(num_vis, FLT);
    fread(uu, FLT, num_vis, f);
    fread(vv, FLT, num_vis, f);
    fread(ww, FLT, num_vis, f);
    fread(vis, 2*FLT, num_vis, f);
    fread(weight, FLT, num_vis, f);

    /* Read grid parameters. */
    fread(&cellsize_rad, DBL, 1, f);
    fread(&w_scale, DBL, 1, f);
    fread(&grid_size, INT, 1, f);
    grid = (float*)calloc(grid_size * grid_size, 2*FLT); /* complex */
    fclose(f);

    /* Call the gridder. */
    printf("Gridding data...\n");
    oskar_grid_wproj_f(num_w_planes, 
          w_support, 
          oversample, 
          conv_size_half,
          w_kernels, 
          num_vis, 
          uu, 
          vv, 
          ww, 
          vis, 
          weight, 
          cellsize_rad,
          w_scale, 
          grid_size, 
          &num_skipped, 
          &norm, grid);

    /* Load the test data to check against. */
    printf("Reading output data file: %s\n", argv[2]);
    f = fopen(argv[2], "rb");
    if (!f)
    {
        error = 1;
        fprintf(stderr, "Unable to open output data file.\n");
        goto fail;
    }
    fread(&test_num_skipped, INT, 1, f);
    fread(&test_norm, DBL, 1, f);
    fread(&test_grid_size, INT, 1, f);
    test_num_grid_points = test_grid_size * test_grid_size;
    test_grid = (float*)calloc(test_num_grid_points, 2*FLT); /* complex */
    fread(test_grid, 2*FLT, test_num_grid_points, f);
    fclose(f);

    /* Check data. */
    printf("Checking results...\n");
    if (test_num_skipped != num_skipped)
    {
        error = 1;
        fprintf(stderr, "Inconsistent number of input points skipped.\n");
        goto fail;
    }
    if (test_grid_size != grid_size)
    {
        error = 1;
        fprintf(stderr, "Inconsistent grid dimensions.\n");
        goto fail;
    }
    if (!check_float(test_norm, norm))
    {
        error = 1;
        fprintf(stderr, "Inconsistent normalisation values.\n");
        goto fail;
    }
    for (i = 0; i < test_num_grid_points; ++i)
    {

        if (!check_float(test_grid[2*i], grid[2*i]) ||
                !check_float(test_grid[2*i + 1], grid[2*i + 1]))
        {
            error = 1;
            fprintf(stderr, "Inconsistent grid values (cell %i).\n", i);
            goto fail;
        }
    }
    printf("All checks passed.\n");

    /* Free memory. */
fail:
    free(w_support);
    free(w_kernels);
    free(uu);
    free(vv);
    free(ww);
    free(vis);
    free(weight);
    free(grid);
    free(test_grid);

    return error;
}
