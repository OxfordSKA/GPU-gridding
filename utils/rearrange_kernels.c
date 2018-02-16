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

#include "rearrange_kernels.h"
#include "binary/oskar_binary.h"

#include <stdlib.h>
#include <stdio.h>

typedef struct { float x, y; } float2;
typedef struct { double x, y; } double2;

/* Rearrange convolution kernel data. */
void rearrange_kernels(const int num_w_planes, const int* support,
        const int oversample, const int conv_size_half, const int prec,
        const void* kernels_in, size_t* rearranged_size, void** kernels_out,
        int* rearranged_kernel_start)
{
    int w, j, k, off_u, off_v;
    size_t element_size = 0;
    float2*  out_f;
    double2* out_d;
    const float2*  in_f = kernels_in;
    const double2* in_d = kernels_in;
    const int oversample_h = oversample / 2;
    const int height = oversample_h + 1;

    /* Allocate enough memory for the rearranged kernels. */
    *rearranged_size = 0;
    for (w = 0; w < num_w_planes; w++)
    {
        const int conv_len = 2 * support[w] + 1;
        const int width = (oversample_h * conv_len + 1) * conv_len;
        rearranged_kernel_start[w] = (int) *rearranged_size;
        *rearranged_size += (width * height);
    }
    element_size = (prec == OSKAR_SINGLE ? sizeof(float) : sizeof(double));
    *kernels_out = realloc(*kernels_out, *rearranged_size * element_size);
    out_f = (float2*)  *kernels_out;
    out_d = (double2*) *kernels_out;

    for (w = 0; w < num_w_planes; w++)
    {
        const int w_support = support[w];
        const int conv_len = 2 * support[w] + 1;
        const int width = (oversample_h * conv_len + 1) * conv_len;
        const int c_in = w * conv_size_half * conv_size_half;
        const int c_out = rearranged_kernel_start[w];

        /* Within each kernel, off_u is slowest varying, so if the
         * rearranged kernel is viewed as a very squashed image, the
         * row index to use for off_u is given by width * abs(off_u).
         *
         * "Reverse" rows and rearrange elements in U dimension
         * for reasonable cache utilisation.
         * For idx_v = abs(off_v + j * oversample), the offset from the
         * start of the row is given by width-1 - (idx_v * conv_len).
         * Stride is +1 for positive or zero values of off_u;
         * stride is -1 for negative values of off_u.
         */
        for (off_u = -oversample_h; off_u <= oversample_h; off_u++)
        {
            const int mid = c_out + (abs(off_u) + 1) * width - 1 - w_support;
            const int stride = (off_u >= 0) ? 1 : -1;
            for (off_v = -oversample_h; off_v <= oversample_h; off_v++)
            {
                for (j = 0; j <= w_support; j++)
                {
                    const int idx_v = abs(off_v + j * oversample);
                    const int p = mid - idx_v * conv_len;
                    for (k = -w_support; k <= w_support; k++)
                    {
                        const int idx_u = abs(off_u + k * oversample);
                        const int a = c_in + idx_v * conv_size_half + idx_u;
                        const int b = p + stride * k;
                        if (prec == OSKAR_SINGLE)
                            out_f[b] = in_f[a];
                        else
                            out_d[b] = in_d[a];
                    }
                }
            }
        }
    }
}
