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

#include "binary/oskar_binary.h"
#include "fitsio.h"

#include <stdlib.h>
#include <stdio.h>

/* Read kernel data from FITS file. */
void* read_kernel(const char* file_name, int precision,
        int* conv_size_half, int* num_w_planes, int** support,
        int* oversample, int* grid_size, double* cellsize_rad,
        double* w_scale, int* status)
{
    fitsfile* f = 0;
    int naxis = 0, datatype;
    long int kernel_dims[] = {0, 0, 0};
    long int fpixel[] = {1, 1, 1};
    long int nelements = 0;
    void* data = 0;

    /* Read dimensions. */
    if (*status) return 0;
    printf("Opening '%s'...\n", file_name);
    fits_open_file(&f, file_name, READONLY, status);
    fits_get_img_dim(f, &naxis,  status);
    if (*status || naxis != 3)
    {
        fprintf(stderr, "Need a 3D kernel FITS cube.\n");
        if (!*status) *status = 1;
        goto done;
    }
    fits_get_img_size(f, 3, kernel_dims, status);
    *conv_size_half = (int)(kernel_dims[0]);
    *num_w_planes = (int)(kernel_dims[2]);

    /* Read the header keywords. */
    fits_read_key(f, TINT, "OVERSAMP", oversample, NULL, status);
    if (*status)
    {
        fprintf(stderr, "Missing 'OVERSAMP' FITS header keyword.\n");
        goto done;
    }
    fits_read_key(f, TINT, "GRIDSIZE", grid_size, NULL, status);
    if (*status)
    {
        fprintf(stderr, "Missing 'GRIDSIZE' FITS header keyword.\n");
        goto done;
    }
    fits_read_key(f, TDOUBLE, "CELLSIZE", cellsize_rad, NULL, status);
    if (*status)
    {
        fprintf(stderr, "Missing 'CELLSIZE' FITS header keyword.\n");
        goto done;
    }
    fits_read_key(f, TDOUBLE, "W_SCALE", w_scale, NULL, status);
    if (*status)
    {
        fprintf(stderr, "Missing 'W_SCALE' FITS header keyword.\n");
        goto done;
    }

    /* Read kernel data. */
    datatype = precision == OSKAR_DOUBLE ? TDOUBLE : TFLOAT;
    nelements = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
    data = calloc(nelements,
            precision == OSKAR_DOUBLE ? sizeof(double) : sizeof(float));
    fits_read_pix(f, datatype, fpixel, nelements, NULL, data, NULL, status);

    /* Read the kernel support sizes from the binary table. */
    if (support)
    {
        *support = (int*) calloc(*num_w_planes, sizeof(int));
        fits_movnam_hdu(f, BINARY_TBL, "W_KERNELS", 0, status);
        if (*status)
        {
            fprintf(stderr, "Missing 'W_KERNELS' FITS binary table.\n");
            goto done;
        }
        fits_read_col(f, TINT, 1, 1, 1, *num_w_planes,
                NULL, *support, NULL, status);
        if (*status)
        {
            fprintf(stderr, "Unable to read kernel support sizes.\n");
            goto done;
        }
    }

done:
    fits_close_file(f, status);
    return data;
}
