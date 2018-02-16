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

#include "oskar_global.h"
#include "write_fits_cube.h"
#include "binary/oskar_binary.h"
#include "fitsio.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Write a block of memory as a FITS image. */
static void write_pixels(int precision, void* data, const char* filename,
        int width, int height, int num_planes, int i_plane, int* status)
{
    long naxes[3], firstpix[3], num_pix;
    int dims_ok = 0;
    fitsfile* f = 0;
    FILE* temp = 0;
    if (*status) return;
    temp = fopen(filename, "rb");
    if (temp)
    {
        int naxis = 0, imagetype = 0;
        fclose(temp);
        fits_open_file(&f, filename, READWRITE, status);
        fits_get_img_param(f, 3, &imagetype, &naxis, naxes, status);
        if (naxis == 3 &&
                naxes[0] == width &&
                naxes[1] == height &&
                naxes[2] == num_planes)
        {
            dims_ok = 1;
        }
        else
        {
            *status = 0;
            fits_close_file(f, status);
            remove(filename);
            f = 0;
        }
    }
    if (!dims_ok)
    {
        naxes[0] = width;
        naxes[1] = height;
        naxes[2] = num_planes;
        fits_create_file(&f, filename, status);
        fits_create_img(f, precision == OSKAR_DOUBLE ? DOUBLE_IMG : FLOAT_IMG,
                3, naxes, status);
    }
    if (*status || !f)
    {
        if (f) fits_close_file(f, status);
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    num_pix = width * height;
    firstpix[0] = 1;
    firstpix[1] = 1;
    firstpix[2] = 1 + i_plane;
    if (i_plane < 0)
    {
        firstpix[2] = 1;
        num_pix *= num_planes;
    }
    fits_write_pix(f, precision == OSKAR_DOUBLE ? TDOUBLE : TFLOAT,
            firstpix, num_pix, data, status);
    fits_close_file(f, status);
}

static void convert_complex(size_t num_elements, int precision,
        const void* input, void* output, int offset, int* status)
{
    size_t i;
    if (*status) return;
    if (precision == OSKAR_SINGLE)
    {
        float *out = (float*) output;
        const float *in = (const float*) input;
        for (i = 0; i < num_elements; ++i) out[i] = in[2*i + offset];
    }
    else
    {
        double *out = (double*) output;
        const double *in = (const double*) input;
        for (i = 0; i < num_elements; ++i) out[i] = in[2*i + offset];
    }
}

void write_fits_cube(int data_type, void* data, const char* root_name,
        int width, int height, int num_planes, int i_plane, int* status)
{
    int precision;
    size_t len;
    char* fname;
    if (*status) return;

    /* Construct the filename. */
    len = strlen(root_name);
    fname = (char*) calloc(11 + len, sizeof(char));

    /* Deal with complex data. */
    precision = oskar_type_precision(data_type);
    if (oskar_type_is_complex(data_type))
    {
        void *temp;
        const size_t num_elements = width * height * num_planes;
        temp = calloc(num_elements,
                precision == OSKAR_DOUBLE ? sizeof(double) : sizeof(float));

        /* Extract the real part and write it. */
        sprintf(fname, "%s_REAL.fits", root_name);
        convert_complex(num_elements, precision, data, temp, 0, status);
        write_pixels(precision, temp, fname, width, height, num_planes,
                i_plane, status);

        /* Extract the imaginary part and write it. */
        sprintf(fname, "%s_IMAG.fits", root_name);
        convert_complex(num_elements, precision, data, temp, 1, status);
        write_pixels(precision, temp, fname, width, height, num_planes,
                i_plane, status);
        free(temp);
    }
    else
    {
        /* No conversion needed. */
        if ((len >= 5) && (
                !strcmp(&(root_name[len-5]), ".fits") ||
                !strcmp(&(root_name[len-5]), ".FITS") ))
        {
            sprintf(fname, "%s", root_name);
        }
        else
        {
            sprintf(fname, "%s.fits", root_name);
        }
        write_pixels(precision, data, fname, width, height, num_planes,
                i_plane, status);
    }
    free(fname);
}
