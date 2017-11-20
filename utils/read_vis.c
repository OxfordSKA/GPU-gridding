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

#include "read_vis.h"

void read_coords(oskar_Binary* h, int i_block, int precision,
        size_t num_coords, void* uu, void* vv, void* ww, int* status)
{
    size_t array_size_bytes;
    array_size_bytes = (precision == OSKAR_DOUBLE ?
            sizeof(double) : sizeof(float)) * num_coords;
    oskar_binary_read(h, precision, (unsigned char) OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_BASELINE_UU, i_block,
            array_size_bytes, uu, status);
    oskar_binary_read(h, precision, (unsigned char) OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_BASELINE_VV, i_block,
            array_size_bytes, vv, status);
    oskar_binary_read(h, precision, (unsigned char) OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_BASELINE_WW, i_block,
            array_size_bytes, ww, status);
}

/* Scale coordinates to wavelengths. */
void scale_coords(double freq_hz, int precision,
        size_t num_coords, void* uu, void* vv, void* ww)
{
    size_t i;
    const double scale = freq_hz / 299792458.0;
    if (precision == OSKAR_DOUBLE)
    {
        double *uu_, *vv_, *ww_;
        uu_ = (double*) uu; vv_ = (double*) vv; ww_ = (double*) ww;
        for (i = 0; i < num_coords; ++i)
        {
            uu_[i] *= scale; vv_[i] *= scale; ww_[i] *= scale;
        }
    }
    else if (precision == OSKAR_SINGLE)
    {
        const float scale_ = (const float) scale;
        float *uu_, *vv_, *ww_;
        uu_ = (float*) uu; vv_ = (float*) vv; ww_ = (float*) ww;
        for (i = 0; i < num_coords; ++i)
        {
            uu_[i] *= scale_; vv_[i] *= scale_; ww_[i] *= scale_;
        }
    }
}
