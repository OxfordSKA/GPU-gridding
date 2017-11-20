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

#ifndef READ_VIS_H_
#define READ_VIS_H_

#include <stddef.h>
#include "binary/oskar_binary.h"

#ifdef __cplusplus
extern "C" {
#endif

enum OSKAR_VIS_HEADER_TAGS
{
    OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK       = 2,
    OSKAR_VIS_HEADER_TAG_AMP_TYPE                 = 5,
    OSKAR_VIS_HEADER_TAG_COORD_PRECISION          = 6,
    OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK      = 7,
    OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL          = 8,
    OSKAR_VIS_HEADER_TAG_MAX_CHANNELS_PER_BLOCK   = 9,
    OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL       = 10,
    OSKAR_VIS_HEADER_TAG_NUM_STATIONS             = 11,
    OSKAR_VIS_HEADER_TAG_FREQ_START_HZ            = 23
};

enum OSKAR_VIS_BLOCK_TAGS
{
    OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE        = 1,
    OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS        = 3,
    OSKAR_VIS_BLOCK_TAG_BASELINE_UU               = 4,
    OSKAR_VIS_BLOCK_TAG_BASELINE_VV               = 5,
    OSKAR_VIS_BLOCK_TAG_BASELINE_WW               = 6
};

/* Return visibility baseline coordinates. */
void read_coords(oskar_Binary* h, int i_block, int precision,
        size_t num_coords, void* uu, void* vv, void* ww, int* status);

/* Scale coordinates to wavelengths. */
void scale_coords(double freq_hz, int precision,
        size_t num_coords, void* uu, void* vv, void* ww);

#ifdef __cplusplus
}
#endif

#endif
