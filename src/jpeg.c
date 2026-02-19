#include "vabackend.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* JPEG Decode Implementation
 *
 * VA-API supplies JPEG data as separate buffers (picture params, IQ tables,
 * Huffman tables, slice data). NVDEC expects a complete JPEG bitstream.
 * This codec reconstructs a minimal JFIF-compliant JPEG from VA buffers and
 * feeds it to NVDEC.
 */

// JPEG marker bytes (JPEG spec: ISO/IEC 10918-1)
#define JPEG_SOI        0xD8    // Start of Image
#define JPEG_EOI        0xD9    // End of Image
#define JPEG_APP0       0xE0    // JFIF application marker
#define JPEG_DQT        0xDB    // Define Quantization Table
#define JPEG_SOF0       0xC0    // Start of Frame (Baseline DCT)
#define JPEG_DHT        0xC4    // Define Huffman Table
#define JPEG_DRI        0xDD    // Define Restart Interval
#define JPEG_SOS        0xDA    // Start of Scan
#define JPEG_MARKER     0xFF    // Marker prefix byte

// JPEG context to store parameters between buffer calls
typedef struct {
    VAPictureParameterBufferJPEGBaseline picParams;
    VAIQMatrixBufferJPEGBaseline         iqMatrix;
    VAHuffmanTableBufferJPEGBaseline     huffmanTable;
    int                                  hasPicParams;
    int                                  hasIQMatrix;
    int                                  hasHuffmanTable;
} JPEGContext;

// Minimal APP0/JFIF header
static const uint8_t jfifHeader[] = {
    JPEG_MARKER, JPEG_SOI,           // Start of Image
    JPEG_MARKER, JPEG_APP0,          // APP0 marker
    0x00, 0x10,                      // Length (16 bytes)
    0x4A, 0x46, 0x49, 0x46, 0x00,    // "JFIF\0"
    0x01, 0x01,                      // Version 1.1
    0x00,                            // Units (0 = none)
    0x00, 0x01,                      // X density
    0x00, 0x01,                      // Y density
    0x00, 0x00                       // No thumbnail
};

// Standard Huffman tables for baseline JPEG (YCbCr)
static const uint8_t dcLuminanceBits[]   = {0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t dcLuminanceVals[]   = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const uint8_t dcChrominanceBits[] = {0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
static const uint8_t dcChrominanceVals[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

static const uint8_t acLuminanceBits[] = {0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
static const uint8_t acLuminanceVals[] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

static const uint8_t acChrominanceBits[] = {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
static const uint8_t acChrominanceVals[] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

// Write 16-bit big-endian value
static void write16be(uint8_t *ptr, uint16_t value) {
    ptr[0] = (value >> 8) & 0xFF;
    ptr[1] = value & 0xFF;
}

// Per-context JPEG state table (keeps vabackend untouched)
#define MAX_JPEG_CONTEXTS 128
typedef struct {
    NVDriver    *drv;
    NVContext   *ctx;
    CUvideodecoder decoder; // used to detect NVContext reuse at same address
    JPEGContext jpegCtx;
} JPEGContextEntry;

static JPEGContextEntry jpegContexts[MAX_JPEG_CONTEXTS];
static pthread_mutex_t jpegContextsMutex = PTHREAD_MUTEX_INITIALIZER;

static bool contextPtrInDriver(NVDriver *drv, NVContext *ctx) {
    bool found = false;
    pthread_mutex_lock(&drv->objectCreationMutex);
    ARRAY_FOR_EACH(Object, o, &drv->objects)
        if (o->type == OBJECT_TYPE_CONTEXT && o->obj == ctx) {
            found = true;
            break;
        }
    END_FOR_EACH
    pthread_mutex_unlock(&drv->objectCreationMutex);
    return found;
}

static void pruneContextsForDriver(NVDriver *drv) {
    for (int i = 0; i < MAX_JPEG_CONTEXTS; i++) {
        if (jpegContexts[i].drv != drv || jpegContexts[i].ctx == NULL) {
            continue;
        }
        if (!contextPtrInDriver(drv, jpegContexts[i].ctx)) {
            jpegContexts[i].drv = NULL;
            jpegContexts[i].ctx = NULL;
            jpegContexts[i].decoder = NULL;
            memset(&jpegContexts[i].jpegCtx, 0, sizeof(jpegContexts[i].jpegCtx));
        }
    }
}

static JPEGContext* getJPEGContext(NVContext *ctx) {
    pthread_mutex_lock(&jpegContextsMutex);

    pruneContextsForDriver(ctx->drv);

    // lookup
    for (int i = 0; i < MAX_JPEG_CONTEXTS; i++) {
        if (jpegContexts[i].ctx == ctx) {
            // If the context was freed and reused at the same address, reset state.
            if (jpegContexts[i].decoder != ctx->decoder) {
                jpegContexts[i].decoder = ctx->decoder;
                memset(&jpegContexts[i].jpegCtx, 0, sizeof(jpegContexts[i].jpegCtx));
            }
            pthread_mutex_unlock(&jpegContextsMutex);
            return &jpegContexts[i].jpegCtx;
        }
    }

    // allocate new slot
    for (int i = 0; i < MAX_JPEG_CONTEXTS; i++) {
        if (jpegContexts[i].ctx == NULL) {
            jpegContexts[i].drv = ctx->drv;
            jpegContexts[i].ctx = ctx;
            jpegContexts[i].decoder = ctx->decoder;
            memset(&jpegContexts[i].jpegCtx, 0, sizeof(jpegContexts[i].jpegCtx));
            pthread_mutex_unlock(&jpegContextsMutex);
            return &jpegContexts[i].jpegCtx;
        }
    }

    pthread_mutex_unlock(&jpegContextsMutex);
    LOG("JPEG: No free context slots");
    return NULL;
}

// Write DQT (Define Quantization Table) segment
// Only write tables that are actually used by components
static uint8_t* writeDQT(uint8_t *ptr, VAIQMatrixBufferJPEGBaseline *iq, VAPictureParameterBufferJPEGBaseline *pic) {
    int tablesUsed[4] = {0, 0, 0, 0};
    for (int i = 0; i < pic->num_components; i++) {
        int tableId = pic->components[i].quantiser_table_selector;
        if (tableId >= 0 && tableId < 4) {
            tablesUsed[tableId] = 1;
        }
    }

    for (int table = 0; table < 4; table++) {
        if (iq->load_quantiser_table[table] && tablesUsed[table]) {
            int sum = 0;
            for (int j = 0; j < 64; j++) {
                sum += iq->quantiser_table[table][j];
            }
            if (sum == 0) continue;  // Skip empty tables

            *ptr++ = JPEG_MARKER;
            *ptr++ = JPEG_DQT;  // Define Quantization Table
            write16be(ptr, 67);  // Length (2 + 1 + 64)
            ptr += 2;
            *ptr++ = table;  // Table ID
            memcpy(ptr, iq->quantiser_table[table], 64);
            ptr += 64;
        }
    }
    return ptr;
}

// Write DRI (Define Restart Interval) segment
static uint8_t* writeDRI(uint8_t *ptr, uint16_t restartInterval) {
    *ptr++ = JPEG_MARKER;
    *ptr++ = JPEG_DRI;  // Define Restart Interval
    write16be(ptr, 4);  // Length (2 + 2 bytes interval)
    ptr += 2;
    write16be(ptr, restartInterval);
    ptr += 2;
    return ptr;
}

// Write SOF0 (Start of Frame) segment
static uint8_t* writeSOF0(uint8_t *ptr, VAPictureParameterBufferJPEGBaseline *pic) {
    *ptr++ = JPEG_MARKER;
    *ptr++ = JPEG_SOF0;  // Start of Frame
    uint16_t length = 2 + 1 + 2 + 2 + 1 + (pic->num_components * 3);
    write16be(ptr, length);
    ptr += 2;
    *ptr++ = 8;  // Precision (8 bits)
    write16be(ptr, pic->picture_height);
    ptr += 2;
    write16be(ptr, pic->picture_width);
    ptr += 2;
    *ptr++ = pic->num_components;

    for (int i = 0; i < pic->num_components; i++) {
        *ptr++ = pic->components[i].component_id;
        *ptr++ = (pic->components[i].h_sampling_factor << 4) |
                 pic->components[i].v_sampling_factor;
        *ptr++ = pic->components[i].quantiser_table_selector;
    }
    return ptr;
}

// Write DHT (Define Huffman Table) segment
static uint8_t* writeDHT(uint8_t *ptr, const uint8_t *bits, const uint8_t *vals,
                         int tableClass, int tableId, int numVals) {
    *ptr++ = JPEG_MARKER;
    *ptr++ = JPEG_DHT;  // Define Huffman Table
    uint16_t length = 2 + 1 + 16 + numVals;
    write16be(ptr, length);
    ptr += 2;
    *ptr++ = (tableClass << 4) | tableId;
    memcpy(ptr, bits, 16);
    ptr += 16;
    memcpy(ptr, vals, numVals);
    ptr += numVals;
    return ptr;
}

// Write standard Huffman tables
static uint8_t* writeStandardHuffmanTables(uint8_t *ptr) {
    ptr = writeDHT(ptr, dcLuminanceBits, dcLuminanceVals, 0, 0, 12);
    ptr = writeDHT(ptr, dcChrominanceBits, dcChrominanceVals, 0, 1, 12);
    ptr = writeDHT(ptr, acLuminanceBits, acLuminanceVals, 1, 0, 162);
    ptr = writeDHT(ptr, acChrominanceBits, acChrominanceVals, 1, 1, 162);
    return ptr;
}

static bool countHuffValues(const uint8_t codes[16], unsigned int maxVals, unsigned int *outCount) {
    unsigned int sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += codes[i];
        if (sum > maxVals) {
            return false;
        }
    }
    *outCount = sum;
    return true;
}

// Write Huffman tables from VA-API. Returns false if tables are invalid/out-of-range.
static bool writeVAHuffmanTables(uint8_t **pptr, VAHuffmanTableBufferJPEGBaseline *huffman) {
    uint8_t *ptr = *pptr;
    for (int tableIdx = 0; tableIdx < 2; tableIdx++) {
        if (huffman->load_huffman_table[tableIdx]) {
            unsigned int numDcValues = 0;
            if (!countHuffValues(huffman->huffman_table[tableIdx].num_dc_codes, 12, &numDcValues) ||
                numDcValues == 0) {
                LOG("JPEG: Invalid DC Huffman table %d (count=%u)", tableIdx, numDcValues);
                return false;
            }
            ptr = writeDHT(ptr,
                          huffman->huffman_table[tableIdx].num_dc_codes,
                          huffman->huffman_table[tableIdx].dc_values,
                          0, tableIdx, (int)numDcValues);

            unsigned int numAcValues = 0;
            if (!countHuffValues(huffman->huffman_table[tableIdx].num_ac_codes, 162, &numAcValues) ||
                numAcValues == 0) {
                LOG("JPEG: Invalid AC Huffman table %d (count=%u)", tableIdx, numAcValues);
                return false;
            }
            ptr = writeDHT(ptr,
                          huffman->huffman_table[tableIdx].num_ac_codes,
                          huffman->huffman_table[tableIdx].ac_values,
                          1, tableIdx, (int)numAcValues);
        }
    }
    *pptr = ptr;
    return true;
}

// Write SOS (Start of Scan) segment
static uint8_t* writeSOS(uint8_t *ptr, VAPictureParameterBufferJPEGBaseline *pic,
                         VASliceParameterBufferJPEGBaseline *slice) {
    *ptr++ = JPEG_MARKER;
    *ptr++ = JPEG_SOS;  // Start of Scan
    uint16_t length = 2 + 1 + (slice->num_components * 2) + 3;
    write16be(ptr, length);
    ptr += 2;
    *ptr++ = slice->num_components;

    for (int i = 0; i < slice->num_components; i++) {
        *ptr++ = slice->components[i].component_selector;
        *ptr++ = (slice->components[i].dc_table_selector << 4) |
                  slice->components[i].ac_table_selector;
    }
    *ptr++ = 0;   // Ss (start of spectral selection)
    *ptr++ = 63;  // Se (end of spectral selection)
    *ptr++ = 0;   // Ah/Al (successive approximation)
    return ptr;
}

// Reconstruct complete JPEG frame
static uint8_t* reconstructJPEG(JPEGContext *jpegCtx,
                                VASliceParameterBufferJPEGBaseline *slices,
                                unsigned int sliceCount,
                                uint8_t *sliceData, uint32_t sliceDataSize, uint32_t *outSize) {
    if (!jpegCtx->hasPicParams || !jpegCtx->hasIQMatrix) {
        LOG("JPEG: Missing picture params or IQ matrix");
        return NULL;
    }

    if (jpegCtx->picParams.picture_width == 0 || jpegCtx->picParams.picture_height == 0) {
        LOG("JPEG: Invalid dimensions: %ux%u", jpegCtx->picParams.picture_width, jpegCtx->picParams.picture_height);
        return NULL;
    }

    if (jpegCtx->picParams.num_components == 0 || jpegCtx->picParams.num_components > 4) {
        LOG("JPEG: Unsupported frame component count: %u", jpegCtx->picParams.num_components);
        return NULL;
    }

    if (sliceCount == 0) {
        LOG("JPEG: No slice parameters");
        return NULL;
    }

    // Validate slice bounds and compute total entropy-coded payload
    uint64_t totalEcsSize = 0;
    for (unsigned int i = 0; i < sliceCount; i++) {
        VASliceParameterBufferJPEGBaseline *slice = &slices[i];

        if (slice->slice_data_flag != VA_SLICE_DATA_FLAG_ALL) {
            LOG("JPEG: slice_data_flag=%u not supported (expected ALL)", slice->slice_data_flag);
            return NULL;
        }

        if (slice->slice_data_offset > sliceDataSize) {
            LOG("JPEG: Invalid slice_data_offset (%u) exceeds buffer size (%u)",
                slice->slice_data_offset, sliceDataSize);
            return NULL;
        }
        uint32_t availableData = sliceDataSize - slice->slice_data_offset;
        if (slice->slice_data_size > availableData) {
            LOG("JPEG: Invalid slice_data_size (%u) exceeds available data (%u)",
            slice->slice_data_size, availableData);
            return NULL;
        }

        if (UINT64_MAX - totalEcsSize < slice->slice_data_size) {
            LOG("JPEG: Total ECS size overflow");
            return NULL;
        }
        totalEcsSize += slice->slice_data_size;
    }

    // Use the first slice as the scan header source. In the common case,
    // VA-API provides multiple slices for a single scan with identical scan header.
    VASliceParameterBufferJPEGBaseline *slice0 = &slices[0];

    if (slice0->num_components == 0 || slice0->num_components > 4) {
        LOG("JPEG: Unsupported scan component count: %u", slice0->num_components);
        return NULL;
    }

    bool allSameSOSHeader = true;
    bool allSameRestartInterval = true;
    for (unsigned int i = 1; i < sliceCount; i++) {
        VASliceParameterBufferJPEGBaseline *s = &slices[i];

        if (s->restart_interval != slice0->restart_interval) {
            allSameRestartInterval = false;
        }

        if (s->num_components != slice0->num_components) {
            allSameSOSHeader = false;
            continue;
        }

        for (int c = 0; c < s->num_components && c < 4; c++) {
            if (s->components[c].component_selector != slice0->components[c].component_selector ||
                s->components[c].dc_table_selector  != slice0->components[c].dc_table_selector ||
                s->components[c].ac_table_selector  != slice0->components[c].ac_table_selector) {
                allSameSOSHeader = false;
                break;
            }
        }
    }

    // Worst-case buffer size calculation (overestimate for safety)
    // All markers are 2 bytes (0xFF + marker byte)
    const uint64_t dqtSize = 4 * (2 + 2 + 1 + 64);           // 4 tables max: marker+length+id+64bytes
    const uint64_t sof0Size = 2 + 2 + 1 + 2 + 2 + 1 + (uint64_t)jpegCtx->picParams.num_components * 3;  // marker+length+precision+height+width+num_comps+comps
    const uint64_t dhtSize = 2 * ((2 + 2 + 1 + 16 + 12) +   // 2 DC tables: marker+length+class_id+16codes+12values
                                  (2 + 2 + 1 + 16 + 162));  // 2 AC tables: marker+length+class_id+16codes+162values
    const uint64_t driSize = (uint64_t)sliceCount * (2 + 2 + 2);  // marker+length+interval
    const uint64_t sosSize = (uint64_t)sliceCount * (2 + 2 + 1 + 4 * 2 + 3);  // marker+length+num_comps+comp_selectors+spectral
    
    uint64_t maxSize64 = sizeof(jfifHeader) + dqtSize + sof0Size + dhtSize + driSize + sosSize + totalEcsSize + 2;

    if (maxSize64 > SIZE_MAX) {
        LOG("JPEG: Frame size too large to allocate (%llu bytes)", (unsigned long long)maxSize64);
        return NULL;
    }

    uint8_t *frame = (uint8_t*) malloc((size_t)maxSize64);
    if (!frame) {
        LOG("JPEG: Failed to allocate frame buffer");
        return NULL;
    }

    uint8_t *ptr = frame;

    // 1. SOI + JFIF header
    memcpy(ptr, jfifHeader, sizeof(jfifHeader));
    ptr += sizeof(jfifHeader);

    // 2. DQT
    ptr = writeDQT(ptr, &jpegCtx->iqMatrix, &jpegCtx->picParams);

    // 3. SOF0
    ptr = writeSOF0(ptr, &jpegCtx->picParams);

    // 4. DHT (VA tables if present, else standard)
    if (jpegCtx->hasHuffmanTable) {
        uint8_t *tmp = ptr;
        if (writeVAHuffmanTables(&tmp, &jpegCtx->huffmanTable)) {
            ptr = tmp;
        } else {
            ptr = writeStandardHuffmanTables(ptr);
        }
    } else {
        ptr = writeStandardHuffmanTables(ptr);
    }

    // 4b. DRI (Restart interval) once if consistent across slices
    if (allSameRestartInterval && slice0->restart_interval != 0) {
        ptr = writeDRI(ptr, slice0->restart_interval);
    }

    // 4b/5/6. Scan(s)
    if (allSameSOSHeader) {
        ptr = writeSOS(ptr, &jpegCtx->picParams, slice0);
        for (unsigned int i = 0; i < sliceCount; i++) {
            VASliceParameterBufferJPEGBaseline *slice = &slices[i];
            memcpy(ptr, sliceData + slice->slice_data_offset, slice->slice_data_size);
            ptr += slice->slice_data_size;
        }
    } else {
        for (unsigned int i = 0; i < sliceCount; i++) {
            VASliceParameterBufferJPEGBaseline *slice = &slices[i];
            if (slice->num_components == 0 || slice->num_components > 4) {
                LOG("JPEG: Unsupported scan component count: %u", slice->num_components);
                free(frame);
                return NULL;
            }

            // If restart_interval wasn't consistent globally, emit per-scan DRI.
            if (!allSameRestartInterval && slice->restart_interval != 0) {
                ptr = writeDRI(ptr, slice->restart_interval);
            }
            ptr = writeSOS(ptr, &jpegCtx->picParams, slice);
            memcpy(ptr, sliceData + slice->slice_data_offset, slice->slice_data_size);
            ptr += slice->slice_data_size;
        }
    }

    // 7. EOI (avoid duplicating if client already included it)
    if (!(ptr - frame >= 2 && ptr[-2] == JPEG_MARKER && ptr[-1] == JPEG_EOI)) {
        *ptr++ = JPEG_MARKER;
        *ptr++ = JPEG_EOI;
    }

    *outSize = (uint32_t)(ptr - frame);
    return frame;
}

static void copyJPEGPicParam(NVContext *ctx, NVBuffer* buffer, CUVIDPICPARAMS *picParams)
{
    VAPictureParameterBufferJPEGBaseline* buf = (VAPictureParameterBufferJPEGBaseline*) buffer->ptr;
    JPEGContext *jpegCtx = getJPEGContext(ctx);

    if (jpegCtx) {
        memcpy(&jpegCtx->picParams, buf, sizeof(VAPictureParameterBufferJPEGBaseline));
        if (buf->picture_width != 0 && buf->picture_height != 0 && buf->num_components > 0 && buf->num_components <= 4) {
            jpegCtx->hasPicParams = 1;
        } else {
            jpegCtx->hasPicParams = 0;
        }
    }

    picParams->PicWidthInMbs    = (int) (buf->picture_width  + 15) / 16;
    picParams->FrameHeightInMbs = (int) (buf->picture_height + 15) / 16;
    picParams->field_pic_flag    = 0;
    picParams->bottom_field_flag = 0;
    picParams->second_field      = 0;
    picParams->intra_pic_flag    = 1;
    picParams->ref_pic_flag      = 0;
}

static void copyJPEGIQMatrix(NVContext *ctx, NVBuffer* buffer, CUVIDPICPARAMS *picParams)
{
    VAIQMatrixBufferJPEGBaseline* buf = (VAIQMatrixBufferJPEGBaseline*) buffer->ptr;
    JPEGContext *jpegCtx = getJPEGContext(ctx);

    if (jpegCtx) {
        memcpy(&jpegCtx->iqMatrix, buf, sizeof(VAIQMatrixBufferJPEGBaseline));
        jpegCtx->hasIQMatrix = 1;
    }
}

static void copyJPEGHuffmanTable(NVContext *ctx, NVBuffer* buffer, CUVIDPICPARAMS *picParams)
{
    VAHuffmanTableBufferJPEGBaseline* buf = (VAHuffmanTableBufferJPEGBaseline*) buffer->ptr;
    JPEGContext *jpegCtx = getJPEGContext(ctx);

    if (jpegCtx) {
        memcpy(&jpegCtx->huffmanTable, buf, sizeof(VAHuffmanTableBufferJPEGBaseline));
        jpegCtx->hasHuffmanTable = 1;
    }
}

static void copyJPEGSliceParam(NVContext *ctx, NVBuffer* buf, CUVIDPICPARAMS *picParams)
{
    ctx->lastSliceParams = buf->ptr;
    ctx->lastSliceParamsCount = buf->elements;
}

static void copyJPEGSliceData(NVContext *ctx, NVBuffer* buf, CUVIDPICPARAMS *picParams)
{
    JPEGContext *jpegCtx = getJPEGContext(ctx);
    if (!jpegCtx) {
        LOG("JPEG: Failed to get context");
        return;
    }

    if (!ctx->lastSliceParams || ctx->lastSliceParamsCount == 0) {
        LOG("JPEG: No slice parameters available");
        return;
    }

    VASliceParameterBufferJPEGBaseline *slices =
        (VASliceParameterBufferJPEGBaseline*) ctx->lastSliceParams;

    LOG("JPEG: Processing %u slice(s), input size %zu bytes", ctx->lastSliceParamsCount, buf->size);

    if (buf->size > UINT32_MAX) {
        LOG("JPEG: Slice data too large (%zu bytes)", buf->size);
        return;
    }

    uint32_t frameSize = 0;
    uint8_t *frame = reconstructJPEG(jpegCtx, slices, ctx->lastSliceParamsCount,
                                     (uint8_t*)buf->ptr, (uint32_t)buf->size, &frameSize);
    if (!frame) {
        LOG("JPEG: Failed to reconstruct JPEG frame");
        return;
    }

    // NVDEC can consume a full JPEG as a single "slice" (same approach as FFmpeg's mjpeg_nvdec)
    picParams->nNumSlices = 1;
    uint32_t offset = (uint32_t) ctx->bitstreamBuffer.size;
    appendBuffer(&ctx->sliceOffsets, &offset, sizeof(offset));
    appendBuffer(&ctx->bitstreamBuffer, frame, frameSize);
    picParams->nBitstreamDataLen = (unsigned int) ctx->bitstreamBuffer.size;

    LOG("JPEG: Reconstructed %u bytes for NVDEC", frameSize);

    free(frame);
}

static cudaVideoCodec computeJPEGCudaCodec(VAProfile profile) {
    switch (profile) {
        case VAProfileJPEGBaseline:
            return cudaVideoCodec_JPEG;
        default:
            return cudaVideoCodec_NONE;
    }
}

static const VAProfile jpegSupportedProfiles[] = {
    VAProfileJPEGBaseline,
};

const DECLARE_CODEC(jpegCodec) = {
    .computeCudaCodec = computeJPEGCudaCodec,
    .handlers = {
        [VAPictureParameterBufferType] = copyJPEGPicParam,
        [VAIQMatrixBufferType]         = copyJPEGIQMatrix,
        [VAHuffmanTableBufferType]     = copyJPEGHuffmanTable,
        [VASliceParameterBufferType]   = copyJPEGSliceParam,
        [VASliceDataBufferType]        = copyJPEGSliceData,
    },
    .supportedProfileCount = ARRAY_SIZE(jpegSupportedProfiles),
    .supportedProfiles     = jpegSupportedProfiles,
};
