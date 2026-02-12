#include "vabackend.h"

static void copyVP8PicParam(NVContext *ctx, NVBuffer* buffer, CUVIDPICPARAMS *picParams)
{
    VAPictureParameterBufferVP8* buf = (VAPictureParameterBufferVP8*) buffer->ptr;

    picParams->PicWidthInMbs    = (buf->frame_width + 15) / 16;
    picParams->FrameHeightInMbs = (buf->frame_height + 15) / 16;

    picParams->CodecSpecific.vp8.width = buf->frame_width;
    picParams->CodecSpecific.vp8.height = buf->frame_height;

    picParams->CodecSpecific.vp8.LastRefIdx = pictureIdxFromSurfaceId(ctx->drv, buf->last_ref_frame);
    picParams->CodecSpecific.vp8.GoldenRefIdx = pictureIdxFromSurfaceId(ctx->drv, buf->golden_ref_frame);
    picParams->CodecSpecific.vp8.AltRefIdx = pictureIdxFromSurfaceId(ctx->drv, buf->alt_ref_frame);

    picParams->CodecSpecific.vp8.vp8_frame_tag.frame_type = buf->pic_fields.bits.key_frame;
    picParams->CodecSpecific.vp8.vp8_frame_tag.version = buf->pic_fields.bits.version;
    // show_frame will be set to 1 by default
    picParams->CodecSpecific.vp8.vp8_frame_tag.update_mb_segmentation_data = buf->pic_fields.bits.segmentation_enabled ? buf->pic_fields.bits.update_segment_feature_data : 0;
}

static void copyVP8SliceParam(NVContext *ctx, NVBuffer* buffer, CUVIDPICPARAMS *picParams)
{
    VASliceParameterBufferVP8* buf = (VASliceParameterBufferVP8*) buffer->ptr;

    // VA-API provides partition_size[0] = header_partition_size - ((macroblock_offset + 7) / 8)
    // NVDEC expects the raw header_partition_size
    // So we need to add the macroblock padding back
    uint32_t macroblock_padding = (buf->macroblock_offset + 7) / 8;
    picParams->CodecSpecific.vp8.first_partition_size = buf->partition_size[0] + macroblock_padding;

    ctx->lastSliceParams = buffer->ptr;
    ctx->lastSliceParamsCount = buffer->elements;

    picParams->nNumSlices += buffer->elements;
}

static void copyVP8SliceData(NVContext *ctx, NVBuffer* buf, CUVIDPICPARAMS *picParams)
{
    // FFmpeg VA-API VP8 decoder skips the VP8 frame header (3 bytes for interframes, 
    // 10 bytes for keyframes) before sending data to VA-API.
    // We need to reconstruct this header according to RFC 6386.
    //
    // VP8 Frame Tag format (3 bytes, little-endian 24-bit value):
    //   Bits 0:     key_frame (0=key, 1=inter)
    //   Bits 1-3:   version (3 bits)
    //   Bit 4:      show_frame (1 bit)
    //   Bits 5-23:  first_partition_size (19 bits)
    //
    // Keyframe additional bytes (7 bytes):
    //   Bytes 3-5:  Sync code: 0x9d 0x01 0x2a
    //   Bytes 6-7:  Width (14 bits) + horizontal_scale (2 bits) as little-endian uint16
    //   Bytes 8-9:  Height (14 bits) + vertical_scale (2 bits) as little-endian uint16
    
    for (unsigned int i = 0; i < ctx->lastSliceParamsCount; i++)
    {
        VASliceParameterBufferVP8 *sliceParams = &((VASliceParameterBufferVP8*) ctx->lastSliceParams)[i];
        uint32_t offset = (uint32_t) ctx->bitstreamBuffer.size;
        appendBuffer(&ctx->sliceOffsets, &offset, sizeof(offset));
        
        uint8_t *sliceData = PTROFF(buf->ptr, sliceParams->slice_data_offset);
        size_t sliceDataSize = sliceParams->slice_data_size;
        
        // Get frame information from picParams
        bool isKeyFrame = (picParams->CodecSpecific.vp8.vp8_frame_tag.frame_type == 0);
        uint8_t version = picParams->CodecSpecific.vp8.vp8_frame_tag.version;
        uint8_t showFrame = 1; // Default to show
        uint32_t firstPartitionSize = picParams->CodecSpecific.vp8.first_partition_size;
        uint16_t width = (uint16_t)picParams->CodecSpecific.vp8.width;
        uint16_t height = (uint16_t)picParams->CodecSpecific.vp8.height;
        
        // Build the VP8 frame tag (24-bit little-endian value)
        // Packed as: [key_frame(1) | version(3) | show_frame(1) | first_partition_size(19)]
        uint32_t frameTag = 0;
        frameTag |= (isKeyFrame ? 0 : 1) << 0;      // key_frame bit (0=key, 1=inter)
        frameTag |= (version & 0x7) << 1;            // version (3 bits)
        frameTag |= (showFrame & 0x1) << 4;          // show_frame (1 bit)
        frameTag |= (firstPartitionSize & 0x7FFFF) << 5; // first_partition_size (19 bits)
        
        // Write frame tag as 3 bytes (little-endian)
        uint8_t frameTagBytes[3];
        frameTagBytes[0] = frameTag & 0xFF;
        frameTagBytes[1] = (frameTag >> 8) & 0xFF;
        frameTagBytes[2] = (frameTag >> 16) & 0xFF;
        
        // Build complete frame header
        if (isKeyFrame) {
            // Keyframe: 10 bytes total
            uint8_t header[10];
            
            // Bytes 0-2: Frame tag
            header[0] = frameTagBytes[0];
            header[1] = frameTagBytes[1];
            header[2] = frameTagBytes[2];
            
            // Bytes 3-5: Sync code
            header[3] = 0x9d;
            header[4] = 0x01;
            header[5] = 0x2a;
            
            // Bytes 6-7: Width (14 bits) + scale (2 bits), little-endian
            // Scale is always 0 for now
            uint16_t widthCode = width & 0x3FFF;  // 14 bits for width
            header[6] = widthCode & 0xFF;
            header[7] = (widthCode >> 8) & 0xFF;
            
            // Bytes 8-9: Height (14 bits) + scale (2 bits), little-endian
            uint16_t heightCode = height & 0x3FFF;  // 14 bits for height
            header[8] = heightCode & 0xFF;
            header[9] = (heightCode >> 8) & 0xFF;
            
            // Prepend header
            appendBuffer(&ctx->bitstreamBuffer, header, sizeof(header));
            picParams->nBitstreamDataLen += sizeof(header);
        } else {
            // Non-keyframe: 3 bytes (frame tag only)
            appendBuffer(&ctx->bitstreamBuffer, frameTagBytes, sizeof(frameTagBytes));
            picParams->nBitstreamDataLen += sizeof(frameTagBytes);
        }
        
        // Append the actual slice data (partition data from FFmpeg)
        appendBuffer(&ctx->bitstreamBuffer, sliceData, sliceDataSize);
        picParams->nBitstreamDataLen += sliceDataSize;
    }
}

static void ignoreVP8Buffer(NVContext *ctx, NVBuffer *buffer, CUVIDPICPARAMS *picParams)
{
    // Intentionally do nothing
    (void)ctx;
    (void)buffer;
    (void)picParams;
}

static cudaVideoCodec computeVP8CudaCodec(VAProfile profile) {
    if (profile == VAProfileVP8Version0_3) {
        return cudaVideoCodec_VP8;
    }

    return cudaVideoCodec_NONE;
}

static const VAProfile vp8SupportedProfiles[] = {
    VAProfileVP8Version0_3,
};

const DECLARE_CODEC(vp8Codec) = {
    .computeCudaCodec = computeVP8CudaCodec,
    .handlers = {
        [VAPictureParameterBufferType] = copyVP8PicParam,
        [VASliceParameterBufferType] = copyVP8SliceParam,
        [VASliceDataBufferType] = copyVP8SliceData,
        [VAIQMatrixBufferType]         = ignoreVP8Buffer,
        [VAProbabilityBufferType]      = ignoreVP8Buffer,
    },
    .supportedProfileCount = ARRAY_SIZE(vp8SupportedProfiles),
    .supportedProfiles = vp8SupportedProfiles,
};
