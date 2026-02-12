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
    // show_frame will be extracted from bitstream in copyVP8SliceData
    picParams->CodecSpecific.vp8.vp8_frame_tag.update_mb_segmentation_data = buf->pic_fields.bits.segmentation_enabled ? buf->pic_fields.bits.update_segment_feature_data : 0;
}

static void copyVP8SliceParam(NVContext *ctx, NVBuffer* buffer, CUVIDPICPARAMS *picParams)
{
    VASliceParameterBufferVP8* buf = (VASliceParameterBufferVP8*) buffer->ptr;

    // VA-API provides partition_size[0] as the first partition size
    // This is what NVDEC expects for first_partition_size
    picParams->CodecSpecific.vp8.first_partition_size = buf->partition_size[0];

    ctx->lastSliceParams = buffer->ptr;
    ctx->lastSliceParamsCount = buffer->elements;

    picParams->nNumSlices += buffer->elements;
}

static void copyVP8SliceData(NVContext *ctx, NVBuffer* buf, CUVIDPICPARAMS *picParams)
{
    // Extract show_frame from the first byte of the bitstream
    // VP8 frame tag: bit 4 (0x10) contains show_frame flag
    uint8_t *firstByte = (uint8_t*) buf->ptr;
    picParams->CodecSpecific.vp8.vp8_frame_tag.show_frame = (firstByte[0] & 0x10) ? 1 : 0;
    
    for (unsigned int i = 0; i < ctx->lastSliceParamsCount; i++)
    {
        VASliceParameterBufferVP8 *sliceParams = &((VASliceParameterBufferVP8*) ctx->lastSliceParams)[i];
        uint32_t offset = (uint32_t) ctx->bitstreamBuffer.size;
        appendBuffer(&ctx->sliceOffsets, &offset, sizeof(offset));
        
        // FFmpeg sends VP8 data WITHOUT the frame header (it skips 3-10 bytes)
        // The VP8 hack in vabackend.c tries to recover this by capturing extra bytes
        // buf->offset contains how many extra bytes were captured before sliceParams->slice_data_offset
        uint8_t *sliceData = PTROFF(buf->ptr, sliceParams->slice_data_offset);
        size_t sliceDataSize = sliceParams->slice_data_size;
        
        // Send the slice data to NVDEC
        // We use sliceData directly since the VP8 hack already adjusted the pointer
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
