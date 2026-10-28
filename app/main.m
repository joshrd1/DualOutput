/*
 * main.m
 * Dual Output Controller — Native macOS AppKit + CoreAudio AUHAL Router
 * Features:
 *   - Full macOS Main Menu Bar (Cmd+Q to Quit, Cmd+W to Hide to Menu Bar, Cmd+S to Save Preset)
 *   - Custom Compact Menu Bar Icon (Icon-only Y-Splitter vector template image)
 *   - Perceptual Cubic Master Volume Curve (matches human logarithmic hearing)
 *   - Named Presets, Per-Output Mono Downmix, Clock-Drift Compensation & 0-500ms Fine Delay
 * Targets macOS Monterey 12.7.x (x86_64)
 */

#import <Cocoa/Cocoa.h>
#import <CoreAudio/CoreAudio.h>
#import <AudioToolbox/AudioToolbox.h>
#import <AVFoundation/AVFoundation.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define kVirtualDeviceUID           "com.dualoutput.driver.device"

/* Lock-free SPSC clock-drift buffer constants (power of 2) */
#define kDriftRingFrames            16384U
#define kDriftRingMask              (kDriftRingFrames - 1U)
#define kDriftTargetFillFrames      1536.0

/* Real-time delay line buffer constants (power of 2; 65536 frames > 500ms at up to 96kHz) */
#define kDelayBufferFrames          65536U
#define kDelayBufferMask            (kDelayBufferFrames - 1U)

/* Maximum frames per CoreAudio callback slice */
#define kMaxSliceFrames             4096U

/* A Linkwitz-Riley crossover needs, at most, four biquad sections per path
 * for the selectable 48 dB/octave slope. */
#define kMaxCrossoverBiquads        4

/* Routing matrices expose up to sixteen hardware destination channels. */
#define kMaxOutputChannels           16U

typedef enum {
    kCrossoverFilterBypass   = 0,
    kCrossoverFilterLowPass  = 1,
    kCrossoverFilterHighPass = 2
} CrossoverFilterMode;

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float z1L, z2L;
    float z1R, z2R;
} CrossoverBiquad;

/* Atomics are written by the AppKit controls and read by the audio callback. */
typedef struct {
    _Atomic bool   enabled;
    _Atomic bool   linked;
    _Atomic bool   linkedOutputAHighPass;
    _Atomic double linkedFrequency;
    _Atomic int    linkedSlopeDb;
    _Atomic int    modeA;
    _Atomic int    modeB;
    _Atomic double frequencyA;
    _Atomic double frequencyB;
    _Atomic int    slopeDbA;
    _Atomic int    slopeDbB;
} CrossoverState;

/* Real-time safe per-output processing state */
typedef struct {
    /* SPSC Clock-Domain Crossing Ring Buffer (written by input callback, read by output callback) */
    float               driftBuffer[kDriftRingFrames * 2];
    _Atomic uint64_t    writeIndex;
    uint64_t            readIndexInt;
    double              readIndexFrac;
    bool                primed;

    /* PI Adaptive Clock-Drift Controller state */
    double              filteredError;
    double              integralError;
    _Atomic double      inputSampleRate;
    _Atomic double      outputSampleRate;
    _Atomic double      currentRatioPPM;
    _Atomic uint32_t    currentFillFrames;

    /* Dedicated Delay Line Ring Buffer (runs at output device native sample rate) */
    float               delayBuffer[kDelayBufferFrames * 2];
    uint32_t            delayWriteIndex;
    double              currentDelaySamples;
    _Atomic double      targetDelayMs;

    /* Gain Stage, Mono Switch & Metering */
    _Atomic float       targetGain;
    float               currentGain;
    _Atomic bool        enabled;
    _Atomic bool        mono;
    _Atomic bool        streamRunning;
    _Atomic float       peakLevel;

    /* Per-output-channel input routing. Bit N routes source L or R to
     * destination channel N. These are consumed atomically by the callback. */
    _Atomic uint32_t    outputChannelCount;
    _Atomic uint32_t    routeMaskLeft;
    _Atomic uint32_t    routeMaskRight;
    _Atomic uint32_t    monoRouteMask;
    _Atomic float       routeGainLeft[kMaxOutputChannels];
    _Atomic float       routeGainRight[kMaxOutputChannels];
    _Atomic float       monoRouteGain[kMaxOutputChannels];

    /* Crossover filter state. Coefficients are reconfigured only when a
     * user-facing setting or the destination sample rate changes. */
    CrossoverBiquad     crossoverBiquads[kMaxCrossoverBiquads];
    int                 crossoverStageCount;
    int                 appliedCrossoverMode;
    int                 appliedCrossoverSlopeDb;
    double              appliedCrossoverFrequency;
    double              appliedCrossoverSampleRate;
    bool                crossoverInvertPolarity;

    /* CoreAudio AUHAL Output Unit */
    AudioUnit           outputUnit;
    AudioDeviceID       deviceID;
} OutputPathState;

/* Global Real-Time Engine State */
typedef struct {
    AudioUnit           inputUnit;
    AudioDeviceID       virtualDeviceID;
    _Atomic bool        inputRunning;
    _Atomic double      virtualSampleRate;
    _Atomic float       inputPeakLevel;
    float               inputScratchBuffer[kMaxSliceFrames * 2];

    OutputPathState     pathA;
    OutputPathState     pathB;

    /* User Parameters */
    float               masterVolume;       /* 0.0 .. 1.0 (perceptual cubic taper) */
    float               balance;            /* -1.0 (A only) .. 0.0 (Equal) .. +1.0 (B only) */
    float               trimDbA;            /* -24.0 .. +6.0 dB */
    float               trimDbB;            /* -24.0 .. +6.0 dB */
    CrossoverState      crossover;
} DualAudioEngine;

static DualAudioEngine gEngine;

typedef struct {
    CrossoverFilterMode mode;
    double              frequency;
    int                 slopeDb;
} CrossoverConfig;

/*===========================================================================
 * Real-Time Safe DSP Helpers (No allocations, no locks, no Objective-C, no I/O)
 *===========================================================================*/

static inline float FastClampSample(float x) {
    if (!isfinite(x)) return 0.0f;
    if (x > 1.0f) return 1.0f;
    if (x < -1.0f) return -1.0f;
    return x;
}

static inline int ClampCrossoverSlope(int slopeDb) {
    switch (slopeDb) {
        case 12:
        case 24:
        case 36:
        case 48:
            return slopeDb;
        default:
            return 24;
    }
}

static inline double ClampCrossoverFrequency(double frequency, double sampleRate) {
    double maximum = fmin(20000.0, sampleRate * 0.45);
    if (maximum < 20.0) maximum = 20.0;
    if (!isfinite(frequency) || frequency < 20.0) return 20.0;
    if (frequency > maximum) return maximum;
    return frequency;
}

static inline double CrossoverFrequencyToSliderValue(double frequency) {
    frequency = fmin(20000.0, fmax(20.0, frequency));
    return log10(frequency);
}

static inline double CrossoverSliderValueToFrequency(double sliderValue) {
    return fmin(20000.0, fmax(20.0, pow(10.0, sliderValue)));
}

static inline float RouteGainDbToLinear(double db) {
    if (!isfinite(db)) db = 0.0;
    if (db < -60.0) db = -60.0;
    if (db > 12.0) db = 12.0;
    return (float)pow(10.0, db / 20.0);
}

static inline double RouteGainLinearToDb(float gain) {
    if (!isfinite(gain) || gain <= 0.000001f) return -60.0;
    return fmin(12.0, fmax(-60.0, 20.0 * log10((double)gain)));
}

/* The Q values below are two identical Butterworth filters cascaded into
 * Linkwitz-Riley alignments. The linked low/high pair therefore shares phase
 * at the crossover point and sums flat at 12, 24, 36, or 48 dB/octave. */
static int GetLinkwitzRileyStageQs(int slopeDb, float *outQs) {
    switch (ClampCrossoverSlope(slopeDb)) {
        case 12:
            /* Two first-order Butterworth sections represented as Q=0.5. */
            outQs[0] = 0.5f;
            return 1;
        case 24:
            outQs[0] = 0.70710678f;
            outQs[1] = 0.70710678f;
            return 2;
        case 36:
            /* Two third-order Butterworth filters: two real poles and two
             * Q=1 complex pairs. */
            outQs[0] = 0.5f;
            outQs[1] = 1.0f;
            outQs[2] = 1.0f;
            return 3;
        case 48:
        default:
            outQs[0] = 0.54119610f;
            outQs[1] = 1.30656296f;
            outQs[2] = 0.54119610f;
            outQs[3] = 1.30656296f;
            return 4;
    }
}

static void CalculateCrossoverBiquad(double sampleRate,
                                     double frequency,
                                     float q,
                                     CrossoverFilterMode mode,
                                     float *b0, float *b1, float *b2,
                                     float *a1, float *a2) {
    frequency = ClampCrossoverFrequency(frequency, sampleRate);
    double omega = 2.0 * M_PI * frequency / sampleRate;
    double sine = sin(omega);
    double cosine = cos(omega);
    double alpha = sine / (2.0 * (double)q);
    double a0 = 1.0 + alpha;

    double rawB0, rawB1, rawB2;
    if (mode == kCrossoverFilterHighPass) {
        rawB0 =  (1.0 + cosine) * 0.5;
        rawB1 = -(1.0 + cosine);
        rawB2 =  (1.0 + cosine) * 0.5;
    } else {
        rawB0 = (1.0 - cosine) * 0.5;
        rawB1 =  1.0 - cosine;
        rawB2 = (1.0 - cosine) * 0.5;
    }

    *b0 = (float)(rawB0 / a0);
    *b1 = (float)(rawB1 / a0);
    *b2 = (float)(rawB2 / a0);
    *a1 = (float)(-2.0 * cosine / a0);
    *a2 = (float)((1.0 - alpha) / a0);
}

static CrossoverConfig GetCrossoverConfigForPath(bool outputA) {
    CrossoverConfig config;
    bool enabled = atomic_load_explicit(&gEngine.crossover.enabled, memory_order_relaxed);
    bool linked  = atomic_load_explicit(&gEngine.crossover.linked, memory_order_relaxed);

    if (!enabled) {
        config.mode = kCrossoverFilterBypass;
        config.frequency = 120.0;
        config.slopeDb = 24;
        return config;
    }

    if (linked) {
        bool outputAHighPass = atomic_load_explicit(&gEngine.crossover.linkedOutputAHighPass, memory_order_relaxed);
        config.mode = (outputA == outputAHighPass) ? kCrossoverFilterHighPass : kCrossoverFilterLowPass;
        config.frequency = atomic_load_explicit(&gEngine.crossover.linkedFrequency, memory_order_relaxed);
        config.slopeDb = atomic_load_explicit(&gEngine.crossover.linkedSlopeDb, memory_order_relaxed);
    } else if (outputA) {
        config.mode = (CrossoverFilterMode)atomic_load_explicit(&gEngine.crossover.modeA, memory_order_relaxed);
        config.frequency = atomic_load_explicit(&gEngine.crossover.frequencyA, memory_order_relaxed);
        config.slopeDb = atomic_load_explicit(&gEngine.crossover.slopeDbA, memory_order_relaxed);
    } else {
        config.mode = (CrossoverFilterMode)atomic_load_explicit(&gEngine.crossover.modeB, memory_order_relaxed);
        config.frequency = atomic_load_explicit(&gEngine.crossover.frequencyB, memory_order_relaxed);
        config.slopeDb = atomic_load_explicit(&gEngine.crossover.slopeDbB, memory_order_relaxed);
    }

    if (config.mode != kCrossoverFilterLowPass && config.mode != kCrossoverFilterHighPass) {
        config.mode = kCrossoverFilterBypass;
    }
    config.slopeDb = ClampCrossoverSlope(config.slopeDb);
    return config;
}

static void ConfigureCrossoverForPath(OutputPathState *path,
                                      CrossoverConfig config,
                                      double sampleRate) {
    config.frequency = ClampCrossoverFrequency(config.frequency, sampleRate);
    bool modeChanged = path->appliedCrossoverMode != (int)config.mode;
    bool stageConfigurationChanged = path->appliedCrossoverSlopeDb != config.slopeDb;
    bool rateChanged = fabs(path->appliedCrossoverSampleRate - sampleRate) > 1.0;
    bool frequencyChanged = fabs(path->appliedCrossoverFrequency - config.frequency) > 0.05;

    if (!modeChanged && !stageConfigurationChanged && !rateChanged && !frequencyChanged) {
        return;
    }

    path->appliedCrossoverMode = (int)config.mode;
    path->appliedCrossoverSlopeDb = config.slopeDb;
    path->appliedCrossoverFrequency = config.frequency;
    path->appliedCrossoverSampleRate = sampleRate;

    if (config.mode == kCrossoverFilterBypass) {
        path->crossoverStageCount = 0;
        path->crossoverInvertPolarity = false;
        if (modeChanged) memset(path->crossoverBiquads, 0, sizeof(path->crossoverBiquads));
        return;
    }

    float qs[kMaxCrossoverBiquads] = {0};
    int stageCount = GetLinkwitzRileyStageQs(config.slopeDb, qs);
    if (modeChanged || stageConfigurationChanged) {
        memset(path->crossoverBiquads, 0, sizeof(path->crossoverBiquads));
    }
    path->crossoverStageCount = stageCount;
    /* Even-order Linkwitz-Riley pairs at 12 and 36 dB/octave need the
     * high-pass branch polarity-normalized to produce a flat linked sum. */
    path->crossoverInvertPolarity = config.mode == kCrossoverFilterHighPass &&
                                    (config.slopeDb == 12 || config.slopeDb == 36);
    for (int i = 0; i < stageCount; ++i) {
        CrossoverBiquad *stage = &path->crossoverBiquads[i];
        CalculateCrossoverBiquad(sampleRate, config.frequency, qs[i], config.mode,
                                 &stage->b0, &stage->b1, &stage->b2, &stage->a1, &stage->a2);
    }
}

static inline float ProcessCrossoverBiquad(float sample, float *z1, float *z2, const CrossoverBiquad *stage) {
    float output = stage->b0 * sample + *z1;
    *z1 = stage->b1 * sample - stage->a1 * output + *z2;
    *z2 = stage->b2 * sample - stage->a2 * output;
    return output;
}

static inline void ProcessCrossoverStereo(OutputPathState *path, float *left, float *right) {
    for (int i = 0; i < path->crossoverStageCount; ++i) {
        CrossoverBiquad *stage = &path->crossoverBiquads[i];
        *left = ProcessCrossoverBiquad(*left, &stage->z1L, &stage->z2L, stage);
        *right = ProcessCrossoverBiquad(*right, &stage->z1R, &stage->z2R, stage);
    }
    if (path->crossoverInvertPolarity) {
        *left = -*left;
        *right = -*right;
    }
}

/* 4-Point Hermite Cubic Interpolation for fractional stereo sample index */
static inline void InterpolateHermiteStereo(const float *ring, uint32_t mask, uint64_t idxInt, double frac, float *outL, float *outR) {
    uint32_t i0 = ((uint32_t)(idxInt - 1)) & mask;
    uint32_t i1 = ((uint32_t)(idxInt))     & mask;
    uint32_t i2 = ((uint32_t)(idxInt + 1)) & mask;
    uint32_t i3 = ((uint32_t)(idxInt + 2)) & mask;

    float f = (float)frac;

    /* Left channel */
    float y0L = ring[i0 * 2];
    float y1L = ring[i1 * 2];
    float y2L = ring[i2 * 2];
    float y3L = ring[i3 * 2];
    float c0L = y1L;
    float c1L = 0.5f * (y2L - y0L);
    float c2L = y0L - 2.5f * y1L + 2.0f * y2L - 0.5f * y3L;
    float c3L = 0.5f * (y3L - y0L) + 1.5f * (y1L - y2L);
    *outL = ((c3L * f + c2L) * f + c1L) * f + c0L;

    /* Right channel */
    float y0R = ring[i0 * 2 + 1];
    float y1R = ring[i1 * 2 + 1];
    float y2R = ring[i2 * 2 + 1];
    float y3R = ring[i3 * 2 + 1];
    float c0R = y1R;
    float c1R = 0.5f * (y2R - y0R);
    float c2R = y0R - 2.5f * y1R + 2.0f * y2R - 0.5f * y3R;
    float c3R = 0.5f * (y3R - y0R) + 1.5f * (y1R - y2R);
    *outR = ((c3R * f + c2R) * f + c1R) * f + c0R;
}

static inline void PushToOutputPath(OutputPathState *path, const float *frames, uint32_t numFrames) {
    if (!atomic_load_explicit(&path->enabled, memory_order_relaxed) ||
        !atomic_load_explicit(&path->streamRunning, memory_order_relaxed)) {
        return;
    }

    uint64_t w = atomic_load_explicit(&path->writeIndex, memory_order_relaxed);
    for (uint32_t i = 0; i < numFrames; ++i) {
        uint32_t slot = ((uint32_t)(w + i)) & kDriftRingMask;
        path->driftBuffer[slot * 2]     = frames[i * 2];
        path->driftBuffer[slot * 2 + 1] = frames[i * 2 + 1];
    }
    atomic_store_explicit(&path->writeIndex, w + numFrames, memory_order_release);
}

static inline uint32_t OutputChannelMask(uint32_t channelCount) {
    if (channelCount >= 32U) return UINT32_MAX;
    return (1U << channelCount) - 1U;
}

static void ClearOutputBufferList(AudioBufferList *bufferList) {
    for (UInt32 b = 0; b < bufferList->mNumberBuffers; ++b) {
        AudioBuffer *buffer = &bufferList->mBuffers[b];
        if (buffer->mData != NULL && buffer->mDataByteSize > 0) {
            memset(buffer->mData, 0, buffer->mDataByteSize);
        }
    }
}

static inline float RoutedSampleForChannel(uint32_t channel,
                                           uint32_t routeMaskLeft,
                                           uint32_t routeMaskRight,
                                           float routeGainLeft,
                                           float routeGainRight,
                                           float sourceLeft,
                                           float sourceRight) {
    uint32_t bit = 1U << channel;
    float mixed = 0.0f;
    if ((routeMaskLeft & bit) != 0U) mixed += sourceLeft * routeGainLeft;
    if ((routeMaskRight & bit) != 0U) mixed += sourceRight * routeGainRight;
    return FastClampSample(mixed);
}

/* The AUHAL normally supplies one interleaved buffer because we request an
 * interleaved client format. The second branch keeps routing correct on a
 * device that presents non-interleaved buffers instead. */
static float WriteRoutedOutputFrame(AudioBufferList *bufferList,
                                    UInt32 frame,
                                    uint32_t channelCount,
                                    uint32_t routeMaskLeft,
                                    uint32_t routeMaskRight,
                                    const float *routeGainsLeft,
                                    const float *routeGainsRight,
                                    float sourceLeft,
                                    float sourceRight) {
    float framePeak = 0.0f;
    if (bufferList->mNumberBuffers == 1 && bufferList->mBuffers[0].mData != NULL &&
        bufferList->mBuffers[0].mNumberChannels >= channelCount) {
        float *samples = (float *)bufferList->mBuffers[0].mData;
        for (uint32_t channel = 0; channel < channelCount; ++channel) {
            float output = RoutedSampleForChannel(channel, routeMaskLeft, routeMaskRight,
                                                   routeGainsLeft[channel], routeGainsRight[channel],
                                                   sourceLeft, sourceRight);
            samples[(size_t)frame * channelCount + channel] = output;
            float magnitude = fabsf(output);
            if (magnitude > framePeak) framePeak = magnitude;
        }
        return framePeak;
    }

    uint32_t channel = 0;
    for (UInt32 b = 0; b < bufferList->mNumberBuffers && channel < channelCount; ++b) {
        AudioBuffer *buffer = &bufferList->mBuffers[b];
        if (buffer->mData == NULL) {
            channel += buffer->mNumberChannels;
            continue;
        }
        float *samples = (float *)buffer->mData;
        for (UInt32 localChannel = 0; localChannel < buffer->mNumberChannels && channel < channelCount; ++localChannel, ++channel) {
            float output = RoutedSampleForChannel(channel, routeMaskLeft, routeMaskRight,
                                                   routeGainsLeft[channel], routeGainsRight[channel],
                                                   sourceLeft, sourceRight);
            samples[(size_t)frame * buffer->mNumberChannels + localChannel] = output;
            float magnitude = fabsf(output);
            if (magnitude > framePeak) framePeak = magnitude;
        }
    }
    return framePeak;
}

/* Input AUHAL Callback: Pulls audio from "Dual Output" virtual device and feeds Path A & Path B */
static OSStatus InputAURenderCallback(void *inRefCon,
                                      AudioUnitRenderActionFlags *ioActionFlags,
                                      const AudioTimeStamp *inTimeStamp,
                                      UInt32 inBusNumber,
                                      UInt32 inNumberFrames,
                                      AudioBufferList *ioData) {
    (void)inRefCon; (void)ioData;
    if (inNumberFrames > kMaxSliceFrames) {
        inNumberFrames = kMaxSliceFrames;
    }

    AudioBufferList bufferList;
    bufferList.mNumberBuffers = 1;
    bufferList.mBuffers[0].mNumberChannels = 2;
    bufferList.mBuffers[0].mDataByteSize = inNumberFrames * 2 * sizeof(float);
    bufferList.mBuffers[0].mData = gEngine.inputScratchBuffer;

    OSStatus status = AudioUnitRender(gEngine.inputUnit,
                                      ioActionFlags,
                                      inTimeStamp,
                                      inBusNumber,
                                      inNumberFrames,
                                      &bufferList);
    if (status != noErr) {
        return noErr;
    }

    const float *src = gEngine.inputScratchBuffer;
    float maxAbs = 0.0f;
    for (UInt32 i = 0; i < inNumberFrames * 2; ++i) {
        float a = fabsf(src[i]);
        if (a > maxAbs) maxAbs = a;
    }
    float prevPeak = atomic_load_explicit(&gEngine.inputPeakLevel, memory_order_relaxed);
    atomic_store_explicit(&gEngine.inputPeakLevel, (maxAbs > prevPeak) ? maxAbs : (prevPeak * 0.92f), memory_order_relaxed);

    PushToOutputPath(&gEngine.pathA, src, inNumberFrames);
    PushToOutputPath(&gEngine.pathB, src, inNumberFrames);
    return noErr;
}

/* Output AUHAL Callback: Runs independently for Output A and Output B */
static OSStatus OutputAURenderCallback(void *inRefCon,
                                       AudioUnitRenderActionFlags *ioActionFlags,
                                       const AudioTimeStamp *inTimeStamp,
                                       UInt32 inBusNumber,
                                       UInt32 inNumberFrames,
                                       AudioBufferList *ioData) {
    (void)ioActionFlags; (void)inTimeStamp; (void)inBusNumber;
    OutputPathState *path = (OutputPathState *)inRefCon;
    if (ioData == NULL || ioData->mNumberBuffers == 0 || ioData->mBuffers[0].mData == NULL) {
        return noErr;
    }

    if (!atomic_load_explicit(&path->enabled, memory_order_relaxed) ||
        !atomic_load_explicit(&gEngine.inputRunning, memory_order_relaxed)) {
        ClearOutputBufferList(ioData);
        atomic_store_explicit(&path->peakLevel, 0.0f, memory_order_relaxed);
        return noErr;
    }

    double inRate  = atomic_load_explicit(&path->inputSampleRate, memory_order_relaxed);
    double outRate = atomic_load_explicit(&path->outputSampleRate, memory_order_relaxed);
    if (inRate < 8000.0)  inRate  = 48000.0;
    if (outRate < 8000.0) outRate = 48000.0;
    double nominalRatio = inRate / outRate;

    uint64_t wIdx = atomic_load_explicit(&path->writeIndex, memory_order_acquire);

    if (!path->primed) {
        if (wIdx >= (uint64_t)kDriftTargetFillFrames) {
            path->readIndexInt  = wIdx - (uint64_t)kDriftTargetFillFrames;
            path->readIndexFrac = 0.0;
            path->filteredError = 0.0;
            path->integralError = 0.0;
            path->primed        = true;
        } else {
            ClearOutputBufferList(ioData);
            return noErr;
        }
    }

    int64_t availSigned = (int64_t)wIdx - (int64_t)path->readIndexInt;
    if (availSigned < (int64_t)(inNumberFrames * nominalRatio) + 8 ||
        availSigned > (int64_t)(kDriftRingFrames - 256)) {
        if (wIdx >= (uint64_t)kDriftTargetFillFrames) {
            path->readIndexInt  = wIdx - (uint64_t)kDriftTargetFillFrames;
            path->readIndexFrac = 0.0;
            path->filteredError = 0.0;
            path->integralError = 0.0;
            availSigned         = (int64_t)kDriftTargetFillFrames;
        } else {
            path->primed = false;
            ClearOutputBufferList(ioData);
            return noErr;
        }
    }

    atomic_store_explicit(&path->currentFillFrames, (uint32_t)availSigned, memory_order_relaxed);

    /* PI Adaptive Clock-Drift Controller */
    double err = (double)availSigned - kDriftTargetFillFrames;
    path->filteredError = 0.98 * path->filteredError + 0.02 * err;
    path->integralError += 0.0000005 * path->filteredError;
    if (path->integralError > 0.002)  path->integralError = 0.002;
    if (path->integralError < -0.002) path->integralError = -0.002;

    double proportional = 0.000015 * (path->filteredError / kDriftTargetFillFrames);
    double driftFactor  = 1.0 + proportional + path->integralError;
    if (driftFactor > 1.005) driftFactor = 1.005;
    if (driftFactor < 0.995) driftFactor = 0.995;

    double stepRatio = nominalRatio * driftFactor;
    atomic_store_explicit(&path->currentRatioPPM, (driftFactor - 1.0) * 1e6, memory_order_relaxed);

    /* Delay, Gain & Mono state */
    double targetDelayMs = atomic_load_explicit(&path->targetDelayMs, memory_order_relaxed);
    if (targetDelayMs < 0.0)   targetDelayMs = 0.0;
    if (targetDelayMs > 500.0) targetDelayMs = 500.0;
    double targetDelaySamples = targetDelayMs * 0.001 * outRate;
    if (targetDelaySamples > (double)(kDelayBufferFrames - 16)) {
        targetDelaySamples = (double)(kDelayBufferFrames - 16);
    }

    double delaySlewCoeff = 1.0 - exp(-1.0 / (0.050 * outRate));
    float  gainSlewCoeff  = (float)(1.0 - exp(-1.0 / (0.010 * outRate)));
    float  targetGain     = atomic_load_explicit(&path->targetGain, memory_order_relaxed);
    bool   isMono         = atomic_load_explicit(&path->mono, memory_order_relaxed);
    uint32_t outputChannels = atomic_load_explicit(&path->outputChannelCount, memory_order_relaxed);
    if (outputChannels < 1U || outputChannels > kMaxOutputChannels) outputChannels = 2U;
    uint32_t validChannelMask = OutputChannelMask(outputChannels);
    uint32_t routeMaskLeft = isMono
        ? atomic_load_explicit(&path->monoRouteMask, memory_order_relaxed) & validChannelMask
        : atomic_load_explicit(&path->routeMaskLeft, memory_order_relaxed) & validChannelMask;
    uint32_t routeMaskRight = isMono ? 0U
        : atomic_load_explicit(&path->routeMaskRight, memory_order_relaxed) & validChannelMask;
    float routeGainsLeft[kMaxOutputChannels];
    float routeGainsRight[kMaxOutputChannels];
    for (uint32_t channel = 0; channel < outputChannels; ++channel) {
        routeGainsLeft[channel] = isMono
            ? atomic_load_explicit(&path->monoRouteGain[channel], memory_order_relaxed)
            : atomic_load_explicit(&path->routeGainLeft[channel], memory_order_relaxed);
        routeGainsRight[channel] = isMono ? 0.0f
            : atomic_load_explicit(&path->routeGainRight[channel], memory_order_relaxed);
    }

    /* The crossover runs at the destination's native rate, after resampling
     * and before the optional mono fold-down. */
    CrossoverConfig crossoverConfig = GetCrossoverConfigForPath(path == &gEngine.pathA);
    ConfigureCrossoverForPath(path, crossoverConfig, outRate);

    float maxAbs = 0.0f;

    for (UInt32 i = 0; i < inNumberFrames; ++i) {
        /* Stage 1: Read from SPSC Clock-Drift buffer with 4-point Hermite cubic interpolation */
        float resampledL = 0.0f, resampledR = 0.0f;
        InterpolateHermiteStereo(path->driftBuffer,
                                 kDriftRingMask,
                                 path->readIndexInt,
                                 path->readIndexFrac,
                                 &resampledL,
                                 &resampledR);

        path->readIndexFrac += stepRatio;
        uint64_t advanceInt = (uint64_t)path->readIndexFrac;
        path->readIndexInt  += advanceInt;
        path->readIndexFrac -= (double)advanceInt;

        /* Stage 2: Write to Delay Line & read at smoothed fractional delay */
        uint32_t dWrite = path->delayWriteIndex & kDelayBufferMask;
        path->delayBuffer[dWrite * 2]     = resampledL;
        path->delayBuffer[dWrite * 2 + 1] = resampledR;

        path->currentDelaySamples += delaySlewCoeff * (targetDelaySamples - path->currentDelaySamples);
        double dSamples = path->currentDelaySamples;

        float delayedL = resampledL;
        float delayedR = resampledR;

        if (dSamples >= 0.01) {
            double readPos = (double)path->delayWriteIndex - dSamples;
            int64_t rFloor = (int64_t)floor(readPos);
            float   rFrac  = (float)(readPos - (double)rFloor);
            uint32_t idxA  = ((uint32_t)rFloor)     & kDelayBufferMask;
            uint32_t idxB  = ((uint32_t)(rFloor+1)) & kDelayBufferMask;

            delayedL = path->delayBuffer[idxA * 2]     * (1.0f - rFrac) + path->delayBuffer[idxB * 2]     * rFrac;
            delayedR = path->delayBuffer[idxA * 2 + 1] * (1.0f - rFrac) + path->delayBuffer[idxB * 2 + 1] * rFrac;
        }
        path->delayWriteIndex += 1;

        /* Stage 3: Per-output Linkwitz-Riley / independent crossover filter */
        if (path->crossoverStageCount > 0) {
            ProcessCrossoverStereo(path, &delayedL, &delayedR);
        }

        /* Stage 4: Optional Per-Output Mono Downmix (-6 dB equal-headroom sum) */
        if (isMono) {
            float monoSum = 0.5f * (delayedL + delayedR);
            delayedL = monoSum;
            delayedR = monoSum;
        }

        /* Stage 5: Per-sample smoothed Gain & Soft-Clip protection */
        path->currentGain += gainSlewCoeff * (targetGain - path->currentGain);
        float g = path->currentGain;

        float outL = FastClampSample(delayedL * g);
        float outR = FastClampSample(delayedR * g);

        float routedPeak = WriteRoutedOutputFrame(ioData, i, outputChannels,
                                                   routeMaskLeft, routeMaskRight,
                                                   routeGainsLeft, routeGainsRight,
                                                   outL, outR);
        if (routedPeak > maxAbs) maxAbs = routedPeak;
    }

    float prevPeak = atomic_load_explicit(&path->peakLevel, memory_order_relaxed);
    atomic_store_explicit(&path->peakLevel, (maxAbs > prevPeak) ? maxAbs : (prevPeak * 0.92f), memory_order_relaxed);

    return noErr;
}

/*===========================================================================
 * Gain Calculation: Perceptual Master Curve + Equal-Power Balance Law
 *===========================================================================*/

static void UpdateEngineTargetGains(void) {
    float s = gEngine.masterVolume;
    if (s < 0.0f) s = 0.0f;
    if (s > 1.0f) s = 1.0f;

    /* Perceptual cubic audio taper (s=0.5 -> -18.06 dB) */
    float masterAmp = (s <= 0.0001f) ? 0.0f : (s * s * s);

    float bal = gEngine.balance;
    if (bal < -1.0f) bal = -1.0f;
    if (bal >  1.0f) bal =  1.0f;

    float balGainA = 1.0f;
    float balGainB = 1.0f;
    if (bal < 0.0f) {
        balGainA = 1.0f;
        balGainB = sinf((float)M_PI_2 * (1.0f + bal));
    } else if (bal > 0.0f) {
        balGainA = cosf((float)M_PI_2 * bal);
        balGainB = 1.0f;
    }

    float trimLinearA = powf(10.0f, gEngine.trimDbA / 20.0f);
    float trimLinearB = powf(10.0f, gEngine.trimDbB / 20.0f);

    float totalA = masterAmp * balGainA * trimLinearA;
    float totalB = masterAmp * balGainB * trimLinearB;
    if (totalA > 2.0f) totalA = 2.0f;
    if (totalB > 2.0f) totalB = 2.0f;
    if (totalA < 0.0f) totalA = 0.0f;
    if (totalB < 0.0f) totalB = 0.0f;

    atomic_store_explicit(&gEngine.pathA.targetGain, totalA, memory_order_relaxed);
    atomic_store_explicit(&gEngine.pathB.targetGain, totalB, memory_order_relaxed);
}

/*===========================================================================
 * CoreAudio Device Enumeration & AUHAL Setup Helpers
 *===========================================================================*/

@interface AudioDeviceItem : NSObject
@property (nonatomic, assign) AudioDeviceID deviceID;
@property (nonatomic, copy)   NSString *name;
@property (nonatomic, copy)   NSString *uid;
@property (nonatomic, assign) double sampleRate;
@end

@implementation AudioDeviceItem
@end

static NSString *CopyDeviceStringProperty(AudioDeviceID devID, AudioObjectPropertySelector selector) {
    AudioObjectPropertyAddress addr = {
        selector,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    CFStringRef cfStr = NULL;
    UInt32 size = sizeof(CFStringRef);
    OSStatus err = AudioObjectGetPropertyData(devID, &addr, 0, NULL, &size, &cfStr);
    if (err != noErr || cfStr == NULL) {
        return nil;
    }
    return (__bridge_transfer NSString *)cfStr;
}

static UInt32 GetDeviceChannelCount(AudioDeviceID devID, AudioObjectPropertyScope scope) {
    AudioObjectPropertyAddress addr = {
        kAudioDevicePropertyStreamConfiguration,
        scope,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(devID, &addr, 0, NULL, &size) != noErr || size == 0) {
        return 0;
    }
    AudioBufferList *abl = (AudioBufferList *)malloc(size);
    if (abl == NULL) return 0;
    UInt32 totalChannels = 0;
    if (AudioObjectGetPropertyData(devID, &addr, 0, NULL, &size, abl) == noErr) {
        for (UInt32 i = 0; i < abl->mNumberBuffers; ++i) {
            totalChannels += abl->mBuffers[i].mNumberChannels;
        }
    }
    free(abl);
    return totalChannels;
}

static double GetDeviceNominalSampleRate(AudioDeviceID devID) {
    AudioObjectPropertyAddress addr = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    Float64 rate = 48000.0;
    UInt32 size = sizeof(Float64);
    AudioObjectGetPropertyData(devID, &addr, 0, NULL, &size, &rate);
    return (double)rate;
}

static AudioDeviceID FindVirtualDeviceID(void) {
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, NULL, &size) != noErr || size == 0) {
        return kAudioObjectUnknown;
    }
    UInt32 count = size / sizeof(AudioDeviceID);
    AudioDeviceID *devices = (AudioDeviceID *)malloc(size);
    if (devices == NULL) return kAudioObjectUnknown;

    AudioDeviceID found = kAudioObjectUnknown;
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, devices) == noErr) {
        for (UInt32 i = 0; i < count; ++i) {
            NSString *uid = CopyDeviceStringProperty(devices[i], kAudioDevicePropertyDeviceUID);
            if ([uid isEqualToString:@kVirtualDeviceUID]) {
                found = devices[i];
                break;
            }
        }
    }
    free(devices);
    return found;
}

static NSArray<AudioDeviceItem *> *EnumeratePhysicalOutputDevices(void) {
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, NULL, &size) != noErr || size == 0) {
        return @[];
    }
    UInt32 count = size / sizeof(AudioDeviceID);
    AudioDeviceID *devices = (AudioDeviceID *)malloc(size);
    if (devices == NULL) return @[];

    NSMutableArray<AudioDeviceItem *> *result = [NSMutableArray array];
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, devices) == noErr) {
        for (UInt32 i = 0; i < count; ++i) {
            AudioDeviceID devID = devices[i];
            NSString *uid = CopyDeviceStringProperty(devID, kAudioDevicePropertyDeviceUID);
            if (uid == nil || [uid isEqualToString:@kVirtualDeviceUID]) {
                continue;
            }
            UInt32 outChannels = GetDeviceChannelCount(devID, kAudioObjectPropertyScopeOutput);
            if (outChannels < 1) {
                continue;
            }
            NSString *name = CopyDeviceStringProperty(devID, kAudioObjectPropertyName) ?: @"Unknown Audio Device";
            AudioDeviceItem *item = [[AudioDeviceItem alloc] init];
            item.deviceID   = devID;
            item.name       = name;
            item.uid        = uid;
            item.sampleRate = GetDeviceNominalSampleRate(devID);
            [result addObject:item];
        }
    }
    free(devices);
    return result;
}

static void StopVirtualInputUnit(void) {
    atomic_store_explicit(&gEngine.inputRunning, false, memory_order_release);
    if (gEngine.inputUnit != NULL) {
        AudioOutputUnitStop(gEngine.inputUnit);
        AudioUnitUninitialize(gEngine.inputUnit);
        AudioComponentInstanceDispose(gEngine.inputUnit);
        gEngine.inputUnit = NULL;
    }
    gEngine.virtualDeviceID = kAudioObjectUnknown;
    atomic_store_explicit(&gEngine.inputPeakLevel, 0.0f, memory_order_relaxed);
}

static OSStatus StartVirtualInputUnit(AudioDeviceID virtualDevID, NSString **outErrorString) {
    StopVirtualInputUnit();
    if (virtualDevID == kAudioObjectUnknown) {
        if (outErrorString) *outErrorString = @"Dual Output driver not found in macOS HAL. Run install.sh first.";
        return kAudioHardwareBadDeviceError;
    }

    AudioComponentDescription desc = {
        .componentType         = kAudioUnitType_Output,
        .componentSubType      = kAudioUnitSubType_HALOutput,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
        .componentFlags        = 0,
        .componentFlagsMask    = 0
    };
    AudioComponent comp = AudioComponentFindNext(NULL, &desc);
    if (comp == NULL) {
        if (outErrorString) *outErrorString = @"Failed to locate Apple AUHAL AudioComponent.";
        return kAudioHardwareIllegalOperationError;
    }

    AudioUnit unit = NULL;
    OSStatus err = AudioComponentInstanceNew(comp, &unit);
    if (err != noErr || unit == NULL) {
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"AudioComponentInstanceNew failed (%d)", (int)err];
        return err;
    }

    UInt32 enableInput = 1;
    UInt32 disableOutput = 0;
    err = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &enableInput, sizeof(enableInput));
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"EnableIO Input failed (%d)", (int)err];
        return err;
    }
    err = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &disableOutput, sizeof(disableOutput));
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"DisableIO Output failed (%d)", (int)err];
        return err;
    }

    err = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &virtualDevID, sizeof(virtualDevID));
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"Failed binding AUHAL to Dual Output device (%d)", (int)err];
        return err;
    }

    double hwRate = GetDeviceNominalSampleRate(virtualDevID);
    AudioStreamBasicDescription clientFormat = {
        .mSampleRate       = hwRate,
        .mFormatID         = kAudioFormatLinearPCM,
        .mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        .mBytesPerPacket   = 8,
        .mFramesPerPacket  = 1,
        .mBytesPerFrame    = 8,
        .mChannelsPerFrame = 2,
        .mBitsPerChannel   = 32
    };
    err = AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &clientFormat, sizeof(clientFormat));
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"Failed setting input stream format (%d)", (int)err];
        return err;
    }

    AURenderCallbackStruct cb = {
        .inputProc       = InputAURenderCallback,
        .inputProcRefCon = NULL
    };
    err = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0, &cb, sizeof(cb));
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"Failed setting input callback (%d)", (int)err];
        return err;
    }

    err = AudioUnitInitialize(unit);
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"AudioUnitInitialize input failed (%d). Check Microphone permission.", (int)err];
        return err;
    }

    gEngine.inputUnit       = unit;
    gEngine.virtualDeviceID = virtualDevID;
    atomic_store_explicit(&gEngine.virtualSampleRate, hwRate, memory_order_relaxed);
    atomic_store_explicit(&gEngine.pathA.inputSampleRate, hwRate, memory_order_relaxed);
    atomic_store_explicit(&gEngine.pathB.inputSampleRate, hwRate, memory_order_relaxed);

    err = AudioOutputUnitStart(unit);
    if (err != noErr) {
        AudioUnitUninitialize(unit);
        AudioComponentInstanceDispose(unit);
        gEngine.inputUnit = NULL;
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"AudioOutputUnitStart input failed (%d)", (int)err];
        return err;
    }

    atomic_store_explicit(&gEngine.inputRunning, true, memory_order_release);
    return noErr;
}

static void StopOutputPath(OutputPathState *path) {
    atomic_store_explicit(&path->streamRunning, false, memory_order_release);
    if (path->outputUnit != NULL) {
        AudioOutputUnitStop(path->outputUnit);
        AudioUnitUninitialize(path->outputUnit);
        AudioComponentInstanceDispose(path->outputUnit);
        path->outputUnit = NULL;
    }
    path->deviceID = kAudioObjectUnknown;
    path->primed   = false;
    atomic_store_explicit(&path->peakLevel, 0.0f, memory_order_relaxed);
    atomic_store_explicit(&path->currentFillFrames, 0, memory_order_relaxed);
    atomic_store_explicit(&path->currentRatioPPM, 0.0, memory_order_relaxed);
    atomic_store_explicit(&path->outputChannelCount, 2U, memory_order_relaxed);
}

static void NormalizeRoutingForOutputPath(OutputPathState *path, uint32_t channelCount) {
    uint32_t validMask = OutputChannelMask(channelCount);
    uint32_t leftMask = atomic_load_explicit(&path->routeMaskLeft, memory_order_relaxed) & validMask;
    uint32_t rightMask = atomic_load_explicit(&path->routeMaskRight, memory_order_relaxed) & validMask;
    uint32_t monoMask = atomic_load_explicit(&path->monoRouteMask, memory_order_relaxed) & validMask;

    atomic_store_explicit(&path->routeMaskLeft, leftMask, memory_order_relaxed);
    atomic_store_explicit(&path->routeMaskRight, rightMask, memory_order_relaxed);
    atomic_store_explicit(&path->monoRouteMask, monoMask, memory_order_relaxed);
    for (uint32_t channel = 0; channel < channelCount; ++channel) {
        uint32_t bit = 1U << channel;
        if ((leftMask & bit) != 0U && atomic_load_explicit(&path->routeGainLeft[channel], memory_order_relaxed) <= 0.0f) {
            atomic_store_explicit(&path->routeGainLeft[channel], 1.0f, memory_order_relaxed);
        }
        if ((rightMask & bit) != 0U && atomic_load_explicit(&path->routeGainRight[channel], memory_order_relaxed) <= 0.0f) {
            atomic_store_explicit(&path->routeGainRight[channel], 1.0f, memory_order_relaxed);
        }
        if ((monoMask & bit) != 0U && atomic_load_explicit(&path->monoRouteGain[channel], memory_order_relaxed) <= 0.0f) {
            atomic_store_explicit(&path->monoRouteGain[channel], 1.0f, memory_order_relaxed);
        }
    }
}

static OSStatus StartOutputPath(OutputPathState *path, AudioDeviceID devID, NSString **outErrorString) {
    StopOutputPath(path);
    if (devID == kAudioObjectUnknown) {
        return noErr;
    }

    AudioComponentDescription desc = {
        .componentType         = kAudioUnitType_Output,
        .componentSubType      = kAudioUnitSubType_HALOutput,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
        .componentFlags        = 0,
        .componentFlagsMask    = 0
    };
    AudioComponent comp = AudioComponentFindNext(NULL, &desc);
    if (comp == NULL) {
        if (outErrorString) *outErrorString = @"Failed locating AUHAL component.";
        return kAudioHardwareIllegalOperationError;
    }

    AudioUnit unit = NULL;
    OSStatus err = AudioComponentInstanceNew(comp, &unit);
    if (err != noErr || unit == NULL) {
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"Output AudioComponentInstanceNew error (%d)", (int)err];
        return err;
    }

    err = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &devID, sizeof(devID));
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"Cannot open output device (%d)", (int)err];
        return err;
    }

    double outRate = GetDeviceNominalSampleRate(devID);
    AudioStreamBasicDescription hwFormat;
    UInt32 fmtSize = sizeof(hwFormat);
    if (AudioUnitGetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, &hwFormat, &fmtSize) == noErr && hwFormat.mSampleRate > 0) {
        outRate = hwFormat.mSampleRate;
    }

    uint32_t outputChannels = GetDeviceChannelCount(devID, kAudioObjectPropertyScopeOutput);
    if (outputChannels < 1U) outputChannels = 2U;
    if (outputChannels > kMaxOutputChannels) outputChannels = kMaxOutputChannels;
    NormalizeRoutingForOutputPath(path, outputChannels);

    AudioStreamBasicDescription clientFormat = {
        .mSampleRate       = outRate,
        .mFormatID         = kAudioFormatLinearPCM,
        .mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        .mBytesPerPacket   = outputChannels * (UInt32)sizeof(float),
        .mFramesPerPacket  = 1,
        .mBytesPerFrame    = outputChannels * (UInt32)sizeof(float),
        .mChannelsPerFrame = outputChannels,
        .mBitsPerChannel   = 32
    };
    err = AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &clientFormat, sizeof(clientFormat));
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"Cannot configure output stream format on device (%d)", (int)err];
        return err;
    }

    memset(path->driftBuffer, 0, sizeof(path->driftBuffer));
    memset(path->delayBuffer, 0, sizeof(path->delayBuffer));
    atomic_store_explicit(&path->writeIndex, 0, memory_order_relaxed);
    path->readIndexInt        = 0;
    path->readIndexFrac       = 0.0;
    path->primed              = false;
    path->filteredError       = 0.0;
    path->integralError       = 0.0;
    path->delayWriteIndex     = 0;
    path->currentDelaySamples = atomic_load_explicit(&path->targetDelayMs, memory_order_relaxed) * 0.001 * outRate;
    path->currentGain         = atomic_load_explicit(&path->targetGain, memory_order_relaxed);
    path->crossoverStageCount = 0;
    path->appliedCrossoverMode = -1;
    path->appliedCrossoverSlopeDb = -1;
    path->appliedCrossoverFrequency = -1.0;
    path->appliedCrossoverSampleRate = -1.0;
    path->crossoverInvertPolarity = false;
    memset(path->crossoverBiquads, 0, sizeof(path->crossoverBiquads));
    atomic_store_explicit(&path->outputSampleRate, outRate, memory_order_relaxed);
    atomic_store_explicit(&path->outputChannelCount, outputChannels, memory_order_relaxed);
    atomic_store_explicit(&path->inputSampleRate, atomic_load_explicit(&gEngine.virtualSampleRate, memory_order_relaxed), memory_order_relaxed);

    AURenderCallbackStruct cb = {
        .inputProc       = OutputAURenderCallback,
        .inputProcRefCon = path
    };
    err = AudioUnitSetProperty(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb, sizeof(cb));
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"Cannot set output render callback (%d)", (int)err];
        return err;
    }

    err = AudioUnitInitialize(unit);
    if (err != noErr) {
        AudioComponentInstanceDispose(unit);
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"AudioUnitInitialize failed for device (%d)", (int)err];
        return err;
    }

    path->outputUnit = unit;
    path->deviceID   = devID;
    atomic_store_explicit(&path->streamRunning, true, memory_order_release);

    err = AudioOutputUnitStart(unit);
    if (err != noErr) {
        atomic_store_explicit(&path->streamRunning, false, memory_order_release);
        AudioUnitUninitialize(unit);
        AudioComponentInstanceDispose(unit);
        path->outputUnit = NULL;
        path->deviceID   = kAudioObjectUnknown;
        if (outErrorString) *outErrorString = [NSString stringWithFormat:@"AudioOutputUnitStart failed (%d)", (int)err];
        return err;
    }

    return noErr;
}

/*===========================================================================
 * Native AppKit User Interface, Main Menu Bar, Status Item & Presets Manager
 *===========================================================================*/

static NSString *const kPrefOutputAUID     = @"OutputA_UID";
static NSString *const kPrefOutputBUID     = @"OutputB_UID";
static NSString *const kPrefOutputAEnabled = @"OutputA_Enabled";
static NSString *const kPrefOutputBEnabled = @"OutputB_Enabled";
static NSString *const kPrefOutputAMono    = @"OutputA_Mono";
static NSString *const kPrefOutputBMono    = @"OutputB_Mono";
static NSString *const kPrefMasterVolume   = @"MasterVolume";
static NSString *const kPrefBalance        = @"Balance";
static NSString *const kPrefTrimDbA        = @"TrimDb_A";
static NSString *const kPrefTrimDbB        = @"TrimDb_B";
static NSString *const kPrefDelayMsA       = @"DelayMs_A";
static NSString *const kPrefDelayMsB       = @"DelayMs_B";
static NSString *const kPrefRouteMaskLeftA = @"Routing_LeftMask_A";
static NSString *const kPrefRouteMaskRightA = @"Routing_RightMask_A";
static NSString *const kPrefRouteMaskLeftB = @"Routing_LeftMask_B";
static NSString *const kPrefRouteMaskRightB = @"Routing_RightMask_B";
static NSString *const kPrefMonoRouteMaskA = @"Routing_MonoMask_A";
static NSString *const kPrefMonoRouteMaskB = @"Routing_MonoMask_B";
static NSString *const kPrefRouteGainDbA = @"Routing_GainDb_A";
static NSString *const kPrefRouteGainDbB = @"Routing_GainDb_B";
static NSString *const kPrefMonoRouteGainDbA = @"Routing_MonoGainDb_A";
static NSString *const kPrefMonoRouteGainDbB = @"Routing_MonoGainDb_B";
static NSString *const kPrefCrossoverEnabled = @"Crossover_Enabled";
static NSString *const kPrefCrossoverLinked  = @"Crossover_Linked";
static NSString *const kPrefCrossoverLinkedOutputAHighPass = @"Crossover_LinkedOutputAHighPass";
static NSString *const kPrefCrossoverFrequency = @"Crossover_LinkedFrequency";
static NSString *const kPrefCrossoverSlope     = @"Crossover_LinkedSlope";
static NSString *const kPrefCrossoverModeA     = @"Crossover_ModeA";
static NSString *const kPrefCrossoverModeB     = @"Crossover_ModeB";
static NSString *const kPrefCrossoverFrequencyA = @"Crossover_FrequencyA";
static NSString *const kPrefCrossoverFrequencyB = @"Crossover_FrequencyB";
static NSString *const kPrefCrossoverSlopeA     = @"Crossover_SlopeA";
static NSString *const kPrefCrossoverSlopeB     = @"Crossover_SlopeB";
static NSString *const kPrefSavedPresets   = @"SavedPresets_v1";
static NSString *const kPrefActivePreset   = @"ActivePresetName_v1";

static double CrossoverMagnitudeDb(CrossoverConfig config, double frequency) {
    if (config.mode == kCrossoverFilterBypass) return 0.0;

    float qs[kMaxCrossoverBiquads] = {0};
    int stages = GetLinkwitzRileyStageQs(config.slopeDb, qs);
    double omega = 2.0 * M_PI * frequency / 48000.0;
    double magnitude = 1.0;

    for (int i = 0; i < stages; ++i) {
        float b0, b1, b2, a1, a2;
        CalculateCrossoverBiquad(48000.0, config.frequency, qs[i], config.mode,
                                 &b0, &b1, &b2, &a1, &a2);
        double cosine1 = cos(omega);
        double sine1 = sin(omega);
        double cosine2 = cos(2.0 * omega);
        double sine2 = sin(2.0 * omega);
        double realNum = b0 + b1 * cosine1 + b2 * cosine2;
        double imagNum = -b1 * sine1 - b2 * sine2;
        double realDen = 1.0 + a1 * cosine1 + a2 * cosine2;
        double imagDen = -a1 * sine1 - a2 * sine2;
        double numerator = realNum * realNum + imagNum * imagNum;
        double denominator = realDen * realDen + imagDen * imagDen;
        magnitude *= sqrt(numerator / fmax(denominator, 1.0e-20));
    }
    return 20.0 * log10(fmax(magnitude, 1.0e-5));
}

@class CrossoverResponseView;
@protocol CrossoverResponseViewDelegate <NSObject>
- (void)crossoverResponseView:(CrossoverResponseView *)view
       didDragFrequency:(double)frequency
             forOutputA:(BOOL)outputA;
@end

/* A compact, interactive log-frequency response display. Dragging a marker
 * changes the shared crossover when linked, or the nearest output filter when
 * unlinked; sliders remain available for precise adjustment. */
@interface CrossoverResponseView : NSView
@property (nonatomic, weak) id<CrossoverResponseViewDelegate> crossoverDelegate;
@end

@implementation CrossoverResponseView

- (BOOL)isFlipped {
    return NO;
}

- (double)frequencyForX:(CGFloat)x inPlot:(NSRect)plot {
    double normalized = (x - NSMinX(plot)) / NSWidth(plot);
    normalized = fmin(1.0, fmax(0.0, normalized));
    return 20.0 * pow(1000.0, normalized);
}

- (CGFloat)xForFrequency:(double)frequency inPlot:(NSRect)plot {
    frequency = fmin(20000.0, fmax(20.0, frequency));
    double normalized = log10(frequency / 20.0) / 3.0;
    return NSMinX(plot) + normalized * NSWidth(plot);
}

- (CGFloat)yForDb:(double)db inPlot:(NSRect)plot {
    db = fmin(0.0, fmax(-48.0, db));
    return NSMinY(plot) + ((db + 48.0) / 48.0) * NSHeight(plot);
}

- (NSRect)responsePlotRect {
    NSRect bounds = self.bounds;
    BOOL compact = NSHeight(bounds) < 100.0;
    CGFloat left = compact ? 30.0 : 36.0;
    CGFloat bottom = compact ? 12.0 : 19.0;
    CGFloat top = compact ? 5.0 : 15.0;
    return NSMakeRect(NSMinX(bounds) + left, NSMinY(bounds) + bottom,
                      NSWidth(bounds) - left - 11.0, NSHeight(bounds) - bottom - top);
}

- (void)drawResponseForConfig:(CrossoverConfig)config
                         color:(NSColor *)color
                        inPlot:(NSRect)plot {
    NSBezierPath *curve = [NSBezierPath bezierPath];
    curve.lineWidth = 2.0;
    const NSInteger steps = 220;
    for (NSInteger i = 0; i <= steps; ++i) {
        CGFloat x = NSMinX(plot) + NSWidth(plot) * ((CGFloat)i / (CGFloat)steps);
        double frequency = [self frequencyForX:x inPlot:plot];
        CGFloat y = [self yForDb:CrossoverMagnitudeDb(config, frequency) inPlot:plot];
        if (i == 0) [curve moveToPoint:NSMakePoint(x, y)];
        else [curve lineToPoint:NSMakePoint(x, y)];
    }
    [color setStroke];
    [curve stroke];
}

- (void)drawMarkerForFrequency:(double)frequency
                          color:(NSColor *)color
                         inPlot:(NSRect)plot
                          label:(NSString *)label {
    CGFloat x = [self xForFrequency:frequency inPlot:plot];
    NSBezierPath *marker = [NSBezierPath bezierPath];
    marker.lineWidth = 1.0;
    CGFloat dash[2] = {3.0, 3.0};
    [marker setLineDash:dash count:2 phase:0.0];
    [marker moveToPoint:NSMakePoint(x, NSMinY(plot))];
    [marker lineToPoint:NSMakePoint(x, NSMaxY(plot))];
    [[color colorWithAlphaComponent:0.75] setStroke];
    [marker stroke];

    if (label.length > 0) {
        NSDictionary *attributes = @{
            NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:9 weight:NSFontWeightMedium],
            NSForegroundColorAttributeName: color
        };
        NSSize size = [label sizeWithAttributes:attributes];
        CGFloat labelX = fmin(NSMaxX(plot) - size.width, fmax(NSMinX(plot), x + 3.0));
        [label drawAtPoint:NSMakePoint(labelX, NSMaxY(plot) - size.height - 2.0) withAttributes:attributes];
    }
}

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    NSRect bounds = self.bounds;
    [[NSColor controlBackgroundColor] setFill];
    NSBezierPath *background = [NSBezierPath bezierPathWithRoundedRect:NSInsetRect(bounds, 0.5, 0.5) xRadius:5.0 yRadius:5.0];
    [background fill];
    [[NSColor separatorColor] setStroke];
    background.lineWidth = 1.0;
    [background stroke];

    BOOL compact = NSHeight(bounds) < 100.0;
    NSRect plot = [self responsePlotRect];
    if (NSWidth(plot) <= 0.0 || NSHeight(plot) <= 0.0) return;

    NSColor *gridColor = [[NSColor secondaryLabelColor] colorWithAlphaComponent:0.18];
    NSDictionary *axisAttributes = @{
        NSFontAttributeName: [NSFont systemFontOfSize:compact ? 7.0 : 8.0],
        NSForegroundColorAttributeName: [NSColor secondaryLabelColor]
    };
    const int dbMarks[] = {0, -24, -48};
    for (int markIndex = 0; markIndex < 3; ++markIndex) {
        int db = dbMarks[markIndex];
        CGFloat y = [self yForDb:(double)db inPlot:plot];
        NSBezierPath *line = [NSBezierPath bezierPath];
        [line moveToPoint:NSMakePoint(NSMinX(plot), y)];
        [line lineToPoint:NSMakePoint(NSMaxX(plot), y)];
        [gridColor setStroke];
        line.lineWidth = 0.75;
        [line stroke];
        [[NSString stringWithFormat:@"%d", db] drawAtPoint:NSMakePoint(NSMinX(bounds) + 3.0, y - 4.0)
                                               withAttributes:axisAttributes];
    }
    const double axisFrequencies[] = {20.0, 30.0, 50.0, 70.0, 100.0, 200.0, 300.0,
                                      500.0, 700.0, 1000.0, 2000.0, 3000.0, 5000.0,
                                      7000.0, 10000.0, 20000.0};
    const NSString *axisLabels[] = {@"20", @"", @"", @"", @"100", @"", @"", @"", @"",
                                    @"1k", @"", @"", @"", @"", @"10k", @"20k"};
    for (int i = 0; i < 16; ++i) {
        CGFloat x = [self xForFrequency:axisFrequencies[i] inPlot:plot];
        NSBezierPath *line = [NSBezierPath bezierPath];
        [line moveToPoint:NSMakePoint(x, NSMinY(plot))];
        [line lineToPoint:NSMakePoint(x, NSMaxY(plot))];
        BOOL primary = axisLabels[i].length > 0;
        line.lineWidth = 0.75;
        if (!primary) {
            CGFloat dash[2] = {1.0, 2.0};
            [line setLineDash:dash count:2 phase:0.0];
            [[[NSColor secondaryLabelColor] colorWithAlphaComponent:0.12] setStroke];
        } else {
            [[[NSColor secondaryLabelColor] colorWithAlphaComponent:0.24] setStroke];
        }
        [line stroke];
        if (primary) {
            NSSize size = [axisLabels[i] sizeWithAttributes:axisAttributes];
            [axisLabels[i] drawAtPoint:NSMakePoint(x - size.width * 0.5, NSMinY(bounds) + 2.0)
                          withAttributes:axisAttributes];
        }
    }

    CrossoverConfig configA = GetCrossoverConfigForPath(true);
    CrossoverConfig configB = GetCrossoverConfigForPath(false);
    NSColor *colorA = [NSColor systemTealColor];
    NSColor *colorB = [NSColor systemOrangeColor];
    [self drawResponseForConfig:configA color:colorA inPlot:plot];
    [self drawResponseForConfig:configB color:colorB inPlot:plot];

    bool enabled = atomic_load_explicit(&gEngine.crossover.enabled, memory_order_relaxed);
    bool linked = atomic_load_explicit(&gEngine.crossover.linked, memory_order_relaxed);
    if (enabled && linked) {
        [self drawMarkerForFrequency:configA.frequency color:[NSColor systemPurpleColor] inPlot:plot
                               label:nil];
    } else if (enabled) {
        if (configA.mode != kCrossoverFilterBypass) {
            [self drawMarkerForFrequency:configA.frequency color:colorA inPlot:plot
                                   label:nil];
        }
        if (configB.mode != kCrossoverFilterBypass) {
            [self drawMarkerForFrequency:configB.frequency color:colorB inPlot:plot
                                   label:nil];
        }
    }

    if (!compact) {
        NSString *legend = enabled
            ? (linked ? [NSString stringWithFormat:@"Linked Linkwitz-Riley  •  A %@  •  B %@",
                         configA.mode == kCrossoverFilterHighPass ? @"high-pass" : @"low-pass",
                         configB.mode == kCrossoverFilterHighPass ? @"high-pass" : @"low-pass"]
                      : @"Independent filters  •  A ●  B ●")
            : @"Crossover bypassed — both outputs are full range";
        NSDictionary *legendAttributes = @{
            NSFontAttributeName: [NSFont systemFontOfSize:9.0 weight:NSFontWeightMedium],
            NSForegroundColorAttributeName: enabled ? [NSColor secondaryLabelColor] : [NSColor tertiaryLabelColor]
        };
        [legend drawAtPoint:NSMakePoint(NSMinX(plot) + 4.0, NSMaxY(plot) - 12.0) withAttributes:legendAttributes];
    }
}

- (void)updateDragAtPoint:(NSPoint)point {
    if (!atomic_load_explicit(&gEngine.crossover.enabled, memory_order_relaxed)) return;
    NSRect plot = [self responsePlotRect];
    if (!NSPointInRect(point, NSInsetRect(plot, -8.0, -8.0))) return;

    bool linked = atomic_load_explicit(&gEngine.crossover.linked, memory_order_relaxed);
    BOOL outputA = YES;
    if (!linked) {
        double xA = [self xForFrequency:atomic_load_explicit(&gEngine.crossover.frequencyA, memory_order_relaxed) inPlot:plot];
        double xB = [self xForFrequency:atomic_load_explicit(&gEngine.crossover.frequencyB, memory_order_relaxed) inPlot:plot];
        outputA = fabs(point.x - xA) <= fabs(point.x - xB);
    }
    double frequency = [self frequencyForX:point.x inPlot:plot];
    [self.crossoverDelegate crossoverResponseView:self didDragFrequency:frequency forOutputA:outputA];
}

- (void)mouseDown:(NSEvent *)event {
    [self updateDragAtPoint:[self convertPoint:event.locationInWindow fromView:nil]];
}

- (void)mouseDragged:(NSEvent *)event {
    [self updateDragAtPoint:[self convertPoint:event.locationInWindow fromView:nil]];
}

@end

@class RoutingMatrixView;
@protocol RoutingMatrixViewDelegate <NSObject>
- (void)routingMatrixView:(RoutingMatrixView *)view
         didToggleInputLeft:(BOOL)inputLeft
         destinationChannel:(uint32_t)destinationChannel;
@end

/* QLab-style crosspoint matrix. Rows are the stereo source inputs and columns
 * are the active destination device's hardware channels. */
@interface RoutingMatrixView : NSView
@property (nonatomic, weak) id<RoutingMatrixViewDelegate> routingDelegate;
@property (nonatomic, assign) BOOL outputA;
@end

@implementation RoutingMatrixView

- (OutputPathState *)routingPath {
    return self.outputA ? &gEngine.pathA : &gEngine.pathB;
}

- (uint32_t)channelCount {
    uint32_t count = atomic_load_explicit(&self.routingPath->outputChannelCount, memory_order_relaxed);
    if (count < 1U || count > kMaxOutputChannels) count = 2U;
    return count;
}

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    NSRect bounds = self.bounds;
    NSBezierPath *background = [NSBezierPath bezierPathWithRoundedRect:NSInsetRect(bounds, 0.5, 0.5) xRadius:5.0 yRadius:5.0];
    [[NSColor controlBackgroundColor] setFill];
    [background fill];
    [[NSColor separatorColor] setStroke];
    background.lineWidth = 1.0;
    [background stroke];

    uint32_t channelCount = [self channelCount];
    BOOL mono = atomic_load_explicit(&self.routingPath->mono, memory_order_relaxed);
    uint32_t leftMask = atomic_load_explicit(&self.routingPath->routeMaskLeft, memory_order_relaxed);
    uint32_t rightMask = atomic_load_explicit(&self.routingPath->routeMaskRight, memory_order_relaxed);
    uint32_t monoMask = atomic_load_explicit(&self.routingPath->monoRouteMask, memory_order_relaxed);
    CGFloat rowLabelWidth = 23.0;
    CGFloat headerHeight = 11.0;
    CGFloat rowHeight = (NSHeight(bounds) - headerHeight) / (mono ? 1.0 : 2.0);
    CGFloat columnWidth = (NSWidth(bounds) - rowLabelWidth - 3.0) / (CGFloat)channelCount;
    NSDictionary *labelAttributes = @{
        NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:9.0 weight:NSFontWeightMedium],
        NSForegroundColorAttributeName: [NSColor secondaryLabelColor]
    };
    [@"IN" drawAtPoint:NSMakePoint(3.0, NSHeight(bounds) - headerHeight + 1.0) withAttributes:labelAttributes];

    for (uint32_t channel = 0; channel < channelCount; ++channel) {
        CGFloat x = rowLabelWidth + (CGFloat)channel * columnWidth;
        NSString *number = [NSString stringWithFormat:@"%u", channel + 1U];
        NSSize numberSize = [number sizeWithAttributes:labelAttributes];
        [number drawAtPoint:NSMakePoint(x + (columnWidth - numberSize.width) * 0.5,
                                       NSHeight(bounds) - headerHeight + 1.0)
           withAttributes:labelAttributes];

        uint32_t rowCount = mono ? 1U : 2U;
        for (uint32_t row = 0; row < rowCount; ++row) {
            BOOL leftInput = mono || row == 1U;
            CGFloat y = mono ? 0.0 : (leftInput ? rowHeight : 0.0);
            NSRect cell = NSMakeRect(x + 1.0, y + 1.0, columnWidth - 2.0, rowHeight - 2.0);
            BOOL connected = mono ? ((monoMask & (1U << channel)) != 0U)
                                  : (leftInput ? ((leftMask & (1U << channel)) != 0U)
                                               : ((rightMask & (1U << channel)) != 0U));
            NSBezierPath *cellPath = [NSBezierPath bezierPathWithRoundedRect:cell xRadius:4.0 yRadius:4.0];
            NSColor *cellColor = connected
                ? [[NSColor controlAccentColor] colorWithAlphaComponent:0.18]
                : [[NSColor tertiaryLabelColor] colorWithAlphaComponent:0.10];
            [cellColor setFill];
            [cellPath fill];
            [[[NSColor separatorColor] colorWithAlphaComponent:(connected ? 0.6 : 0.35)] setStroke];
            cellPath.lineWidth = 0.5;
            [cellPath stroke];
            if (connected) {
                float gain = mono
                    ? atomic_load_explicit(&self.routingPath->monoRouteGain[channel], memory_order_relaxed)
                    : (leftInput
                        ? atomic_load_explicit(&self.routingPath->routeGainLeft[channel], memory_order_relaxed)
                        : atomic_load_explicit(&self.routingPath->routeGainRight[channel], memory_order_relaxed));
                double db = RouteGainLinearToDb(gain);
                NSString *gainText = fabs(db) < 0.05 ? @"0" : [NSString stringWithFormat:@"%+.0f", db];
                NSDictionary *activeAttributes = @{
                    NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:10.0 weight:NSFontWeightSemibold],
                    NSForegroundColorAttributeName: [NSColor controlAccentColor]
                };
                NSSize gainSize = [gainText sizeWithAttributes:activeAttributes];
                [gainText drawAtPoint:NSMakePoint(NSMidX(cell) - gainSize.width * 0.5,
                                                NSMidY(cell) - gainSize.height * 0.5)
                    withAttributes:activeAttributes];
            }
        }
    }

    if (mono) {
        [@"M" drawAtPoint:NSMakePoint(7.0, (rowHeight - 11.0) * 0.5) withAttributes:labelAttributes];
    } else {
        [@"L" drawAtPoint:NSMakePoint(7.0, rowHeight + (rowHeight - 11.0) * 0.5) withAttributes:labelAttributes];
        [@"R" drawAtPoint:NSMakePoint(7.0, (rowHeight - 11.0) * 0.5) withAttributes:labelAttributes];
    }
}

- (void)mouseDown:(NSEvent *)event {
    NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
    uint32_t channelCount = [self channelCount];
    BOOL mono = atomic_load_explicit(&self.routingPath->mono, memory_order_relaxed);
    CGFloat rowLabelWidth = 23.0;
    CGFloat headerHeight = 11.0;
    CGFloat rowHeight = (NSHeight(self.bounds) - headerHeight) / (mono ? 1.0 : 2.0);
    CGFloat matrixHeight = rowHeight * (mono ? 1.0 : 2.0);
    if (point.x < rowLabelWidth || point.y < 0.0 || point.y >= matrixHeight) return;
    CGFloat columnWidth = (NSWidth(self.bounds) - rowLabelWidth - 3.0) / (CGFloat)channelCount;
    uint32_t channel = (uint32_t)((point.x - rowLabelWidth) / columnWidth);
    if (channel >= channelCount) return;
    BOOL inputLeft = mono || point.y >= rowHeight;
    [self.routingDelegate routingMatrixView:self didToggleInputLeft:inputLeft destinationChannel:channel];
}

@end

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate, CrossoverResponseViewDelegate, RoutingMatrixViewDelegate>
@property (nonatomic, strong) NSWindow          *window;
@property (nonatomic, strong) NSScrollView      *controllerScrollView;

/* Menu Bar Status Item */
@property (nonatomic, strong) NSStatusItem      *statusItem;
@property (nonatomic, strong) NSMenuItem        *statusMenuHeaderItem;
@property (nonatomic, strong) NSMenuItem        *statusMenuPresetsItem;
@property (nonatomic, strong) NSMenuItem        *statusMenuEnableAItem;
@property (nonatomic, strong) NSMenuItem        *statusMenuMonoAItem;
@property (nonatomic, strong) NSMenuItem        *statusMenuEnableBItem;
@property (nonatomic, strong) NSMenuItem        *statusMenuMonoBItem;

/* Virtual Device Banner */
@property (nonatomic, strong) NSTextField       *virtualStatusLabel;
@property (nonatomic, strong) NSLevelIndicator  *inputMeter;
@property (nonatomic, strong) NSButton          *setSystemDefaultButton;

/* Presets Controls */
@property (nonatomic, strong) NSPopUpButton     *presetPopup;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSDictionary *> *savedPresets;
@property (nonatomic, copy)   NSString          *activePresetName;

/* Output A Controls */
@property (nonatomic, strong) NSPopUpButton     *popupA;
@property (nonatomic, strong) NSButton          *enableCheckboxA;
@property (nonatomic, strong) NSButton          *monoCheckboxA;
@property (nonatomic, strong) NSTextField       *statusLabelA;
@property (nonatomic, strong) NSSlider          *trimSliderA;
@property (nonatomic, strong) NSTextField       *trimValueLabelA;
@property (nonatomic, strong) NSSlider          *delaySliderA;
@property (nonatomic, strong) NSStepper         *delayStepperA;
@property (nonatomic, strong) NSTextField       *delayValueLabelA;
@property (nonatomic, strong) NSLevelIndicator  *meterA;
@property (nonatomic, strong) RoutingMatrixView *routingMatrixA;

/* Output B Controls */
@property (nonatomic, strong) NSPopUpButton     *popupB;
@property (nonatomic, strong) NSButton          *enableCheckboxB;
@property (nonatomic, strong) NSButton          *monoCheckboxB;
@property (nonatomic, strong) NSTextField       *statusLabelB;
@property (nonatomic, strong) NSSlider          *trimSliderB;
@property (nonatomic, strong) NSTextField       *trimValueLabelB;
@property (nonatomic, strong) NSSlider          *delaySliderB;
@property (nonatomic, strong) NSStepper         *delayStepperB;
@property (nonatomic, strong) NSTextField       *delayValueLabelB;
@property (nonatomic, strong) NSLevelIndicator  *meterB;
@property (nonatomic, strong) RoutingMatrixView *routingMatrixB;
@property (nonatomic, strong) NSPopover         *routingGainPopover;
@property (nonatomic, strong) NSSlider          *routingGainFader;
@property (nonatomic, strong) NSTextField       *routingGainValueField;
@property (nonatomic, weak)   RoutingMatrixView *activeRoutingMatrix;
@property (nonatomic, assign) BOOL              activeRoutingInputLeft;
@property (nonatomic, assign) BOOL              activeRoutingMono;
@property (nonatomic, assign) uint32_t          activeRoutingChannel;

/* Master & Balance Controls */
@property (nonatomic, strong) NSSlider          *masterSlider;
@property (nonatomic, strong) NSTextField       *masterValueLabel;
@property (nonatomic, strong) NSSlider          *balanceSlider;
@property (nonatomic, strong) NSTextField       *balanceValueLabel;
@property (nonatomic, strong) NSTextField       *errorBannerLabel;

/* Crossover Controls */
@property (nonatomic, strong) NSButton          *crossoverEnabledCheckbox;
@property (nonatomic, strong) NSButton          *crossoverLinkedCheckbox;
@property (nonatomic, strong) NSSlider          *linkedCrossoverFrequencySlider;
@property (nonatomic, strong) NSPopUpButton     *linkedCrossoverSlopePopup;
@property (nonatomic, strong) NSPopUpButton     *linkedCrossoverAssignmentPopup;
@property (nonatomic, strong) NSTextField       *linkedCrossoverFrequencyLabel;
@property (nonatomic, strong) NSPopUpButton     *crossoverModePopupA;
@property (nonatomic, strong) NSPopUpButton     *crossoverModePopupB;
@property (nonatomic, strong) NSSlider          *crossoverFrequencySliderA;
@property (nonatomic, strong) NSSlider          *crossoverFrequencySliderB;
@property (nonatomic, strong) NSPopUpButton     *crossoverSlopePopupA;
@property (nonatomic, strong) NSPopUpButton     *crossoverSlopePopupB;
@property (nonatomic, strong) NSTextField       *crossoverFrequencyLabelA;
@property (nonatomic, strong) NSTextField       *crossoverFrequencyLabelB;
@property (nonatomic, strong) NSTextField       *crossoverGraphFrequencyLabel;
@property (nonatomic, strong) NSArray<NSView *> *linkedCrossoverControls;
@property (nonatomic, strong) NSArray<NSView *> *independentCrossoverControls;
@property (nonatomic, strong) CrossoverResponseView *crossoverResponseView;

@property (nonatomic, strong) NSArray<AudioDeviceItem *> *availableDevices;
@property (nonatomic, copy)   NSString          *desiredUID_A;
@property (nonatomic, copy)   NSString          *desiredUID_B;
@property (nonatomic, strong) NSTimer           *telemetryTimer;
@end

@implementation AppDelegate

- (NSTextField *)makeLabel:(NSString *)text bold:(BOOL)bold size:(CGFloat)size color:(NSColor *)color {
    NSTextField *tf = [[NSTextField alloc] initWithFrame:NSZeroRect];
    tf.stringValue     = text ?: @"";
    tf.bezeled         = NO;
    tf.drawsBackground = NO;
    tf.editable        = NO;
    tf.selectable      = NO;
    tf.font            = bold ? [NSFont boldSystemFontOfSize:size] : [NSFont systemFontOfSize:size];
    if (color) tf.textColor = color;
    return tf;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    (void)notification;
    memset(&gEngine, 0, sizeof(gEngine));
    gEngine.virtualDeviceID = kAudioObjectUnknown;
    gEngine.pathA.deviceID  = kAudioObjectUnknown;
    gEngine.pathB.deviceID  = kAudioObjectUnknown;

    [self loadPreferences];
    [self setupMainMenu];
    [self setupStatusBarItem];
    [self buildWindowAndUI];
    [self rebuildPresetsUI];
    [self requestMicrophonePermissionAndStartAudio];
    [self registerCoreAudioListeners];

    self.telemetryTimer = [NSTimer scheduledTimerWithTimeInterval:0.06
                                                           target:self
                                                         selector:@selector(updateTelemetryUI)
                                                         userInfo:nil
                                                          repeats:YES];
}

- (void)applicationWillTerminate:(NSNotification *)notification {
    (void)notification;
    [self.telemetryTimer invalidate];
    StopOutputPath(&gEngine.pathA);
    StopOutputPath(&gEngine.pathB);
    StopVirtualInputUnit();
    [self savePreferences];
}

/* Keep running in background when GUI window is closed */
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender;
    return NO;
}

/* Reopen window if user clicks app in Finder / Spotlight while hidden */
- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)flag {
    (void)sender;
    if (!flag) {
        [self showControllerWindow:nil];
    }
    return YES;
}

/* Intercept red window close button & Cmd+W: hide window & stay active in Menu Bar */
- (BOOL)windowShouldClose:(NSWindow *)sender {
    (void)sender;
    [self.window orderOut:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    return NO;
}

- (void)showControllerWindow:(id)sender {
    (void)sender;
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    self.window.collectionBehavior |= NSWindowCollectionBehaviorMoveToActiveSpace;
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)quitApplication:(id)sender {
    (void)sender;
    [NSApp terminate:nil];
}

/*===========================================================================
 * Standard macOS Application Menu Bar (Enables Cmd+Q, Cmd+W, Cmd+S, Cmd+C/V)
 *===========================================================================*/

- (void)setupMainMenu {
    NSMenu *mainMenu = [[NSMenu alloc] initWithTitle:@"MainMenu"];

    /* 1. Application Menu (Dual Output) */
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] initWithTitle:@"Dual Output" action:nil keyEquivalent:@""];
    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"Dual Output"];

    [appMenu addItemWithTitle:@"About Dual Output"
                       action:@selector(orderFrontStandardAboutPanel:)
                keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Hide Dual Output"
                       action:@selector(hide:)
                keyEquivalent:@"h"];

    NSMenuItem *hideOthersItem = [appMenu addItemWithTitle:@"Hide Others"
                                                    action:@selector(hideOtherApplications:)
                                             keyEquivalent:@"h"];
    hideOthersItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagOption;

    [appMenu addItemWithTitle:@"Show All"
                       action:@selector(unhideAllApplications:)
                keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *quitItem = [[NSMenuItem alloc] initWithTitle:@"Quit Dual Output"
                                                      action:@selector(quitApplication:)
                                               keyEquivalent:@"q"];
    quitItem.target = self;
    [appMenu addItem:quitItem];

    appMenuItem.submenu = appMenu;
    [mainMenu addItem:appMenuItem];

    /* 2. File / Window Menu (Enables Cmd+W & Cmd+S) */
    NSMenuItem *fileMenuItem = [[NSMenuItem alloc] initWithTitle:@"Window" action:nil keyEquivalent:@""];
    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"Window"];

    NSMenuItem *savePresetItem = [[NSMenuItem alloc] initWithTitle:@"Save Preset..."
                                                            action:@selector(savePresetClicked:)
                                                     keyEquivalent:@"s"];
    savePresetItem.target = self;
    [fileMenu addItem:savePresetItem];

    [fileMenu addItem:[NSMenuItem separatorItem]];
    [fileMenu addItemWithTitle:@"Minimize"
                        action:@selector(performMiniaturize:)
                 keyEquivalent:@"m"];
    [fileMenu addItemWithTitle:@"Close Window (Keep in Menu Bar)"
                        action:@selector(performClose:)
                 keyEquivalent:@"w"];

    fileMenuItem.submenu = fileMenu;
    [mainMenu addItem:fileMenuItem];

    /* 3. Edit Menu (Enables Cmd+X/C/V/A in Preset Name Dialog) */
    NSMenuItem *editMenuItem = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
    NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
    [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
    editMenuItem.submenu = editMenu;
    [mainMenu addItem:editMenuItem];

    [NSApp setMainMenu:mainMenu];
}

/*===========================================================================
 * Custom Compact Menu Bar Status Icon (Icon-Only Y-Splitter Vector Template)
 *===========================================================================*/

- (NSImage *)createDualSplitterStatusIcon {
    NSSize iconSize = NSMakeSize(18.0, 18.0);
    NSImage *img = [NSImage imageWithSize:iconSize flipped:NO drawingHandler:^BOOL(NSRect dstRect) {
        (void)dstRect;
        [[NSColor blackColor] setStroke];
        [[NSColor blackColor] setFill];

        /* Left audio input node circle */
        NSBezierPath *inNode = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(1.5, 7.2, 3.6, 3.6)];
        [inNode fill];

        /* Input stem from left node to center split point */
        NSBezierPath *stem = [NSBezierPath bezierPath];
        stem.lineWidth = 1.75;
        stem.lineCapStyle = NSLineCapStyleRound;
        stem.lineJoinStyle = NSLineJoinStyleRound;
        [stem moveToPoint:NSMakePoint(4.5, 9.0)];
        [stem lineToPoint:NSMakePoint(7.8, 9.0)];
        [stem stroke];

        /* Upper branch (Output A) & Lower branch (Output B) */
        NSBezierPath *branches = [NSBezierPath bezierPath];
        branches.lineWidth = 1.65;
        branches.lineCapStyle = NSLineCapStyleRound;
        branches.lineJoinStyle = NSLineJoinStyleRound;

        /* Output A path */
        [branches moveToPoint:NSMakePoint(7.8, 9.0)];
        [branches lineToPoint:NSMakePoint(11.0, 13.5)];
        [branches lineToPoint:NSMakePoint(15.2, 13.5)];

        /* Output B path */
        [branches moveToPoint:NSMakePoint(7.8, 9.0)];
        [branches lineToPoint:NSMakePoint(11.0, 4.5)];
        [branches lineToPoint:NSMakePoint(15.2, 4.5)];
        [branches stroke];

        /* Arrowheads for Output A and Output B */
        NSBezierPath *arrows = [NSBezierPath bezierPath];
        arrows.lineWidth = 1.55;
        arrows.lineCapStyle = NSLineCapStyleRound;
        arrows.lineJoinStyle = NSLineJoinStyleRound;

        /* Top arrowhead */
        [arrows moveToPoint:NSMakePoint(13.4, 15.3)];
        [arrows lineToPoint:NSMakePoint(15.8, 13.5)];
        [arrows lineToPoint:NSMakePoint(13.4, 11.7)];

        /* Bottom arrowhead */
        [arrows moveToPoint:NSMakePoint(13.4, 6.3)];
        [arrows lineToPoint:NSMakePoint(15.8, 4.5)];
        [arrows lineToPoint:NSMakePoint(13.4, 2.7)];
        [arrows stroke];

        return YES;
    }];
    [img setTemplate:YES];
    return img;
}

- (void)setupStatusBarItem {
    self.statusItem = [[NSStatusBar systemStatusBar] statusItemWithLength:NSSquareStatusItemLength];
    self.statusItem.button.image = [self createDualSplitterStatusIcon];
    self.statusItem.button.title = @"";
    self.statusItem.button.toolTip = @"Dual Output Audio Controller";

    NSMenu *menu = [[NSMenu alloc] initWithTitle:@"Dual Output"];

    self.statusMenuHeaderItem = [[NSMenuItem alloc] initWithTitle:@"Dual Output: Initializing..."
                                                           action:nil
                                                    keyEquivalent:@""];
    self.statusMenuHeaderItem.enabled = NO;
    [menu addItem:self.statusMenuHeaderItem];
    [menu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *openItem = [[NSMenuItem alloc] initWithTitle:@"Open Controller Window..."
                                                      action:@selector(showControllerWindow:)
                                               keyEquivalent:@"o"];
    openItem.target = self;
    [menu addItem:openItem];

    NSMenuItem *setSysItem = [[NSMenuItem alloc] initWithTitle:@"Set Dual Output as System Output"
                                                        action:@selector(setVirtualDeviceAsSystemOutput:)
                                                 keyEquivalent:@""];
    setSysItem.target = self;
    [menu addItem:setSysItem];

    [menu addItem:[NSMenuItem separatorItem]];

    /* Presets Submenu in Menu Bar */
    self.statusMenuPresetsItem = [[NSMenuItem alloc] initWithTitle:@"Presets"
                                                            action:nil
                                                     keyEquivalent:@""];
    NSMenu *presetsSubmenu = [[NSMenu alloc] initWithTitle:@"Presets"];
    self.statusMenuPresetsItem.submenu = presetsSubmenu;
    [menu addItem:self.statusMenuPresetsItem];

    [menu addItem:[NSMenuItem separatorItem]];

    /* Quick Output A & B Toggles */
    self.statusMenuEnableAItem = [[NSMenuItem alloc] initWithTitle:@"Output A Enabled"
                                                            action:@selector(menuToggleEnableA:)
                                                     keyEquivalent:@""];
    self.statusMenuEnableAItem.target = self;
    [menu addItem:self.statusMenuEnableAItem];

    self.statusMenuMonoAItem = [[NSMenuItem alloc] initWithTitle:@"Output A Mono (Sum L+R)"
                                                          action:@selector(menuToggleMonoA:)
                                                   keyEquivalent:@""];
    self.statusMenuMonoAItem.target = self;
    [menu addItem:self.statusMenuMonoAItem];

    [menu addItem:[NSMenuItem separatorItem]];

    self.statusMenuEnableBItem = [[NSMenuItem alloc] initWithTitle:@"Output B Enabled"
                                                            action:@selector(menuToggleEnableB:)
                                                     keyEquivalent:@""];
    self.statusMenuEnableBItem.target = self;
    [menu addItem:self.statusMenuEnableBItem];

    self.statusMenuMonoBItem = [[NSMenuItem alloc] initWithTitle:@"Output B Mono (Sum L+R)"
                                                          action:@selector(menuToggleMonoB:)
                                                   keyEquivalent:@""];
    self.statusMenuMonoBItem.target = self;
    [menu addItem:self.statusMenuMonoBItem];

    [menu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *quitItem = [[NSMenuItem alloc] initWithTitle:@"Quit Dual Output"
                                                      action:@selector(quitApplication:)
                                               keyEquivalent:@"q"];
    quitItem.target = self;
    [menu addItem:quitItem];

    self.statusItem.menu = menu;
    [self syncStatusBarMenuChecks];
}

- (void)syncStatusBarMenuChecks {
    self.statusMenuEnableAItem.state = atomic_load(&gEngine.pathA.enabled) ? NSControlStateValueOn : NSControlStateValueOff;
    self.statusMenuMonoAItem.state   = atomic_load(&gEngine.pathA.mono)    ? NSControlStateValueOn : NSControlStateValueOff;
    self.statusMenuEnableBItem.state = atomic_load(&gEngine.pathB.enabled) ? NSControlStateValueOn : NSControlStateValueOff;
    self.statusMenuMonoBItem.state   = atomic_load(&gEngine.pathB.mono)    ? NSControlStateValueOn : NSControlStateValueOff;
}

/*===========================================================================
 * Preferences & Presets Persistence
 *===========================================================================*/

- (void)loadPreferences {
    NSUserDefaults *defs = [NSUserDefaults standardUserDefaults];
    [defs registerDefaults:@{
        kPrefOutputAEnabled : @YES,
        kPrefOutputBEnabled : @YES,
        kPrefOutputAMono    : @NO,
        kPrefOutputBMono    : @NO,
        kPrefMasterVolume   : @(1.0f),
        kPrefBalance        : @(0.0f),
        kPrefTrimDbA        : @(0.0f),
        kPrefTrimDbB        : @(0.0f),
        kPrefDelayMsA       : @(0.0),
        kPrefDelayMsB       : @(0.0),
        kPrefRouteMaskLeftA : @(1U),
        kPrefRouteMaskRightA : @(2U),
        kPrefRouteMaskLeftB : @(1U),
        kPrefRouteMaskRightB : @(2U),
        kPrefMonoRouteMaskA : @(3U),
        kPrefMonoRouteMaskB : @(3U),
        /* Keep the new processor off for existing setups until deliberately
         * enabled; new controls are otherwise ready with a sensible split. */
        kPrefCrossoverEnabled : @NO,
        kPrefCrossoverLinked  : @YES,
        kPrefCrossoverLinkedOutputAHighPass : @NO,
        kPrefCrossoverFrequency : @(120.0),
        kPrefCrossoverSlope     : @(24),
        kPrefCrossoverModeA     : @(kCrossoverFilterLowPass),
        kPrefCrossoverModeB     : @(kCrossoverFilterHighPass),
        kPrefCrossoverFrequencyA : @(120.0),
        kPrefCrossoverFrequencyB : @(120.0),
        kPrefCrossoverSlopeA     : @(24),
        kPrefCrossoverSlopeB     : @(24)
    }];

    self.desiredUID_A = [defs stringForKey:kPrefOutputAUID];
    self.desiredUID_B = [defs stringForKey:kPrefOutputBUID];

    atomic_store(&gEngine.pathA.enabled, [defs boolForKey:kPrefOutputAEnabled]);
    atomic_store(&gEngine.pathB.enabled, [defs boolForKey:kPrefOutputBEnabled]);
    atomic_store(&gEngine.pathA.mono,    [defs boolForKey:kPrefOutputAMono]);
    atomic_store(&gEngine.pathB.mono,    [defs boolForKey:kPrefOutputBMono]);
    gEngine.masterVolume = [defs floatForKey:kPrefMasterVolume];
    gEngine.balance      = [defs floatForKey:kPrefBalance];
    gEngine.trimDbA      = [defs floatForKey:kPrefTrimDbA];
    gEngine.trimDbB      = [defs floatForKey:kPrefTrimDbB];
    atomic_store(&gEngine.pathA.targetDelayMs, [defs doubleForKey:kPrefDelayMsA]);
    atomic_store(&gEngine.pathB.targetDelayMs, [defs doubleForKey:kPrefDelayMsB]);
    atomic_store(&gEngine.pathA.routeMaskLeft, (uint32_t)[defs integerForKey:kPrefRouteMaskLeftA]);
    atomic_store(&gEngine.pathA.routeMaskRight, (uint32_t)[defs integerForKey:kPrefRouteMaskRightA]);
    atomic_store(&gEngine.pathB.routeMaskLeft, (uint32_t)[defs integerForKey:kPrefRouteMaskLeftB]);
    atomic_store(&gEngine.pathB.routeMaskRight, (uint32_t)[defs integerForKey:kPrefRouteMaskRightB]);
    atomic_store(&gEngine.pathA.monoRouteMask, (uint32_t)[defs integerForKey:kPrefMonoRouteMaskA]);
    atomic_store(&gEngine.pathB.monoRouteMask, (uint32_t)[defs integerForKey:kPrefMonoRouteMaskB]);
    atomic_store(&gEngine.pathA.outputChannelCount, 2U);
    atomic_store(&gEngine.pathB.outputChannelCount, 2U);
    for (uint32_t channel = 0; channel < kMaxOutputChannels; ++channel) {
        atomic_store(&gEngine.pathA.routeGainLeft[channel], 1.0f);
        atomic_store(&gEngine.pathA.routeGainRight[channel], 1.0f);
        atomic_store(&gEngine.pathB.routeGainLeft[channel], 1.0f);
        atomic_store(&gEngine.pathB.routeGainRight[channel], 1.0f);
        atomic_store(&gEngine.pathA.monoRouteGain[channel], 1.0f);
        atomic_store(&gEngine.pathB.monoRouteGain[channel], 1.0f);
    }
    [self applyRouteGainDbValues:[defs arrayForKey:kPrefRouteGainDbA] toPath:&gEngine.pathA];
    [self applyRouteGainDbValues:[defs arrayForKey:kPrefRouteGainDbB] toPath:&gEngine.pathB];
    [self applyMonoRouteGainDbValues:[defs arrayForKey:kPrefMonoRouteGainDbA] toPath:&gEngine.pathA];
    [self applyMonoRouteGainDbValues:[defs arrayForKey:kPrefMonoRouteGainDbB] toPath:&gEngine.pathB];
    atomic_store(&gEngine.crossover.enabled, [defs boolForKey:kPrefCrossoverEnabled]);
    atomic_store(&gEngine.crossover.linked, [defs boolForKey:kPrefCrossoverLinked]);
    atomic_store(&gEngine.crossover.linkedOutputAHighPass, [defs boolForKey:kPrefCrossoverLinkedOutputAHighPass]);
    atomic_store(&gEngine.crossover.linkedFrequency, [defs doubleForKey:kPrefCrossoverFrequency]);
    atomic_store(&gEngine.crossover.linkedSlopeDb, ClampCrossoverSlope([defs integerForKey:kPrefCrossoverSlope]));
    atomic_store(&gEngine.crossover.modeA, [defs integerForKey:kPrefCrossoverModeA]);
    atomic_store(&gEngine.crossover.modeB, [defs integerForKey:kPrefCrossoverModeB]);
    atomic_store(&gEngine.crossover.frequencyA, [defs doubleForKey:kPrefCrossoverFrequencyA]);
    atomic_store(&gEngine.crossover.frequencyB, [defs doubleForKey:kPrefCrossoverFrequencyB]);
    atomic_store(&gEngine.crossover.slopeDbA, ClampCrossoverSlope([defs integerForKey:kPrefCrossoverSlopeA]));
    atomic_store(&gEngine.crossover.slopeDbB, ClampCrossoverSlope([defs integerForKey:kPrefCrossoverSlopeB]));

    NSDictionary *storedPresets = [defs dictionaryForKey:kPrefSavedPresets];
    if (storedPresets != nil) {
        self.savedPresets = [storedPresets mutableCopy];
    } else {
        self.savedPresets = [NSMutableDictionary dictionary];
    }
    self.activePresetName = [defs stringForKey:kPrefActivePreset] ?: @"";

    UpdateEngineTargetGains();
}

- (void)savePreferences {
    NSUserDefaults *defs = [NSUserDefaults standardUserDefaults];
    if (self.desiredUID_A) [defs setObject:self.desiredUID_A forKey:kPrefOutputAUID];
    if (self.desiredUID_B) [defs setObject:self.desiredUID_B forKey:kPrefOutputBUID];
    [defs setBool:atomic_load(&gEngine.pathA.enabled) forKey:kPrefOutputAEnabled];
    [defs setBool:atomic_load(&gEngine.pathB.enabled) forKey:kPrefOutputBEnabled];
    [defs setBool:atomic_load(&gEngine.pathA.mono)    forKey:kPrefOutputAMono];
    [defs setBool:atomic_load(&gEngine.pathB.mono)    forKey:kPrefOutputBMono];
    [defs setFloat:gEngine.masterVolume forKey:kPrefMasterVolume];
    [defs setFloat:gEngine.balance forKey:kPrefBalance];
    [defs setFloat:gEngine.trimDbA forKey:kPrefTrimDbA];
    [defs setFloat:gEngine.trimDbB forKey:kPrefTrimDbB];
    [defs setDouble:atomic_load(&gEngine.pathA.targetDelayMs) forKey:kPrefDelayMsA];
    [defs setDouble:atomic_load(&gEngine.pathB.targetDelayMs) forKey:kPrefDelayMsB];
    [defs setInteger:atomic_load(&gEngine.pathA.routeMaskLeft) forKey:kPrefRouteMaskLeftA];
    [defs setInteger:atomic_load(&gEngine.pathA.routeMaskRight) forKey:kPrefRouteMaskRightA];
    [defs setInteger:atomic_load(&gEngine.pathB.routeMaskLeft) forKey:kPrefRouteMaskLeftB];
    [defs setInteger:atomic_load(&gEngine.pathB.routeMaskRight) forKey:kPrefRouteMaskRightB];
    [defs setInteger:atomic_load(&gEngine.pathA.monoRouteMask) forKey:kPrefMonoRouteMaskA];
    [defs setInteger:atomic_load(&gEngine.pathB.monoRouteMask) forKey:kPrefMonoRouteMaskB];
    [defs setObject:[self routeGainDbValuesForPath:&gEngine.pathA] forKey:kPrefRouteGainDbA];
    [defs setObject:[self routeGainDbValuesForPath:&gEngine.pathB] forKey:kPrefRouteGainDbB];
    [defs setObject:[self monoRouteGainDbValuesForPath:&gEngine.pathA] forKey:kPrefMonoRouteGainDbA];
    [defs setObject:[self monoRouteGainDbValuesForPath:&gEngine.pathB] forKey:kPrefMonoRouteGainDbB];
    [defs setBool:atomic_load(&gEngine.crossover.enabled) forKey:kPrefCrossoverEnabled];
    [defs setBool:atomic_load(&gEngine.crossover.linked) forKey:kPrefCrossoverLinked];
    [defs setBool:atomic_load(&gEngine.crossover.linkedOutputAHighPass) forKey:kPrefCrossoverLinkedOutputAHighPass];
    [defs setDouble:atomic_load(&gEngine.crossover.linkedFrequency) forKey:kPrefCrossoverFrequency];
    [defs setInteger:atomic_load(&gEngine.crossover.linkedSlopeDb) forKey:kPrefCrossoverSlope];
    [defs setInteger:atomic_load(&gEngine.crossover.modeA) forKey:kPrefCrossoverModeA];
    [defs setInteger:atomic_load(&gEngine.crossover.modeB) forKey:kPrefCrossoverModeB];
    [defs setDouble:atomic_load(&gEngine.crossover.frequencyA) forKey:kPrefCrossoverFrequencyA];
    [defs setDouble:atomic_load(&gEngine.crossover.frequencyB) forKey:kPrefCrossoverFrequencyB];
    [defs setInteger:atomic_load(&gEngine.crossover.slopeDbA) forKey:kPrefCrossoverSlopeA];
    [defs setInteger:atomic_load(&gEngine.crossover.slopeDbB) forKey:kPrefCrossoverSlopeB];
    if (self.savedPresets) [defs setObject:self.savedPresets forKey:kPrefSavedPresets];
    if (self.activePresetName) [defs setObject:self.activePresetName forKey:kPrefActivePreset];
    [defs synchronize];
}

- (NSDictionary *)captureCurrentStateAsPresetDictionary {
    return @{
        kPrefOutputAUID     : self.desiredUID_A ?: @"",
        kPrefOutputBUID     : self.desiredUID_B ?: @"",
        kPrefOutputAEnabled : @(atomic_load(&gEngine.pathA.enabled)),
        kPrefOutputBEnabled : @(atomic_load(&gEngine.pathB.enabled)),
        kPrefOutputAMono    : @(atomic_load(&gEngine.pathA.mono)),
        kPrefOutputBMono    : @(atomic_load(&gEngine.pathB.mono)),
        kPrefMasterVolume   : @(gEngine.masterVolume),
        kPrefBalance        : @(gEngine.balance),
        kPrefTrimDbA        : @(gEngine.trimDbA),
        kPrefTrimDbB        : @(gEngine.trimDbB),
        kPrefDelayMsA       : @(atomic_load(&gEngine.pathA.targetDelayMs)),
        kPrefDelayMsB       : @(atomic_load(&gEngine.pathB.targetDelayMs)),
        kPrefRouteMaskLeftA : @(atomic_load(&gEngine.pathA.routeMaskLeft)),
        kPrefRouteMaskRightA : @(atomic_load(&gEngine.pathA.routeMaskRight)),
        kPrefRouteMaskLeftB : @(atomic_load(&gEngine.pathB.routeMaskLeft)),
        kPrefRouteMaskRightB : @(atomic_load(&gEngine.pathB.routeMaskRight)),
        kPrefMonoRouteMaskA : @(atomic_load(&gEngine.pathA.monoRouteMask)),
        kPrefMonoRouteMaskB : @(atomic_load(&gEngine.pathB.monoRouteMask)),
        kPrefRouteGainDbA : [self routeGainDbValuesForPath:&gEngine.pathA],
        kPrefRouteGainDbB : [self routeGainDbValuesForPath:&gEngine.pathB],
        kPrefMonoRouteGainDbA : [self monoRouteGainDbValuesForPath:&gEngine.pathA],
        kPrefMonoRouteGainDbB : [self monoRouteGainDbValuesForPath:&gEngine.pathB],
        kPrefCrossoverEnabled : @(atomic_load(&gEngine.crossover.enabled)),
        kPrefCrossoverLinked  : @(atomic_load(&gEngine.crossover.linked)),
        kPrefCrossoverLinkedOutputAHighPass : @(atomic_load(&gEngine.crossover.linkedOutputAHighPass)),
        kPrefCrossoverFrequency : @(atomic_load(&gEngine.crossover.linkedFrequency)),
        kPrefCrossoverSlope     : @(atomic_load(&gEngine.crossover.linkedSlopeDb)),
        kPrefCrossoverModeA     : @(atomic_load(&gEngine.crossover.modeA)),
        kPrefCrossoverModeB     : @(atomic_load(&gEngine.crossover.modeB)),
        kPrefCrossoverFrequencyA : @(atomic_load(&gEngine.crossover.frequencyA)),
        kPrefCrossoverFrequencyB : @(atomic_load(&gEngine.crossover.frequencyB)),
        kPrefCrossoverSlopeA     : @(atomic_load(&gEngine.crossover.slopeDbA)),
        kPrefCrossoverSlopeB     : @(atomic_load(&gEngine.crossover.slopeDbB))
    };
}

- (void)applyPresetNamed:(NSString *)presetName {
    NSDictionary *preset = self.savedPresets[presetName];
    if (preset == nil) return;

    self.activePresetName = presetName;
    self.desiredUID_A     = preset[kPrefOutputAUID] ?: @"";
    self.desiredUID_B     = preset[kPrefOutputBUID] ?: @"";

    bool enA   = [preset[kPrefOutputAEnabled] boolValue];
    bool enB   = [preset[kPrefOutputBEnabled] boolValue];
    bool monoA = [preset[kPrefOutputAMono] boolValue];
    bool monoB = [preset[kPrefOutputBMono] boolValue];

    atomic_store(&gEngine.pathA.enabled, enA);
    atomic_store(&gEngine.pathB.enabled, enB);
    atomic_store(&gEngine.pathA.mono,    monoA);
    atomic_store(&gEngine.pathB.mono,    monoB);

    gEngine.masterVolume = [preset[kPrefMasterVolume] floatValue];
    gEngine.balance      = [preset[kPrefBalance] floatValue];
    gEngine.trimDbA      = [preset[kPrefTrimDbA] floatValue];
    gEngine.trimDbB      = [preset[kPrefTrimDbB] floatValue];

    double delA = [preset[kPrefDelayMsA] doubleValue];
    double delB = [preset[kPrefDelayMsB] doubleValue];
    atomic_store(&gEngine.pathA.targetDelayMs, delA);
    atomic_store(&gEngine.pathB.targetDelayMs, delB);

    /* Older presets did not contain routing cells, so retain the current
     * matrix when those entries are absent. */
    NSNumber *routeMaskLeftA = preset[kPrefRouteMaskLeftA];
    NSNumber *routeMaskRightA = preset[kPrefRouteMaskRightA];
    NSNumber *routeMaskLeftB = preset[kPrefRouteMaskLeftB];
    NSNumber *routeMaskRightB = preset[kPrefRouteMaskRightB];
    NSNumber *monoRouteMaskA = preset[kPrefMonoRouteMaskA];
    NSNumber *monoRouteMaskB = preset[kPrefMonoRouteMaskB];
    if (routeMaskLeftA != nil) atomic_store(&gEngine.pathA.routeMaskLeft, routeMaskLeftA.unsignedIntValue);
    if (routeMaskRightA != nil) atomic_store(&gEngine.pathA.routeMaskRight, routeMaskRightA.unsignedIntValue);
    if (routeMaskLeftB != nil) atomic_store(&gEngine.pathB.routeMaskLeft, routeMaskLeftB.unsignedIntValue);
    if (routeMaskRightB != nil) atomic_store(&gEngine.pathB.routeMaskRight, routeMaskRightB.unsignedIntValue);
    if (monoRouteMaskA != nil) atomic_store(&gEngine.pathA.monoRouteMask, monoRouteMaskA.unsignedIntValue);
    if (monoRouteMaskB != nil) atomic_store(&gEngine.pathB.monoRouteMask, monoRouteMaskB.unsignedIntValue);
    [self applyRouteGainDbValues:preset[kPrefRouteGainDbA] toPath:&gEngine.pathA];
    [self applyRouteGainDbValues:preset[kPrefRouteGainDbB] toPath:&gEngine.pathB];
    [self applyMonoRouteGainDbValues:preset[kPrefMonoRouteGainDbA] toPath:&gEngine.pathA];
    [self applyMonoRouteGainDbValues:preset[kPrefMonoRouteGainDbB] toPath:&gEngine.pathB];

    /* Presets saved before the crossover feature simply leave the currently
     * loaded crossover configuration intact. */
    NSNumber *crossoverEnabled = preset[kPrefCrossoverEnabled];
    if (crossoverEnabled != nil) {
        atomic_store(&gEngine.crossover.enabled, crossoverEnabled.boolValue);
        atomic_store(&gEngine.crossover.linked, [preset[kPrefCrossoverLinked] boolValue]);
        atomic_store(&gEngine.crossover.linkedOutputAHighPass, [preset[kPrefCrossoverLinkedOutputAHighPass] boolValue]);
        atomic_store(&gEngine.crossover.linkedFrequency, [preset[kPrefCrossoverFrequency] doubleValue]);
        atomic_store(&gEngine.crossover.linkedSlopeDb, ClampCrossoverSlope([preset[kPrefCrossoverSlope] integerValue]));
        atomic_store(&gEngine.crossover.modeA, [preset[kPrefCrossoverModeA] integerValue]);
        atomic_store(&gEngine.crossover.modeB, [preset[kPrefCrossoverModeB] integerValue]);
        atomic_store(&gEngine.crossover.frequencyA, [preset[kPrefCrossoverFrequencyA] doubleValue]);
        atomic_store(&gEngine.crossover.frequencyB, [preset[kPrefCrossoverFrequencyB] doubleValue]);
        atomic_store(&gEngine.crossover.slopeDbA, ClampCrossoverSlope([preset[kPrefCrossoverSlopeA] integerValue]));
        atomic_store(&gEngine.crossover.slopeDbB, ClampCrossoverSlope([preset[kPrefCrossoverSlopeB] integerValue]));
    }

    UpdateEngineTargetGains();

    self.enableCheckboxA.state  = enA   ? NSControlStateValueOn : NSControlStateValueOff;
    self.enableCheckboxB.state  = enB   ? NSControlStateValueOn : NSControlStateValueOff;
    self.monoCheckboxA.state    = monoA ? NSControlStateValueOn : NSControlStateValueOff;
    self.monoCheckboxB.state    = monoB ? NSControlStateValueOn : NSControlStateValueOff;
    self.masterSlider.floatValue  = gEngine.masterVolume;
    self.balanceSlider.floatValue = gEngine.balance;
    self.trimSliderA.floatValue   = gEngine.trimDbA;
    self.trimSliderB.floatValue   = gEngine.trimDbB;
    self.delaySliderA.doubleValue  = delA;
    self.delayStepperA.doubleValue = delA;
    self.delaySliderB.doubleValue  = delB;
    self.delayStepperB.doubleValue = delB;

    [self updateCrossoverControls];
    [self updateRoutingMatrixViews];

    [self updateValueLabels];
    [self syncStatusBarMenuChecks];
    [self rebuildPresetsUI];
    [self refreshDevicesAndStreams];
    [self savePreferences];

    self.errorBannerLabel.textColor = [NSColor systemGreenColor];
    self.errorBannerLabel.stringValue = [NSString stringWithFormat:@"Loaded preset \"%@\".", presetName];
}

- (void)rebuildPresetsUI {
    NSArray<NSString *> *sortedNames = [[self.savedPresets allKeys] sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];

    if (self.presetPopup != nil) {
        [self.presetPopup removeAllItems];
        [self.presetPopup addItemWithTitle:@"— Select Preset —"];
        for (NSString *name in sortedNames) {
            [self.presetPopup addItemWithTitle:name];
        }
        if (self.activePresetName.length > 0 && [sortedNames containsObject:self.activePresetName]) {
            [self.presetPopup selectItemWithTitle:self.activePresetName];
        } else {
            [self.presetPopup selectItemAtIndex:0];
        }
    }

    NSMenu *sub = self.statusMenuPresetsItem.submenu;
    [sub removeAllItems];
    if (sortedNames.count == 0) {
        NSMenuItem *emptyItem = [[NSMenuItem alloc] initWithTitle:@"No Saved Presets" action:nil keyEquivalent:@""];
        emptyItem.enabled = NO;
        [sub addItem:emptyItem];
    } else {
        for (NSString *name in sortedNames) {
            NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:name
                                                          action:@selector(menuPresetClicked:)
                                                   keyEquivalent:@""];
            item.target = self;
            item.representedObject = name;
            if ([name isEqualToString:self.activePresetName]) {
                item.state = NSControlStateValueOn;
            }
            [sub addItem:item];
        }
    }
}

/*===========================================================================
 * Main Window & Controls Construction
 *===========================================================================*/

- (void)buildWindowAndUI {
    /* The content remains a deliberately spacious fixed layout, while the
     * surrounding window can shrink or grow on any display. */
    NSRect frame = NSMakeRect(0, 0, 780, 820);
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                       NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                              styleMask:style
                                                backing:NSBackingStoreBuffered
                                                  defer:NO];
    self.window.title = @"Dual Output — Virtual Audio Splitter & Synchronizer";
    self.window.releasedWhenClosed = NO;
    self.window.delegate = self;
    self.window.contentMinSize = NSMakeSize(480.0, 480.0);
    /* Reopening from the menu bar follows the current desktop instead of
     * making macOS switch back to the Space where it was last hidden. */
    self.window.collectionBehavior = NSWindowCollectionBehaviorMoveToActiveSpace;
    [self.window center];

    self.controllerScrollView = [[NSScrollView alloc] initWithFrame:self.window.contentView.bounds];
    self.controllerScrollView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.controllerScrollView.hasVerticalScroller = YES;
    self.controllerScrollView.hasHorizontalScroller = YES;
    self.controllerScrollView.autohidesScrollers = YES;
    self.controllerScrollView.borderType = NSNoBorder;
    self.controllerScrollView.drawsBackground = YES;
    [self.window.contentView addSubview:self.controllerScrollView];

    NSView *cv = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 740, 792)];
    self.controllerScrollView.documentView = cv;

    /* 1. Top Virtual Device Banner Box */
    NSBox *topBox = [[NSBox alloc] initWithFrame:NSMakeRect(20, 708, 700, 76)];
    topBox.title = @"Virtual Audio Device Status (System Input Source)";
    topBox.titleFont = [NSFont boldSystemFontOfSize:12];
    [cv addSubview:topBox];

    self.virtualStatusLabel = [self makeLabel:@"Checking Dual Output driver..." bold:YES size:12 color:[NSColor labelColor]];
    self.virtualStatusLabel.frame = NSMakeRect(14, 29, 470, 18);
    [topBox.contentView addSubview:self.virtualStatusLabel];

    NSTextField *inMeterLbl = [self makeLabel:@"Input Level:" bold:NO size:11 color:[NSColor secondaryLabelColor]];
    inMeterLbl.frame = NSMakeRect(14, 7, 75, 16);
    [topBox.contentView addSubview:inMeterLbl];

    self.inputMeter = [[NSLevelIndicator alloc] initWithFrame:NSMakeRect(92, 7, 380, 16)];
    self.inputMeter.levelIndicatorStyle = NSLevelIndicatorStyleContinuousCapacity;
    self.inputMeter.minValue = 0.0;
    self.inputMeter.maxValue = 1.0;
    self.inputMeter.warningValue = 0.75;
    self.inputMeter.criticalValue = 0.92;
    [topBox.contentView addSubview:self.inputMeter];

    self.setSystemDefaultButton = [[NSButton alloc] initWithFrame:NSMakeRect(495, 16, 185, 28)];
    self.setSystemDefaultButton.bezelStyle = NSBezelStyleRounded;
    self.setSystemDefaultButton.title = @"Set as System Output";
    self.setSystemDefaultButton.target = self;
    self.setSystemDefaultButton.action = @selector(setVirtualDeviceAsSystemOutput:);
    [topBox.contentView addSubview:self.setSystemDefaultButton];

    /* 2. Presets Management Bar */
    NSBox *presetBox = [[NSBox alloc] initWithFrame:NSMakeRect(20, 650, 700, 50)];
    presetBox.title = @"Configuration Presets";
    presetBox.titleFont = [NSFont boldSystemFontOfSize:12];
    [cv addSubview:presetBox];

    NSTextField *presetLbl = [self makeLabel:@"Active Preset:" bold:YES size:12 color:nil];
    presetLbl.frame = NSMakeRect(14, 4, 95, 18);
    [presetBox.contentView addSubview:presetLbl];

    self.presetPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(112, 1, 300, 24) pullsDown:NO];
    self.presetPopup.target = self;
    self.presetPopup.action = @selector(presetPopupChanged:);
    [presetBox.contentView addSubview:self.presetPopup];

    NSButton *savePresetBtn = [[NSButton alloc] initWithFrame:NSMakeRect(422, 1, 130, 24)];
    savePresetBtn.bezelStyle = NSBezelStyleRounded;
    savePresetBtn.title = @"Save Preset...";
    savePresetBtn.target = self;
    savePresetBtn.action = @selector(savePresetClicked:);
    [presetBox.contentView addSubview:savePresetBtn];

    NSButton *deletePresetBtn = [[NSButton alloc] initWithFrame:NSMakeRect(558, 1, 122, 24)];
    deletePresetBtn.bezelStyle = NSBezelStyleRounded;
    deletePresetBtn.title = @"Delete Preset";
    deletePresetBtn.target = self;
    deletePresetBtn.action = @selector(deletePresetClicked:);
    [presetBox.contentView addSubview:deletePresetBtn];

    /* 3. Master Volume & Relative Balance Box */
    NSBox *mixBox = [[NSBox alloc] initWithFrame:NSMakeRect(20, 534, 700, 108)];
    mixBox.title = @"Master Volume & Equal-Power Output Balance (Output A vs Output B)";
    mixBox.titleFont = [NSFont boldSystemFontOfSize:12];
    [cv addSubview:mixBox];

    NSTextField *masterLbl = [self makeLabel:@"Master Volume:" bold:YES size:12 color:nil];
    masterLbl.frame = NSMakeRect(14, 52, 110, 20);
    [mixBox.contentView addSubview:masterLbl];

    self.masterSlider = [NSSlider sliderWithValue:gEngine.masterVolume
                                         minValue:0.0
                                         maxValue:1.0
                                           target:self
                                           action:@selector(masterVolumeChanged:)];
    self.masterSlider.frame = NSMakeRect(130, 52, 410, 22);
    self.masterSlider.continuous = YES;
    [mixBox.contentView addSubview:self.masterSlider];

    self.masterValueLabel = [self makeLabel:@"" bold:NO size:12 color:nil];
    self.masterValueLabel.frame = NSMakeRect(550, 52, 135, 20);
    [mixBox.contentView addSubview:self.masterValueLabel];

    NSTextField *balLbl = [self makeLabel:@"A / B Balance:" bold:YES size:12 color:nil];
    balLbl.frame = NSMakeRect(14, 16, 110, 20);
    [mixBox.contentView addSubview:balLbl];

    self.balanceSlider = [NSSlider sliderWithValue:gEngine.balance
                                          minValue:-1.0
                                          maxValue:1.0
                                            target:self
                                            action:@selector(balanceChanged:)];
    self.balanceSlider.frame = NSMakeRect(130, 16, 330, 22);
    self.balanceSlider.continuous = YES;
    self.balanceSlider.numberOfTickMarks = 3;
    [mixBox.contentView addSubview:self.balanceSlider];

    NSButton *centerBalBtn = [[NSButton alloc] initWithFrame:NSMakeRect(468, 13, 72, 24)];
    centerBalBtn.bezelStyle = NSBezelStyleRounded;
    centerBalBtn.title = @"Center";
    centerBalBtn.target = self;
    centerBalBtn.action = @selector(resetBalanceCenter:);
    [mixBox.contentView addSubview:centerBalBtn];

    self.balanceValueLabel = [self makeLabel:@"" bold:NO size:12 color:nil];
    self.balanceValueLabel.frame = NSMakeRect(550, 16, 135, 20);
    [mixBox.contentView addSubview:self.balanceValueLabel];

    /* 4. Crossover network: linked A low-pass / B high-pass or independent filters. */
    NSBox *crossoverBox = [[NSBox alloc] initWithFrame:NSMakeRect(20, 340, 700, 186)];
    crossoverBox.title = @"Crossover Network — Linkwitz-Riley";
    crossoverBox.titleFont = [NSFont boldSystemFontOfSize:12];
    [cv addSubview:crossoverBox];

    self.crossoverEnabledCheckbox = [NSButton checkboxWithTitle:@"Enable crossover processing"
                                                          target:self
                                                          action:@selector(crossoverEnabledChanged:)];
    self.crossoverEnabledCheckbox.frame = NSMakeRect(14, 132, 168, 22);
    self.crossoverEnabledCheckbox.state = atomic_load(&gEngine.crossover.enabled) ? NSControlStateValueOn : NSControlStateValueOff;
    self.crossoverEnabledCheckbox.toolTip = @"When off, both outputs remain full range.";
    [crossoverBox.contentView addSubview:self.crossoverEnabledCheckbox];

    self.crossoverLinkedCheckbox = [NSButton checkboxWithTitle:@"Link crossover filters"
                                                         target:self
                                                         action:@selector(crossoverLinkedChanged:)];
    self.crossoverLinkedCheckbox.frame = NSMakeRect(190, 132, 150, 22);
    self.crossoverLinkedCheckbox.state = atomic_load(&gEngine.crossover.linked) ? NSControlStateValueOn : NSControlStateValueOff;
    self.crossoverLinkedCheckbox.toolTip = @"Uses one frequency and slope for a phase-aligned A/B crossover pair.";
    [crossoverBox.contentView addSubview:self.crossoverLinkedCheckbox];

    NSTextField *linkedFrequencyTitle = [self makeLabel:@"Shared freq:" bold:YES size:11 color:nil];
    linkedFrequencyTitle.frame = NSMakeRect(14, 101, 78, 18);
    [crossoverBox.contentView addSubview:linkedFrequencyTitle];

    self.linkedCrossoverFrequencySlider = [NSSlider sliderWithValue:CrossoverFrequencyToSliderValue(atomic_load(&gEngine.crossover.linkedFrequency))
                                                              minValue:log10(20.0)
                                                              maxValue:log10(20000.0)
                                                                target:self
                                                                action:@selector(linkedCrossoverFrequencyChanged:)];
    self.linkedCrossoverFrequencySlider.frame = NSMakeRect(96, 97, 275, 22);
    self.linkedCrossoverFrequencySlider.continuous = YES;
    self.linkedCrossoverFrequencySlider.toolTip = @"20 Hz to 20 kHz, logarithmic scale";
    [crossoverBox.contentView addSubview:self.linkedCrossoverFrequencySlider];

    self.linkedCrossoverFrequencyLabel = [self makeLabel:@"" bold:YES size:11 color:nil];
    self.linkedCrossoverFrequencyLabel.frame = NSMakeRect(378, 101, 62, 18);
    [crossoverBox.contentView addSubview:self.linkedCrossoverFrequencyLabel];

    NSTextField *linkedSlopeTitle = [self makeLabel:@"Slope:" bold:YES size:11 color:nil];
    linkedSlopeTitle.frame = NSMakeRect(443, 101, 42, 18);
    [crossoverBox.contentView addSubview:linkedSlopeTitle];

    self.linkedCrossoverSlopePopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(488, 95, 192, 26) pullsDown:NO];
    for (NSInteger slope = 12; slope <= 48; slope += 12) {
        [self.linkedCrossoverSlopePopup addItemWithTitle:[NSString stringWithFormat:@"%ld dB / octave", (long)slope]];
        self.linkedCrossoverSlopePopup.lastItem.tag = slope;
    }
    self.linkedCrossoverSlopePopup.target = self;
    self.linkedCrossoverSlopePopup.action = @selector(linkedCrossoverSlopeChanged:);
    [crossoverBox.contentView addSubview:self.linkedCrossoverSlopePopup];

    NSTextField *linkedAssignmentTitle = [self makeLabel:@"Band assignment:" bold:YES size:11 color:nil];
    linkedAssignmentTitle.frame = NSMakeRect(344, 137, 90, 18);
    [crossoverBox.contentView addSubview:linkedAssignmentTitle];

    self.linkedCrossoverAssignmentPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(436, 128, 244, 26) pullsDown:NO];
    [self.linkedCrossoverAssignmentPopup addItemWithTitle:@"Output A: Low-Pass  •  Output B: High-Pass"];
    self.linkedCrossoverAssignmentPopup.lastItem.tag = 0;
    [self.linkedCrossoverAssignmentPopup addItemWithTitle:@"Output A: High-Pass  •  Output B: Low-Pass"];
    self.linkedCrossoverAssignmentPopup.lastItem.tag = 1;
    self.linkedCrossoverAssignmentPopup.target = self;
    self.linkedCrossoverAssignmentPopup.action = @selector(linkedCrossoverAssignmentChanged:);
    self.linkedCrossoverAssignmentPopup.toolTip = @"Choose which physical output receives the low or high band while filters are linked.";
    [crossoverBox.contentView addSubview:self.linkedCrossoverAssignmentPopup];

    NSTextField *independentLabelA = [self makeLabel:@"Output A:" bold:YES size:11 color:nil];
    independentLabelA.frame = NSMakeRect(14, 102, 58, 18);
    [crossoverBox.contentView addSubview:independentLabelA];
    self.crossoverModePopupA = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(75, 98, 111, 26) pullsDown:NO];
    [self.crossoverModePopupA addItemWithTitle:@"Bypass"]; self.crossoverModePopupA.lastItem.tag = kCrossoverFilterBypass;
    [self.crossoverModePopupA addItemWithTitle:@"Low-Pass"]; self.crossoverModePopupA.lastItem.tag = kCrossoverFilterLowPass;
    [self.crossoverModePopupA addItemWithTitle:@"High-Pass"]; self.crossoverModePopupA.lastItem.tag = kCrossoverFilterHighPass;
    self.crossoverModePopupA.target = self;
    self.crossoverModePopupA.action = @selector(crossoverModeAChanged:);
    [crossoverBox.contentView addSubview:self.crossoverModePopupA];
    NSTextField *independentFrequencyTitleA = [self makeLabel:@"Freq:" bold:NO size:10 color:[NSColor secondaryLabelColor]];
    independentFrequencyTitleA.frame = NSMakeRect(196, 104, 34, 16);
    [crossoverBox.contentView addSubview:independentFrequencyTitleA];
    self.crossoverFrequencySliderA = [NSSlider sliderWithValue:CrossoverFrequencyToSliderValue(atomic_load(&gEngine.crossover.frequencyA))
                                                        minValue:log10(20.0) maxValue:log10(20000.0)
                                                          target:self action:@selector(crossoverFrequencyAChanged:)];
    self.crossoverFrequencySliderA.frame = NSMakeRect(232, 100, 160, 22);
    self.crossoverFrequencySliderA.continuous = YES;
    [crossoverBox.contentView addSubview:self.crossoverFrequencySliderA];
    self.crossoverFrequencyLabelA = [self makeLabel:@"" bold:YES size:10 color:nil];
    self.crossoverFrequencyLabelA.frame = NSMakeRect(398, 104, 68, 16);
    [crossoverBox.contentView addSubview:self.crossoverFrequencyLabelA];
    NSTextField *independentSlopeTitleA = [self makeLabel:@"Slope:" bold:NO size:10 color:[NSColor secondaryLabelColor]];
    independentSlopeTitleA.frame = NSMakeRect(470, 104, 42, 16);
    [crossoverBox.contentView addSubview:independentSlopeTitleA];
    self.crossoverSlopePopupA = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(514, 98, 166, 26) pullsDown:NO];
    for (NSInteger slope = 12; slope <= 48; slope += 12) {
        [self.crossoverSlopePopupA addItemWithTitle:[NSString stringWithFormat:@"%ld dB / octave", (long)slope]];
        self.crossoverSlopePopupA.lastItem.tag = slope;
    }
    self.crossoverSlopePopupA.target = self;
    self.crossoverSlopePopupA.action = @selector(crossoverSlopeAChanged:);
    [crossoverBox.contentView addSubview:self.crossoverSlopePopupA];

    NSTextField *independentLabelB = [self makeLabel:@"Output B:" bold:YES size:11 color:nil];
    independentLabelB.frame = NSMakeRect(14, 74, 58, 18);
    [crossoverBox.contentView addSubview:independentLabelB];
    self.crossoverModePopupB = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(75, 68, 111, 26) pullsDown:NO];
    [self.crossoverModePopupB addItemWithTitle:@"Bypass"]; self.crossoverModePopupB.lastItem.tag = kCrossoverFilterBypass;
    [self.crossoverModePopupB addItemWithTitle:@"Low-Pass"]; self.crossoverModePopupB.lastItem.tag = kCrossoverFilterLowPass;
    [self.crossoverModePopupB addItemWithTitle:@"High-Pass"]; self.crossoverModePopupB.lastItem.tag = kCrossoverFilterHighPass;
    self.crossoverModePopupB.target = self;
    self.crossoverModePopupB.action = @selector(crossoverModeBChanged:);
    [crossoverBox.contentView addSubview:self.crossoverModePopupB];
    NSTextField *independentFrequencyTitleB = [self makeLabel:@"Freq:" bold:NO size:10 color:[NSColor secondaryLabelColor]];
    independentFrequencyTitleB.frame = NSMakeRect(196, 76, 34, 16);
    [crossoverBox.contentView addSubview:independentFrequencyTitleB];
    self.crossoverFrequencySliderB = [NSSlider sliderWithValue:CrossoverFrequencyToSliderValue(atomic_load(&gEngine.crossover.frequencyB))
                                                        minValue:log10(20.0) maxValue:log10(20000.0)
                                                          target:self action:@selector(crossoverFrequencyBChanged:)];
    self.crossoverFrequencySliderB.frame = NSMakeRect(232, 70, 160, 22);
    self.crossoverFrequencySliderB.continuous = YES;
    [crossoverBox.contentView addSubview:self.crossoverFrequencySliderB];
    self.crossoverFrequencyLabelB = [self makeLabel:@"" bold:YES size:10 color:nil];
    self.crossoverFrequencyLabelB.frame = NSMakeRect(398, 76, 68, 16);
    [crossoverBox.contentView addSubview:self.crossoverFrequencyLabelB];
    NSTextField *independentSlopeTitleB = [self makeLabel:@"Slope:" bold:NO size:10 color:[NSColor secondaryLabelColor]];
    independentSlopeTitleB.frame = NSMakeRect(470, 76, 42, 16);
    [crossoverBox.contentView addSubview:independentSlopeTitleB];
    self.crossoverSlopePopupB = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(514, 68, 166, 26) pullsDown:NO];
    for (NSInteger slope = 12; slope <= 48; slope += 12) {
        [self.crossoverSlopePopupB addItemWithTitle:[NSString stringWithFormat:@"%ld dB / octave", (long)slope]];
        self.crossoverSlopePopupB.lastItem.tag = slope;
    }
    self.crossoverSlopePopupB.target = self;
    self.crossoverSlopePopupB.action = @selector(crossoverSlopeBChanged:);
    [crossoverBox.contentView addSubview:self.crossoverSlopePopupB];

    self.linkedCrossoverControls = @[linkedFrequencyTitle, self.linkedCrossoverFrequencySlider,
                                     self.linkedCrossoverFrequencyLabel, linkedSlopeTitle,
                                     self.linkedCrossoverSlopePopup, linkedAssignmentTitle,
                                     self.linkedCrossoverAssignmentPopup];
    self.independentCrossoverControls = @[independentLabelA, self.crossoverModePopupA,
                                           independentFrequencyTitleA, self.crossoverFrequencySliderA,
                                           self.crossoverFrequencyLabelA, independentSlopeTitleA,
                                           self.crossoverSlopePopupA, independentLabelB, self.crossoverModePopupB,
                                           independentFrequencyTitleB, self.crossoverFrequencySliderB,
                                           self.crossoverFrequencyLabelB, independentSlopeTitleB,
                                           self.crossoverSlopePopupB];

    self.crossoverGraphFrequencyLabel = [self makeLabel:@"" bold:YES size:10 color:[NSColor systemPurpleColor]];
    self.crossoverGraphFrequencyLabel.alignment = NSTextAlignmentCenter;
    self.crossoverGraphFrequencyLabel.frame = NSMakeRect(44, 66, 612, 14);
    [crossoverBox.contentView addSubview:self.crossoverGraphFrequencyLabel];

    self.crossoverResponseView = [[CrossoverResponseView alloc] initWithFrame:NSMakeRect(14, 4, 672, 60)];
    self.crossoverResponseView.crossoverDelegate = self;
    self.crossoverResponseView.toolTip = @"Response graph. Drag a marker to set its crossover frequency.";
    [crossoverBox.contentView addSubview:self.crossoverResponseView];
    [self updateCrossoverControls];

    /* 5. Output A Box (Left Column) */
    NSBox *boxA = [[NSBox alloc] initWithFrame:NSMakeRect(20, 50, 342, 280)];
    boxA.title = @"Output A Configuration";
    boxA.titleFont = [NSFont boldSystemFontOfSize:12];
    [cv addSubview:boxA];

    self.enableCheckboxA = [NSButton checkboxWithTitle:@"Enable Output A"
                                                target:self
                                                action:@selector(toggleEnableA:)];
    self.enableCheckboxA.state = atomic_load(&gEngine.pathA.enabled) ? NSControlStateValueOn : NSControlStateValueOff;
    self.enableCheckboxA.frame = NSMakeRect(14, 224, 145, 22);
    [boxA.contentView addSubview:self.enableCheckboxA];

    self.monoCheckboxA = [NSButton checkboxWithTitle:@"Mono (Sum L+R)"
                                              target:self
                                              action:@selector(toggleMonoA:)];
    self.monoCheckboxA.state = atomic_load(&gEngine.pathA.mono) ? NSControlStateValueOn : NSControlStateValueOff;
    self.monoCheckboxA.frame = NSMakeRect(175, 224, 145, 22);
    [boxA.contentView addSubview:self.monoCheckboxA];

    NSTextField *devLblA = [self makeLabel:@"Destination Device:" bold:NO size:11 color:[NSColor secondaryLabelColor]];
    devLblA.frame = NSMakeRect(14, 200, 200, 16);
    [boxA.contentView addSubview:devLblA];

    self.popupA = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(14, 174, 310, 26) pullsDown:NO];
    self.popupA.target = self;
    self.popupA.action = @selector(outputADeviceSelected:);
    [boxA.contentView addSubview:self.popupA];

    self.statusLabelA = [self makeLabel:@"Status: Idle" bold:NO size:11 color:[NSColor secondaryLabelColor]];
    self.statusLabelA.frame = NSMakeRect(14, 154, 184, 16);
    [boxA.contentView addSubview:self.statusLabelA];

    self.routingMatrixA = [[RoutingMatrixView alloc] initWithFrame:NSMakeRect(14, 108, 310, 40)];
    self.routingMatrixA.outputA = YES;
    self.routingMatrixA.routingDelegate = self;
    self.routingMatrixA.toolTip = @"Click a cell to route Left or Right input to that destination output channel.";
    [boxA.contentView addSubview:self.routingMatrixA];

    NSTextField *trimLblA = [self makeLabel:@"Output A Gain Trim (-24 to +6 dB):" bold:NO size:11 color:nil];
    trimLblA.frame = NSMakeRect(14, 82, 210, 16);
    [boxA.contentView addSubview:trimLblA];

    self.trimValueLabelA = [self makeLabel:@"0.0 dB" bold:YES size:11 color:nil];
    self.trimValueLabelA.frame = NSMakeRect(230, 82, 90, 16);
    [boxA.contentView addSubview:self.trimValueLabelA];

    self.trimSliderA = [NSSlider sliderWithValue:gEngine.trimDbA
                                        minValue:-24.0
                                        maxValue:6.0
                                          target:self
                                          action:@selector(trimAChanged:)];
    self.trimSliderA.frame = NSMakeRect(14, 58, 240, 22);
    self.trimSliderA.continuous = YES;
    [boxA.contentView addSubview:self.trimSliderA];

    NSButton *unityBtnA = [[NSButton alloc] initWithFrame:NSMakeRect(260, 56, 64, 24)];
    unityBtnA.bezelStyle = NSBezelStyleRounded;
    unityBtnA.title = @"0 dB";
    unityBtnA.target = self;
    unityBtnA.action = @selector(resetTrimA:);
    [boxA.contentView addSubview:unityBtnA];

    NSTextField *delayLblA = [self makeLabel:@"Time Delay (0 - 500 ms):" bold:NO size:11 color:nil];
    delayLblA.frame = NSMakeRect(14, 32, 175, 16);
    [boxA.contentView addSubview:delayLblA];

    self.delayValueLabelA = [self makeLabel:@"0.0 ms" bold:YES size:11 color:nil];
    self.delayValueLabelA.frame = NSMakeRect(200, 32, 110, 16);
    [boxA.contentView addSubview:self.delayValueLabelA];

    double initDelayA = atomic_load(&gEngine.pathA.targetDelayMs);
    self.delaySliderA = [NSSlider sliderWithValue:initDelayA
                                         minValue:0.0
                                         maxValue:500.0
                                           target:self
                                           action:@selector(delayASliderChanged:)];
    self.delaySliderA.frame = NSMakeRect(14, 10, 275, 22);
    self.delaySliderA.continuous = YES;
    [boxA.contentView addSubview:self.delaySliderA];

    self.delayStepperA = [[NSStepper alloc] initWithFrame:NSMakeRect(296, 8, 28, 26)];
    self.delayStepperA.minValue = 0.0;
    self.delayStepperA.maxValue = 500.0;
    self.delayStepperA.increment = 0.5;
    self.delayStepperA.valueWraps = NO;
    self.delayStepperA.doubleValue = initDelayA;
    self.delayStepperA.target = self;
    self.delayStepperA.action = @selector(delayAStepperChanged:);
    [boxA.contentView addSubview:self.delayStepperA];

    NSTextField *meterLblA = [self makeLabel:@"Level:" bold:NO size:11 color:[NSColor secondaryLabelColor]];
    meterLblA.frame = NSMakeRect(202, 154, 38, 16);
    [boxA.contentView addSubview:meterLblA];

    self.meterA = [[NSLevelIndicator alloc] initWithFrame:NSMakeRect(240, 154, 84, 16)];
    self.meterA.levelIndicatorStyle = NSLevelIndicatorStyleContinuousCapacity;
    self.meterA.minValue = 0.0;
    self.meterA.maxValue = 1.0;
    self.meterA.warningValue = 0.75;
    self.meterA.criticalValue = 0.92;
    [boxA.contentView addSubview:self.meterA];

    /* 5. Output B Box (Right Column) */
    NSBox *boxB = [[NSBox alloc] initWithFrame:NSMakeRect(378, 50, 342, 280)];
    boxB.title = @"Output B Configuration";
    boxB.titleFont = [NSFont boldSystemFontOfSize:12];
    [cv addSubview:boxB];

    self.enableCheckboxB = [NSButton checkboxWithTitle:@"Enable Output B"
                                                target:self
                                                action:@selector(toggleEnableB:)];
    self.enableCheckboxB.state = atomic_load(&gEngine.pathB.enabled) ? NSControlStateValueOn : NSControlStateValueOff;
    self.enableCheckboxB.frame = NSMakeRect(14, 224, 145, 22);
    [boxB.contentView addSubview:self.enableCheckboxB];

    self.monoCheckboxB = [NSButton checkboxWithTitle:@"Mono (Sum L+R)"
                                              target:self
                                              action:@selector(toggleMonoB:)];
    self.monoCheckboxB.state = atomic_load(&gEngine.pathB.mono) ? NSControlStateValueOn : NSControlStateValueOff;
    self.monoCheckboxB.frame = NSMakeRect(175, 224, 145, 22);
    [boxB.contentView addSubview:self.monoCheckboxB];

    NSTextField *devLblB = [self makeLabel:@"Destination Device:" bold:NO size:11 color:[NSColor secondaryLabelColor]];
    devLblB.frame = NSMakeRect(14, 200, 200, 16);
    [boxB.contentView addSubview:devLblB];

    self.popupB = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(14, 174, 310, 26) pullsDown:NO];
    self.popupB.target = self;
    self.popupB.action = @selector(outputBDeviceSelected:);
    [boxB.contentView addSubview:self.popupB];

    self.statusLabelB = [self makeLabel:@"Status: Idle" bold:NO size:11 color:[NSColor secondaryLabelColor]];
    self.statusLabelB.frame = NSMakeRect(14, 154, 184, 16);
    [boxB.contentView addSubview:self.statusLabelB];

    self.routingMatrixB = [[RoutingMatrixView alloc] initWithFrame:NSMakeRect(14, 108, 310, 40)];
    self.routingMatrixB.outputA = NO;
    self.routingMatrixB.routingDelegate = self;
    self.routingMatrixB.toolTip = @"Click a cell to route Left or Right input to that destination output channel.";
    [boxB.contentView addSubview:self.routingMatrixB];

    NSTextField *trimLblB = [self makeLabel:@"Output B Gain Trim (-24 to +6 dB):" bold:NO size:11 color:nil];
    trimLblB.frame = NSMakeRect(14, 82, 210, 16);
    [boxB.contentView addSubview:trimLblB];

    self.trimValueLabelB = [self makeLabel:@"0.0 dB" bold:YES size:11 color:nil];
    self.trimValueLabelB.frame = NSMakeRect(230, 82, 90, 16);
    [boxB.contentView addSubview:self.trimValueLabelB];

    self.trimSliderB = [NSSlider sliderWithValue:gEngine.trimDbB
                                        minValue:-24.0
                                        maxValue:6.0
                                          target:self
                                          action:@selector(trimBChanged:)];
    self.trimSliderB.frame = NSMakeRect(14, 58, 240, 22);
    self.trimSliderB.continuous = YES;
    [boxB.contentView addSubview:self.trimSliderB];

    NSButton *unityBtnB = [[NSButton alloc] initWithFrame:NSMakeRect(260, 56, 64, 24)];
    unityBtnB.bezelStyle = NSBezelStyleRounded;
    unityBtnB.title = @"0 dB";
    unityBtnB.target = self;
    unityBtnB.action = @selector(resetTrimB:);
    [boxB.contentView addSubview:unityBtnB];

    NSTextField *delayLblB = [self makeLabel:@"Time Delay (0 - 500 ms):" bold:NO size:11 color:nil];
    delayLblB.frame = NSMakeRect(14, 32, 175, 16);
    [boxB.contentView addSubview:delayLblB];

    self.delayValueLabelB = [self makeLabel:@"0.0 ms" bold:YES size:11 color:nil];
    self.delayValueLabelB.frame = NSMakeRect(200, 32, 110, 16);
    [boxB.contentView addSubview:self.delayValueLabelB];

    double initDelayB = atomic_load(&gEngine.pathB.targetDelayMs);
    self.delaySliderB = [NSSlider sliderWithValue:initDelayB
                                         minValue:0.0
                                         maxValue:500.0
                                           target:self
                                           action:@selector(delayBSliderChanged:)];
    self.delaySliderB.frame = NSMakeRect(14, 10, 275, 22);
    self.delaySliderB.continuous = YES;
    [boxB.contentView addSubview:self.delaySliderB];

    self.delayStepperB = [[NSStepper alloc] initWithFrame:NSMakeRect(296, 8, 28, 26)];
    self.delayStepperB.minValue = 0.0;
    self.delayStepperB.maxValue = 500.0;
    self.delayStepperB.increment = 0.5;
    self.delayStepperB.valueWraps = NO;
    self.delayStepperB.doubleValue = initDelayB;
    self.delayStepperB.target = self;
    self.delayStepperB.action = @selector(delayBStepperChanged:);
    [boxB.contentView addSubview:self.delayStepperB];

    NSTextField *meterLblB = [self makeLabel:@"Level:" bold:NO size:11 color:[NSColor secondaryLabelColor]];
    meterLblB.frame = NSMakeRect(202, 154, 38, 16);
    [boxB.contentView addSubview:meterLblB];

    self.meterB = [[NSLevelIndicator alloc] initWithFrame:NSMakeRect(240, 154, 84, 16)];
    self.meterB.levelIndicatorStyle = NSLevelIndicatorStyleContinuousCapacity;
    self.meterB.minValue = 0.0;
    self.meterB.maxValue = 1.0;
    self.meterB.warningValue = 0.75;
    self.meterB.criticalValue = 0.92;
    [boxB.contentView addSubview:self.meterB];

    /* 6. Bottom Status & Info Bar */
    self.errorBannerLabel = [self makeLabel:@"Tip: Press Cmd+W to hide this window to the Menu Bar, or Cmd+Q to quit Dual Output completely." bold:NO size:11 color:[NSColor secondaryLabelColor]];
    self.errorBannerLabel.frame = NSMakeRect(24, 14, 692, 18);
    [cv addSubview:self.errorBannerLabel];

    [self updateValueLabels];
    /* AppKit's default origin is at the bottom of an unflipped document view;
     * open at the status section at the top instead. */
    [self.window layoutIfNeeded];
    NSRect clipBounds = self.controllerScrollView.contentView.bounds;
    CGFloat topOrigin = fmax(0.0, NSHeight(cv.bounds) - NSHeight(clipBounds));
    [self.controllerScrollView.contentView scrollToPoint:NSMakePoint(0.0, topOrigin)];
    [self.controllerScrollView reflectScrolledClipView:self.controllerScrollView.contentView];
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

/*===========================================================================
 * Audio Lifecycle & Hot-Plug Listeners
 *===========================================================================*/

- (void)requestMicrophonePermissionAndStartAudio {
    AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
    if (status == AVAuthorizationStatusAuthorized) {
        [self refreshDevicesAndStreams];
    } else if (status == AVAuthorizationStatusNotDetermined) {
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
            dispatch_async(dispatch_get_main_queue(), ^{
                if (granted) {
                    [self refreshDevicesAndStreams];
                } else {
                    self.errorBannerLabel.textColor = [NSColor systemRedColor];
                    self.errorBannerLabel.stringValue = @"Microphone access denied. Enable Dual Output in System Preferences → Security & Privacy → Privacy → Microphone.";
                }
            });
        }];
    } else {
        self.errorBannerLabel.textColor = [NSColor systemRedColor];
        self.errorBannerLabel.stringValue = @"Microphone access is required to read from Dual Output. Grant access in System Preferences → Security & Privacy → Microphone.";
        [self refreshDevicesAndStreams];
    }
}

- (void)registerCoreAudioListeners {
    AudioObjectPropertyAddress devListAddr = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectAddPropertyListenerBlock(kAudioObjectSystemObject, &devListAddr, dispatch_get_main_queue(), ^(UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses) {
        (void)inNumberAddresses; (void)inAddresses;
        [self refreshDevicesAndStreams];
    });
}

- (void)attachSampleRateListenerForDevice:(AudioDeviceID)devID {
    if (devID == kAudioObjectUnknown) return;
    AudioObjectPropertyAddress rateAddr = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectAddPropertyListenerBlock(devID, &rateAddr, dispatch_get_main_queue(), ^(UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses) {
        (void)inNumberAddresses; (void)inAddresses;
        [self refreshDevicesAndStreams];
    });
}

- (void)refreshDevicesAndStreams {
    AudioDeviceID vDev = FindVirtualDeviceID();
    if (vDev == kAudioObjectUnknown) {
        StopVirtualInputUnit();
        self.virtualStatusLabel.textColor = [NSColor systemRedColor];
        self.virtualStatusLabel.stringValue = @"Dual Output driver NOT installed in /Library/Audio/Plug-Ins/HAL/";
        self.statusMenuHeaderItem.title = @"Dual Output: Driver Not Installed";
    } else {
        double vRate = GetDeviceNominalSampleRate(vDev);
        if (!atomic_load(&gEngine.inputRunning) || gEngine.virtualDeviceID != vDev ||
            fabs(vRate - atomic_load(&gEngine.virtualSampleRate)) > 1.0) {
            NSString *errStr = nil;
            if (StartVirtualInputUnit(vDev, &errStr) == noErr) {
                [self attachSampleRateListenerForDevice:vDev];
                self.virtualStatusLabel.textColor = [NSColor systemGreenColor];
                self.virtualStatusLabel.stringValue = [NSString stringWithFormat:@"Dual Output Active — %.0f Hz Stereo Float32", vRate];
                self.statusMenuHeaderItem.title = [NSString stringWithFormat:@"Dual Output: Active (%.0f Hz)", vRate];
            } else {
                self.virtualStatusLabel.textColor = [NSColor systemRedColor];
                self.virtualStatusLabel.stringValue = errStr ?: @"Failed starting Dual Output capture";
                self.statusMenuHeaderItem.title = @"Dual Output: Stream Error";
            }
        }
    }

    self.availableDevices = EnumeratePhysicalOutputDevices();

    [self.popupA removeAllItems];
    [self.popupB removeAllItems];
    [self.popupA addItemWithTitle:@"— None (Disabled) —"];
    [self.popupB addItemWithTitle:@"— None (Disabled) —"];

    for (AudioDeviceItem *item in self.availableDevices) {
        NSString *title = [NSString stringWithFormat:@"%@ (%.0f Hz)", item.name, item.sampleRate];
        [self.popupA addItemWithTitle:title];
        [self.popupB addItemWithTitle:title];
    }

    if (self.desiredUID_A == nil && self.availableDevices.count >= 1) {
        self.desiredUID_A = self.availableDevices[0].uid;
    }
    if (self.desiredUID_B == nil && self.availableDevices.count >= 2) {
        self.desiredUID_B = self.availableDevices[1].uid;
    }

    [self reconcileOutputPath:&gEngine.pathA
                        popup:self.popupA
                   desiredUID:self.desiredUID_A
                  statusLabel:self.statusLabelA
                        label:@"Output A"];

    [self reconcileOutputPath:&gEngine.pathB
                        popup:self.popupB
                   desiredUID:self.desiredUID_B
                  statusLabel:self.statusLabelB
                        label:@"Output B"];
    [self updateRoutingMatrixViews];
}

- (void)reconcileOutputPath:(OutputPathState *)path
                      popup:(NSPopUpButton *)popup
                 desiredUID:(NSString *)desiredUID
                statusLabel:(NSTextField *)statusLabel
                      label:(NSString *)label {
    if (desiredUID == nil || desiredUID.length == 0) {
        [popup selectItemAtIndex:0];
        StopOutputPath(path);
        statusLabel.textColor = [NSColor secondaryLabelColor];
        statusLabel.stringValue = [NSString stringWithFormat:@"%@: No device selected", label];
        return;
    }

    NSInteger foundIdx = -1;
    AudioDeviceItem *foundItem = nil;
    for (NSUInteger i = 0; i < self.availableDevices.count; ++i) {
        if ([self.availableDevices[i].uid isEqualToString:desiredUID]) {
            foundIdx = (NSInteger)i;
            foundItem = self.availableDevices[i];
            break;
        }
    }

    if (foundItem == nil) {
        [popup selectItemAtIndex:0];
        StopOutputPath(path);
        statusLabel.textColor = [NSColor systemOrangeColor];
        statusLabel.stringValue = @"Device disconnected — waiting for reconnect...";
        return;
    }

    [popup selectItemAtIndex:foundIdx + 1];

    double currentOutRate = atomic_load(&path->outputSampleRate);
    if (!atomic_load(&path->streamRunning) || path->deviceID != foundItem.deviceID ||
        fabs(currentOutRate - foundItem.sampleRate) > 1.0) {
        NSString *errStr = nil;
        if (StartOutputPath(path, foundItem.deviceID, &errStr) == noErr) {
            [self attachSampleRateListenerForDevice:foundItem.deviceID];
            self.errorBannerLabel.textColor = [NSColor secondaryLabelColor];
            self.errorBannerLabel.stringValue = [NSString stringWithFormat:@"%@ connected to %@ (%.0f Hz).", label, foundItem.name, foundItem.sampleRate];
        } else {
            statusLabel.textColor = [NSColor systemRedColor];
            statusLabel.stringValue = errStr ?: @"Error starting stream";
            self.errorBannerLabel.textColor = [NSColor systemRedColor];
            self.errorBannerLabel.stringValue = errStr ?: @"Error starting output stream";
        }
    }
}

/*===========================================================================
 * Preset UI Actions
 *===========================================================================*/

- (void)presetPopupChanged:(id)sender {
    (void)sender;
    NSInteger idx = self.presetPopup.indexOfSelectedItem;
    if (idx <= 0) return;
    NSString *name = self.presetPopup.titleOfSelectedItem;
    if (name.length > 0) {
        [self applyPresetNamed:name];
    }
}

- (void)menuPresetClicked:(NSMenuItem *)sender {
    NSString *name = sender.representedObject;
    if (name.length > 0) {
        [self applyPresetNamed:name];
    }
}

- (void)savePresetClicked:(id)sender {
    (void)sender;
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Save Dual Output Preset";
    alert.informativeText = @"Enter a name for this preset (saves devices, routing matrices, enable/mono toggles, volume, balance, trim, and delays):";
    [alert addButtonWithTitle:@"Save"];
    [alert addButtonWithTitle:@"Cancel"];

    NSTextField *inputField = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 260, 24)];
    inputField.stringValue = (self.activePresetName.length > 0) ? self.activePresetName : @"Desk + Studio Monitors";
    alert.accessoryView = inputField;

    [self showControllerWindow:nil];
    NSModalResponse resp = [alert runModal];
    if (resp == NSAlertFirstButtonReturn) {
        NSString *name = [inputField.stringValue stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        if (name.length == 0) {
            self.errorBannerLabel.textColor = [NSColor systemRedColor];
            self.errorBannerLabel.stringValue = @"Preset name cannot be empty.";
            return;
        }
        self.savedPresets[name] = [self captureCurrentStateAsPresetDictionary];
        self.activePresetName = name;
        [self rebuildPresetsUI];
        [self savePreferences];
        self.errorBannerLabel.textColor = [NSColor systemGreenColor];
        self.errorBannerLabel.stringValue = [NSString stringWithFormat:@"Saved preset \"%@\".", name];
    }
}

- (void)deletePresetClicked:(id)sender {
    (void)sender;
    if (self.activePresetName.length == 0 || self.savedPresets[self.activePresetName] == nil) {
        self.errorBannerLabel.textColor = [NSColor systemOrangeColor];
        self.errorBannerLabel.stringValue = @"Select a saved preset first to delete it.";
        return;
    }
    NSString *deletedName = self.activePresetName;
    [self.savedPresets removeObjectForKey:deletedName];
    self.activePresetName = @"";
    [self rebuildPresetsUI];
    [self savePreferences];
    self.errorBannerLabel.textColor = [NSColor secondaryLabelColor];
    self.errorBannerLabel.stringValue = [NSString stringWithFormat:@"Deleted preset \"%@\".", deletedName];
}

/*===========================================================================
 * Audio Control Actions
 *===========================================================================*/

- (void)setVirtualDeviceAsSystemOutput:(id)sender {
    (void)sender;
    AudioDeviceID vDev = FindVirtualDeviceID();
    if (vDev == kAudioObjectUnknown) {
        self.errorBannerLabel.textColor = [NSColor systemRedColor];
        self.errorBannerLabel.stringValue = @"Dual Output driver is not installed. Run ./install.sh first.";
        return;
    }
    AudioObjectPropertyAddress defaultOutAddr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    OSStatus err = AudioObjectSetPropertyData(kAudioObjectSystemObject, &defaultOutAddr, 0, NULL, sizeof(AudioDeviceID), &vDev);
    if (err == noErr) {
        self.errorBannerLabel.textColor = [NSColor systemGreenColor];
        self.errorBannerLabel.stringValue = @"Dual Output is now the active macOS System Output device.";
    } else {
        self.errorBannerLabel.textColor = [NSColor systemRedColor];
        self.errorBannerLabel.stringValue = [NSString stringWithFormat:@"Could not set default output device (%d).", (int)err];
    }
}

- (void)outputADeviceSelected:(id)sender {
    (void)sender;
    NSInteger idx = self.popupA.indexOfSelectedItem;
    if (idx <= 0 || (idx - 1) >= (NSInteger)self.availableDevices.count) {
        self.desiredUID_A = @"";
    } else {
        self.desiredUID_A = self.availableDevices[idx - 1].uid;
    }
    [self savePreferences];
    [self reconcileOutputPath:&gEngine.pathA
                        popup:self.popupA
                   desiredUID:self.desiredUID_A
                  statusLabel:self.statusLabelA
                        label:@"Output A"];
    [self updateRoutingMatrixViews];
}

- (void)outputBDeviceSelected:(id)sender {
    (void)sender;
    NSInteger idx = self.popupB.indexOfSelectedItem;
    if (idx <= 0 || (idx - 1) >= (NSInteger)self.availableDevices.count) {
        self.desiredUID_B = @"";
    } else {
        self.desiredUID_B = self.availableDevices[idx - 1].uid;
    }
    [self savePreferences];
    [self reconcileOutputPath:&gEngine.pathB
                        popup:self.popupB
                   desiredUID:self.desiredUID_B
                  statusLabel:self.statusLabelB
                        label:@"Output B"];
    [self updateRoutingMatrixViews];
}

- (void)toggleEnableA:(id)sender {
    (void)sender;
    bool en = (self.enableCheckboxA.state == NSControlStateValueOn);
    atomic_store_explicit(&gEngine.pathA.enabled, en, memory_order_relaxed);
    [self syncStatusBarMenuChecks];
    [self savePreferences];
}

- (void)menuToggleEnableA:(id)sender {
    (void)sender;
    bool en = !atomic_load_explicit(&gEngine.pathA.enabled, memory_order_relaxed);
    atomic_store_explicit(&gEngine.pathA.enabled, en, memory_order_relaxed);
    self.enableCheckboxA.state = en ? NSControlStateValueOn : NSControlStateValueOff;
    [self syncStatusBarMenuChecks];
    [self savePreferences];
}

- (void)toggleMonoA:(id)sender {
    (void)sender;
    bool m = (self.monoCheckboxA.state == NSControlStateValueOn);
    atomic_store_explicit(&gEngine.pathA.mono, m, memory_order_relaxed);
    [self updateRoutingMatrixViews];
    [self syncStatusBarMenuChecks];
    [self savePreferences];
}

- (void)menuToggleMonoA:(id)sender {
    (void)sender;
    bool m = !atomic_load_explicit(&gEngine.pathA.mono, memory_order_relaxed);
    atomic_store_explicit(&gEngine.pathA.mono, m, memory_order_relaxed);
    self.monoCheckboxA.state = m ? NSControlStateValueOn : NSControlStateValueOff;
    [self updateRoutingMatrixViews];
    [self syncStatusBarMenuChecks];
    [self savePreferences];
}

- (void)toggleEnableB:(id)sender {
    (void)sender;
    bool en = (self.enableCheckboxB.state == NSControlStateValueOn);
    atomic_store_explicit(&gEngine.pathB.enabled, en, memory_order_relaxed);
    [self syncStatusBarMenuChecks];
    [self savePreferences];
}

- (void)menuToggleEnableB:(id)sender {
    (void)sender;
    bool en = !atomic_load_explicit(&gEngine.pathB.enabled, memory_order_relaxed);
    atomic_store_explicit(&gEngine.pathB.enabled, en, memory_order_relaxed);
    self.enableCheckboxB.state = en ? NSControlStateValueOn : NSControlStateValueOff;
    [self syncStatusBarMenuChecks];
    [self savePreferences];
}

- (void)toggleMonoB:(id)sender {
    (void)sender;
    bool m = (self.monoCheckboxB.state == NSControlStateValueOn);
    atomic_store_explicit(&gEngine.pathB.mono, m, memory_order_relaxed);
    [self updateRoutingMatrixViews];
    [self syncStatusBarMenuChecks];
    [self savePreferences];
}

- (void)menuToggleMonoB:(id)sender {
    (void)sender;
    bool m = !atomic_load_explicit(&gEngine.pathB.mono, memory_order_relaxed);
    atomic_store_explicit(&gEngine.pathB.mono, m, memory_order_relaxed);
    self.monoCheckboxB.state = m ? NSControlStateValueOn : NSControlStateValueOff;
    [self updateRoutingMatrixViews];
    [self syncStatusBarMenuChecks];
    [self savePreferences];
}

/*===========================================================================
 * QLab-Style Input / Output Routing Matrices
 *===========================================================================*/

- (void)updateRoutingMatrixViews {
    [self.routingMatrixA setNeedsDisplay:YES];
    [self.routingMatrixB setNeedsDisplay:YES];
}

- (NSArray<NSNumber *> *)routeGainDbValuesForPath:(OutputPathState *)path {
    NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:kMaxOutputChannels * 2U];
    for (uint32_t channel = 0; channel < kMaxOutputChannels; ++channel) {
        [values addObject:@(RouteGainLinearToDb(atomic_load_explicit(&path->routeGainLeft[channel], memory_order_relaxed)))];
    }
    for (uint32_t channel = 0; channel < kMaxOutputChannels; ++channel) {
        [values addObject:@(RouteGainLinearToDb(atomic_load_explicit(&path->routeGainRight[channel], memory_order_relaxed)))];
    }
    return values;
}

- (void)applyRouteGainDbValues:(NSArray<NSNumber *> *)values toPath:(OutputPathState *)path {
    if (![values isKindOfClass:[NSArray class]]) return;
    for (uint32_t channel = 0; channel < kMaxOutputChannels; ++channel) {
        if (channel < values.count) {
            atomic_store_explicit(&path->routeGainLeft[channel], RouteGainDbToLinear(values[channel].doubleValue), memory_order_relaxed);
        }
        NSUInteger rightIndex = kMaxOutputChannels + channel;
        if (rightIndex < values.count) {
            atomic_store_explicit(&path->routeGainRight[channel], RouteGainDbToLinear(values[rightIndex].doubleValue), memory_order_relaxed);
        }
    }
}

- (NSArray<NSNumber *> *)monoRouteGainDbValuesForPath:(OutputPathState *)path {
    NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:kMaxOutputChannels];
    for (uint32_t channel = 0; channel < kMaxOutputChannels; ++channel) {
        [values addObject:@(RouteGainLinearToDb(atomic_load_explicit(&path->monoRouteGain[channel], memory_order_relaxed)))];
    }
    return values;
}

- (void)applyMonoRouteGainDbValues:(NSArray<NSNumber *> *)values toPath:(OutputPathState *)path {
    if (![values isKindOfClass:[NSArray class]]) return;
    for (uint32_t channel = 0; channel < kMaxOutputChannels && channel < values.count; ++channel) {
        atomic_store_explicit(&path->monoRouteGain[channel], RouteGainDbToLinear(values[channel].doubleValue), memory_order_relaxed);
    }
}

- (void)ensureRoutingGainPopover {
    if (self.routingGainPopover != nil) return;

    self.routingGainPopover = [[NSPopover alloc] init];
    self.routingGainPopover.behavior = NSPopoverBehaviorTransient;
    NSViewController *controller = [[NSViewController alloc] init];
    NSView *content = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 190, 146)];
    controller.view = content;
    self.routingGainPopover.contentViewController = controller;

    NSTextField *title = [self makeLabel:@"Route Gain" bold:YES size:12 color:nil];
    title.frame = NSMakeRect(62, 112, 110, 20);
    [content addSubview:title];

    self.routingGainFader = [[NSSlider alloc] initWithFrame:NSMakeRect(20, 18, 28, 108)];
    self.routingGainFader.vertical = YES;
    self.routingGainFader.minValue = -60.0;
    self.routingGainFader.maxValue = 12.0;
    self.routingGainFader.continuous = YES;
    self.routingGainFader.target = self;
    self.routingGainFader.action = @selector(routingGainFaderChanged:);
    self.routingGainFader.toolTip = @"Route gain, from −60 dB to +12 dB";
    [content addSubview:self.routingGainFader];

    self.routingGainValueField = [[NSTextField alloc] initWithFrame:NSMakeRect(62, 80, 106, 25)];
    self.routingGainValueField.alignment = NSTextAlignmentRight;
    self.routingGainValueField.target = self;
    self.routingGainValueField.action = @selector(routingGainValueChanged:);
    [content addSubview:self.routingGainValueField];

    NSTextField *dbLabel = [self makeLabel:@"dB" bold:NO size:11 color:[NSColor secondaryLabelColor]];
    dbLabel.frame = NSMakeRect(172, 84, 18, 16);
    [content addSubview:dbLabel];

    NSTextField *hint = [self makeLabel:@"Use the fader or enter an exact value." bold:NO size:10 color:[NSColor secondaryLabelColor]];
    hint.frame = NSMakeRect(62, 52, 116, 22);
    hint.lineBreakMode = NSLineBreakByWordWrapping;
    [content addSubview:hint];

    NSButton *disableButton = [[NSButton alloc] initWithFrame:NSMakeRect(62, 18, 106, 26)];
    disableButton.bezelStyle = NSBezelStyleRounded;
    disableButton.title = @"Disable Route";
    disableButton.target = self;
    disableButton.action = @selector(disableActiveRoutingCrosspoint:);
    [content addSubview:disableButton];
}

- (OutputPathState *)activeRoutingPath {
    return self.activeRoutingMatrix.outputA ? &gEngine.pathA : &gEngine.pathB;
}

- (void)setActiveRoutingGainDb:(double)db {
    db = fmin(12.0, fmax(-60.0, db));
    OutputPathState *path = [self activeRoutingPath];
    _Atomic float *gains = self.activeRoutingMono ? path->monoRouteGain
                          : (self.activeRoutingInputLeft ? path->routeGainLeft : path->routeGainRight);
    atomic_store_explicit(&gains[self.activeRoutingChannel], RouteGainDbToLinear(db), memory_order_relaxed);
    self.routingGainFader.doubleValue = db;
    self.routingGainValueField.stringValue = [NSString stringWithFormat:@"%.1f", db];
    [self updateRoutingMatrixViews];
    [self savePreferences];
}

- (void)routingMatrixView:(RoutingMatrixView *)view
         didToggleInputLeft:(BOOL)inputLeft
         destinationChannel:(uint32_t)destinationChannel {
    if (destinationChannel >= kMaxOutputChannels) return;
    OutputPathState *path = view.outputA ? &gEngine.pathA : &gEngine.pathB;
    uint32_t bit = 1U << destinationChannel;
    BOOL mono = atomic_load_explicit(&path->mono, memory_order_relaxed);
    _Atomic uint32_t *mask = mono ? &path->monoRouteMask
                             : (inputLeft ? &path->routeMaskLeft : &path->routeMaskRight);
    uint32_t current = atomic_load_explicit(mask, memory_order_relaxed);
    if ((current & bit) == 0U) {
        atomic_store_explicit(mask, current | bit, memory_order_relaxed);
    }

    self.activeRoutingMatrix = view;
    self.activeRoutingInputLeft = inputLeft;
    self.activeRoutingMono = mono;
    self.activeRoutingChannel = destinationChannel;
    _Atomic float *gains = mono ? path->monoRouteGain
                          : (inputLeft ? path->routeGainLeft : path->routeGainRight);
    double db = RouteGainLinearToDb(atomic_load_explicit(&gains[destinationChannel], memory_order_relaxed));
    [self ensureRoutingGainPopover];
    self.routingGainFader.doubleValue = db;
    self.routingGainValueField.stringValue = [NSString stringWithFormat:@"%.1f", db];
    [self.routingGainPopover performClose:nil];
    [self.routingGainPopover showRelativeToRect:view.bounds ofView:view preferredEdge:NSMaxYEdge];
    [self updateRoutingMatrixViews];
    [self savePreferences];
}

- (void)routingGainFaderChanged:(id)sender {
    (void)sender;
    [self setActiveRoutingGainDb:self.routingGainFader.doubleValue];
}

- (void)routingGainValueChanged:(id)sender {
    (void)sender;
    [self setActiveRoutingGainDb:self.routingGainValueField.doubleValue];
}

- (void)disableActiveRoutingCrosspoint:(id)sender {
    (void)sender;
    OutputPathState *path = [self activeRoutingPath];
    _Atomic uint32_t *mask = self.activeRoutingMono ? &path->monoRouteMask
                             : (self.activeRoutingInputLeft ? &path->routeMaskLeft : &path->routeMaskRight);
    uint32_t bit = 1U << self.activeRoutingChannel;
    uint32_t current = atomic_load_explicit(mask, memory_order_relaxed);
    atomic_store_explicit(mask, current & ~bit, memory_order_relaxed);
    [self.routingGainPopover performClose:nil];
    [self updateRoutingMatrixViews];
    [self savePreferences];
}

- (void)masterVolumeChanged:(id)sender {
    (void)sender;
    gEngine.masterVolume = self.masterSlider.floatValue;
    UpdateEngineTargetGains();
    [self updateValueLabels];
    [self savePreferences];
}

- (void)balanceChanged:(id)sender {
    (void)sender;
    gEngine.balance = self.balanceSlider.floatValue;
    UpdateEngineTargetGains();
    [self updateValueLabels];
    [self savePreferences];
}

- (void)resetBalanceCenter:(id)sender {
    (void)sender;
    gEngine.balance = 0.0f;
    self.balanceSlider.floatValue = 0.0f;
    UpdateEngineTargetGains();
    [self updateValueLabels];
    [self savePreferences];
}

- (void)trimAChanged:(id)sender {
    (void)sender;
    gEngine.trimDbA = self.trimSliderA.floatValue;
    UpdateEngineTargetGains();
    [self updateValueLabels];
    [self savePreferences];
}

- (void)resetTrimA:(id)sender {
    (void)sender;
    gEngine.trimDbA = 0.0f;
    self.trimSliderA.floatValue = 0.0f;
    UpdateEngineTargetGains();
    [self updateValueLabels];
    [self savePreferences];
}

- (void)trimBChanged:(id)sender {
    (void)sender;
    gEngine.trimDbB = self.trimSliderB.floatValue;
    UpdateEngineTargetGains();
    [self updateValueLabels];
    [self savePreferences];
}

- (void)resetTrimB:(id)sender {
    (void)sender;
    gEngine.trimDbB = 0.0f;
    self.trimSliderB.floatValue = 0.0f;
    UpdateEngineTargetGains();
    [self updateValueLabels];
    [self savePreferences];
}

- (void)delayASliderChanged:(id)sender {
    (void)sender;
    double ms = round(self.delaySliderA.doubleValue * 2.0) / 2.0;
    self.delayStepperA.doubleValue = ms;
    atomic_store_explicit(&gEngine.pathA.targetDelayMs, ms, memory_order_relaxed);
    [self updateValueLabels];
    [self savePreferences];
}

- (void)delayAStepperChanged:(id)sender {
    (void)sender;
    double ms = self.delayStepperA.doubleValue;
    self.delaySliderA.doubleValue = ms;
    atomic_store_explicit(&gEngine.pathA.targetDelayMs, ms, memory_order_relaxed);
    [self updateValueLabels];
    [self savePreferences];
}

- (void)delayBSliderChanged:(id)sender {
    (void)sender;
    double ms = round(self.delaySliderB.doubleValue * 2.0) / 2.0;
    self.delayStepperB.doubleValue = ms;
    atomic_store_explicit(&gEngine.pathB.targetDelayMs, ms, memory_order_relaxed);
    [self updateValueLabels];
    [self savePreferences];
}

- (void)delayBStepperChanged:(id)sender {
    (void)sender;
    double ms = self.delayStepperB.doubleValue;
    self.delaySliderB.doubleValue = ms;
    atomic_store_explicit(&gEngine.pathB.targetDelayMs, ms, memory_order_relaxed);
    [self updateValueLabels];
    [self savePreferences];
}

/*===========================================================================
 * Crossover Controls
 *===========================================================================*/

- (void)updateCrossoverControls {
    bool enabled = atomic_load_explicit(&gEngine.crossover.enabled, memory_order_relaxed);
    bool linked = atomic_load_explicit(&gEngine.crossover.linked, memory_order_relaxed);
    double linkedFrequency = ClampCrossoverFrequency(atomic_load_explicit(&gEngine.crossover.linkedFrequency, memory_order_relaxed), 48000.0);
    double frequencyA = ClampCrossoverFrequency(atomic_load_explicit(&gEngine.crossover.frequencyA, memory_order_relaxed), 48000.0);
    double frequencyB = ClampCrossoverFrequency(atomic_load_explicit(&gEngine.crossover.frequencyB, memory_order_relaxed), 48000.0);

    self.crossoverEnabledCheckbox.state = enabled ? NSControlStateValueOn : NSControlStateValueOff;
    self.crossoverLinkedCheckbox.state = linked ? NSControlStateValueOn : NSControlStateValueOff;
    self.crossoverLinkedCheckbox.enabled = enabled;

    self.linkedCrossoverFrequencySlider.doubleValue = CrossoverFrequencyToSliderValue(linkedFrequency);
    self.linkedCrossoverFrequencyLabel.stringValue = [NSString stringWithFormat:@"%.0f Hz", linkedFrequency];
    [self.linkedCrossoverSlopePopup selectItemWithTag:ClampCrossoverSlope(atomic_load_explicit(&gEngine.crossover.linkedSlopeDb, memory_order_relaxed))];
    [self.linkedCrossoverAssignmentPopup selectItemWithTag:
        atomic_load_explicit(&gEngine.crossover.linkedOutputAHighPass, memory_order_relaxed) ? 1 : 0];

    [self.crossoverModePopupA selectItemWithTag:atomic_load_explicit(&gEngine.crossover.modeA, memory_order_relaxed)];
    [self.crossoverModePopupB selectItemWithTag:atomic_load_explicit(&gEngine.crossover.modeB, memory_order_relaxed)];
    self.crossoverFrequencySliderA.doubleValue = CrossoverFrequencyToSliderValue(frequencyA);
    self.crossoverFrequencySliderB.doubleValue = CrossoverFrequencyToSliderValue(frequencyB);
    self.crossoverFrequencyLabelA.stringValue = [NSString stringWithFormat:@"%.0f Hz", frequencyA];
    self.crossoverFrequencyLabelB.stringValue = [NSString stringWithFormat:@"%.0f Hz", frequencyB];
    [self.crossoverSlopePopupA selectItemWithTag:ClampCrossoverSlope(atomic_load_explicit(&gEngine.crossover.slopeDbA, memory_order_relaxed))];
    [self.crossoverSlopePopupB selectItemWithTag:ClampCrossoverSlope(atomic_load_explicit(&gEngine.crossover.slopeDbB, memory_order_relaxed))];

    for (NSView *control in self.linkedCrossoverControls) {
        control.hidden = !linked;
        if ([control respondsToSelector:@selector(setEnabled:)]) {
            [(id)control setEnabled:enabled];
        }
    }
    for (NSView *control in self.independentCrossoverControls) {
        control.hidden = linked;
        if ([control respondsToSelector:@selector(setEnabled:)]) {
            [(id)control setEnabled:enabled];
        }
    }
    self.crossoverGraphFrequencyLabel.hidden = !linked;
    if (linked) {
        self.crossoverGraphFrequencyLabel.stringValue = [NSString stringWithFormat:@"Crossover marker  •  %.0f Hz", linkedFrequency];
    }
    [self.crossoverResponseView setNeedsDisplay:YES];
}

- (void)crossoverEnabledChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.enabled,
                          self.crossoverEnabledCheckbox.state == NSControlStateValueOn,
                          memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)crossoverLinkedChanged:(id)sender {
    (void)sender;
    bool linked = self.crossoverLinkedCheckbox.state == NSControlStateValueOn;
    if (linked) {
        /* Relinking adopts Output A's timing and slope, then restores the
         * selected A/B band roles. */
        atomic_store_explicit(&gEngine.crossover.linkedFrequency,
                              atomic_load_explicit(&gEngine.crossover.frequencyA, memory_order_relaxed),
                              memory_order_relaxed);
        atomic_store_explicit(&gEngine.crossover.linkedSlopeDb,
                              ClampCrossoverSlope(atomic_load_explicit(&gEngine.crossover.slopeDbA, memory_order_relaxed)),
                              memory_order_relaxed);
        CrossoverFilterMode outputAMode = (CrossoverFilterMode)atomic_load_explicit(&gEngine.crossover.modeA, memory_order_relaxed);
        if (outputAMode == kCrossoverFilterLowPass || outputAMode == kCrossoverFilterHighPass) {
            atomic_store_explicit(&gEngine.crossover.linkedOutputAHighPass,
                                  outputAMode == kCrossoverFilterHighPass,
                                  memory_order_relaxed);
        }
    } else {
        double frequency = atomic_load_explicit(&gEngine.crossover.linkedFrequency, memory_order_relaxed);
        int slope = ClampCrossoverSlope(atomic_load_explicit(&gEngine.crossover.linkedSlopeDb, memory_order_relaxed));
        bool outputAHighPass = atomic_load_explicit(&gEngine.crossover.linkedOutputAHighPass, memory_order_relaxed);
        atomic_store_explicit(&gEngine.crossover.modeA,
                              outputAHighPass ? kCrossoverFilterHighPass : kCrossoverFilterLowPass,
                              memory_order_relaxed);
        atomic_store_explicit(&gEngine.crossover.modeB,
                              outputAHighPass ? kCrossoverFilterLowPass : kCrossoverFilterHighPass,
                              memory_order_relaxed);
        atomic_store_explicit(&gEngine.crossover.frequencyA, frequency, memory_order_relaxed);
        atomic_store_explicit(&gEngine.crossover.frequencyB, frequency, memory_order_relaxed);
        atomic_store_explicit(&gEngine.crossover.slopeDbA, slope, memory_order_relaxed);
        atomic_store_explicit(&gEngine.crossover.slopeDbB, slope, memory_order_relaxed);
    }
    atomic_store_explicit(&gEngine.crossover.linked, linked, memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)linkedCrossoverFrequencyChanged:(id)sender {
    (void)sender;
    double frequency = CrossoverSliderValueToFrequency(self.linkedCrossoverFrequencySlider.doubleValue);
    atomic_store_explicit(&gEngine.crossover.linkedFrequency, frequency, memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)linkedCrossoverSlopeChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.linkedSlopeDb,
                          ClampCrossoverSlope(self.linkedCrossoverSlopePopup.selectedItem.tag),
                          memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)linkedCrossoverAssignmentChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.linkedOutputAHighPass,
                          self.linkedCrossoverAssignmentPopup.selectedItem.tag == 1,
                          memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)crossoverModeAChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.modeA, self.crossoverModePopupA.selectedItem.tag, memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)crossoverModeBChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.modeB, self.crossoverModePopupB.selectedItem.tag, memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)crossoverFrequencyAChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.frequencyA,
                          CrossoverSliderValueToFrequency(self.crossoverFrequencySliderA.doubleValue),
                          memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)crossoverFrequencyBChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.frequencyB,
                          CrossoverSliderValueToFrequency(self.crossoverFrequencySliderB.doubleValue),
                          memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)crossoverSlopeAChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.slopeDbA,
                          ClampCrossoverSlope(self.crossoverSlopePopupA.selectedItem.tag),
                          memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)crossoverSlopeBChanged:(id)sender {
    (void)sender;
    atomic_store_explicit(&gEngine.crossover.slopeDbB,
                          ClampCrossoverSlope(self.crossoverSlopePopupB.selectedItem.tag),
                          memory_order_relaxed);
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)crossoverResponseView:(CrossoverResponseView *)view
       didDragFrequency:(double)frequency
             forOutputA:(BOOL)outputA {
    (void)view;
    if (atomic_load_explicit(&gEngine.crossover.linked, memory_order_relaxed)) {
        atomic_store_explicit(&gEngine.crossover.linkedFrequency, frequency, memory_order_relaxed);
    } else if (outputA) {
        atomic_store_explicit(&gEngine.crossover.frequencyA, frequency, memory_order_relaxed);
    } else {
        atomic_store_explicit(&gEngine.crossover.frequencyB, frequency, memory_order_relaxed);
    }
    [self updateCrossoverControls];
    [self savePreferences];
}

- (void)updateValueLabels {
    float s = gEngine.masterVolume;
    if (s <= 0.0001f) {
        self.masterValueLabel.stringValue = @"0% (Muted)";
    } else {
        float db = 60.0f * log10f(s);
        self.masterValueLabel.stringValue = [NSString stringWithFormat:@"%.0f%% (%+.1f dB)", s * 100.0f, db];
    }

    float b = gEngine.balance;
    if (fabsf(b) < 0.01f) {
        self.balanceValueLabel.stringValue = @"Center (A=100%, B=100%)";
    } else if (b < 0.0f) {
        float pctB = sinf((float)M_PI_2 * (1.0f + b)) * 100.0f;
        self.balanceValueLabel.stringValue = [NSString stringWithFormat:@"Left (A=100%%, B=%.0f%%)", pctB];
    } else {
        float pctA = cosf((float)M_PI_2 * b) * 100.0f;
        self.balanceValueLabel.stringValue = [NSString stringWithFormat:@"Right (A=%.0f%%, B=100%%)", pctA];
    }

    self.trimValueLabelA.stringValue = [NSString stringWithFormat:@"%+.1f dB", gEngine.trimDbA];
    self.trimValueLabelB.stringValue = [NSString stringWithFormat:@"%+.1f dB", gEngine.trimDbB];

    self.delayValueLabelA.stringValue = [NSString stringWithFormat:@"%.1f ms", atomic_load(&gEngine.pathA.targetDelayMs)];
    self.delayValueLabelB.stringValue = [NSString stringWithFormat:@"%.1f ms", atomic_load(&gEngine.pathB.targetDelayMs)];
}

- (void)updateTelemetryUI {
    self.inputMeter.doubleValue = (double)atomic_load_explicit(&gEngine.inputPeakLevel, memory_order_relaxed);
    self.meterA.doubleValue     = (double)atomic_load_explicit(&gEngine.pathA.peakLevel, memory_order_relaxed);
    self.meterB.doubleValue     = (double)atomic_load_explicit(&gEngine.pathB.peakLevel, memory_order_relaxed);

    if (atomic_load_explicit(&gEngine.pathA.streamRunning, memory_order_relaxed)) {
        bool en   = atomic_load_explicit(&gEngine.pathA.enabled, memory_order_relaxed);
        bool mono = atomic_load_explicit(&gEngine.pathA.mono, memory_order_relaxed);
        double rate = atomic_load_explicit(&gEngine.pathA.outputSampleRate, memory_order_relaxed);
        double ppm  = atomic_load_explicit(&gEngine.pathA.currentRatioPPM, memory_order_relaxed);
        uint32_t fill = atomic_load_explicit(&gEngine.pathA.currentFillFrames, memory_order_relaxed);
        if (en) {
            self.statusLabelA.textColor = [NSColor systemGreenColor];
            self.statusLabelA.stringValue = [NSString stringWithFormat:@"Active (%@) | %.0f Hz | Drift: %+.1f ppm | Buf: %u f",
                                             mono ? @"Mono" : @"Stereo", rate, ppm, fill];
        } else {
            self.statusLabelA.textColor = [NSColor secondaryLabelColor];
            self.statusLabelA.stringValue = [NSString stringWithFormat:@"Muted (Disabled) | %.0f Hz", rate];
        }
    }

    if (atomic_load_explicit(&gEngine.pathB.streamRunning, memory_order_relaxed)) {
        bool en   = atomic_load_explicit(&gEngine.pathB.enabled, memory_order_relaxed);
        bool mono = atomic_load_explicit(&gEngine.pathB.mono, memory_order_relaxed);
        double rate = atomic_load_explicit(&gEngine.pathB.outputSampleRate, memory_order_relaxed);
        double ppm  = atomic_load_explicit(&gEngine.pathB.currentRatioPPM, memory_order_relaxed);
        uint32_t fill = atomic_load_explicit(&gEngine.pathB.currentFillFrames, memory_order_relaxed);
        if (en) {
            self.statusLabelB.textColor = [NSColor systemGreenColor];
            self.statusLabelB.stringValue = [NSString stringWithFormat:@"Active (%@) | %.0f Hz | Drift: %+.1f ppm | Buf: %u f",
                                             mono ? @"Mono" : @"Stereo", rate, ppm, fill];
        } else {
            self.statusLabelB.textColor = [NSColor secondaryLabelColor];
            self.statusLabelB.stringValue = [NSString stringWithFormat:@"Muted (Disabled) | %.0f Hz", rate];
        }
    }
}

@end

int main(int argc, const char *argv[]) {
    (void)argc; (void)argv;
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        AppDelegate *delegate = [[AppDelegate alloc] init];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
