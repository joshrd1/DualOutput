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

#define kVirtualDeviceUID		   "com.dualoutput.driver.device"
#define kVirtualDriverBundleID	   "com.dualoutput.driver"
#define kDualOutputPlugInPropertyDevicePresence 'dopr'

/* Lock-free SPSC clock-drift buffer constants (power of 2) */
#define kDriftRingFrames			16384U
#define kDriftRingMask			  (kDriftRingFrames - 1U)
#define kDriftTargetFillFrames	  1536.0

/* Real-time delay line buffer constants (power of 2; 65536 frames > 500ms at up to 96kHz) */
#define kDelayBufferFrames		  65536U
#define kDelayBufferMask			(kDelayBufferFrames - 1U)

/* Maximum frames per CoreAudio callback slice */
#define kMaxSliceFrames			 4096U

/* A Linkwitz-Riley crossover needs, at most, eight biquad sections per path
 * for the selectable 96 dB/octave slope. */
#define kMaxCrossoverBiquads		8

/* Routing matrices expose up to sixteen hardware destination channels. */
#define kMaxOutputChannels		   16U
#define kGraphicEQBandCount		 31
#define kGraphicEQMinDb			 (-12.0f)
#define kGraphicEQMaxDb			 12.0f
/* 1/3-octave bandwidth Q factor: sqrt(2^(1/3)) / (2^(1/3) - 1) */
#define kGraphicEQOneThirdOctaveQ   4.318477f

static const double kGraphicEQFrequencies[kGraphicEQBandCount] = {
	20.0,   25.0,   31.5,   40.0,   50.0,   63.0,   80.0,   100.0,
	125.0,  160.0,  200.0,  250.0,  315.0,  400.0,  500.0,  630.0,
	800.0,  1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0,
	5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0
};

static NSString * const kGraphicEQBandLabels[kGraphicEQBandCount] = {
	@"20", @"25", @"31.5", @"40", @"50", @"63", @"80", @"100",
	@"125", @"160", @"200", @"250", @"315", @"400", @"500", @"630",
	@"800", @"1k", @"1.25k", @"1.6k", @"2k", @"2.5k", @"3.15k", @"4k",
	@"5k", @"6.3k", @"8k", @"10k", @"12.5k", @"16k", @"20k"
};

typedef enum {
	kCrossoverFilterBypass   = 0,
	kCrossoverFilterLowPass  = 1,
	kCrossoverFilterHighPass = 2
} CrossoverFilterMode;

typedef NS_ENUM(NSInteger, StatusLevel) {
	StatusLevelHint,
	StatusLevelInfo,
	StatusLevelSuccess,
	StatusLevelWarning,
	StatusLevelError
};

static NSString * const kDefaultStatusHint = @"Tip: Cmd+Z undoes a setting change; Shift-Cmd+Z redoes it.";
static NSString * const kStatusIssueMicrophone = @"microphone";
static NSString * const kStatusIssueVirtualDevice = @"virtual-device";
static NSString * const kStatusIssueVirtualDeviceStream = @"virtual-device-stream";
static NSString * const kStatusIssueOutputA = @"output-a";
static NSString * const kStatusIssueOutputB = @"output-b";
static NSString * const kStatusIssueSystemOutput = @"system-output";

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
	_Atomic int	linkedSlopeDb;
	_Atomic int	modeA;
	_Atomic int	modeB;
	_Atomic double frequencyA;
	_Atomic double frequencyB;
	_Atomic int	slopeDbA;
	_Atomic int	slopeDbB;
} CrossoverState;

typedef struct {
	_Atomic bool	 enabled;
	_Atomic float	bandGainDb[kGraphicEQBandCount];
	_Atomic uint32_t revision;
} GraphicEQState;

/* Real-time safe per-output processing state */
typedef struct {
	/* SPSC Clock-Domain Crossing Ring Buffer (written by input callback, read by output callback) */
	float			   driftBuffer[kDriftRingFrames * 2];
	_Atomic uint64_t	writeIndex;
	uint64_t			readIndexInt;
	double			  readIndexFrac;
	bool				primed;

	/* PI Adaptive Clock-Drift Controller state */
	double			  filteredError;
	double			  integralError;
	_Atomic double	  inputSampleRate;
	_Atomic double	  outputSampleRate;
	_Atomic double	  currentRatioPPM;
	_Atomic uint32_t	currentFillFrames;

	/* Dedicated Delay Line Ring Buffer (runs at output device native sample rate) */
	float			   delayBuffer[kDelayBufferFrames * 2];
	uint32_t			delayWriteIndex;
	double			  currentDelaySamples;
	double			  delaySlewCoeff;
	float			   gainSlewCoeff;
	_Atomic double	  targetDelayMs;

	/* Gain Stage, Mono Switch & Metering */
	_Atomic float	   targetGain;
	float			   currentGain;
	_Atomic bool		enabled;
	_Atomic bool		mono;
	_Atomic bool		streamRunning;
	_Atomic float	   peakLevelL;
	_Atomic float	   peakLevelR;
	_Atomic bool		clippedL;
	_Atomic bool		clippedR;

	/* Per-output-channel input routing. Bit N routes source L or R to
	 * destination channel N. These are consumed atomically by the callback. */
	_Atomic uint32_t	outputChannelCount;
	_Atomic uint32_t	routeMaskLeft;
	_Atomic uint32_t	routeMaskRight;
	_Atomic uint32_t	monoRouteMask;
	_Atomic float	   routeGainLeft[kMaxOutputChannels];
	_Atomic float	   routeGainRight[kMaxOutputChannels];
	_Atomic float	   monoRouteGain[kMaxOutputChannels];

	/* 31-Band Graphic EQ filter state (runs right before Crossover) */
	CrossoverBiquad	 eqBiquads[kGraphicEQBandCount];
	bool				eqStageActive[kGraphicEQBandCount];
	uint8_t			 eqStageTailFrames[kGraphicEQBandCount];
	bool				eqActive;
	uint32_t			appliedEQRevision;
	double			  appliedEQSampleRate;

	/* Crossover filter state. Coefficients are reconfigured only when a
	 * user-facing setting or the destination sample rate changes. */
	CrossoverBiquad	 crossoverBiquads[kMaxCrossoverBiquads];
	int				 crossoverStageCount;
	int				 appliedCrossoverMode;
	int				 appliedCrossoverSlopeDb;
	double			  appliedCrossoverFrequency;
	double			  appliedCrossoverSampleRate;
	bool				crossoverInvertPolarity;

	/* CoreAudio AUHAL Output Unit */
	AudioUnit		   outputUnit;
	AudioDeviceID	   deviceID;
} OutputPathState;

/* Global Real-Time Engine State */
typedef struct {
	AudioUnit		   inputUnit;
	AudioDeviceID	   virtualDeviceID;
	_Atomic bool		inputRunning;
	/* Meter peaks are only useful while the controller is visible. Keeping this
	 * separate from inputRunning lets the audio router continue untouched when
	 * the UI is hidden. */
	_Atomic bool		meteringEnabled;
	_Atomic double	  virtualSampleRate;
	_Atomic float	   peakLevelL;
	_Atomic float	   peakLevelR;
	_Atomic bool		clippedL;
	_Atomic bool		clippedR;
	float			   inputScratchBuffer[kMaxSliceFrames * 2];

	OutputPathState	 pathA;
	OutputPathState	 pathB;

	/* User Parameters */
	float			   masterVolume;	   /* 0.0 .. 1.0 (perceptual cubic taper) */
	float			   balance;			/* -1.0 (A only) .. 0.0 (Equal) .. +1.0 (B only) */
	float			   trimDbA;			/* -24.0 .. +6.0 dB */
	float			   trimDbB;			/* -24.0 .. +6.0 dB */
	GraphicEQState	  graphicEQ;
	CrossoverState	  crossover;
} DualAudioEngine;

static DualAudioEngine gEngine;

typedef struct {
	CrossoverFilterMode mode;
	double			  frequency;
	int				 slopeDb;
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
	slopeDb = fmin(96, fmax(12, slopeDb));
	return 12 * (int)llround((double)slopeDb / 12.0);
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

static const double kRouteGainDisabledDb = -60.0;

static inline BOOL RouteGainIsDisabled(double db) {
	return (isinf(db) && db < 0.0) || db <= kRouteGainDisabledDb;
}

static BOOL RouteGainTextIsDisabled(NSString *text) {
	NSString *normalized = [[text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
	normalized = [normalized stringByReplacingOccurrencesOfString:@"−" withString:@"-"];
	return [normalized isEqualToString:@"-inf"] ||
		   [normalized isEqualToString:@"-infinity"] ||
		   [normalized isEqualToString:@"-∞"];
}

static inline float RouteGainDbToLinear(double db) {
	if (!isfinite(db)) db = 0.0;
	if (db < kRouteGainDisabledDb) db = kRouteGainDisabledDb;
	if (db > 12.0) db = 12.0;
	return (float)pow(10.0, db / 20.0);
}

static inline double RouteGainLinearToDb(float gain) {
	if (!isfinite(gain) || gain <= 0.000001f) return kRouteGainDisabledDb;
	return fmin(12.0, fmax(kRouteGainDisabledDb, 20.0 * log10((double)gain)));
}

/* The Q values below are two identical Butterworth filters cascaded into
 * Linkwitz-Riley alignments. The linked low/high pair therefore shares phase
 * at the crossover point and sums flat at 12 through 96 dB/octave, in
 * 12 dB/octave increments. */
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
			outQs[0] = 0.54119610f;
			outQs[1] = 1.30656296f;
			outQs[2] = 0.54119610f;
			outQs[3] = 1.30656296f;
			return 4;
		case 60:
			outQs[0] = 0.5f;
			outQs[1] = 0.61803399f;
			outQs[2] = 1.61803399f;
			outQs[3] = 0.61803399f;
			outQs[4] = 1.61803399f;
			return 5;
		case 72:
			outQs[0] = 0.51763809f;
			outQs[1] = 0.70710678f;
			outQs[2] = 1.93185165f;
			outQs[3] = 0.51763809f;
			outQs[4] = 0.70710678f;
			outQs[5] = 1.93185165f;
			return 6;
		case 84:
			outQs[0] = 0.5f;
			outQs[1] = 0.55495813f;
			outQs[2] = 0.80193774f;
			outQs[3] = 2.24697960f;
			outQs[4] = 0.55495813f;
			outQs[5] = 0.80193774f;
			outQs[6] = 2.24697960f;
			return 7;
		case 96:
		default:
			outQs[0] = 0.50979558f;
			outQs[1] = 0.60134489f;
			outQs[2] = 0.89997622f;
			outQs[3] = 2.56291545f;
			outQs[4] = 0.50979558f;
			outQs[5] = 0.60134489f;
			outQs[6] = 0.89997622f;
			outQs[7] = 2.56291545f;
			return 8;
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

static CrossoverConfig GetConfiguredCrossoverConfigForPath(bool outputA) {
	CrossoverConfig config;
	bool linked  = atomic_load_explicit(&gEngine.crossover.linked, memory_order_relaxed);

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

static CrossoverConfig GetCrossoverConfigForPath(bool outputA) {
	if (!atomic_load_explicit(&gEngine.crossover.enabled, memory_order_relaxed)) {
		CrossoverConfig config = { kCrossoverFilterBypass, 120.0, 24 };
		return config;
	}
	return GetConfiguredCrossoverConfigForPath(outputA);
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
	/* Linked alignments with an odd number of biquad sections need the
	 * high-pass branch polarity-normalized to produce a flat sum. */
	path->crossoverInvertPolarity = config.mode == kCrossoverFilterHighPass &&
									(config.slopeDb % 24 == 12);
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

static inline float ClampGraphicEQGainDb(float db) {
	if (!isfinite(db)) return 0.0f;
	if (db < kGraphicEQMinDb) return kGraphicEQMinDb;
	if (db > kGraphicEQMaxDb) return kGraphicEQMaxDb;
	return db;
}

static void CalculatePeakingEQBiquad(double sampleRate,
									 double frequency,
									 float gainDb,
									 float q,
									 float *b0, float *b1, float *b2,
									 float *a1, float *a2) {
	if (fabsf(gainDb) < 0.01f || frequency >= sampleRate * 0.475) {
		*b0 = 1.0f; *b1 = 0.0f; *b2 = 0.0f;
		*a1 = 0.0f; *a2 = 0.0f;
		return;
	}
	double A	 = pow(10.0, (double)gainDb / 40.0);
	double omega = 2.0 * M_PI * frequency / sampleRate;
	double sine  = sin(omega);
	double cs	= cos(omega);
	double alpha = sine / (2.0 * (double)q);

	double rawB0 =  1.0 + alpha * A;
	double rawB1 = -2.0 * cs;
	double rawB2 =  1.0 - alpha * A;
	double a0	=  1.0 + alpha / A;
	double rawA1 = -2.0 * cs;
	double rawA2 =  1.0 - alpha / A;

	*b0 = (float)(rawB0 / a0);
	*b1 = (float)(rawB1 / a0);
	*b2 = (float)(rawB2 / a0);
	*a1 = (float)(rawA1 / a0);
	*a2 = (float)(rawA2 / a0);
}

/* A flat band is an identity biquad. It can be omitted once its two delay
 * elements have drained, but a band just returned to 0 dB must process its
 * remaining two samples so its filter tail stays continuous. */
static inline uint8_t GraphicEQTailFrames(const CrossoverBiquad *stage) {
	uint8_t frames = 0;
	if (stage->z1L != 0.0f || stage->z1R != 0.0f) frames = 1;
	if (stage->z2L != 0.0f || stage->z2R != 0.0f) frames = 2;
	return frames;
}

static void ConfigureGraphicEQForPath(OutputPathState *path, double sampleRate) {
	bool enabled = atomic_load_explicit(&gEngine.graphicEQ.enabled, memory_order_relaxed);
	uint32_t rev = atomic_load_explicit(&gEngine.graphicEQ.revision, memory_order_relaxed);
	bool rateChanged = fabs(path->appliedEQSampleRate - sampleRate) > 1.0;

	if (!enabled) {
		if (path->eqActive) {
			path->eqActive = false;
			memset(path->eqBiquads, 0, sizeof(path->eqBiquads));
			memset(path->eqStageActive, 0, sizeof(path->eqStageActive));
			memset(path->eqStageTailFrames, 0, sizeof(path->eqStageTailFrames));
		}
		path->appliedEQRevision   = rev;
		path->appliedEQSampleRate = sampleRate;
		return;
	}

	/* A flat but enabled EQ is still fully configured. Do not keep rebuilding
	 * all 31 identity filters once per audio callback. */
	if (path->appliedEQRevision == rev && !rateChanged) {
		return;
	}

	bool wasActive = path->eqActive;
	if (!wasActive || rateChanged) {
		memset(path->eqBiquads, 0, sizeof(path->eqBiquads));
		memset(path->eqStageActive, 0, sizeof(path->eqStageActive));
		memset(path->eqStageTailFrames, 0, sizeof(path->eqStageTailFrames));
	}

	bool anyNonZero = false;
	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		float db = ClampGraphicEQGainDb(atomic_load_explicit(&gEngine.graphicEQ.bandGainDb[i], memory_order_relaxed));
		bool bandActive = fabsf(db) >= 0.01f;
		if (bandActive) anyNonZero = true;
		CrossoverBiquad *stage = &path->eqBiquads[i];
		CalculatePeakingEQBiquad(sampleRate,
								 kGraphicEQFrequencies[i],
								 db,
								 kGraphicEQOneThirdOctaveQ,
								 &stage->b0, &stage->b1, &stage->b2,
								 &stage->a1, &stage->a2);
		path->eqStageActive[i] = bandActive;
		path->eqStageTailFrames[i] = 0;
		if (!bandActive) {
			uint8_t tailFrames = GraphicEQTailFrames(stage);
			path->eqStageActive[i] = tailFrames > 0;
			path->eqStageTailFrames[i] = tailFrames;
		}
	}

	path->eqActive			= anyNonZero;
	path->appliedEQRevision   = rev;
	path->appliedEQSampleRate = sampleRate;
	/* Existing behavior bypasses the entire EQ as soon as its final non-flat
	 * band is reset, so do not retain a tail in that fully-flat case. */
	if (!anyNonZero) {
		memset(path->eqStageActive, 0, sizeof(path->eqStageActive));
		memset(path->eqStageTailFrames, 0, sizeof(path->eqStageTailFrames));
	}
}

static inline void ProcessGraphicEQStereo(OutputPathState *path, float *left, float *right) {
	float l = *left;
	float r = *right;
	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		if (!path->eqStageActive[i]) continue;
		CrossoverBiquad *stage = &path->eqBiquads[i];
		l = ProcessCrossoverBiquad(l, &stage->z1L, &stage->z2L, stage);
		r = ProcessCrossoverBiquad(r, &stage->z1R, &stage->z2R, stage);
		if (path->eqStageTailFrames[i] > 0 && --path->eqStageTailFrames[i] == 0) {
			path->eqStageActive[i] = false;
		}
	}
	*left  = l;
	*right = r;
}

/* 4-Point Hermite Cubic Interpolation for fractional stereo sample index */
static inline void InterpolateHermiteStereo(const float *ring, uint32_t mask, uint64_t idxInt, double frac, float *outL, float *outR) {
	uint32_t i0 = ((uint32_t)(idxInt - 1)) & mask;
	uint32_t i1 = ((uint32_t)(idxInt))	 & mask;
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
	uint32_t writeSlot = ((uint32_t)w) & kDriftRingMask;
	uint32_t firstFrames = kDriftRingFrames - writeSlot;
	if (firstFrames > numFrames) firstFrames = numFrames;
	memcpy(&path->driftBuffer[writeSlot * 2], frames, (size_t)firstFrames * 2U * sizeof(float));
	if (firstFrames < numFrames) {
		memcpy(path->driftBuffer,
			   &frames[firstFrames * 2],
			   (size_t)(numFrames - firstFrames) * 2U * sizeof(float));
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
static void WriteRoutedOutputFrame(AudioBufferList *bufferList,
								  UInt32 frame,
								  uint32_t channelCount,
								  uint32_t routeMaskLeft,
								  uint32_t routeMaskRight,
								  const float *routeGainsLeft,
								  const float *routeGainsRight,
								  float sourceLeft,
								  float sourceRight) {
	if (bufferList->mNumberBuffers == 1 && bufferList->mBuffers[0].mData != NULL &&
		bufferList->mBuffers[0].mNumberChannels >= channelCount) {
		float *samples = (float *)bufferList->mBuffers[0].mData;
		for (uint32_t channel = 0; channel < channelCount; ++channel) {
			float output = RoutedSampleForChannel(channel, routeMaskLeft, routeMaskRight,
												   routeGainsLeft[channel], routeGainsRight[channel],
												   sourceLeft, sourceRight);
			samples[(size_t)frame * channelCount + channel] = output;
		}
		return;
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
		}
	}
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
	if (atomic_load_explicit(&gEngine.meteringEnabled, memory_order_relaxed)) {
		float maxAbsL = 0.0f;
		float maxAbsR = 0.0f;
		for (UInt32 i = 0; i < inNumberFrames; ++i) {
			float absL = fabsf(src[i * 2]);
			float absR = fabsf(src[i * 2 + 1]);
			if (absL > maxAbsL) maxAbsL = absL;
			if (absR > maxAbsR) maxAbsR = absR;
		}
		float prevPeakL = atomic_load_explicit(&gEngine.peakLevelL, memory_order_relaxed);
		float prevPeakR = atomic_load_explicit(&gEngine.peakLevelR, memory_order_relaxed);
		atomic_store_explicit(&gEngine.peakLevelL, (maxAbsL > prevPeakL) ? maxAbsL : (prevPeakL * 0.92f), memory_order_relaxed);
		atomic_store_explicit(&gEngine.peakLevelR, (maxAbsR > prevPeakR) ? maxAbsR : (prevPeakR * 0.92f), memory_order_relaxed);
		if (maxAbsL >= 0.999f) atomic_store_explicit(&gEngine.clippedL, true, memory_order_relaxed);
		if (maxAbsR >= 0.999f) atomic_store_explicit(&gEngine.clippedR, true, memory_order_relaxed);
	}

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
	bool collectMetering = atomic_load_explicit(&gEngine.meteringEnabled, memory_order_relaxed);
	if (ioData == NULL || ioData->mNumberBuffers == 0 || ioData->mBuffers[0].mData == NULL) {
		return noErr;
	}

	if (!atomic_load_explicit(&path->enabled, memory_order_relaxed) ||
		!atomic_load_explicit(&gEngine.inputRunning, memory_order_relaxed)) {
		ClearOutputBufferList(ioData);
		if (collectMetering) {
			atomic_store_explicit(&path->peakLevelL, 0.0f, memory_order_relaxed);
			atomic_store_explicit(&path->peakLevelR, 0.0f, memory_order_relaxed);
		}
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
			path->primed		= true;
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
			availSigned		 = (int64_t)kDriftTargetFillFrames;
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

	double delaySlewCoeff = path->delaySlewCoeff;
	float  gainSlewCoeff  = path->gainSlewCoeff;
	float  targetGain	 = atomic_load_explicit(&path->targetGain, memory_order_relaxed);
	bool   isMono		 = atomic_load_explicit(&path->mono, memory_order_relaxed);
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

	ConfigureGraphicEQForPath(path, outRate);
	/* The crossover runs at the destination's native rate, after resampling
	 * and before the optional mono fold-down. */
	CrossoverConfig crossoverConfig = GetCrossoverConfigForPath(path == &gEngine.pathA);
	ConfigureCrossoverForPath(path, crossoverConfig, outRate);

	/* The default stereo map is by far the common case. Its samples are
	 * already clamped, so write them directly instead of rebuilding the same
	 * two-channel matrix for every frame. All custom, mono, and multichannel
	 * maps retain the generic renderer below. */
	float *directStereoOutput = NULL;
	bool directMonoOutput = false;
	if (outputChannels == 2U && ioData->mNumberBuffers == 1 &&
		ioData->mBuffers[0].mData != NULL && ioData->mBuffers[0].mNumberChannels == 2U) {
		if (!isMono && routeMaskLeft == 0x1U && routeMaskRight == 0x2U &&
			routeGainsLeft[0] == 1.0f && routeGainsRight[1] == 1.0f) {
			directStereoOutput = (float *)ioData->mBuffers[0].mData;
		} else if (isMono && routeMaskLeft == 0x3U &&
				   routeGainsLeft[0] == 1.0f && routeGainsLeft[1] == 1.0f) {
			directStereoOutput = (float *)ioData->mBuffers[0].mData;
			directMonoOutput = true;
		}
	}

	float maxAbsL = 0.0f;
	float maxAbsR = 0.0f;

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
		path->delayBuffer[dWrite * 2]	 = resampledL;
		path->delayBuffer[dWrite * 2 + 1] = resampledR;

		path->currentDelaySamples += delaySlewCoeff * (targetDelaySamples - path->currentDelaySamples);
		double dSamples = path->currentDelaySamples;

		float delayedL = resampledL;
		float delayedR = resampledR;

		if (dSamples >= 0.01) {
			double readPos = (double)path->delayWriteIndex - dSamples;
			int64_t rFloor = (int64_t)floor(readPos);
			float   rFrac  = (float)(readPos - (double)rFloor);
			uint32_t idxA  = ((uint32_t)rFloor)	 & kDelayBufferMask;
			uint32_t idxB  = ((uint32_t)(rFloor+1)) & kDelayBufferMask;

			delayedL = path->delayBuffer[idxA * 2]	 * (1.0f - rFrac) + path->delayBuffer[idxB * 2]	 * rFrac;
			delayedR = path->delayBuffer[idxA * 2 + 1] * (1.0f - rFrac) + path->delayBuffer[idxB * 2 + 1] * rFrac;
		}
		path->delayWriteIndex += 1;

		/* Stage 2.5: 31-Band Graphic EQ (Pre-Crossover) */
		if (path->eqActive) {
			ProcessGraphicEQStereo(path, &delayedL, &delayedR);
		}

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

		if (directStereoOutput != NULL) {
			directStereoOutput[(size_t)i * 2U] = outL;
			directStereoOutput[(size_t)i * 2U + 1U] = directMonoOutput ? outL : outR;
		} else {
			WriteRoutedOutputFrame(ioData, i, outputChannels,
								   routeMaskLeft, routeMaskRight,
								   routeGainsLeft, routeGainsRight,
								   outL, outR);
		}
		if (collectMetering) {
			float absL = fabsf(outL);
			float absR = fabsf(outR);
			if (absL > maxAbsL) maxAbsL = absL;
			if (absR > maxAbsR) maxAbsR = absR;
		}
	}

	if (collectMetering) {
		float prevPeakL = atomic_load_explicit(&path->peakLevelL, memory_order_relaxed);
		float prevPeakR = atomic_load_explicit(&path->peakLevelR, memory_order_relaxed);
		atomic_store_explicit(&path->peakLevelL, (maxAbsL > prevPeakL) ? maxAbsL : (prevPeakL * 0.92f), memory_order_relaxed);
		atomic_store_explicit(&path->peakLevelR, (maxAbsR > prevPeakR) ? maxAbsR : (prevPeakR * 0.92f), memory_order_relaxed);
		if (maxAbsL >= 0.999f) atomic_store_explicit(&path->clippedL, true, memory_order_relaxed);
		if (maxAbsR >= 0.999f) atomic_store_explicit(&path->clippedR, true, memory_order_relaxed);
	}

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

/* The driver is always installed, but only publishes its virtual endpoint
 * while this controller process is alive. Resolving the plug-in by bundle ID
 * still works when the device itself is absent from the hardware list. */
static BOOL SetVirtualDevicePresence(BOOL present) {
	AudioObjectPropertyAddress plugInAddress = {
		kAudioHardwarePropertyTranslateBundleIDToPlugIn,
		kAudioObjectPropertyScopeGlobal,
		kAudioObjectPropertyElementMain
	};
	CFStringRef bundleID = CFSTR(kVirtualDriverBundleID);
	AudioObjectID plugInID = kAudioObjectUnknown;
	UInt32 size = sizeof(plugInID);
	OSStatus err = AudioObjectGetPropertyData(kAudioObjectSystemObject,
										 &plugInAddress,
										 sizeof(bundleID),
										 &bundleID,
										 &size,
										 &plugInID);
	if (err != noErr || plugInID == kAudioObjectUnknown) return NO;

	AudioObjectPropertyAddress presenceAddress = {
		kDualOutputPlugInPropertyDevicePresence,
		kAudioObjectPropertyScopeGlobal,
		kAudioObjectPropertyElementMain
	};
	CFPropertyListRef value = present ? kCFBooleanTrue : kCFBooleanFalse;
	return AudioObjectSetPropertyData(plugInID, &presenceAddress, 0, NULL,
									  sizeof(value), &value) == noErr;
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
			item.name	   = name;
			item.uid		= uid;
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
	atomic_store_explicit(&gEngine.peakLevelL, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&gEngine.peakLevelR, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&gEngine.clippedL, false, memory_order_relaxed);
	atomic_store_explicit(&gEngine.clippedR, false, memory_order_relaxed);
}

static OSStatus StartVirtualInputUnit(AudioDeviceID virtualDevID, NSString **outErrorString) {
	StopVirtualInputUnit();
	if (virtualDevID == kAudioObjectUnknown) {
		if (outErrorString) *outErrorString = @"Dual Output driver not found in macOS HAL. Run install.sh first.";
		return kAudioHardwareBadDeviceError;
	}

	AudioComponentDescription desc = {
		.componentType		 = kAudioUnitType_Output,
		.componentSubType	  = kAudioUnitSubType_HALOutput,
		.componentManufacturer = kAudioUnitManufacturer_Apple,
		.componentFlags		= 0,
		.componentFlagsMask	= 0
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
		.mSampleRate	   = hwRate,
		.mFormatID		 = kAudioFormatLinearPCM,
		.mFormatFlags	  = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
		.mBytesPerPacket   = 8,
		.mFramesPerPacket  = 1,
		.mBytesPerFrame	= 8,
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
		.inputProc	   = InputAURenderCallback,
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

	gEngine.inputUnit	   = unit;
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
	atomic_store_explicit(&path->peakLevelL, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&path->peakLevelR, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&path->clippedL, false, memory_order_relaxed);
	atomic_store_explicit(&path->clippedR, false, memory_order_relaxed);
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
		.componentType		 = kAudioUnitType_Output,
		.componentSubType	  = kAudioUnitSubType_HALOutput,
		.componentManufacturer = kAudioUnitManufacturer_Apple,
		.componentFlags		= 0,
		.componentFlagsMask	= 0
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
		.mSampleRate	   = outRate,
		.mFormatID		 = kAudioFormatLinearPCM,
		.mFormatFlags	  = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
		.mBytesPerPacket   = outputChannels * (UInt32)sizeof(float),
		.mFramesPerPacket  = 1,
		.mBytesPerFrame	= outputChannels * (UInt32)sizeof(float),
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
	path->readIndexInt		= 0;
	path->readIndexFrac	   = 0.0;
	path->primed			  = false;
	path->filteredError	   = 0.0;
	path->integralError	   = 0.0;
	path->delayWriteIndex	 = 0;
	path->currentDelaySamples = atomic_load_explicit(&path->targetDelayMs, memory_order_relaxed) * 0.001 * outRate;
	path->delaySlewCoeff = 1.0 - exp(-1.0 / (0.050 * outRate));
	path->gainSlewCoeff  = (float)(1.0 - exp(-1.0 / (0.010 * outRate)));
	path->currentGain		 = atomic_load_explicit(&path->targetGain, memory_order_relaxed);
	path->crossoverStageCount = 0;
	path->appliedCrossoverMode = -1;
	path->appliedCrossoverSlopeDb = -1;
	path->appliedCrossoverFrequency = -1.0;
	path->appliedCrossoverSampleRate = -1.0;
	path->crossoverInvertPolarity = false;
	memset(path->crossoverBiquads, 0, sizeof(path->crossoverBiquads));
	path->eqActive = false;
	path->appliedEQRevision = 0;
	path->appliedEQSampleRate = -1.0;
	memset(path->eqBiquads, 0, sizeof(path->eqBiquads));
	memset(path->eqStageActive, 0, sizeof(path->eqStageActive));
	memset(path->eqStageTailFrames, 0, sizeof(path->eqStageTailFrames));
	atomic_store_explicit(&path->outputSampleRate, outRate, memory_order_relaxed);
	atomic_store_explicit(&path->outputChannelCount, outputChannels, memory_order_relaxed);
	atomic_store_explicit(&path->inputSampleRate, atomic_load_explicit(&gEngine.virtualSampleRate, memory_order_relaxed), memory_order_relaxed);

	AURenderCallbackStruct cb = {
		.inputProc	   = OutputAURenderCallback,
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

static NSString *const kPrefOutputAUID	 = @"OutputA_UID";
static NSString *const kPrefOutputBUID	 = @"OutputB_UID";
static NSString *const kPrefOutputAEnabled = @"OutputA_Enabled";
static NSString *const kPrefOutputBEnabled = @"OutputB_Enabled";
static NSString *const kPrefOutputAMono	= @"OutputA_Mono";
static NSString *const kPrefOutputBMono	= @"OutputB_Mono";
static NSString *const kPrefMasterVolume   = @"MasterVolume";
static NSString *const kPrefBalance		= @"Balance";
static NSString *const kPrefTrimDbA		= @"TrimDb_A";
static NSString *const kPrefTrimDbB		= @"TrimDb_B";
static NSString *const kPrefDelayMsA	   = @"DelayMs_A";
static NSString *const kPrefDelayMsB	   = @"DelayMs_B";
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
static NSString *const kPrefCrossoverSlope	 = @"Crossover_LinkedSlope";
static NSString *const kPrefCrossoverModeA	 = @"Crossover_ModeA";
static NSString *const kPrefCrossoverModeB	 = @"Crossover_ModeB";
static NSString *const kPrefCrossoverFrequencyA = @"Crossover_FrequencyA";
static NSString *const kPrefCrossoverFrequencyB = @"Crossover_FrequencyB";
static NSString *const kPrefCrossoverSlopeA	 = @"Crossover_SlopeA";
static NSString *const kPrefCrossoverSlopeB	 = @"Crossover_SlopeB";
static NSString *const kPrefGraphicEQEnabled	 = @"GraphicEQ_Enabled";
static NSString *const kPrefGraphicEQBandGainsDb = @"GraphicEQ_BandGainsDb";
static NSString *const kPrefSavedPresets   = @"SavedPresets_v1";
static NSString *const kPrefActivePreset   = @"ActivePresetName_v1";

static double CrossoverMagnitudeDb(CrossoverConfig config, double frequency, double sampleRate) {
	if (config.mode == kCrossoverFilterBypass) return 0.0;

	if (!isfinite(sampleRate) || sampleRate < 8000.0) sampleRate = 48000.0;
	float qs[kMaxCrossoverBiquads] = {0};
	int stages = GetLinkwitzRileyStageQs(config.slopeDb, qs);
	double omega = 2.0 * M_PI * frequency / sampleRate;
	double magnitude = 1.0;

	for (int i = 0; i < stages; ++i) {
		float b0, b1, b2, a1, a2;
		CalculateCrossoverBiquad(sampleRate, config.frequency, qs[i], config.mode,
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

static double GraphicEQMagnitudeDb(double frequency, double sampleRate) {
	if (!isfinite(sampleRate) || sampleRate < 8000.0) sampleRate = 48000.0;
	double omega   = 2.0 * M_PI * frequency / sampleRate;
	double cosine1 = cos(omega);
	double sine1   = sin(omega);
	double cosine2 = cos(2.0 * omega);
	double sine2   = sin(2.0 * omega);
	double magnitude = 1.0;

	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		float gainDb = ClampGraphicEQGainDb(atomic_load_explicit(&gEngine.graphicEQ.bandGainDb[i], memory_order_relaxed));
		if (fabsf(gainDb) < 0.01f) continue;
		float b0, b1, b2, a1, a2;
		CalculatePeakingEQBiquad(sampleRate, kGraphicEQFrequencies[i], gainDb, kGraphicEQOneThirdOctaveQ,
								 &b0, &b1, &b2, &a1, &a2);
		double realNum = b0 + b1 * cosine1 + b2 * cosine2;
		double imagNum = -b1 * sine1 - b2 * sine2;
		double realDen = 1.0 + a1 * cosine1 + a2 * cosine2;
		double imagDen = -a1 * sine1 - a2 * sine2;
		double num = realNum * realNum + imagNum * imagNum;
		double den = realDen * realDen + imagDen * imagDen;
		magnitude *= sqrt(num / fmax(den, 1.0e-20));
	}
	return 20.0 * log10(fmax(magnitude, 1.0e-5));
}

@interface GraphicEQBandSlider : NSSlider
@end

@implementation GraphicEQBandSlider
- (void)mouseDown:(NSEvent *)event {
	if (event.clickCount == 2 || (event.modifierFlags & NSEventModifierFlagOption) != 0) {
		self.floatValue = 0.0f;
		[NSApp sendAction:self.action to:self.target from:self];
		return;
	}
	[super mouseDown:event];
}
@end

@class GraphicEQPreviewView;
@protocol GraphicEQPreviewViewDelegate <NSObject>
- (void)graphicEQPreviewViewDidClick:(GraphicEQPreviewView *)view;
@end

@interface GraphicEQPreviewView : NSView
@property (nonatomic, weak) id<GraphicEQPreviewViewDelegate> eqDelegate;
@property (nonatomic, assign) BOOL clickable;
@end

@implementation GraphicEQPreviewView

- (BOOL)isOpaque { return NO; }

- (BOOL)usesDarkAppearance {
	NSString *appearance = [self.effectiveAppearance bestMatchFromAppearancesWithNames:@[
		NSAppearanceNameAqua, NSAppearanceNameDarkAqua
	]];
	return [appearance isEqualToString:NSAppearanceNameDarkAqua];
}

- (NSRect)plotRect {
	NSRect b = self.bounds;
	BOOL compact = NSHeight(b) < 100.0;
	CGFloat left = compact ? 30.0 : 36.0;
	CGFloat bottom = compact ? 12.0 : 19.0;
	CGFloat top = compact ? 5.0 : 15.0;
	return NSMakeRect(NSMinX(b) + left, NSMinY(b) + bottom,
					  NSWidth(b) - left - 10.0, NSHeight(b) - bottom - top);
}

- (double)frequencyForX:(CGFloat)x inPlot:(NSRect)plot {
	double norm = fmin(1.0, fmax(0.0, (x - NSMinX(plot)) / NSWidth(plot)));
	return 20.0 * pow(1000.0, norm);
}

- (CGFloat)xForFrequency:(double)freq inPlot:(NSRect)plot {
	freq = fmin(20000.0, fmax(20.0, freq));
	double norm = log10(freq / 20.0) / 3.0;
	return NSMinX(plot) + (CGFloat)norm * NSWidth(plot);
}

- (CGFloat)yForDb:(double)db inPlot:(NSRect)plot {
	db = fmin(15.0, fmax(-15.0, db));
	return NSMinY(plot) + (CGFloat)((db + 15.0) / 30.0) * NSHeight(plot);
}

- (void)resetCursorRects {
	[super resetCursorRects];
	if (self.clickable) {
		[self addCursorRect:self.bounds cursor:[NSCursor pointingHandCursor]];
	}
}

- (void)mouseDown:(NSEvent *)event {
	(void)event;
	if (self.clickable) {
		[self.eqDelegate graphicEQPreviewViewDidClick:self];
	}
}

- (void)drawRect:(NSRect)dirtyRect {
	(void)dirtyRect;
	BOOL dark = [self usesDarkAppearance];
	NSRect bounds = self.bounds;
	BOOL compact = NSHeight(bounds) < 100.0;
	NSRect plot = [self plotRect];
	if (NSWidth(plot) <= 0.0 || NSHeight(plot) <= 0.0) return;

	NSColor *gridColor = dark
		? [[NSColor whiteColor] colorWithAlphaComponent:0.115]
		: [[NSColor secondaryLabelColor] colorWithAlphaComponent:0.16];
	NSDictionary *axisAttrs = @{
		NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:compact ? 7.5 : 8.5 weight:NSFontWeightMedium],
		NSForegroundColorAttributeName: dark ? [[NSColor whiteColor] colorWithAlphaComponent:0.64] : [NSColor secondaryLabelColor]
	};

	const int dbMarks[] = { 12, 6, 0, -6, -12 };
	for (int i = 0; i < 5; ++i) {
		CGFloat y = [self yForDb:dbMarks[i] inPlot:plot];
		NSBezierPath *line = [NSBezierPath bezierPath];
		[line moveToPoint:NSMakePoint(NSMinX(plot), y)];
		[line lineToPoint:NSMakePoint(NSMaxX(plot), y)];
		[gridColor setStroke];
		line.lineWidth = 0.65;
		[line stroke];
		NSString *label = dbMarks[i] > 0 ? [NSString stringWithFormat:@"+%d", dbMarks[i]]
			: [NSString stringWithFormat:@"%d", dbMarks[i]];
		NSSize labelSize = [label sizeWithAttributes:axisAttrs];
		[label drawAtPoint:NSMakePoint(NSMinX(plot) - labelSize.width - 5.0, y - 4.0)
		 withAttributes:axisAttrs];
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
		line.lineWidth = primary ? 0.8 : 0.55;
		[(primary ? gridColor : [gridColor colorWithAlphaComponent:0.55]) setStroke];
		[line stroke];
		if (primary) {
			NSSize size = [axisLabels[i] sizeWithAttributes:axisAttrs];
			[axisLabels[i] drawAtPoint:NSMakePoint(x - size.width * 0.5, NSMinY(bounds) + 2.0)
						  withAttributes:axisAttrs];
		}
	}

	bool enabled = atomic_load_explicit(&gEngine.graphicEQ.enabled, memory_order_relaxed);
	NSColor *accent = enabled
		? (dark ? [NSColor colorWithSRGBRed:0.250 green:0.620 blue:0.880 alpha:1.0]
				: [NSColor colorWithSRGBRed:0.100 green:0.430 blue:0.720 alpha:1.0])
		: (dark ? [NSColor colorWithSRGBRed:0.620 green:0.660 blue:0.680 alpha:1.0]
				: [NSColor colorWithSRGBRed:0.430 green:0.460 blue:0.480 alpha:1.0]);

	CGFloat zeroY = [self yForDb:0.0 inPlot:plot];

	double sampleRate = atomic_load_explicit(&gEngine.virtualSampleRate, memory_order_relaxed);
	if (!isfinite(sampleRate) || sampleRate < 8000.0) sampleRate = 48000.0;

	NSBezierPath *curve = [NSBezierPath bezierPath];
	NSBezierPath *fill  = [NSBezierPath bezierPath];
	const NSInteger steps = 200;
	for (NSInteger i = 0; i <= steps; ++i) {
		CGFloat x = NSMinX(plot) + NSWidth(plot) * ((CGFloat)i / (CGFloat)steps);
		double f  = [self frequencyForX:x inPlot:plot];
		double db = GraphicEQMagnitudeDb(f, sampleRate);
		NSPoint pt = NSMakePoint(x, [self yForDb:db inPlot:plot]);
		if (i == 0) {
			[curve moveToPoint:pt];
			[fill moveToPoint:NSMakePoint(x, zeroY)];
			[fill lineToPoint:pt];
		} else {
			[curve lineToPoint:pt];
			[fill lineToPoint:pt];
		}
	}
	[fill lineToPoint:NSMakePoint(NSMaxX(plot), zeroY)];
	[fill closePath];

	NSGradient *gradient = [[NSGradient alloc] initWithStartingColor:[accent colorWithAlphaComponent:enabled ? 0.035 : 0.015]
										   endingColor:[accent colorWithAlphaComponent:enabled ? 0.22 : 0.075]];
	[gradient drawInBezierPath:fill angle:90.0];

	curve.lineWidth = 1.8;
	[[accent colorWithAlphaComponent:enabled ? 1.0 : 0.48] setStroke];
	[curve stroke];
}

@end

@implementation CrossoverResponseView

- (BOOL)isFlipped {
	return NO;
}

- (BOOL)isOpaque {
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
					  NSWidth(bounds) - left - 10.0, NSHeight(bounds) - bottom - top);
}

- (BOOL)usesDarkAppearance {
	NSString *appearance = [self.effectiveAppearance bestMatchFromAppearancesWithNames:@[
		NSAppearanceNameAqua, NSAppearanceNameDarkAqua
	]];
	return [appearance isEqualToString:NSAppearanceNameDarkAqua];
}

- (NSBezierPath *)responsePathForConfig:(CrossoverConfig)config
							  sampleRate:(double)sampleRate
								  inPlot:(NSRect)plot
								   start:(NSPoint *)outStart
									 end:(NSPoint *)outEnd {
	NSBezierPath *curve = [NSBezierPath bezierPath];
	const NSInteger steps = 220;
	const double floorDb = -48.0;
	BOOL drawing = NO;
	BOOL hasVisibleSegment = NO;
	double previousDb = 0.0;
	CGFloat previousX = 0.0;

	for (NSInteger i = 0; i <= steps; ++i) {
		CGFloat x = NSMinX(plot) + NSWidth(plot) * ((CGFloat)i / (CGFloat)steps);
		double frequency = [self frequencyForX:x inPlot:plot];
		double db = CrossoverMagnitudeDb(config, frequency, sampleRate);
		BOOL visible = db >= floorDb;

		if (i == 0 && visible) {
			NSPoint point = NSMakePoint(x, [self yForDb:db inPlot:plot]);
			[curve moveToPoint:point];
			if (outStart != NULL) *outStart = point;
			if (outEnd != NULL) *outEnd = point;
			drawing = YES;
			hasVisibleSegment = YES;
		} else if (i > 0 && visible && drawing) {
			NSPoint point = NSMakePoint(x, [self yForDb:db inPlot:plot]);
			[curve lineToPoint:point];
			if (outEnd != NULL) *outEnd = point;
		} else if (i > 0 && visible && !drawing) {
			CGFloat fraction = (floorDb - previousDb) / (db - previousDb);
			CGFloat crossingX = previousX + (x - previousX) * fraction;
			NSPoint crossing = NSMakePoint(crossingX, NSMinY(plot));
			NSPoint point = NSMakePoint(x, [self yForDb:db inPlot:plot]);
			[curve moveToPoint:crossing];
			[curve lineToPoint:point];
			if (!hasVisibleSegment && outStart != NULL) *outStart = crossing;
			if (outEnd != NULL) *outEnd = point;
			drawing = YES;
			hasVisibleSegment = YES;
		} else if (i > 0 && !visible && drawing) {
			CGFloat fraction = (floorDb - previousDb) / (db - previousDb);
			CGFloat crossingX = previousX + (x - previousX) * fraction;
			NSPoint crossing = NSMakePoint(crossingX, NSMinY(plot));
			[curve lineToPoint:crossing];
			if (outEnd != NULL) *outEnd = crossing;
			drawing = NO;
		}
		previousDb = db;
		previousX = x;
	}
	return curve;
}

- (void)drawResponseForConfig:(CrossoverConfig)config
						 color:(NSColor *)color
					 sampleRate:(double)sampleRate
						inPlot:(NSRect)plot
						 muted:(BOOL)muted {
	NSPoint start = NSZeroPoint;
	NSPoint end = NSZeroPoint;
	NSBezierPath *curve = [self responsePathForConfig:config sampleRate:sampleRate inPlot:plot start:&start end:&end];
	if (curve.elementCount == 0) return;
	NSBezierPath *fill = [curve copy];
	[fill lineToPoint:NSMakePoint(end.x, NSMinY(plot))];
	[fill lineToPoint:NSMakePoint(start.x, NSMinY(plot))];
	[fill closePath];
	NSGradient *gradient = [[NSGradient alloc] initWithStartingColor:[color colorWithAlphaComponent:muted ? 0.015 : 0.035]
														   endingColor:[color colorWithAlphaComponent:muted ? 0.075 : 0.22]];
	[gradient drawInBezierPath:fill angle:90.0];

	curve.lineWidth = 1.8;
	[[color colorWithAlphaComponent:muted ? 0.48 : 1.0] setStroke];
	[curve stroke];
}

- (void)drawMarkerForFrequency:(double)frequency
						  color:(NSColor *)color
						 inPlot:(NSRect)plot
						  label:(NSString *)label {
	CGFloat x = [self xForFrequency:frequency inPlot:plot];
	NSBezierPath *marker = [NSBezierPath bezierPath];
	[marker moveToPoint:NSMakePoint(x, NSMinY(plot))];
	[marker lineToPoint:NSMakePoint(x, NSMaxY(plot))];
	marker.lineWidth = 0.8;
	[[color colorWithAlphaComponent:0.9] setStroke];
	[marker stroke];

	CGFloat knobY = [self yForDb:-6.0 inPlot:plot];
	NSRect knobRect = NSMakeRect(x - 3.0, knobY - 3.0, 6.0, 6.0);
	[[color colorWithAlphaComponent:0.95] setFill];
	[[NSBezierPath bezierPathWithOvalInRect:knobRect] fill];
	[[NSColor controlBackgroundColor] setStroke];
	[[NSBezierPath bezierPathWithOvalInRect:NSInsetRect(knobRect, 0.5, 0.5)] stroke];

	if (label.length > 0) {
		BOOL dark = [self usesDarkAppearance];
		NSDictionary *attributes = @{
			NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:8.5 weight:NSFontWeightSemibold],
			NSForegroundColorAttributeName: dark ? [NSColor labelColor] : [NSColor secondaryLabelColor]
		};
		NSSize size = [label sizeWithAttributes:attributes];
		CGFloat labelX = fmin(NSMaxX(plot) - size.width - 5.0, fmax(NSMinX(plot) + 5.0, x + 7.0));
		CGFloat labelY = fmax(NSMinY(plot) + 2.0, knobY - size.height - 4.0);
		NSRect labelBackground = NSMakeRect(labelX - 3.0, labelY - 1.5, size.width + 6.0, size.height + 3.0);
		NSColor *labelBackgroundColor = dark
			? [[NSColor blackColor] colorWithAlphaComponent:0.74]
			: [[NSColor whiteColor] colorWithAlphaComponent:0.86];
		[labelBackgroundColor setFill];
		[[NSBezierPath bezierPathWithRoundedRect:labelBackground xRadius:2.0 yRadius:2.0] fill];
		[label drawAtPoint:NSMakePoint(labelX, labelY) withAttributes:attributes];
	}
}

- (void)drawRect:(NSRect)dirtyRect {
	(void)dirtyRect;
	BOOL dark = [self usesDarkAppearance];
	NSRect bounds = self.bounds;

	BOOL compact = NSHeight(bounds) < 100.0;
	NSRect plot = [self responsePlotRect];
	if (NSWidth(plot) <= 0.0 || NSHeight(plot) <= 0.0) return;

	NSColor *gridColor = dark
		? [[NSColor whiteColor] colorWithAlphaComponent:0.115]
		: [[NSColor secondaryLabelColor] colorWithAlphaComponent:0.16];
	NSDictionary *axisAttributes = @{
		NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:compact ? 7.5 : 8.5 weight:NSFontWeightMedium],
		NSForegroundColorAttributeName: dark ? [[NSColor whiteColor] colorWithAlphaComponent:0.64] : [NSColor secondaryLabelColor]
	};
	const int dbMarks[] = {0, -12, -24, -36, -48};
	for (int markIndex = 0; markIndex < 5; ++markIndex) {
		int db = dbMarks[markIndex];
		CGFloat y = [self yForDb:(double)db inPlot:plot];
		NSBezierPath *line = [NSBezierPath bezierPath];
		[line moveToPoint:NSMakePoint(NSMinX(plot), y)];
		[line lineToPoint:NSMakePoint(NSMaxX(plot), y)];
		[gridColor setStroke];
		line.lineWidth = 0.65;
		[line stroke];
		NSString *label = [NSString stringWithFormat:@"%d", db];
		NSSize labelSize = [label sizeWithAttributes:axisAttributes];
		[label drawAtPoint:NSMakePoint(NSMinX(plot) - labelSize.width - 5.0, y - 4.0)
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
		line.lineWidth = primary ? 0.8 : 0.55;
		[(primary ? gridColor : [gridColor colorWithAlphaComponent:0.55]) setStroke];
		[line stroke];
		if (primary) {
			NSSize size = [axisLabels[i] sizeWithAttributes:axisAttributes];
			[axisLabels[i] drawAtPoint:NSMakePoint(x - size.width * 0.5, NSMinY(bounds) + 2.0)
						  withAttributes:axisAttributes];
		}
	}

	CrossoverConfig configA = GetConfiguredCrossoverConfigForPath(true);
	CrossoverConfig configB = GetConfiguredCrossoverConfigForPath(false);
	double sampleRateA = atomic_load_explicit(&gEngine.pathA.outputSampleRate, memory_order_relaxed);
	double sampleRateB = atomic_load_explicit(&gEngine.pathB.outputSampleRate, memory_order_relaxed);
	double fallbackRate = atomic_load_explicit(&gEngine.virtualSampleRate, memory_order_relaxed);
	if (!isfinite(fallbackRate) || fallbackRate < 8000.0) fallbackRate = 48000.0;
	if (!isfinite(sampleRateA) || sampleRateA < 8000.0) sampleRateA = fallbackRate;
	if (!isfinite(sampleRateB) || sampleRateB < 8000.0) sampleRateB = fallbackRate;
	configA.frequency = ClampCrossoverFrequency(configA.frequency, sampleRateA);
	configB.frequency = ClampCrossoverFrequency(configB.frequency, sampleRateB);
	NSColor *highPassColor = dark
		? [NSColor colorWithSRGBRed:0.250 green:0.620 blue:0.880 alpha:1.0]
		: [NSColor colorWithSRGBRed:0.100 green:0.430 blue:0.720 alpha:1.0];
	NSColor *lowPassColor = dark
		? [NSColor colorWithSRGBRed:0.890 green:0.690 blue:0.330 alpha:1.0]
		: [NSColor colorWithSRGBRed:0.720 green:0.500 blue:0.100 alpha:1.0];
	NSColor *colorA = configA.mode == kCrossoverFilterHighPass ? highPassColor : lowPassColor;
	NSColor *colorB = configB.mode == kCrossoverFilterHighPass ? highPassColor : lowPassColor;
	bool enabled = atomic_load_explicit(&gEngine.crossover.enabled, memory_order_relaxed);
	NSColor *disabledCurveColor = dark
		? [NSColor colorWithSRGBRed:0.620 green:0.660 blue:0.680 alpha:1.0]
		: [NSColor colorWithSRGBRed:0.430 green:0.460 blue:0.480 alpha:1.0];
	if (configA.mode != kCrossoverFilterBypass) {
		[self drawResponseForConfig:configA
							   color:enabled ? colorA : disabledCurveColor
						  sampleRate:sampleRateA
							  inPlot:plot
							   muted:!enabled];
	}
	if (configB.mode != kCrossoverFilterBypass) {
		[self drawResponseForConfig:configB
							   color:enabled ? colorB : disabledCurveColor
						  sampleRate:sampleRateB
							  inPlot:plot
							   muted:!enabled];
	}

	bool linked = atomic_load_explicit(&gEngine.crossover.linked, memory_order_relaxed);
	if (linked) {
		NSString *label = [NSString stringWithFormat:@"%.0f Hz", configA.frequency];
		[self drawMarkerForFrequency:configA.frequency
							   color:enabled ? [NSColor whiteColor] : disabledCurveColor
							  inPlot:plot
							   label:label];
	} else {
		if (configA.mode != kCrossoverFilterBypass) {
			[self drawMarkerForFrequency:configA.frequency color:enabled ? colorA : disabledCurveColor inPlot:plot
								   label:[NSString stringWithFormat:@"%.0f Hz", configA.frequency]];
		}
		if (configB.mode != kCrossoverFilterBypass) {
			[self drawMarkerForFrequency:configB.frequency color:enabled ? colorB : disabledCurveColor inPlot:plot
								   label:[NSString stringWithFormat:@"%.0f Hz", configB.frequency]];
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

/* Stereo peak meter with a dBFS scale. The fill is logarithmic so quiet
 * program material remains readable, while the thin hold line preserves fast
 * transients that would otherwise disappear between UI updates. */
static const CGFloat kAudioMeterFloorDb = -60.0;
static const CFTimeInterval kAudioMeterPeakHoldSeconds = 1.5;
static const CGFloat kAudioMeterPeakDecayDbPerSecond = 24.0;

static inline CGFloat AudioMeterDbFS(float peak) {
	if (!isfinite(peak) || peak <= 0.0f) return kAudioMeterFloorDb;
	return fmin(0.0, fmax(kAudioMeterFloorDb, 20.0 * log10((double)peak)));
}

@interface AudioLevelMeterView : NSView
- (void)setPeakLevelL:(float)peakLevelL
		   peakLevelR:(float)peakLevelR
				 clipL:(BOOL)clipL
				 clipR:(BOOL)clipR;
@end

@implementation AudioLevelMeterView {
	CGFloat		_levelDbL;
	CGFloat		_levelDbR;
	CGFloat		_heldDbL;
	CGFloat		_heldDbR;
	CFTimeInterval _holdUntilL;
	CFTimeInterval _holdUntilR;
	CFTimeInterval _lastUpdateTime;
	BOOL		   _clipL;
	BOOL		   _clipR;
}

- (instancetype)initWithFrame:(NSRect)frame {
	self = [super initWithFrame:frame];
	if (self) {
		_levelDbL = kAudioMeterFloorDb;
		_levelDbR = kAudioMeterFloorDb;
		_heldDbL = kAudioMeterFloorDb;
		_heldDbR = kAudioMeterFloorDb;
		self.toolTip = @"Stereo dBFS peak meter. White ticks hold peaks for 1.5 seconds; red dots mark a clip event.";
	}
	return self;
}

- (BOOL)isOpaque {
	return NO;
}

- (void)updatePeakHoldForLevel:(CGFloat)levelDb
						 heldDb:(CGFloat *)heldDb
					  holdUntil:(CFTimeInterval *)holdUntil
							now:(CFTimeInterval)now
					elapsedTime:(CFTimeInterval)elapsedTime {
	if (levelDb >= *heldDb) {
		*heldDb = levelDb;
		*holdUntil = now + kAudioMeterPeakHoldSeconds;
	} else if (now > *holdUntil) {
		*heldDb = fmax(levelDb, *heldDb - kAudioMeterPeakDecayDbPerSecond * elapsedTime);
	}
}

- (void)setPeakLevelL:(float)peakLevelL
		   peakLevelR:(float)peakLevelR
				 clipL:(BOOL)clipL
				 clipR:(BOOL)clipR {
	CFTimeInterval now = CFAbsoluteTimeGetCurrent();
	CFTimeInterval elapsedTime = _lastUpdateTime > 0.0 ? fmax(0.0, now - _lastUpdateTime) : 0.0;
	_lastUpdateTime = now;

	_levelDbL = AudioMeterDbFS(peakLevelL);
	_levelDbR = AudioMeterDbFS(peakLevelR);
	[self updatePeakHoldForLevel:_levelDbL heldDb:&_heldDbL holdUntil:&_holdUntilL now:now elapsedTime:elapsedTime];
	[self updatePeakHoldForLevel:_levelDbR heldDb:&_heldDbR holdUntil:&_holdUntilR now:now elapsedTime:elapsedTime];

	/* A clip event is latched in the audio callback so one-buffer overloads
	 * remain visible even when they occur between telemetry refreshes. */
	_clipL = clipL || peakLevelL >= 0.999f;
	_clipR = clipR || peakLevelR >= 0.999f;
	[self setNeedsDisplay:YES];
}

- (void)drawMeterRow:(NSRect)row
				label:(NSString *)label
			  levelDb:(CGFloat)levelDb
			   heldDb:(CGFloat)heldDb
				 clip:(BOOL)clip {
	CGFloat labelWidth = NSWidth(self.bounds) >= 56.0 ? 10.0 : 0.0;
	if (labelWidth > 0.0) {
		NSDictionary *labelAttributes = @{
			NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:7.0 weight:NSFontWeightSemibold],
			NSForegroundColorAttributeName: [NSColor secondaryLabelColor]
		};
		NSSize labelSize = [label sizeWithAttributes:labelAttributes];
		[label drawAtPoint:NSMakePoint(NSMinX(row), NSMidY(row) - labelSize.height * 0.5)
			withAttributes:labelAttributes];
	}

	NSRect track = NSMakeRect(NSMinX(row) + labelWidth, NSMinY(row),
							  NSWidth(row) - labelWidth, NSHeight(row));
	if (NSWidth(track) <= 1.0 || NSHeight(track) <= 1.0) return;

	CGFloat radius = fmin(2.0, NSHeight(track) * 0.5);
	NSBezierPath *trackPath = [NSBezierPath bezierPathWithRoundedRect:track xRadius:radius yRadius:radius];
	[[[NSColor blackColor] colorWithAlphaComponent:0.48] setFill];
	[trackPath fill];

	CGFloat normalizedLevel = (levelDb - kAudioMeterFloorDb) / -kAudioMeterFloorDb;
	normalizedLevel = fmin(1.0, fmax(0.0, normalizedLevel));
	if (normalizedLevel > 0.0) {
		NSRect fill = NSMakeRect(NSMinX(track), NSMinY(track), NSWidth(track) * normalizedLevel, NSHeight(track));
		NSGradient *gradient = [[NSGradient alloc] initWithColorsAndLocations:
			[NSColor colorWithCalibratedRed:0.10 green:0.72 blue:0.29 alpha:1.0], 0.0,
			[NSColor colorWithCalibratedRed:0.22 green:0.88 blue:0.36 alpha:1.0], 0.80,
			[NSColor colorWithCalibratedRed:0.98 green:0.82 blue:0.10 alpha:1.0], 0.8001,
			[NSColor colorWithCalibratedRed:0.98 green:0.77 blue:0.12 alpha:1.0], 0.95,
			[NSColor colorWithCalibratedRed:0.93 green:0.19 blue:0.16 alpha:1.0], 0.9501,
			[NSColor colorWithCalibratedRed:0.98 green:0.25 blue:0.20 alpha:1.0], 1.0,
			nil];
		[NSGraphicsContext saveGraphicsState];
		[trackPath addClip];
		[[NSBezierPath bezierPathWithRect:fill] addClip];
		[gradient drawInRect:track angle:0.0];
		[NSGraphicsContext restoreGraphicsState];
	}

	/* -12 and -3 dBFS boundaries remain visible even where the fill has not
	 * reached them, making the green/yellow/red ranges easy to read. */
	[[NSColor whiteColor] setStroke];
	for (CGFloat boundary = 0.80; boundary < 1.0; boundary += 0.15) {
		CGFloat x = NSMinX(track) + NSWidth(track) * boundary;
		NSBezierPath *line = [NSBezierPath bezierPath];
		[line moveToPoint:NSMakePoint(x, NSMinY(track) + 0.5)];
		[line lineToPoint:NSMakePoint(x, NSMaxY(track) - 0.5)];
		line.lineWidth = 0.5;
		[[[NSColor whiteColor] colorWithAlphaComponent:0.25] setStroke];
		[line stroke];
	}

	CGFloat normalizedHold = (heldDb - kAudioMeterFloorDb) / -kAudioMeterFloorDb;
	normalizedHold = fmin(1.0, fmax(0.0, normalizedHold));
	CGFloat holdX = NSMinX(track) + NSWidth(track) * normalizedHold;
	NSBezierPath *holdLine = [NSBezierPath bezierPath];
	[holdLine moveToPoint:NSMakePoint(holdX, NSMinY(track))];
	[holdLine lineToPoint:NSMakePoint(holdX, NSMaxY(track))];
	[[NSColor whiteColor] setStroke];
	holdLine.lineWidth = 1.0;
	[holdLine stroke];

	[[NSColor separatorColor] setStroke];
	trackPath.lineWidth = 0.5;
	[trackPath stroke];

	if (clip) {
		CGFloat diameter = fmin(4.5, NSHeight(track) - 1.0);
		NSRect dot = NSMakeRect(NSMaxX(track) - diameter - 1.0,
								NSMidY(track) - diameter * 0.5,
								diameter, diameter);
		NSBezierPath *dotPath = [NSBezierPath bezierPathWithOvalInRect:dot];
		[[NSColor systemRedColor] setFill];
		[dotPath fill];
		[[[NSColor whiteColor] colorWithAlphaComponent:0.70] setStroke];
		dotPath.lineWidth = 0.5;
		[dotPath stroke];
	}
}

- (void)drawRect:(NSRect)dirtyRect {
	(void)dirtyRect;
	NSRect bounds = NSInsetRect(self.bounds, 0.5, 0.5);
	CGFloat rowGap = NSHeight(bounds) >= 12.0 ? 2.0 : 1.0;
	CGFloat verticalInset = 1.0;
	CGFloat rowHeight = (NSHeight(bounds) - rowGap - verticalInset * 2.0) * 0.5;
	if (rowHeight <= 0.0) return;

	NSRect rowL = NSMakeRect(NSMinX(bounds), NSMaxY(bounds) - verticalInset - rowHeight,
							 NSWidth(bounds), rowHeight);
	NSRect rowR = NSMakeRect(NSMinX(bounds), NSMinY(bounds) + verticalInset,
							 NSWidth(bounds), rowHeight);
	[self drawMeterRow:rowL label:@"L" levelDb:_levelDbL heldDb:_heldDbL clip:_clipL];
	[self drawMeterRow:rowR label:@"R" levelDb:_levelDbR heldDb:_heldDbR clip:_clipR];
}

@end

@class RoutingMatrixView;
@protocol RoutingMatrixViewDelegate <NSObject>
- (void)routingMatrixView:(RoutingMatrixView *)view
		 didToggleInputLeft:(BOOL)inputLeft
		 destinationChannel:(uint32_t)destinationChannel;
- (void)routingMatrixView:(RoutingMatrixView *)view
	didRequestDisableInputLeft:(BOOL)inputLeft
		 destinationChannel:(uint32_t)destinationChannel;
- (void)routingMatrixView:(RoutingMatrixView *)view
	 didRequestToggleInputLeft:(BOOL)inputLeft
		destinationChannel:(uint32_t)destinationChannel;
@end

/* QLab-style crosspoint matrix. Rows are the stereo source inputs and columns
 * are the active destination device's hardware channels. */
@interface RoutingMatrixView : NSView
@property (nonatomic, weak) id<RoutingMatrixViewDelegate> routingDelegate;
@property (nonatomic, assign) BOOL outputA;
- (NSRect)rectForInputLeft:(BOOL)inputLeft channel:(uint32_t)channel;
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

- (BOOL)isFlipped {
	return YES;
}

- (CGFloat)headerHeight {
	return 15.0;
}

- (CGFloat)rowHeightForMono:(BOOL)mono {
	return (NSHeight(self.bounds) - self.headerHeight) / (mono ? 1.0 : 2.0);
}

- (NSRect)rectForInputLeft:(BOOL)inputLeft channel:(uint32_t)channel {
	uint32_t channelCount = [self channelCount];
	if (channel >= channelCount) return NSZeroRect;

	BOOL mono = atomic_load_explicit(&self.routingPath->mono, memory_order_relaxed);
	CGFloat rowLabelWidth = 23.0;
	CGFloat rowHeight = [self rowHeightForMono:mono];
	CGFloat columnWidth = (NSWidth(self.bounds) - rowLabelWidth - 3.0) / (CGFloat)channelCount;
	CGFloat y = self.headerHeight + ((mono || inputLeft) ? 0.0 : rowHeight);
	CGFloat verticalInset = 1.5;
	return NSMakeRect(rowLabelWidth + (CGFloat)channel * columnWidth + 1.0,
					  y + verticalInset,
					  columnWidth - 2.0,
					  rowHeight - 2.0 * verticalInset);
}

- (void)drawString:(NSString *)string centeredInRect:(NSRect)rect attributes:(NSDictionary<NSAttributedStringKey, id> *)attributes {
	NSSize size = [string sizeWithAttributes:attributes];
	[string drawAtPoint:NSMakePoint(NSMidX(rect) - size.width * 0.5,
								   NSMidY(rect) - size.height * 0.5)
		withAttributes:attributes];
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
	CGFloat headerHeight = self.headerHeight;
	CGFloat rowHeight = [self rowHeightForMono:mono];
	CGFloat columnWidth = (NSWidth(bounds) - rowLabelWidth - 3.0) / (CGFloat)channelCount;
	NSDictionary *labelAttributes = @{
		NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:9.0 weight:NSFontWeightMedium],
		NSForegroundColorAttributeName: [NSColor secondaryLabelColor]
	};
	[self drawString:@"IN"
	  centeredInRect:NSMakeRect(0.0, 0.0, rowLabelWidth, headerHeight)
		  attributes:labelAttributes];

	CGFloat gainFontSize = 9.5;
	if (channelCount > 8U) {
		gainFontSize = MAX(7.5, 9.5 - 0.25 * (CGFloat)(channelCount - 8U));
	}
	NSMutableDictionary<NSAttributedStringKey, id> *activeAttributes = [@{
		NSFontAttributeName: [NSFont monospacedDigitSystemFontOfSize:gainFontSize weight:NSFontWeightSemibold],
		NSForegroundColorAttributeName: [NSColor controlAccentColor]
	} mutableCopy];
	/* At 16 channels, a four-character label is wider than its 17.75pt column
	 * at 7.5pt. Tighten its tracking very slightly to preserve the requested
	 * minimum font size without letting labels overlap adjacent crosspoints. */
	if (channelCount == kMaxOutputChannels) {
		activeAttributes[NSKernAttributeName] = @(-0.9);
	}

	for (uint32_t channel = 0; channel < channelCount; ++channel) {
		CGFloat x = rowLabelWidth + (CGFloat)channel * columnWidth;
		NSString *number = [NSString stringWithFormat:@"%u", channel + 1U];
		[self drawString:number
		  centeredInRect:NSMakeRect(x, 0.0, columnWidth, headerHeight)
			  attributes:labelAttributes];

		uint32_t rowCount = mono ? 1U : 2U;
		for (uint32_t row = 0; row < rowCount; ++row) {
			BOOL leftInput = mono || row == 0U;
			NSRect cell = [self rectForInputLeft:leftInput channel:channel];
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
				NSString *gainText = fabs(db) < 0.05 ? @"0.0"
									: (fabs(db) < 10.0 ? [NSString stringWithFormat:@"%+.1f", db]
													   : [NSString stringWithFormat:@"%+.0f", db]);
				[self drawString:gainText centeredInRect:cell attributes:activeAttributes];
			}
		}
	}

	if (mono) {
		[self drawString:@"M"
		  centeredInRect:NSMakeRect(0.0, headerHeight, rowLabelWidth, rowHeight)
			  attributes:labelAttributes];
	} else {
		[self drawString:@"L"
		  centeredInRect:NSMakeRect(0.0, headerHeight, rowLabelWidth, rowHeight)
			  attributes:labelAttributes];
		[self drawString:@"R"
		  centeredInRect:NSMakeRect(0.0, headerHeight + rowHeight, rowLabelWidth, rowHeight)
			  attributes:labelAttributes];
	}
}

- (void)handleMouseDown:(NSEvent *)event rightClick:(BOOL)rightClick {
	NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
	uint32_t channelCount = [self channelCount];
	BOOL mono = atomic_load_explicit(&self.routingPath->mono, memory_order_relaxed);
	CGFloat rowLabelWidth = 23.0;
	CGFloat headerHeight = self.headerHeight;
	CGFloat rowHeight = [self rowHeightForMono:mono];
	if (point.x < rowLabelWidth || point.y < headerHeight || point.y >= NSHeight(self.bounds)) return;
	CGFloat columnWidth = (NSWidth(self.bounds) - rowLabelWidth - 3.0) / (CGFloat)channelCount;
	uint32_t channel = (uint32_t)((point.x - rowLabelWidth) / columnWidth);
	if (channel >= channelCount) return;
	BOOL inputLeft = mono || point.y < headerHeight + rowHeight;
	BOOL commandClick = (event.modifierFlags & NSEventModifierFlagCommand) != 0;
	if (commandClick) {
		[self.routingDelegate routingMatrixView:self
					 didRequestDisableInputLeft:inputLeft
							  destinationChannel:channel];
		return;
	}
	if (rightClick) {
		[self.routingDelegate routingMatrixView:self
			  didRequestToggleInputLeft:inputLeft
					   destinationChannel:channel];
		return;
	}
	[self.routingDelegate routingMatrixView:self
						   didToggleInputLeft:inputLeft
							destinationChannel:channel];
}

- (void)mouseDown:(NSEvent *)event {
	[self handleMouseDown:event rightClick:NO];
}

- (void)rightMouseDown:(NSEvent *)event {
	[self handleMouseDown:event rightClick:YES];
}

@end

@interface RoutingGainPopoverContentView : NSView
@property (nonatomic, weak) NSSlider *routingGainFader;
@end

@implementation RoutingGainPopoverContentView

- (void)scrollWheel:(NSEvent *)event {
	NSSlider *fader = self.routingGainFader;
	if (fader == nil) return;

	CGFloat delta = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY : event.deltaY;
	if (delta == 0.0) return;
	double step = event.hasPreciseScrollingDeltas ? 0.1 : 1.0;
	fader.doubleValue = fmin(fader.maxValue, fmax(fader.minValue, fader.doubleValue + delta * step));
	[NSApp sendAction:fader.action to:fader.target from:fader];
}

@end

/* Rounded gain fields are intentionally compact. The standard cell reserves
 * more horizontal inset than the short signed value needs. */
@interface CompactRoundedTextFieldCell : NSTextFieldCell
@end

@implementation CompactRoundedTextFieldCell

- (NSRect)drawingRectForBounds:(NSRect)rect {
	return NSInsetRect([super drawingRectForBounds:rect], -3.0, 0.0);
}

@end

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate, CrossoverResponseViewDelegate, RoutingMatrixViewDelegate, GraphicEQPreviewViewDelegate>
@property (nonatomic, strong) NSWindow		  *window;
@property (nonatomic, strong) NSScrollView	  *controllerScrollView;

/* Menu Bar Status Item */
@property (nonatomic, strong) NSStatusItem	  *statusItem;
@property (nonatomic, strong) NSMenuItem		*statusMenuHeaderItem;
@property (nonatomic, strong) NSMenuItem		*statusMenuPresetsItem;
@property (nonatomic, strong) NSMenuItem		*statusMenuEnableAItem;
@property (nonatomic, strong) NSMenuItem		*statusMenuMonoAItem;
@property (nonatomic, strong) NSMenuItem		*statusMenuEnableBItem;
@property (nonatomic, strong) NSMenuItem		*statusMenuMonoBItem;

/* Virtual Device Banner */
@property (nonatomic, strong) NSTextField	   *virtualStatusLabel;
@property (nonatomic, strong) AudioLevelMeterView *inputMeter;
@property (nonatomic, strong) NSButton		  *setSystemDefaultButton;

/* Presets Controls */
@property (nonatomic, strong) NSPopUpButton	 *presetPopup;
@property (nonatomic, strong) NSMenuItem		*presetPopupIndicatorItem;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSDictionary *> *savedPresets;
@property (nonatomic, copy)   NSString		  *activePresetName;

/* Settings changes are kept together in one document-style undo history. */
@property (nonatomic, strong) NSUndoManager	  *settingsUndoManager;

/* Output A Controls */
@property (nonatomic, strong) NSPopUpButton	 *popupA;
@property (nonatomic, strong) NSButton		  *enableCheckboxA;
@property (nonatomic, strong) NSButton		  *monoCheckboxA;
@property (nonatomic, strong) NSTextField	   *statusLabelA;
@property (nonatomic, strong) NSSlider		  *trimSliderA;
@property (nonatomic, strong) NSTextField	   *trimValueLabelA;
@property (nonatomic, strong) id				gainFieldFocusMonitor;
@property (nonatomic, strong) NSSlider		  *delaySliderA;
@property (nonatomic, strong) NSStepper		 *delayStepperA;
@property (nonatomic, strong) NSTextField	   *delayValueLabelA;
@property (nonatomic, strong) AudioLevelMeterView *meterA;
@property (nonatomic, strong) RoutingMatrixView *routingMatrixA;

/* Output B Controls */
@property (nonatomic, strong) NSPopUpButton	 *popupB;
@property (nonatomic, strong) NSButton		  *enableCheckboxB;
@property (nonatomic, strong) NSButton		  *monoCheckboxB;
@property (nonatomic, strong) NSTextField	   *statusLabelB;
@property (nonatomic, strong) NSSlider		  *trimSliderB;
@property (nonatomic, strong) NSTextField	   *trimValueLabelB;
@property (nonatomic, strong) NSSlider		  *delaySliderB;
@property (nonatomic, strong) NSStepper		 *delayStepperB;
@property (nonatomic, strong) NSTextField	   *delayValueLabelB;
@property (nonatomic, strong) AudioLevelMeterView *meterB;
@property (nonatomic, strong) RoutingMatrixView *routingMatrixB;
@property (nonatomic, strong) NSPopover		 *routingGainPopover;
@property (nonatomic, strong) NSTextField	   *routingGainPopoverTitleLabel;
@property (nonatomic, strong) NSSlider		  *routingGainFader;
@property (nonatomic, strong) NSTextField	   *routingGainValueField;
@property (nonatomic, weak)   RoutingMatrixView *activeRoutingMatrix;
@property (nonatomic, assign) BOOL			  activeRoutingInputLeft;
@property (nonatomic, assign) BOOL			  activeRoutingMono;
@property (nonatomic, assign) uint32_t		  activeRoutingChannel;

/* Master & Balance Controls */
@property (nonatomic, strong) NSSlider		  *masterSlider;
@property (nonatomic, strong) NSTextField	   *masterValueLabel;
@property (nonatomic, strong) NSSlider		  *balanceSlider;
@property (nonatomic, strong) NSTextField	   *balanceValueLabel;
@property (nonatomic, strong) NSTextField	   *errorBannerLabel;
@property (nonatomic, strong) NSView			*statusBannerBadge;
@property (nonatomic, strong) NSTimer		   *statusBannerClearTimer;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSString *> *pinnedStatusIssues;
@property (nonatomic, strong) NSMutableOrderedSet<NSString *> *pinnedStatusIssueOrder;
@property (nonatomic, assign) NSUInteger		statusBannerGeneration;

/* 31-Band Graphic EQ Controls (Main Preview + Pop-Out Editor Window) */
@property (nonatomic, strong) NSButton			  *graphicEQEnabledCheckbox;
@property (nonatomic, strong) NSTextField		   *graphicEQSummaryLabel;
@property (nonatomic, strong) GraphicEQPreviewView  *graphicEQPreviewView;
@property (nonatomic, strong) NSWindow			  *graphicEQWindow;
@property (nonatomic, strong) NSButton			  *graphicEQWindowEnabledCheckbox;
@property (nonatomic, strong) GraphicEQPreviewView  *graphicEQWindowPreviewView;
@property (nonatomic, strong) NSArray<NSSlider *>   *graphicEQSliders;
@property (nonatomic, strong) NSArray<NSTextField *> *graphicEQValueLabels;

/* Crossover Controls */
@property (nonatomic, strong) NSButton		  *crossoverEnabledCheckbox;
@property (nonatomic, strong) NSButton		  *crossoverLinkedCheckbox;
@property (nonatomic, strong) NSSlider		  *linkedCrossoverFrequencySlider;
@property (nonatomic, strong) NSSlider		  *linkedCrossoverSlopeSlider;
@property (nonatomic, strong) NSPopUpButton	 *linkedCrossoverAssignmentPopup;
@property (nonatomic, strong) NSTextField	   *linkedCrossoverFrequencyLabel;
@property (nonatomic, strong) NSTextField	   *linkedCrossoverSlopeLabel;
@property (nonatomic, strong) NSPopUpButton	 *crossoverModePopupA;
@property (nonatomic, strong) NSPopUpButton	 *crossoverModePopupB;
@property (nonatomic, strong) NSSlider		  *crossoverFrequencySliderA;
@property (nonatomic, strong) NSSlider		  *crossoverFrequencySliderB;
@property (nonatomic, strong) NSSlider		  *crossoverSlopeSliderA;
@property (nonatomic, strong) NSSlider		  *crossoverSlopeSliderB;
@property (nonatomic, strong) NSTextField	   *crossoverFrequencyLabelA;
@property (nonatomic, strong) NSTextField	   *crossoverFrequencyLabelB;
@property (nonatomic, strong) NSTextField	   *crossoverSlopeLabelA;
@property (nonatomic, strong) NSTextField	   *crossoverSlopeLabelB;
@property (nonatomic, strong) NSArray<NSTextField *> *crossoverPrimaryLabels;
@property (nonatomic, strong) NSArray<NSView *> *linkedCrossoverControls;
@property (nonatomic, strong) NSArray<NSView *> *independentCrossoverControls;
@property (nonatomic, strong) CrossoverResponseView *crossoverResponseView;
@property (nonatomic, assign) double			displayedCrossoverSampleRateA;
@property (nonatomic, assign) double			displayedCrossoverSampleRateB;

@property (nonatomic, strong) NSArray<AudioDeviceItem *> *availableDevices;
@property (nonatomic, copy)   NSString		  *desiredUID_A;
@property (nonatomic, copy)   NSString		  *desiredUID_B;
@property (nonatomic, strong) NSTimer		   *telemetryTimer;
@end

@implementation AppDelegate

- (NSTextField *)makeLabel:(NSString *)text bold:(BOOL)bold size:(CGFloat)size color:(NSColor *)color {
	NSTextField *tf = [[NSTextField alloc] initWithFrame:NSZeroRect];
	tf.stringValue	 = text ?: @"";
	tf.bezeled		 = NO;
	tf.drawsBackground = NO;
	tf.editable		= NO;
	tf.selectable	  = NO;
	tf.font			= bold ? [NSFont boldSystemFontOfSize:size] : [NSFont systemFontOfSize:size];
	if (color) tf.textColor = color;
	return tf;
}

- (void)showDefaultStatusHintAnimated:(BOOL)animated {
	[self.statusBannerClearTimer invalidate];
	self.statusBannerClearTimer = nil;
	NSUInteger generation = ++self.statusBannerGeneration;

	void (^showHint)(void) = ^{
		if (self.statusBannerGeneration != generation) return;
		self.statusBannerBadge.hidden = YES;
		self.statusBannerBadge.alphaValue = 1.0;
		self.errorBannerLabel.stringValue = kDefaultStatusHint;
		self.errorBannerLabel.textColor = [NSColor secondaryLabelColor];
		self.errorBannerLabel.alphaValue = 1.0;
	};

	if (!animated) {
		showHint();
		return;
	}

	[NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
		context.duration = 0.20;
		self.statusBannerBadge.animator.alphaValue = 0.0;
		self.errorBannerLabel.animator.alphaValue = 0.0;
	} completionHandler:showHint];
}

- (void)autoClearStatusMessage:(NSTimer *)timer {
	if (timer != self.statusBannerClearTimer || self.pinnedStatusIssues.count > 0) return;
	self.statusBannerClearTimer = nil;
	[self showDefaultStatusHintAnimated:YES];
}

- (void)showStatusMessage:(NSString *)msg
					 level:(StatusLevel)level
			autoClearAfter:(NSTimeInterval)seconds {
	if (msg.length == 0) {
		[self showDefaultStatusHintAnimated:NO];
		return;
	}

	/* Hardware and permission failures take precedence over routine notices
	 * until the specific condition which caused them has been resolved. */
	if (self.pinnedStatusIssues.count > 0 && level != StatusLevelError) return;

	[self.statusBannerClearTimer invalidate];
	self.statusBannerClearTimer = nil;
	self.statusBannerGeneration += 1;

	NSColor *badgeColor = nil;
	NSColor *textColor = [NSColor labelColor];
	NSString *displayMessage = msg;
	switch (level) {
		case StatusLevelSuccess:
			badgeColor = [[NSColor systemGreenColor] colorWithAlphaComponent:0.14];
			break;
		case StatusLevelWarning:
			badgeColor = [[NSColor systemOrangeColor] colorWithAlphaComponent:0.16];
			textColor = [NSColor systemOrangeColor];
			break;
		case StatusLevelError:
			badgeColor = [[NSColor systemRedColor] colorWithAlphaComponent:0.16];
			textColor = [NSColor systemRedColor];
			displayMessage = [@"⚠︎  " stringByAppendingString:msg];
			break;
		case StatusLevelInfo:
			badgeColor = [[NSColor systemBlueColor] colorWithAlphaComponent:0.12];
			break;
		case StatusLevelHint:
			[self showDefaultStatusHintAnimated:NO];
			return;
	}

	self.statusBannerBadge.hidden = NO;
	self.statusBannerBadge.layer.backgroundColor = badgeColor.CGColor;
	self.statusBannerBadge.alphaValue = 1.0;
	self.errorBannerLabel.alphaValue = 1.0;
	self.errorBannerLabel.textColor = textColor;
	self.errorBannerLabel.stringValue = displayMessage;

	if (level != StatusLevelError && seconds > 0.0) {
		self.statusBannerClearTimer = [NSTimer scheduledTimerWithTimeInterval:seconds
																		 target:self
																	   selector:@selector(autoClearStatusMessage:)
																	   userInfo:nil
																		repeats:NO];
	}
}

- (void)showPinnedStatusMessage:(NSString *)msg issue:(NSString *)issue {
	if (issue.length == 0) return;
	if (self.pinnedStatusIssues == nil) self.pinnedStatusIssues = [NSMutableDictionary dictionary];
	if (self.pinnedStatusIssueOrder == nil) self.pinnedStatusIssueOrder = [NSMutableOrderedSet orderedSet];
	self.pinnedStatusIssues[issue] = msg;
	[self.pinnedStatusIssueOrder removeObject:issue];
	[self.pinnedStatusIssueOrder addObject:issue];
	[self showStatusMessage:msg level:StatusLevelError autoClearAfter:0.0];
}

- (void)resolvePinnedStatusIssue:(NSString *)issue {
	if (issue.length == 0 || self.pinnedStatusIssues[issue] == nil) return;
	[self.pinnedStatusIssues removeObjectForKey:issue];
	[self.pinnedStatusIssueOrder removeObject:issue];

	NSString *remainingIssue = self.pinnedStatusIssueOrder.lastObject;
	NSString *remainingMessage = self.pinnedStatusIssues[remainingIssue];
	if (remainingMessage.length > 0) {
		[self showStatusMessage:remainingMessage level:StatusLevelError autoClearAfter:0.0];
	} else {
		[self showDefaultStatusHintAnimated:NO];
	}
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
	(void)notification;
	memset(&gEngine, 0, sizeof(gEngine));
	gEngine.virtualDeviceID = kAudioObjectUnknown;
	gEngine.pathA.deviceID  = kAudioObjectUnknown;
	gEngine.pathB.deviceID  = kAudioObjectUnknown;
	self.settingsUndoManager = [[NSUndoManager alloc] init];
	self.settingsUndoManager.levelsOfUndo = 100;
	self.settingsUndoManager.groupsByEvent = YES;

	[self loadPreferences];
	[self setupMainMenu];
	[self setupStatusBarItem];
	[self buildWindowAndUI];
	[self rebuildPresetsUI];
	[self registerCoreAudioListeners];
	SetVirtualDevicePresence(YES);
	[self requestMicrophonePermissionAndStartAudio];
	[self startTelemetryUpdates];
}

- (void)applicationWillTerminate:(NSNotification *)notification {
	(void)notification;
	if (self.gainFieldFocusMonitor != nil) {
		[NSEvent removeMonitor:self.gainFieldFocusMonitor];
		self.gainFieldFocusMonitor = nil;
	}
	[self stopTelemetryUpdates];
	[self.statusBannerClearTimer invalidate];
	StopOutputPath(&gEngine.pathA);
	StopOutputPath(&gEngine.pathB);
	StopVirtualInputUnit();
	SetVirtualDevicePresence(NO);
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

/* Telemetry drives only visual state. Stop both its 60 ms timer and the peak
 * calculations in the real-time callbacks while the controller is not on
 * screen; the audio routing and all device listeners continue to run. */
- (void)resetMeteringData {
	atomic_store_explicit(&gEngine.peakLevelL, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&gEngine.peakLevelR, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&gEngine.clippedL, false, memory_order_relaxed);
	atomic_store_explicit(&gEngine.clippedR, false, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathA.peakLevelL, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathA.peakLevelR, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathA.clippedL, false, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathA.clippedR, false, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathB.peakLevelL, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathB.peakLevelR, 0.0f, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathB.clippedL, false, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathB.clippedR, false, memory_order_relaxed);
}

- (void)startTelemetryUpdates {
	if (self.telemetryTimer != nil) return;
	[self resetMeteringData];
	atomic_store_explicit(&gEngine.meteringEnabled, true, memory_order_relaxed);
	[self updateTelemetryUI];
	self.telemetryTimer = [NSTimer scheduledTimerWithTimeInterval:0.06
											   target:self
											 selector:@selector(updateTelemetryUI)
											 userInfo:nil
											  repeats:YES];
}

- (void)stopTelemetryUpdates {
	atomic_store_explicit(&gEngine.meteringEnabled, false, memory_order_relaxed);
	[self.telemetryTimer invalidate];
	self.telemetryTimer = nil;
}

/* AppKit may leave a window object alive while it is hidden, minimized, or
 * its application is hidden. Treat each of those states like a closed GUI. */
- (void)applicationDidHide:(NSNotification *)notification {
	(void)notification;
	[self stopTelemetryUpdates];
}

- (void)applicationDidUnhide:(NSNotification *)notification {
	(void)notification;
	if (self.window.isVisible && !self.window.isMiniaturized) {
		[self startTelemetryUpdates];
	}
}

- (void)windowDidMiniaturize:(NSNotification *)notification {
	if (notification.object == self.window) {
		[self stopTelemetryUpdates];
	}
}

- (void)windowDidDeminiaturize:(NSNotification *)notification {
	if (notification.object == self.window && ![NSApp isHidden]) {
		[self startTelemetryUpdates];
	}
}

/* Intercept red window close button & Cmd+W: hide window & stay active in Menu Bar */
- (BOOL)windowShouldClose:(NSWindow *)sender {
	if (sender != self.window) return YES;
	[self stopTelemetryUpdates];
	[self.window orderOut:nil];
	[NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
	return NO;
}

/* Supplying the window's undo manager lets the standard responder-chain
 * actions serve both editable text fields and the controller settings. */
- (NSUndoManager *)windowWillReturnUndoManager:(NSWindow *)window {
	(void)window;
	return self.settingsUndoManager;
}

- (void)showControllerWindow:(id)sender {
	(void)sender;
	BOOL wasExplicitlyHidden = !self.window.isVisible;
	[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
	if (wasExplicitlyHidden) {
		/* Move a restored controller to the current Space once, then return
		 * to the default behavior so later activation leaves it pinned there. */
		self.window.collectionBehavior = NSWindowCollectionBehaviorMoveToActiveSpace;
	}
	[self.window makeKeyAndOrderFront:nil];
	[NSApp activateIgnoringOtherApps:YES];
	[self startTelemetryUpdates];
	if (wasExplicitlyHidden) {
		dispatch_async(dispatch_get_main_queue(), ^{
			self.window.collectionBehavior = NSWindowCollectionBehaviorDefault;
		});
	}
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

	/* 3. Edit Menu (standard Cmd+Z / Shift-Cmd+Z and text editing commands) */
	NSMenuItem *editMenuItem = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
	NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
	[editMenu addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
	NSMenuItem *redoItem = [editMenu addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"z"];
	redoItem.keyEquivalentModifierMask = NSEventModifierFlagCommand | NSEventModifierFlagShift;
	[editMenu addItem:[NSMenuItem separatorItem]];
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
	self.statusMenuMonoAItem.state   = atomic_load(&gEngine.pathA.mono)	? NSControlStateValueOn : NSControlStateValueOff;
	self.statusMenuEnableBItem.state = atomic_load(&gEngine.pathB.enabled) ? NSControlStateValueOn : NSControlStateValueOff;
	self.statusMenuMonoBItem.state   = atomic_load(&gEngine.pathB.mono)	? NSControlStateValueOn : NSControlStateValueOff;
}

/*===========================================================================
 * Preferences & Presets Persistence
 *===========================================================================*/

- (void)loadPreferences {
	NSUserDefaults *defs = [NSUserDefaults standardUserDefaults];
	[defs registerDefaults:@{
		kPrefOutputAEnabled : @YES,
		kPrefOutputBEnabled : @YES,
		kPrefOutputAMono	: @NO,
		kPrefOutputBMono	: @NO,
		kPrefMasterVolume   : @(1.0f),
		kPrefBalance		: @(0.0f),
		kPrefTrimDbA		: @(0.0f),
		kPrefTrimDbB		: @(0.0f),
		kPrefDelayMsA	   : @(0.0),
		kPrefDelayMsB	   : @(0.0),
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
		kPrefCrossoverSlope	 : @(24),
		kPrefCrossoverModeA	 : @(kCrossoverFilterLowPass),
		kPrefCrossoverModeB	 : @(kCrossoverFilterHighPass),
		kPrefCrossoverFrequencyA : @(120.0),
		kPrefCrossoverFrequencyB : @(120.0),
		kPrefCrossoverSlopeA	 : @(24),
		kPrefCrossoverSlopeB	 : @(24)
	}];

	self.desiredUID_A = [defs stringForKey:kPrefOutputAUID];
	self.desiredUID_B = [defs stringForKey:kPrefOutputBUID];

	atomic_store(&gEngine.pathA.enabled, [defs boolForKey:kPrefOutputAEnabled]);
	atomic_store(&gEngine.pathB.enabled, [defs boolForKey:kPrefOutputBEnabled]);
	atomic_store(&gEngine.pathA.mono,	[defs boolForKey:kPrefOutputAMono]);
	atomic_store(&gEngine.pathB.mono,	[defs boolForKey:kPrefOutputBMono]);
	gEngine.masterVolume = [defs floatForKey:kPrefMasterVolume];
	gEngine.balance	  = [defs floatForKey:kPrefBalance];
	gEngine.trimDbA	  = [defs floatForKey:kPrefTrimDbA];
	gEngine.trimDbB	  = [defs floatForKey:kPrefTrimDbB];
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

	atomic_store(&gEngine.graphicEQ.enabled, [defs boolForKey:kPrefGraphicEQEnabled]);
	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		atomic_store(&gEngine.graphicEQ.bandGainDb[i], 0.0f);
	}
	[self applyGraphicEQBandGainValues:[defs arrayForKey:kPrefGraphicEQBandGainsDb]];
	atomic_fetch_add(&gEngine.graphicEQ.revision, 1U);

	UpdateEngineTargetGains();
}

- (void)savePreferences {
	NSUserDefaults *defs = [NSUserDefaults standardUserDefaults];
	if (self.desiredUID_A) [defs setObject:self.desiredUID_A forKey:kPrefOutputAUID];
	if (self.desiredUID_B) [defs setObject:self.desiredUID_B forKey:kPrefOutputBUID];
	[defs setBool:atomic_load(&gEngine.pathA.enabled) forKey:kPrefOutputAEnabled];
	[defs setBool:atomic_load(&gEngine.pathB.enabled) forKey:kPrefOutputBEnabled];
	[defs setBool:atomic_load(&gEngine.pathA.mono)	forKey:kPrefOutputAMono];
	[defs setBool:atomic_load(&gEngine.pathB.mono)	forKey:kPrefOutputBMono];
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
	[defs setBool:atomic_load(&gEngine.graphicEQ.enabled) forKey:kPrefGraphicEQEnabled];
	[defs setObject:[self graphicEQBandGainValues] forKey:kPrefGraphicEQBandGainsDb];
	if (self.savedPresets) [defs setObject:self.savedPresets forKey:kPrefSavedPresets];
	if (self.activePresetName) [defs setObject:self.activePresetName forKey:kPrefActivePreset];
	[defs synchronize];
	[self updatePresetIndicator];
}

- (NSDictionary *)captureCurrentStateAsPresetDictionary {
	return @{
		kPrefOutputAUID	 : self.desiredUID_A ?: @"",
		kPrefOutputBUID	 : self.desiredUID_B ?: @"",
		kPrefOutputAEnabled : @(atomic_load(&gEngine.pathA.enabled)),
		kPrefOutputBEnabled : @(atomic_load(&gEngine.pathB.enabled)),
		kPrefOutputAMono	: @(atomic_load(&gEngine.pathA.mono)),
		kPrefOutputBMono	: @(atomic_load(&gEngine.pathB.mono)),
		kPrefMasterVolume   : @(gEngine.masterVolume),
		kPrefBalance		: @(gEngine.balance),
		kPrefTrimDbA		: @(gEngine.trimDbA),
		kPrefTrimDbB		: @(gEngine.trimDbB),
		kPrefDelayMsA	   : @(atomic_load(&gEngine.pathA.targetDelayMs)),
		kPrefDelayMsB	   : @(atomic_load(&gEngine.pathB.targetDelayMs)),
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
		kPrefCrossoverSlope	 : @(atomic_load(&gEngine.crossover.linkedSlopeDb)),
		kPrefCrossoverModeA	 : @(atomic_load(&gEngine.crossover.modeA)),
		kPrefCrossoverModeB	 : @(atomic_load(&gEngine.crossover.modeB)),
		kPrefCrossoverFrequencyA : @(atomic_load(&gEngine.crossover.frequencyA)),
		kPrefCrossoverFrequencyB : @(atomic_load(&gEngine.crossover.frequencyB)),
		kPrefCrossoverSlopeA	 : @(atomic_load(&gEngine.crossover.slopeDbA)),
		kPrefCrossoverSlopeB	 : @(atomic_load(&gEngine.crossover.slopeDbB)),
		kPrefGraphicEQEnabled	 : @(atomic_load(&gEngine.graphicEQ.enabled)),
		kPrefGraphicEQBandGainsDb : [self graphicEQBandGainValues]
	};
}

- (NSDictionary *)captureUndoSnapshot {
	return @{
		@"settings" : [self captureCurrentStateAsPresetDictionary],
		@"savedPresets" : [self.savedPresets copy] ?: @{},
		@"activePresetName" : self.activePresetName ?: @""
	};
}

- (void)registerUndoForCurrentStateWithActionName:(NSString *)actionName {
	if (self.settingsUndoManager == nil || self.settingsUndoManager.isUndoing || self.settingsUndoManager.isRedoing) return;
	NSDictionary *snapshot = [self captureUndoSnapshot];
	[[self.settingsUndoManager prepareWithInvocationTarget:self] restoreUndoSnapshot:snapshot
													 actionName:actionName];
	[self.settingsUndoManager setActionName:actionName];
}

- (void)restoreUndoSnapshot:(NSDictionary *)snapshot actionName:(NSString *)actionName {
	if (![snapshot[@"settings"] isKindOfClass:[NSDictionary class]]) return;

	NSDictionary *inverseSnapshot = [self captureUndoSnapshot];
	NSDictionary *savedPresets = snapshot[@"savedPresets"];
	self.savedPresets = [savedPresets isKindOfClass:[NSDictionary class]]
		? [savedPresets mutableCopy] : [NSMutableDictionary dictionary];
	[self applyControllerStateDictionary:snapshot[@"settings"]
					 activePresetName:[snapshot[@"activePresetName"] isKindOfClass:[NSString class]]
									? snapshot[@"activePresetName"] : @""];

	[[self.settingsUndoManager prepareWithInvocationTarget:self] restoreUndoSnapshot:inverseSnapshot
													 actionName:actionName];
	[self.settingsUndoManager setActionName:actionName];
}

- (void)applyControllerStateDictionary:(NSDictionary *)preset activePresetName:(NSString *)activePresetName {
	if (![preset isKindOfClass:[NSDictionary class]]) return;

	self.activePresetName = activePresetName ?: @"";
	self.desiredUID_A	 = preset[kPrefOutputAUID] ?: @"";
	self.desiredUID_B	 = preset[kPrefOutputBUID] ?: @"";

	bool enA   = [preset[kPrefOutputAEnabled] boolValue];
	bool enB   = [preset[kPrefOutputBEnabled] boolValue];
	bool monoA = [preset[kPrefOutputAMono] boolValue];
	bool monoB = [preset[kPrefOutputBMono] boolValue];

	atomic_store(&gEngine.pathA.enabled, enA);
	atomic_store(&gEngine.pathB.enabled, enB);
	atomic_store(&gEngine.pathA.mono,	monoA);
	atomic_store(&gEngine.pathB.mono,	monoB);

	gEngine.masterVolume = [preset[kPrefMasterVolume] floatValue];
	gEngine.balance	  = [preset[kPrefBalance] floatValue];
	gEngine.trimDbA	  = [preset[kPrefTrimDbA] floatValue];
	gEngine.trimDbB	  = [preset[kPrefTrimDbB] floatValue];

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

	NSNumber *graphicEQEnabled = preset[kPrefGraphicEQEnabled];
	if (graphicEQEnabled != nil) {
		atomic_store(&gEngine.graphicEQ.enabled, graphicEQEnabled.boolValue);
		[self applyGraphicEQBandGainValues:preset[kPrefGraphicEQBandGainsDb]];
		atomic_fetch_add(&gEngine.graphicEQ.revision, 1U);
	}

	UpdateEngineTargetGains();

	self.enableCheckboxA.state  = enA   ? NSControlStateValueOn : NSControlStateValueOff;
	self.enableCheckboxB.state  = enB   ? NSControlStateValueOn : NSControlStateValueOff;
	self.monoCheckboxA.state	= monoA ? NSControlStateValueOn : NSControlStateValueOff;
	self.monoCheckboxB.state	= monoB ? NSControlStateValueOn : NSControlStateValueOff;
	self.masterSlider.floatValue  = gEngine.masterVolume;
	self.balanceSlider.floatValue = gEngine.balance;
	self.trimSliderA.floatValue   = gEngine.trimDbA;
	self.trimSliderB.floatValue   = gEngine.trimDbB;
	self.delaySliderA.doubleValue  = delA;
	self.delayStepperA.doubleValue = delA;
	self.delaySliderB.doubleValue  = delB;
	self.delayStepperB.doubleValue = delB;

	[self updateCrossoverControls];
	[self updateGraphicEQControls];
	[self updateRoutingMatrixViews];

	[self updateValueLabels];
	[self syncStatusBarMenuChecks];
	[self rebuildPresetsUI];
	[self refreshDevicesAndStreams];
	[self savePreferences];
}

- (void)applyPresetNamed:(NSString *)presetName {
	NSDictionary *preset = self.savedPresets[presetName];
	if (preset == nil) return;

	[self registerUndoForCurrentStateWithActionName:@"Load Preset"];
	[self applyControllerStateDictionary:preset activePresetName:presetName];
	[self showStatusMessage:[NSString stringWithFormat:@"Loaded preset \"%@\".", presetName]
					 level:StatusLevelSuccess
			autoClearAfter:4.0];
}

- (BOOL)presetValue:(id)presetValue matchesCurrentValue:(id)currentValue {
	if (presetValue == currentValue) return YES;
	if (presetValue == nil || currentValue == nil) return NO;

	if ([presetValue isKindOfClass:[NSNumber class]] && [currentValue isKindOfClass:[NSNumber class]]) {
		/* Route gains are serialized as dB, then converted back to linear
		 * values when loaded.  A small tolerance prevents round-trip noise
		 * from incorrectly marking an untouched preset as edited. */
		return fabs(((NSNumber *)presetValue).doubleValue - ((NSNumber *)currentValue).doubleValue) < 0.00001;
	}
	if ([presetValue isKindOfClass:[NSArray class]] && [currentValue isKindOfClass:[NSArray class]]) {
		NSArray *presetValues = presetValue;
		NSArray *currentValues = currentValue;
		if (presetValues.count != currentValues.count) return NO;
		for (NSUInteger index = 0; index < presetValues.count; ++index) {
			if (![self presetValue:presetValues[index] matchesCurrentValue:currentValues[index]]) return NO;
		}
		return YES;
	}
	return [presetValue isEqual:currentValue];
}

- (BOOL)activePresetHasUnsavedChanges {
	NSDictionary *preset = self.savedPresets[self.activePresetName];
	if (self.activePresetName.length == 0 || preset == nil) return NO;

	NSDictionary *currentState = [self captureCurrentStateAsPresetDictionary];
	if (preset.count != currentState.count) return YES;
	for (NSString *key in currentState) {
		if (![self presetValue:preset[key] matchesCurrentValue:currentState[key]]) return YES;
	}
	return NO;
}

- (void)updatePresetIndicator {
	if (self.presetPopup == nil || self.presetPopupIndicatorItem == nil) return;

	BOOL edited = [self activePresetHasUnsavedChanges];
	for (NSMenuItem *item in self.presetPopup.itemArray) {
		NSString *name = [item.representedObject isKindOfClass:[NSString class]] ? item.representedObject : nil;
		if (name.length == 0) continue;
		BOOL isActive = [name isEqualToString:self.activePresetName];
		item.state = isActive && !edited ? NSControlStateValueOn : NSControlStateValueOff;
	}
	if (self.activePresetName.length > 0 && edited) {
		NSMutableAttributedString *title = [[NSMutableAttributedString alloc] initWithString:self.activePresetName];
		[title appendAttributedString:[[NSAttributedString alloc] initWithString:@" • Edited"
																	  attributes:@{ NSForegroundColorAttributeName : [NSColor secondaryLabelColor] }]];
		self.presetPopupIndicatorItem.attributedTitle = title;
	} else {
		self.presetPopupIndicatorItem.title = self.activePresetName ?: @"";
	}
	[self.presetPopup selectItem:self.presetPopupIndicatorItem];
}

- (void)rebuildPresetsUI {
	NSArray<NSString *> *sortedNames = [[self.savedPresets allKeys] sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];

	if (self.presetPopup != nil) {
		NSMenu *popupMenu = self.presetPopup.menu;
		[popupMenu removeAllItems];

		/* Keep the popup's displayed title separate from the menu rows, so
		 * an edited preset can show a suffix without altering its menu name. */
		self.presetPopupIndicatorItem = [[NSMenuItem alloc] initWithTitle:@""
																	 action:nil
															  keyEquivalent:@""];
		self.presetPopupIndicatorItem.hidden = YES;
		[popupMenu addItem:self.presetPopupIndicatorItem];

		NSMenuItem *saveItem = [[NSMenuItem alloc] initWithTitle:@"Save Preset..."
														   action:nil
													keyEquivalent:@""];
		NSMenu *saveMenu = [[NSMenu alloc] initWithTitle:@"Save Preset..."];
		NSMenuItem *newPresetItem = [[NSMenuItem alloc] initWithTitle:@"Save as New Preset..."
															   action:@selector(savePresetClicked:)
														keyEquivalent:@""];
		newPresetItem.target = self;
		[saveMenu addItem:newPresetItem];
		if (sortedNames.count > 0) {
			[saveMenu addItem:[NSMenuItem separatorItem]];
			for (NSString *name in sortedNames) {
				NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:name
															  action:@selector(overwritePresetClicked:)
													   keyEquivalent:@""];
				item.target = self;
				item.representedObject = name;
				[saveMenu addItem:item];
			}
		}
		saveItem.submenu = saveMenu;
		[popupMenu addItem:saveItem];

		NSMenuItem *deleteItem = [[NSMenuItem alloc] initWithTitle:@"Delete Preset"
															 action:nil
													  keyEquivalent:@""];
		NSMenu *deleteMenu = [[NSMenu alloc] initWithTitle:@"Delete Preset"];
		NSMenuItem *renameItem = [[NSMenuItem alloc] initWithTitle:@"Rename Preset"
															 action:nil
													  keyEquivalent:@""];
		NSMenu *renameMenu = [[NSMenu alloc] initWithTitle:@"Rename Preset"];
		for (NSString *name in sortedNames) {
			NSMenuItem *deleteNameItem = [[NSMenuItem alloc] initWithTitle:name
																	  action:@selector(deletePresetClicked:)
															   keyEquivalent:@""];
			deleteNameItem.target = self;
			deleteNameItem.representedObject = name;
			[deleteMenu addItem:deleteNameItem];

			NSMenuItem *renameNameItem = [[NSMenuItem alloc] initWithTitle:name
																	  action:@selector(renamePresetClicked:)
															   keyEquivalent:@""];
			renameNameItem.target = self;
			renameNameItem.representedObject = name;
			[renameMenu addItem:renameNameItem];
		}
		if (sortedNames.count == 0) {
			deleteItem.enabled = NO;
			renameItem.enabled = NO;
		}
		deleteItem.submenu = deleteMenu;
		renameItem.submenu = renameMenu;
		[popupMenu addItem:deleteItem];
		[popupMenu addItem:renameItem];

		if (sortedNames.count > 0) {
			[popupMenu addItem:[NSMenuItem separatorItem]];
			BOOL edited = [self activePresetHasUnsavedChanges];
			for (NSString *name in sortedNames) {
				NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:name
															  action:@selector(presetPopupChanged:)
													   keyEquivalent:@""];
				item.target = self;
				item.representedObject = name;
				item.state = [name isEqualToString:self.activePresetName] && !edited
					? NSControlStateValueOn : NSControlStateValueOff;
				[popupMenu addItem:item];
			}
		}
		[self updatePresetIndicator];
	}

	NSMenu *sub = self.statusMenuPresetsItem.submenu;
	[sub removeAllItems];
	BOOL edited = [self activePresetHasUnsavedChanges];
	for (NSString *name in sortedNames) {
		NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:name
													  action:@selector(menuPresetClicked:)
											   keyEquivalent:@""];
		item.target = self;
		item.representedObject = name;
		if ([name isEqualToString:self.activePresetName] && !edited) {
			item.state = NSControlStateValueOn;
		}
		[sub addItem:item];
	}
}

/*===========================================================================
 * Main Window & Controls Construction
 *===========================================================================*/

- (void)buildWindowAndUI {
	/* The content remains a deliberately spacious fixed layout, while the
	 * surrounding window can shrink or grow on any display. */
	NSRect frame = NSMakeRect(0, 0, 780, 790);
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
	[self.window center];

	self.controllerScrollView = [[NSScrollView alloc] initWithFrame:self.window.contentView.bounds];
	self.controllerScrollView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
	self.controllerScrollView.hasVerticalScroller = YES;
	self.controllerScrollView.hasHorizontalScroller = YES;
	self.controllerScrollView.autohidesScrollers = YES;
	self.controllerScrollView.borderType = NSNoBorder;
	self.controllerScrollView.drawsBackground = YES;
	[self.window.contentView addSubview:self.controllerScrollView];

	NSView *cv = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 740, 790)];
	self.controllerScrollView.documentView = cv;

	/* 1. Preset selection and virtual-device status share the unboxed top row. */
	self.presetPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(20, 758, 395, 24) pullsDown:NO];
	self.presetPopup.target = self;
	self.presetPopup.action = @selector(presetPopupChanged:);
	[cv addSubview:self.presetPopup];

	self.virtualStatusLabel = [self makeLabel:@"Checking Dual Output driver..." bold:YES size:12 color:[NSColor labelColor]];
	self.virtualStatusLabel.frame = NSMakeRect(430, 761, 290, 18);
	self.virtualStatusLabel.alignment = NSTextAlignmentRight;
	self.virtualStatusLabel.lineBreakMode = NSLineBreakByTruncatingMiddle;
	[cv addSubview:self.virtualStatusLabel];

	/* 2. Global level and input controls share one compact, title-free box. */
	NSBox *mixBox = [[NSBox alloc] initWithFrame:NSMakeRect(20, 660, 700, 90)];
	mixBox.titlePosition = NSNoTitle;
	[cv addSubview:mixBox];

	NSTextField *inMeterLbl = [self makeLabel:@"Input Level:" bold:NO size:11 color:[NSColor secondaryLabelColor]];
	inMeterLbl.frame = NSMakeRect(14, 55, 72, 16);
	[mixBox.contentView addSubview:inMeterLbl];

	self.inputMeter = [[AudioLevelMeterView alloc] initWithFrame:NSMakeRect(90, 53, 380, 20)];
	[mixBox.contentView addSubview:self.inputMeter];

	self.setSystemDefaultButton = [[NSButton alloc] initWithFrame:NSMakeRect(495, 49, 185, 28)];
	self.setSystemDefaultButton.bezelStyle = NSBezelStyleRounded;
	self.setSystemDefaultButton.title = @"Set as System Output";
	self.setSystemDefaultButton.target = self;
	self.setSystemDefaultButton.action = @selector(setVirtualDeviceAsSystemOutput:);
	[mixBox.contentView addSubview:self.setSystemDefaultButton];

	/* Keep the global level controls together: compact labels and sliders let
	 * both values remain visible on one row without a separate reset button. */
	NSTextField *masterLbl = [self makeLabel:@"Master:" bold:YES size:12 color:nil];
	masterLbl.frame = NSMakeRect(14, 10, 60, 20);
	[mixBox.contentView addSubview:masterLbl];

	self.masterSlider = [NSSlider sliderWithValue:gEngine.masterVolume
										 minValue:0.0
										 maxValue:1.0
										   target:self
										   action:@selector(masterVolumeChanged:)];
	self.masterSlider.frame = NSMakeRect(76, 10, 182, 22);
	self.masterSlider.continuous = YES;
	[mixBox.contentView addSubview:self.masterSlider];

	self.masterValueLabel = [self makeLabel:@"" bold:NO size:12 color:nil];
	self.masterValueLabel.frame = NSMakeRect(262, 10, 102, 20);
	[mixBox.contentView addSubview:self.masterValueLabel];

	NSTextField *balLbl = [self makeLabel:@"A / B:" bold:YES size:12 color:nil];
	balLbl.frame = NSMakeRect(374, 10, 58, 20);
	[mixBox.contentView addSubview:balLbl];

	self.balanceSlider = [NSSlider sliderWithValue:gEngine.balance
										  minValue:-1.0
										  maxValue:1.0
											target:self
											action:@selector(balanceChanged:)];
	self.balanceSlider.frame = NSMakeRect(434, 10, 142, 22);
	self.balanceSlider.continuous = YES;
	self.balanceSlider.numberOfTickMarks = 3;
	[mixBox.contentView addSubview:self.balanceSlider];

	self.balanceValueLabel = [self makeLabel:@"" bold:NO size:12 color:nil];
	self.balanceValueLabel.frame = NSMakeRect(580, 10, 100, 20);
	[mixBox.contentView addSubview:self.balanceValueLabel];

	/* 3. 31-Band Graphic EQ (Pre-Crossover): compact graphic preview in main view;
	 *	clicking the plot (or button) opens the full 31-fader editor window. */
	NSBox *graphicEQBox = [[NSBox alloc] initWithFrame:NSMakeRect(20, 534, 700, 118)];
	graphicEQBox.title = @"31-Band Graphic EQ — 1/3-Octave Pre-Crossover";
	graphicEQBox.titleFont = [NSFont boldSystemFontOfSize:12];
	[cv addSubview:graphicEQBox];

	self.graphicEQEnabledCheckbox = [NSButton checkboxWithTitle:@"Enable 31-band Graphic EQ"
														 target:self
														 action:@selector(graphicEQEnabledChanged:)];
	self.graphicEQEnabledCheckbox.frame = NSMakeRect(14, 70, 190, 20);
	[graphicEQBox.contentView addSubview:self.graphicEQEnabledCheckbox];

	self.graphicEQSummaryLabel = [self makeLabel:@"" bold:NO size:11 color:[NSColor secondaryLabelColor]];
	self.graphicEQSummaryLabel.frame = NSMakeRect(210, 71, 340, 16);
	[graphicEQBox.contentView addSubview:self.graphicEQSummaryLabel];

	NSButton *openEQButton = [[NSButton alloc] initWithFrame:NSMakeRect(558, 66, 128, 26)];
	openEQButton.bezelStyle = NSBezelStyleRounded;
	openEQButton.title = @"Open EQ Sliders...";
	openEQButton.target = self;
	openEQButton.action = @selector(showGraphicEQWindow:);
	[graphicEQBox.contentView addSubview:openEQButton];

	self.graphicEQPreviewView = [[GraphicEQPreviewView alloc] initWithFrame:NSMakeRect(14, 6, 672, 58)];
	self.graphicEQPreviewView.clickable = YES;
	self.graphicEQPreviewView.eqDelegate = self;
	self.graphicEQPreviewView.toolTip = @"Click anywhere on this Graphic EQ response curve to open the full 31-band slider window.";
	[graphicEQBox.contentView addSubview:self.graphicEQPreviewView];

	[self updateGraphicEQControls];

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

	self.linkedCrossoverAssignmentPopup = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(350, 128, 330, 26) pullsDown:NO];
	[self.linkedCrossoverAssignmentPopup addItemWithTitle:@"Output A: Low-Pass  •  Output B: High-Pass"];
	self.linkedCrossoverAssignmentPopup.lastItem.tag = 0;
	[self.linkedCrossoverAssignmentPopup addItemWithTitle:@"Output A: High-Pass  •  Output B: Low-Pass"];
	self.linkedCrossoverAssignmentPopup.lastItem.tag = 1;
	self.linkedCrossoverAssignmentPopup.target = self;
	self.linkedCrossoverAssignmentPopup.action = @selector(linkedCrossoverAssignmentChanged:);
	self.linkedCrossoverAssignmentPopup.toolTip = @"Choose which physical output receives the low or high band while filters are linked.";
	[crossoverBox.contentView addSubview:self.linkedCrossoverAssignmentPopup];

	NSTextField *linkedFrequencyTitle = [self makeLabel:@"Crossover freq:" bold:YES size:11 color:nil];
	linkedFrequencyTitle.frame = NSMakeRect(14, 101, 86, 18);
	[crossoverBox.contentView addSubview:linkedFrequencyTitle];

	self.linkedCrossoverFrequencySlider = [NSSlider sliderWithValue:CrossoverFrequencyToSliderValue(atomic_load(&gEngine.crossover.linkedFrequency))
															  minValue:log10(20.0)
															  maxValue:log10(20000.0)
																target:self
																action:@selector(linkedCrossoverFrequencyChanged:)];
	self.linkedCrossoverFrequencySlider.frame = NSMakeRect(104, 97, 260, 22);
	self.linkedCrossoverFrequencySlider.continuous = YES;
	self.linkedCrossoverFrequencySlider.toolTip = @"20 Hz to 20 kHz, logarithmic scale";
	[crossoverBox.contentView addSubview:self.linkedCrossoverFrequencySlider];

	self.linkedCrossoverFrequencyLabel = [self makeLabel:@"" bold:YES size:11 color:nil];
	self.linkedCrossoverFrequencyLabel.frame = NSMakeRect(370, 101, 62, 18);
	[crossoverBox.contentView addSubview:self.linkedCrossoverFrequencyLabel];

	NSTextField *linkedSlopeTitle = [self makeLabel:@"Slope:" bold:YES size:11 color:nil];
	linkedSlopeTitle.frame = NSMakeRect(437, 101, 42, 18);
	[crossoverBox.contentView addSubview:linkedSlopeTitle];

	self.linkedCrossoverSlopeSlider = [NSSlider sliderWithValue:atomic_load(&gEngine.crossover.linkedSlopeDb)
														minValue:12.0
														maxValue:96.0
														  target:self
														  action:@selector(linkedCrossoverSlopeChanged:)];
	self.linkedCrossoverSlopeSlider.frame = NSMakeRect(480, 97, 130, 22);
	self.linkedCrossoverSlopeSlider.continuous = YES;
	self.linkedCrossoverSlopeSlider.numberOfTickMarks = 8;
	self.linkedCrossoverSlopeSlider.allowsTickMarkValuesOnly = YES;
	self.linkedCrossoverSlopeSlider.toolTip = @"Linkwitz-Riley slope: 12 to 96 dB per octave";
	[crossoverBox.contentView addSubview:self.linkedCrossoverSlopeSlider];

	self.linkedCrossoverSlopeLabel = [self makeLabel:@"" bold:YES size:10 color:nil];
	self.linkedCrossoverSlopeLabel.frame = NSMakeRect(616, 101, 64, 18);
	[crossoverBox.contentView addSubview:self.linkedCrossoverSlopeLabel];

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
	self.crossoverSlopeSliderA = [NSSlider sliderWithValue:atomic_load(&gEngine.crossover.slopeDbA)
												   minValue:12.0
												   maxValue:96.0
													 target:self
													 action:@selector(crossoverSlopeAChanged:)];
	self.crossoverSlopeSliderA.frame = NSMakeRect(514, 100, 96, 22);
	self.crossoverSlopeSliderA.continuous = YES;
	self.crossoverSlopeSliderA.numberOfTickMarks = 8;
	self.crossoverSlopeSliderA.allowsTickMarkValuesOnly = YES;
	self.crossoverSlopeSliderA.toolTip = @"Linkwitz-Riley slope: 12 to 96 dB per octave";
	[crossoverBox.contentView addSubview:self.crossoverSlopeSliderA];
	self.crossoverSlopeLabelA = [self makeLabel:@"" bold:YES size:10 color:nil];
	self.crossoverSlopeLabelA.frame = NSMakeRect(617, 104, 63, 16);
	[crossoverBox.contentView addSubview:self.crossoverSlopeLabelA];

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
	self.crossoverSlopeSliderB = [NSSlider sliderWithValue:atomic_load(&gEngine.crossover.slopeDbB)
												   minValue:12.0
												   maxValue:96.0
													 target:self
													 action:@selector(crossoverSlopeBChanged:)];
	self.crossoverSlopeSliderB.frame = NSMakeRect(514, 72, 96, 22);
	self.crossoverSlopeSliderB.continuous = YES;
	self.crossoverSlopeSliderB.numberOfTickMarks = 8;
	self.crossoverSlopeSliderB.allowsTickMarkValuesOnly = YES;
	self.crossoverSlopeSliderB.toolTip = @"Linkwitz-Riley slope: 12 to 96 dB per octave";
	[crossoverBox.contentView addSubview:self.crossoverSlopeSliderB];
	self.crossoverSlopeLabelB = [self makeLabel:@"" bold:YES size:10 color:nil];
	self.crossoverSlopeLabelB.frame = NSMakeRect(617, 76, 63, 16);
	[crossoverBox.contentView addSubview:self.crossoverSlopeLabelB];

	self.linkedCrossoverControls = @[linkedFrequencyTitle, self.linkedCrossoverFrequencySlider,
									 self.linkedCrossoverFrequencyLabel, linkedSlopeTitle,
									 self.linkedCrossoverSlopeSlider, self.linkedCrossoverSlopeLabel,
									 self.linkedCrossoverAssignmentPopup];
	self.independentCrossoverControls = @[independentLabelA, self.crossoverModePopupA,
										   independentFrequencyTitleA, self.crossoverFrequencySliderA,
										   self.crossoverFrequencyLabelA, independentSlopeTitleA,
										   self.crossoverSlopeSliderA, self.crossoverSlopeLabelA,
										   independentLabelB, self.crossoverModePopupB,
										   independentFrequencyTitleB, self.crossoverFrequencySliderB,
										   self.crossoverFrequencyLabelB, independentSlopeTitleB,
										   self.crossoverSlopeSliderB, self.crossoverSlopeLabelB];
	self.crossoverPrimaryLabels = @[linkedFrequencyTitle, linkedSlopeTitle,
									independentLabelA, independentLabelB];

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
	self.statusLabelA.frame = NSMakeRect(14, 154, 222, 16);
	[boxA.contentView addSubview:self.statusLabelA];

	NSTextField *gainLblA = [self makeLabel:@"Gain" bold:NO size:11 color:nil];
	gainLblA.frame = NSMakeRect(14, 111, 32, 16);
	[boxA.contentView addSubview:gainLblA];

	self.trimSliderA = [NSSlider sliderWithValue:gEngine.trimDbA
										minValue:-24.0
										maxValue:6.0
										  target:self
										  action:@selector(trimAChanged:)];
	self.trimSliderA.frame = NSMakeRect(50, 108, 189, 22);
	self.trimSliderA.continuous = YES;
	[boxA.contentView addSubview:self.trimSliderA];

	self.trimValueLabelA = [[NSTextField alloc] initWithFrame:NSMakeRect(253, 106, 50, 24)];
	self.trimValueLabelA.cell = [[CompactRoundedTextFieldCell alloc] initTextCell:@""];
	self.trimValueLabelA.editable = YES;
	self.trimValueLabelA.selectable = YES;
	self.trimValueLabelA.alignment = NSTextAlignmentLeft;
	self.trimValueLabelA.bezeled = YES;
	self.trimValueLabelA.bezelStyle = NSTextFieldRoundedBezel;
	self.trimValueLabelA.drawsBackground = YES;
	self.trimValueLabelA.backgroundColor = [NSColor textBackgroundColor];
	self.trimValueLabelA.font = [NSFont monospacedDigitSystemFontOfSize:11 weight:NSFontWeightMedium];
	self.trimValueLabelA.target = self;
	self.trimValueLabelA.action = @selector(trimAValueChanged:);
	self.trimValueLabelA.toolTip = @"Enter a gain from −24.0 to +6.0 dB";
	[boxA.contentView addSubview:self.trimValueLabelA];
	NSTextField *trimUnitA = [self makeLabel:@"dB" bold:NO size:10 color:[NSColor secondaryLabelColor]];
	trimUnitA.frame = NSMakeRect(306, 111, 18, 16);
	[boxA.contentView addSubview:trimUnitA];

	NSTextField *delayLblA = [self makeLabel:@"Time Delay (0 - 500 ms):" bold:NO size:11 color:nil];
	delayLblA.frame = NSMakeRect(14, 90, 175, 16);
	[boxA.contentView addSubview:delayLblA];

	self.delayValueLabelA = [self makeLabel:@"0.0 ms" bold:YES size:11 color:nil];
	self.delayValueLabelA.frame = NSMakeRect(200, 90, 110, 16);
	[boxA.contentView addSubview:self.delayValueLabelA];

	double initDelayA = atomic_load(&gEngine.pathA.targetDelayMs);
	self.delaySliderA = [NSSlider sliderWithValue:initDelayA
										 minValue:0.0
										 maxValue:500.0
										   target:self
										   action:@selector(delayASliderChanged:)];
	self.delaySliderA.frame = NSMakeRect(14, 64, 275, 22);
	self.delaySliderA.continuous = YES;
	[boxA.contentView addSubview:self.delaySliderA];

	self.delayStepperA = [[NSStepper alloc] initWithFrame:NSMakeRect(296, 62, 28, 26)];
	self.delayStepperA.minValue = 0.0;
	self.delayStepperA.maxValue = 500.0;
	self.delayStepperA.increment = 0.5;
	self.delayStepperA.valueWraps = NO;
	self.delayStepperA.doubleValue = initDelayA;
	self.delayStepperA.target = self;
	self.delayStepperA.action = @selector(delayAStepperChanged:);
	[boxA.contentView addSubview:self.delayStepperA];

	self.meterA = [[AudioLevelMeterView alloc] initWithFrame:NSMakeRect(240, 152, 84, 20)];
	[boxA.contentView addSubview:self.meterA];

	/* The routing matrix is the final stage: source L/R reaches the selected
	 * destination channels only after gain and delay have been configured. */
	self.routingMatrixA = [[RoutingMatrixView alloc] initWithFrame:NSMakeRect(14, 4, 310, 56)];
	self.routingMatrixA.outputA = YES;
	self.routingMatrixA.routingDelegate = self;
	self.routingMatrixA.toolTip = @"Final routing stage. Click a cell to inspect or set its gain; right-click to toggle a route; Command-click to disable.";
	[boxA.contentView addSubview:self.routingMatrixA];

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
	self.statusLabelB.frame = NSMakeRect(14, 154, 222, 16);
	[boxB.contentView addSubview:self.statusLabelB];

	NSTextField *gainLblB = [self makeLabel:@"Gain" bold:NO size:11 color:nil];
	gainLblB.frame = NSMakeRect(14, 111, 32, 16);
	[boxB.contentView addSubview:gainLblB];

	self.trimSliderB = [NSSlider sliderWithValue:gEngine.trimDbB
										minValue:-24.0
										maxValue:6.0
										  target:self
										  action:@selector(trimBChanged:)];
	self.trimSliderB.frame = NSMakeRect(50, 108, 189, 22);
	self.trimSliderB.continuous = YES;
	[boxB.contentView addSubview:self.trimSliderB];

	self.trimValueLabelB = [[NSTextField alloc] initWithFrame:NSMakeRect(253, 106, 50, 24)];
	self.trimValueLabelB.cell = [[CompactRoundedTextFieldCell alloc] initTextCell:@""];
	self.trimValueLabelB.editable = YES;
	self.trimValueLabelB.selectable = YES;
	self.trimValueLabelB.alignment = NSTextAlignmentLeft;
	self.trimValueLabelB.bezeled = YES;
	self.trimValueLabelB.bezelStyle = NSTextFieldRoundedBezel;
	self.trimValueLabelB.drawsBackground = YES;
	self.trimValueLabelB.backgroundColor = [NSColor textBackgroundColor];
	self.trimValueLabelB.font = [NSFont monospacedDigitSystemFontOfSize:11 weight:NSFontWeightMedium];
	self.trimValueLabelB.target = self;
	self.trimValueLabelB.action = @selector(trimBValueChanged:);
	self.trimValueLabelB.toolTip = @"Enter a gain from −24.0 to +6.0 dB";
	[boxB.contentView addSubview:self.trimValueLabelB];
	NSTextField *trimUnitB = [self makeLabel:@"dB" bold:NO size:10 color:[NSColor secondaryLabelColor]];
	trimUnitB.frame = NSMakeRect(306, 111, 18, 16);
	[boxB.contentView addSubview:trimUnitB];

	NSTextField *delayLblB = [self makeLabel:@"Time Delay (0 - 500 ms):" bold:NO size:11 color:nil];
	delayLblB.frame = NSMakeRect(14, 90, 175, 16);
	[boxB.contentView addSubview:delayLblB];

	self.delayValueLabelB = [self makeLabel:@"0.0 ms" bold:YES size:11 color:nil];
	self.delayValueLabelB.frame = NSMakeRect(200, 90, 110, 16);
	[boxB.contentView addSubview:self.delayValueLabelB];

	double initDelayB = atomic_load(&gEngine.pathB.targetDelayMs);
	self.delaySliderB = [NSSlider sliderWithValue:initDelayB
										 minValue:0.0
										 maxValue:500.0
										   target:self
										   action:@selector(delayBSliderChanged:)];
	self.delaySliderB.frame = NSMakeRect(14, 64, 275, 22);
	self.delaySliderB.continuous = YES;
	[boxB.contentView addSubview:self.delaySliderB];

	self.delayStepperB = [[NSStepper alloc] initWithFrame:NSMakeRect(296, 62, 28, 26)];
	self.delayStepperB.minValue = 0.0;
	self.delayStepperB.maxValue = 500.0;
	self.delayStepperB.increment = 0.5;
	self.delayStepperB.valueWraps = NO;
	self.delayStepperB.doubleValue = initDelayB;
	self.delayStepperB.target = self;
	self.delayStepperB.action = @selector(delayBStepperChanged:);
	[boxB.contentView addSubview:self.delayStepperB];

	self.meterB = [[AudioLevelMeterView alloc] initWithFrame:NSMakeRect(240, 152, 84, 20)];
	[boxB.contentView addSubview:self.meterB];

	self.routingMatrixB = [[RoutingMatrixView alloc] initWithFrame:NSMakeRect(14, 4, 310, 56)];
	self.routingMatrixB.outputA = NO;
	self.routingMatrixB.routingDelegate = self;
	self.routingMatrixB.toolTip = @"Final routing stage. Click a cell to inspect or set its gain; right-click to toggle a route; Command-click to disable.";
	[boxB.contentView addSubview:self.routingMatrixB];

	/* 6. Bottom Status & Info Bar. Routine messages use a quiet pill and
	 * return to this keyboard-shortcut hint automatically. */
	self.statusBannerBadge = [[NSView alloc] initWithFrame:NSMakeRect(20, 10, 700, 26)];
	self.statusBannerBadge.wantsLayer = YES;
	self.statusBannerBadge.layer.cornerRadius = 13.0;
	self.statusBannerBadge.layer.masksToBounds = YES;
	self.statusBannerBadge.hidden = YES;
	[cv addSubview:self.statusBannerBadge];

	self.errorBannerLabel = [self makeLabel:kDefaultStatusHint bold:NO size:11 color:[NSColor secondaryLabelColor]];
	self.errorBannerLabel.frame = NSMakeRect(30, 14, 680, 18);
	[cv addSubview:self.errorBannerLabel];

	[self updateValueLabels];
	/* AppKit's default origin is at the bottom of an unflipped document view;
	 * open at the status section at the top instead. */
	[self.window layoutIfNeeded];
	NSRect clipBounds = self.controllerScrollView.contentView.bounds;
	CGFloat topOrigin = fmax(0.0, NSHeight(cv.bounds) - NSHeight(clipBounds));
	[self.controllerScrollView.contentView scrollToPoint:NSMakePoint(0.0, topOrigin)];
	[self.controllerScrollView reflectScrolledClipView:self.controllerScrollView.contentView];
	self.window.initialFirstResponder = self.controllerScrollView;
	[self installGainFieldFocusMonitor];
	[self.window makeKeyAndOrderFront:nil];
	[NSApp activateIgnoringOtherApps:YES];
}

- (void)installGainFieldFocusMonitor {
	if (self.gainFieldFocusMonitor != nil) {
		[NSEvent removeMonitor:self.gainFieldFocusMonitor];
	}

	__weak AppDelegate *weakSelf = self;
	self.gainFieldFocusMonitor = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskLeftMouseDown
																		 handler:^NSEvent *(NSEvent *event) {
		AppDelegate *strongSelf = weakSelf;
		if (strongSelf == nil || event.window != strongSelf.window) return event;

		NSResponder *firstResponder = strongSelf.window.firstResponder;
		BOOL editingGain = firstResponder == strongSelf.trimValueLabelA ||
						   firstResponder == strongSelf.trimValueLabelB ||
						   firstResponder == strongSelf.trimValueLabelA.currentEditor ||
						   firstResponder == strongSelf.trimValueLabelB.currentEditor;
		if (!editingGain) return event;

		NSPoint location = [strongSelf.window.contentView convertPoint:event.locationInWindow fromView:nil];
		NSView *hitView = [strongSelf.window.contentView hitTest:location];
		BOOL clickedGainField = hitView == strongSelf.trimValueLabelA ||
								hitView == strongSelf.trimValueLabelB ||
								[hitView isDescendantOf:strongSelf.trimValueLabelA] ||
								[hitView isDescendantOf:strongSelf.trimValueLabelB];
		if (!clickedGainField) [strongSelf.window makeFirstResponder:nil];
		return event;
	}];
}

/*===========================================================================
 * Audio Lifecycle & Hot-Plug Listeners
 *===========================================================================*/

- (void)requestMicrophonePermissionAndStartAudio {
	AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
	if (status == AVAuthorizationStatusAuthorized) {
		[self resolvePinnedStatusIssue:kStatusIssueMicrophone];
		[self refreshDevicesAndStreams];
	} else if (status == AVAuthorizationStatusNotDetermined) {
		[AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL granted) {
			dispatch_async(dispatch_get_main_queue(), ^{
				if (granted) {
					[self resolvePinnedStatusIssue:kStatusIssueMicrophone];
					[self refreshDevicesAndStreams];
				} else {
					[self showPinnedStatusMessage:@"Microphone access denied. Enable Dual Output in System Preferences → Security & Privacy → Privacy → Microphone."
											 issue:kStatusIssueMicrophone];
				}
			});
		}];
	} else {
		[self showPinnedStatusMessage:@"Microphone access is required to read from Dual Output. Grant access in System Preferences → Security & Privacy → Microphone."
								 issue:kStatusIssueMicrophone];
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
		[self showPinnedStatusMessage:@"Dual Output driver is not installed. Run ./install.sh first."
								 issue:kStatusIssueVirtualDevice];
	} else {
		[self resolvePinnedStatusIssue:kStatusIssueVirtualDevice];
		double vRate = GetDeviceNominalSampleRate(vDev);
		if (!atomic_load(&gEngine.inputRunning) || gEngine.virtualDeviceID != vDev ||
			fabs(vRate - atomic_load(&gEngine.virtualSampleRate)) > 1.0) {
			NSString *errStr = nil;
			if (StartVirtualInputUnit(vDev, &errStr) == noErr) {
				[self attachSampleRateListenerForDevice:vDev];
				self.virtualStatusLabel.textColor = [NSColor systemGreenColor];
				self.virtualStatusLabel.stringValue = [NSString stringWithFormat:@"Dual Output Active — %.0f Hz Stereo Float32", vRate];
				self.statusMenuHeaderItem.title = [NSString stringWithFormat:@"Dual Output: Active (%.0f Hz)", vRate];
				[self resolvePinnedStatusIssue:kStatusIssueVirtualDeviceStream];
			} else {
				self.virtualStatusLabel.textColor = [NSColor systemRedColor];
				self.virtualStatusLabel.stringValue = errStr ?: @"Failed starting Dual Output capture";
				self.statusMenuHeaderItem.title = @"Dual Output: Stream Error";
				[self showPinnedStatusMessage:errStr ?: @"Failed starting Dual Output capture"
										 issue:kStatusIssueVirtualDeviceStream];
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
	[self updatePresetIndicator];
}

- (void)reconcileOutputPath:(OutputPathState *)path
					  popup:(NSPopUpButton *)popup
				 desiredUID:(NSString *)desiredUID
				statusLabel:(NSTextField *)statusLabel
					  label:(NSString *)label {
	NSString *statusIssue = path == &gEngine.pathA ? kStatusIssueOutputA : kStatusIssueOutputB;
	if (desiredUID == nil || desiredUID.length == 0) {
		[popup selectItemAtIndex:0];
		StopOutputPath(path);
		statusLabel.textColor = [NSColor secondaryLabelColor];
		statusLabel.stringValue = [NSString stringWithFormat:@"%@: No device selected", label];
		[self resolvePinnedStatusIssue:statusIssue];
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
		[self showPinnedStatusMessage:[NSString stringWithFormat:@"%@ device disconnected — waiting for reconnect.", label]
								 issue:statusIssue];
		return;
	}

	[popup selectItemAtIndex:foundIdx + 1];

	double currentOutRate = atomic_load(&path->outputSampleRate);
	if (!atomic_load(&path->streamRunning) || path->deviceID != foundItem.deviceID ||
		fabs(currentOutRate - foundItem.sampleRate) > 1.0) {
		NSString *errStr = nil;
		if (StartOutputPath(path, foundItem.deviceID, &errStr) == noErr) {
			[self attachSampleRateListenerForDevice:foundItem.deviceID];
			[self resolvePinnedStatusIssue:statusIssue];
			[self showStatusMessage:[NSString stringWithFormat:@"%@ connected to %@ (%.0f Hz).", label, foundItem.name, foundItem.sampleRate]
							 level:StatusLevelInfo
					autoClearAfter:4.0];
		} else {
			statusLabel.textColor = [NSColor systemRedColor];
			statusLabel.stringValue = errStr ?: @"Error starting stream";
			[self showPinnedStatusMessage:errStr ?: @"Error starting output stream"
									 issue:statusIssue];
		}
	}
}

/*===========================================================================
 * Preset UI Actions
 *===========================================================================*/

- (void)presetPopupChanged:(id)sender {
	NSMenuItem *item = [sender isKindOfClass:[NSMenuItem class]] ? sender : self.presetPopup.selectedItem;
	NSString *name = [item.representedObject isKindOfClass:[NSString class]] ? item.representedObject : nil;
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
	alert.messageText = @"Save New Dual Output Preset";
	alert.informativeText = @"Enter a name for this new preset (saves devices, routing matrices, enable/mono toggles, volume, balance, trim, delays, and crossover settings):";
	[alert addButtonWithTitle:@"Save"];
	[alert addButtonWithTitle:@"Cancel"];

	NSTextField *inputField = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 260, 24)];
	inputField.stringValue = @"New Preset";
	alert.accessoryView = inputField;

	[self showControllerWindow:nil];
	NSModalResponse resp = [alert runModal];
	if (resp == NSAlertFirstButtonReturn) {
		NSString *name = [inputField.stringValue stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
		if (name.length == 0) {
			[self showStatusMessage:@"Preset name cannot be empty."
							 level:StatusLevelWarning
					autoClearAfter:4.0];
			return;
		}
		if (self.savedPresets[name] != nil) {
			if (![self confirmOverwritePresetNamed:name]) return;
		}
		[self saveCurrentStateAsPresetNamed:name];
	}
}

- (void)deletePresetClicked:(id)sender {
	NSString *name = [sender isKindOfClass:[NSMenuItem class]] &&
					 [((NSMenuItem *)sender).representedObject isKindOfClass:[NSString class]]
		? ((NSMenuItem *)sender).representedObject : self.activePresetName;
	if (name.length == 0 || self.savedPresets[name] == nil) {
		[self showStatusMessage:@"Choose a saved preset to delete."
						 level:StatusLevelWarning
				autoClearAfter:4.0];
		return;
	}
	if (![self confirmDeletePresetNamed:name]) return;

	[self registerUndoForCurrentStateWithActionName:@"Delete Preset"];
	[self.savedPresets removeObjectForKey:name];
	if ([name isEqualToString:self.activePresetName]) self.activePresetName = @"";
	[self rebuildPresetsUI];
	[self savePreferences];
	[self showStatusMessage:[NSString stringWithFormat:@"Deleted preset \"%@\".", name]
					 level:StatusLevelSuccess
			autoClearAfter:4.0];
}

- (void)overwritePresetClicked:(NSMenuItem *)sender {
	NSString *name = [sender.representedObject isKindOfClass:[NSString class]] ? sender.representedObject : nil;
	if (name.length == 0 || self.savedPresets[name] == nil) return;
	if (![self confirmOverwritePresetNamed:name]) return;
	[self saveCurrentStateAsPresetNamed:name];
}

- (void)renamePresetClicked:(NSMenuItem *)sender {
	NSString *oldName = [sender.representedObject isKindOfClass:[NSString class]] ? sender.representedObject : nil;
	if (oldName.length == 0 || self.savedPresets[oldName] == nil) return;

	NSAlert *alert = [[NSAlert alloc] init];
	alert.messageText = @"Rename Dual Output Preset";
	alert.informativeText = [NSString stringWithFormat:@"Enter a new name for \"%@\".", oldName];
	[alert addButtonWithTitle:@"Rename"];
	[alert addButtonWithTitle:@"Cancel"];
	NSTextField *inputField = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 260, 24)];
	inputField.stringValue = oldName;
	alert.accessoryView = inputField;

	[self showControllerWindow:nil];
	if ([alert runModal] != NSAlertFirstButtonReturn) return;

	NSString *newName = [inputField.stringValue stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
	if (newName.length == 0) {
		[self showStatusMessage:@"Preset name cannot be empty."
						 level:StatusLevelWarning
				autoClearAfter:4.0];
		return;
	}
	if ([newName isEqualToString:oldName]) return;
	if (self.savedPresets[newName] != nil && ![self confirmReplacePresetWhileRenaming:oldName withName:newName]) return;

	[self registerUndoForCurrentStateWithActionName:@"Rename Preset"];
	NSDictionary *preset = self.savedPresets[oldName];
	[self.savedPresets removeObjectForKey:oldName];
	self.savedPresets[newName] = preset;
	if ([self.activePresetName isEqualToString:oldName]) self.activePresetName = newName;
	[self rebuildPresetsUI];
	[self savePreferences];
	[self showStatusMessage:[NSString stringWithFormat:@"Renamed preset \"%@\" to \"%@\".", oldName, newName]
					 level:StatusLevelSuccess
			autoClearAfter:4.0];
}

- (void)saveCurrentStateAsPresetNamed:(NSString *)name {
	[self registerUndoForCurrentStateWithActionName:@"Save Preset"];
	self.savedPresets[name] = [self captureCurrentStateAsPresetDictionary];
	self.activePresetName = name;
	[self rebuildPresetsUI];
	[self savePreferences];
	[self showStatusMessage:[NSString stringWithFormat:@"Saved preset \"%@\".", name]
					 level:StatusLevelSuccess
			autoClearAfter:4.0];
}

- (BOOL)confirmOverwritePresetNamed:(NSString *)name {
	NSAlert *alert = [[NSAlert alloc] init];
	alert.messageText = @"Overwrite Preset?";
	alert.informativeText = [NSString stringWithFormat:@"\"%@\" will be replaced with the current settings. This cannot be undone.", name];
	NSButton *overwriteButton = [alert addButtonWithTitle:@"Overwrite"];
	overwriteButton.hasDestructiveAction = YES;
	[alert addButtonWithTitle:@"Cancel"];
	[self showControllerWindow:nil];
	return [alert runModal] == NSAlertFirstButtonReturn;
}

- (BOOL)confirmDeletePresetNamed:(NSString *)name {
	NSAlert *alert = [[NSAlert alloc] init];
	alert.messageText = @"Delete Preset?";
	alert.informativeText = [NSString stringWithFormat:@"\"%@\" will be permanently deleted. This cannot be undone.", name];
	NSButton *deleteButton = [alert addButtonWithTitle:@"Delete"];
	deleteButton.hasDestructiveAction = YES;
	[alert addButtonWithTitle:@"Cancel"];
	[self showControllerWindow:nil];
	return [alert runModal] == NSAlertFirstButtonReturn;
}

- (BOOL)confirmReplacePresetWhileRenaming:(NSString *)oldName withName:(NSString *)newName {
	NSAlert *alert = [[NSAlert alloc] init];
	alert.messageText = @"Replace Existing Preset?";
	alert.informativeText = [NSString stringWithFormat:@"Renaming \"%@\" to \"%@\" will replace the existing preset. This cannot be undone.", oldName, newName];
	NSButton *replaceButton = [alert addButtonWithTitle:@"Replace"];
	replaceButton.hasDestructiveAction = YES;
	[alert addButtonWithTitle:@"Cancel"];
	[self showControllerWindow:nil];
	return [alert runModal] == NSAlertFirstButtonReturn;
}

/*===========================================================================
 * Audio Control Actions
 *===========================================================================*/

- (void)setVirtualDeviceAsSystemOutput:(id)sender {
	(void)sender;
	AudioDeviceID vDev = FindVirtualDeviceID();
	if (vDev == kAudioObjectUnknown) {
		[self showPinnedStatusMessage:@"Dual Output driver is not installed. Run ./install.sh first."
								 issue:kStatusIssueVirtualDevice];
		return;
	}
	[self resolvePinnedStatusIssue:kStatusIssueVirtualDevice];
	AudioObjectPropertyAddress defaultOutAddr = {
		kAudioHardwarePropertyDefaultOutputDevice,
		kAudioObjectPropertyScopeGlobal,
		kAudioObjectPropertyElementMain
	};
	OSStatus err = AudioObjectSetPropertyData(kAudioObjectSystemObject, &defaultOutAddr, 0, NULL, sizeof(AudioDeviceID), &vDev);
	if (err == noErr) {
		[self resolvePinnedStatusIssue:kStatusIssueSystemOutput];
		[self showStatusMessage:@"Dual Output is now the active macOS System Output device."
						 level:StatusLevelSuccess
				autoClearAfter:4.0];
	} else {
		[self showPinnedStatusMessage:[NSString stringWithFormat:@"Could not set default output device (%d).", (int)err]
								 issue:kStatusIssueSystemOutput];
	}
}

- (void)outputADeviceSelected:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output A Device"];
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
	[self registerUndoForCurrentStateWithActionName:@"Change Output B Device"];
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
	[self registerUndoForCurrentStateWithActionName:@"Toggle Output A"];
	bool en = (self.enableCheckboxA.state == NSControlStateValueOn);
	atomic_store_explicit(&gEngine.pathA.enabled, en, memory_order_relaxed);
	[self syncStatusBarMenuChecks];
	[self savePreferences];
}

- (void)menuToggleEnableA:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Output A"];
	bool en = !atomic_load_explicit(&gEngine.pathA.enabled, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathA.enabled, en, memory_order_relaxed);
	self.enableCheckboxA.state = en ? NSControlStateValueOn : NSControlStateValueOff;
	[self syncStatusBarMenuChecks];
	[self savePreferences];
}

- (void)toggleMonoA:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Output A Mono"];
	bool m = (self.monoCheckboxA.state == NSControlStateValueOn);
	atomic_store_explicit(&gEngine.pathA.mono, m, memory_order_relaxed);
	[self updateRoutingMatrixViews];
	[self syncStatusBarMenuChecks];
	[self savePreferences];
}

- (void)menuToggleMonoA:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Output A Mono"];
	bool m = !atomic_load_explicit(&gEngine.pathA.mono, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathA.mono, m, memory_order_relaxed);
	self.monoCheckboxA.state = m ? NSControlStateValueOn : NSControlStateValueOff;
	[self updateRoutingMatrixViews];
	[self syncStatusBarMenuChecks];
	[self savePreferences];
}

- (void)toggleEnableB:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Output B"];
	bool en = (self.enableCheckboxB.state == NSControlStateValueOn);
	atomic_store_explicit(&gEngine.pathB.enabled, en, memory_order_relaxed);
	[self syncStatusBarMenuChecks];
	[self savePreferences];
}

- (void)menuToggleEnableB:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Output B"];
	bool en = !atomic_load_explicit(&gEngine.pathB.enabled, memory_order_relaxed);
	atomic_store_explicit(&gEngine.pathB.enabled, en, memory_order_relaxed);
	self.enableCheckboxB.state = en ? NSControlStateValueOn : NSControlStateValueOff;
	[self syncStatusBarMenuChecks];
	[self savePreferences];
}

- (void)toggleMonoB:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Output B Mono"];
	bool m = (self.monoCheckboxB.state == NSControlStateValueOn);
	atomic_store_explicit(&gEngine.pathB.mono, m, memory_order_relaxed);
	[self updateRoutingMatrixViews];
	[self syncStatusBarMenuChecks];
	[self savePreferences];
}

- (void)menuToggleMonoB:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Output B Mono"];
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
	RoutingGainPopoverContentView *content = [[RoutingGainPopoverContentView alloc] initWithFrame:NSMakeRect(0, 0, 250, 156)];
	controller.view = content;
	self.routingGainPopover.contentViewController = controller;

	self.routingGainPopoverTitleLabel = [self makeLabel:@"Route Gain" bold:YES size:11 color:nil];
	self.routingGainPopoverTitleLabel.alignment = NSTextAlignmentCenter;
	self.routingGainPopoverTitleLabel.frame = NSMakeRect(50, 128, 190, 20);
	[content addSubview:self.routingGainPopoverTitleLabel];

	self.routingGainFader = [[NSSlider alloc] initWithFrame:NSMakeRect(14, 18, 28, 112)];
	self.routingGainFader.vertical = YES;
	self.routingGainFader.minValue = -60.0;
	self.routingGainFader.maxValue = 12.0;
	self.routingGainFader.continuous = YES;
	self.routingGainFader.target = self;
	self.routingGainFader.action = @selector(routingGainFaderChanged:);
	self.routingGainFader.toolTip = @"Route gain, from −∞ (disabled) to +12 dB";
	[content addSubview:self.routingGainFader];
	content.routingGainFader = self.routingGainFader;

	self.routingGainValueField = [[NSTextField alloc] initWithFrame:NSMakeRect(50, 92, 135, 25)];
	self.routingGainValueField.alignment = NSTextAlignmentRight;
	self.routingGainValueField.target = self;
	self.routingGainValueField.action = @selector(routingGainValueChanged:);
	[content addSubview:self.routingGainValueField];

	NSTextField *dbLabel = [self makeLabel:@"dB" bold:NO size:11 color:[NSColor secondaryLabelColor]];
	dbLabel.frame = NSMakeRect(189, 96, 18, 16);
	[content addSubview:dbLabel];

	NSTextField *hint = [self makeLabel:@"Use the fader or enter an exact value." bold:NO size:10 color:[NSColor secondaryLabelColor]];
	hint.frame = NSMakeRect(50, 55, 180, 22);
	hint.lineBreakMode = NSLineBreakByWordWrapping;
	[content addSubview:hint];

	NSButton *unityButton = [[NSButton alloc] initWithFrame:NSMakeRect(50, 18, 94, 26)];
	unityButton.bezelStyle = NSBezelStyleRounded;
	unityButton.title = @"0 dB (Unity)";
	unityButton.target = self;
	unityButton.action = @selector(resetActiveRoutingGainToUnity:);
	[content addSubview:unityButton];

	NSButton *disableButton = [[NSButton alloc] initWithFrame:NSMakeRect(148, 18, 92, 26)];
	disableButton.bezelStyle = NSBezelStyleRounded;
	disableButton.title = @"-∞";
	disableButton.toolTip = @"Disable Route";
	disableButton.target = self;
	disableButton.action = @selector(disableActiveRoutingCrosspoint:);
	[content addSubview:disableButton];
}

- (void)updateRoutingGainPopoverHeader {
	NSString *outputLabel = self.activeRoutingMatrix.outputA ? @"A" : @"B";
	NSString *inputLabel = self.activeRoutingMono ? @"M" : (self.activeRoutingInputLeft ? @"L" : @"R");
	self.routingGainPopoverTitleLabel.stringValue = [NSString stringWithFormat:@"Output %@ \u2022 In %@ \u2192 Out %u",
													   outputLabel, inputLabel, self.activeRoutingChannel + 1U];
}

- (OutputPathState *)activeRoutingPath {
	return self.activeRoutingMatrix.outputA ? &gEngine.pathA : &gEngine.pathB;
}

- (void)updateRoutingGainPopoverControlsForDb:(double)db {
	BOOL disabled = RouteGainIsDisabled(db);
	self.routingGainFader.doubleValue = disabled ? kRouteGainDisabledDb : db;
	self.routingGainValueField.stringValue = disabled ? @"-∞" : [NSString stringWithFormat:@"%.1f", db];
}

- (void)setActiveRoutingGainDb:(double)db {
	[self registerUndoForCurrentStateWithActionName:@"Change Route Gain"];
	OutputPathState *path = [self activeRoutingPath];
	_Atomic uint32_t *mask = self.activeRoutingMono ? &path->monoRouteMask
							 : (self.activeRoutingInputLeft ? &path->routeMaskLeft : &path->routeMaskRight);
	_Atomic float *gains = self.activeRoutingMono ? path->monoRouteGain
						  : (self.activeRoutingInputLeft ? path->routeGainLeft : path->routeGainRight);
	uint32_t bit = 1U << self.activeRoutingChannel;
	uint32_t current = atomic_load_explicit(mask, memory_order_relaxed);
	if (RouteGainIsDisabled(db)) {
		db = kRouteGainDisabledDb;
		atomic_store_explicit(mask, current & ~bit, memory_order_relaxed);
	} else {
		if (!isfinite(db)) db = 0.0;
		db = fmin(12.0, fmax(kRouteGainDisabledDb, db));
		atomic_store_explicit(&gains[self.activeRoutingChannel], RouteGainDbToLinear(db), memory_order_relaxed);
		atomic_store_explicit(mask, current | bit, memory_order_relaxed);
	}
	[self updateRoutingGainPopoverControlsForDb:db];
	[self updateRoutingMatrixViews];
	[self savePreferences];
}

- (void)resetActiveRoutingGainToUnity:(id)sender {
	(void)sender;
	[self setActiveRoutingGainDb:0.0];
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

	self.activeRoutingMatrix = view;
	self.activeRoutingInputLeft = inputLeft;
	self.activeRoutingMono = mono;
	self.activeRoutingChannel = destinationChannel;
	_Atomic float *gains = mono ? path->monoRouteGain
						  : (inputLeft ? path->routeGainLeft : path->routeGainRight);
	double db = (current & bit) != 0U
		? RouteGainLinearToDb(atomic_load_explicit(&gains[destinationChannel], memory_order_relaxed))
		: kRouteGainDisabledDb;
	[self ensureRoutingGainPopover];
	[self updateRoutingGainPopoverHeader];
	[self updateRoutingGainPopoverControlsForDb:db];
	[self.routingGainPopover performClose:nil];
	[self.routingGainPopover showRelativeToRect:[view rectForInputLeft:inputLeft channel:destinationChannel]
										 ofView:view
								  preferredEdge:NSMaxYEdge];
}

- (void)routingMatrixView:(RoutingMatrixView *)view
	didRequestDisableInputLeft:(BOOL)inputLeft
		 destinationChannel:(uint32_t)destinationChannel {
	if (destinationChannel >= kMaxOutputChannels) return;
	[self registerUndoForCurrentStateWithActionName:@"Disable Route"];
	OutputPathState *path = view.outputA ? &gEngine.pathA : &gEngine.pathB;
	BOOL mono = atomic_load_explicit(&path->mono, memory_order_relaxed);
	_Atomic uint32_t *mask = mono ? &path->monoRouteMask
							 : (inputLeft ? &path->routeMaskLeft : &path->routeMaskRight);
	uint32_t bit = 1U << destinationChannel;
	uint32_t current = atomic_load_explicit(mask, memory_order_relaxed);
	atomic_store_explicit(mask, current & ~bit, memory_order_relaxed);
	[self.routingGainPopover performClose:nil];
	[self updateRoutingMatrixViews];
	[self savePreferences];
}

- (void)routingMatrixView:(RoutingMatrixView *)view
	 didRequestToggleInputLeft:(BOOL)inputLeft
		destinationChannel:(uint32_t)destinationChannel {
	if (destinationChannel >= kMaxOutputChannels) return;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Route"];
	OutputPathState *path = view.outputA ? &gEngine.pathA : &gEngine.pathB;
	BOOL mono = atomic_load_explicit(&path->mono, memory_order_relaxed);
	_Atomic uint32_t *mask = mono ? &path->monoRouteMask
							 : (inputLeft ? &path->routeMaskLeft : &path->routeMaskRight);
	uint32_t bit = 1U << destinationChannel;
	uint32_t current = atomic_load_explicit(mask, memory_order_relaxed);
	if ((current & bit) != 0U) {
		atomic_store_explicit(mask, current & ~bit, memory_order_relaxed);
	} else {
		atomic_store_explicit(mask, current | bit, memory_order_relaxed);
	}
	[self.routingGainPopover performClose:nil];
	[self updateRoutingMatrixViews];
	[self savePreferences];
}

- (void)routingGainFaderChanged:(id)sender {
	(void)sender;
	[self setActiveRoutingGainDb:self.routingGainFader.doubleValue];
}

- (void)routingGainValueChanged:(id)sender {
	(void)sender;
	double db = RouteGainTextIsDisabled(self.routingGainValueField.stringValue)
		? -INFINITY : self.routingGainValueField.doubleValue;
	[self setActiveRoutingGainDb:db];
}

- (void)disableActiveRoutingCrosspoint:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Disable Route"];
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
	[self registerUndoForCurrentStateWithActionName:@"Change Master Volume"];
	gEngine.masterVolume = self.masterSlider.floatValue;
	UpdateEngineTargetGains();
	[self updateValueLabels];
	[self savePreferences];
}

- (void)balanceChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Balance"];
	gEngine.balance = self.balanceSlider.floatValue;
	UpdateEngineTargetGains();
	[self updateValueLabels];
	[self savePreferences];
}

- (void)trimAChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output A Trim"];
	gEngine.trimDbA = self.trimSliderA.floatValue;
	UpdateEngineTargetGains();
	[self updateValueLabels];
	[self savePreferences];
}

- (void)trimAValueChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output A Trim"];
	gEngine.trimDbA = fminf(6.0f, fmaxf(-24.0f, self.trimValueLabelA.floatValue));
	self.trimSliderA.floatValue = gEngine.trimDbA;
	UpdateEngineTargetGains();
	[self updateValueLabels];
	[self savePreferences];
}

- (void)trimBChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output B Trim"];
	gEngine.trimDbB = self.trimSliderB.floatValue;
	UpdateEngineTargetGains();
	[self updateValueLabels];
	[self savePreferences];
}

- (void)trimBValueChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output B Trim"];
	gEngine.trimDbB = fminf(6.0f, fmaxf(-24.0f, self.trimValueLabelB.floatValue));
	self.trimSliderB.floatValue = gEngine.trimDbB;
	UpdateEngineTargetGains();
	[self updateValueLabels];
	[self savePreferences];
}

- (void)delayASliderChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output A Delay"];
	double ms = round(self.delaySliderA.doubleValue * 2.0) / 2.0;
	self.delayStepperA.doubleValue = ms;
	atomic_store_explicit(&gEngine.pathA.targetDelayMs, ms, memory_order_relaxed);
	[self updateValueLabels];
	[self savePreferences];
}

- (void)delayAStepperChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output A Delay"];
	double ms = self.delayStepperA.doubleValue;
	self.delaySliderA.doubleValue = ms;
	atomic_store_explicit(&gEngine.pathA.targetDelayMs, ms, memory_order_relaxed);
	[self updateValueLabels];
	[self savePreferences];
}

- (void)delayBSliderChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output B Delay"];
	double ms = round(self.delaySliderB.doubleValue * 2.0) / 2.0;
	self.delayStepperB.doubleValue = ms;
	atomic_store_explicit(&gEngine.pathB.targetDelayMs, ms, memory_order_relaxed);
	[self updateValueLabels];
	[self savePreferences];
}

- (void)delayBStepperChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output B Delay"];
	double ms = self.delayStepperB.doubleValue;
	self.delaySliderB.doubleValue = ms;
	atomic_store_explicit(&gEngine.pathB.targetDelayMs, ms, memory_order_relaxed);
	[self updateValueLabels];
	[self savePreferences];
}

/*===========================================================================
 * 31-Band Graphic EQ Controls & Pop-Out Editor Window
 *===========================================================================*/

- (NSArray<NSNumber *> *)graphicEQBandGainValues {
	NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:kGraphicEQBandCount];
	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		float db = ClampGraphicEQGainDb(atomic_load_explicit(&gEngine.graphicEQ.bandGainDb[i], memory_order_relaxed));
		[values addObject:@(round((double)db * 10.0) / 10.0)];
	}
	return values;
}

- (void)applyGraphicEQBandGainValues:(NSArray<NSNumber *> *)values {
	if (![values isKindOfClass:[NSArray class]]) return;
	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		float db = (i < (int)values.count) ? ClampGraphicEQGainDb(values[(NSUInteger)i].floatValue) : 0.0f;
		atomic_store_explicit(&gEngine.graphicEQ.bandGainDb[i], db, memory_order_relaxed);
	}
}

- (void)updateGraphicEQControls {
	bool enabled = atomic_load_explicit(&gEngine.graphicEQ.enabled, memory_order_relaxed);
	self.graphicEQEnabledCheckbox.state = enabled ? NSControlStateValueOn : NSControlStateValueOff;
	if (self.graphicEQWindowEnabledCheckbox != nil) {
		self.graphicEQWindowEnabledCheckbox.state = enabled ? NSControlStateValueOn : NSControlStateValueOff;
	}

	int activeBands = 0;
	float maxBoost = 0.0f;
	float maxCut   = 0.0f;
	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		float db = ClampGraphicEQGainDb(atomic_load_explicit(&gEngine.graphicEQ.bandGainDb[i], memory_order_relaxed));
		if (fabsf(db) >= 0.05f) {
			activeBands += 1;
			if (db > maxBoost) maxBoost = db;
			if (db < maxCut)   maxCut   = db;
		}
		if (i < (int)self.graphicEQSliders.count) {
			self.graphicEQSliders[(NSUInteger)i].floatValue = db;
			self.graphicEQSliders[(NSUInteger)i].enabled = enabled;
		}
		if (i < (int)self.graphicEQValueLabels.count) {
			self.graphicEQValueLabels[(NSUInteger)i].stringValue =
				(fabsf(db) < 0.05f) ? @"0.0" : [NSString stringWithFormat:@"%+.1f", db];
			self.graphicEQValueLabels[(NSUInteger)i].textColor =
				enabled ? [NSColor labelColor] : [NSColor secondaryLabelColor];
		}
	}

	if (!enabled) {
		self.graphicEQSummaryLabel.stringValue = @"Bypassed — click graph to edit 31-band sliders";
	} else if (activeBands == 0) {
		self.graphicEQSummaryLabel.stringValue = @"Active (Flat 0.0 dB) — click graph to edit 31-band sliders";
	} else {
		self.graphicEQSummaryLabel.stringValue = [NSString stringWithFormat:
			@"Active (%d/31 bands shaped, %+.1f / %+.1f dB) — click graph to edit",
			activeBands, maxBoost, maxCut];
	}

	[self.graphicEQPreviewView setNeedsDisplay:YES];
	[self.graphicEQWindowPreviewView setNeedsDisplay:YES];
}

- (void)graphicEQEnabledChanged:(NSButton *)sender {
	[self registerUndoForCurrentStateWithActionName:@"Toggle Graphic EQ"];
	bool enabled = (sender.state == NSControlStateValueOn);
	atomic_store_explicit(&gEngine.graphicEQ.enabled, enabled, memory_order_relaxed);
	atomic_fetch_add_explicit(&gEngine.graphicEQ.revision, 1U, memory_order_relaxed);
	[self updateGraphicEQControls];
	[self savePreferences];
}

- (void)graphicEQSliderChanged:(NSSlider *)sender {
	NSInteger band = sender.tag;
	if (band < 0 || band >= kGraphicEQBandCount) return;
	[self registerUndoForCurrentStateWithActionName:@"Adjust Graphic EQ"];
	float db = roundf(ClampGraphicEQGainDb(sender.floatValue) * 10.0f) / 10.0f;
	atomic_store_explicit(&gEngine.graphicEQ.bandGainDb[band], db, memory_order_relaxed);
	atomic_fetch_add_explicit(&gEngine.graphicEQ.revision, 1U, memory_order_relaxed);
	[self updateGraphicEQControls];
	[self savePreferences];
}

- (void)flattenGraphicEQ:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Flatten Graphic EQ"];
	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		atomic_store_explicit(&gEngine.graphicEQ.bandGainDb[i], 0.0f, memory_order_relaxed);
	}
	atomic_fetch_add_explicit(&gEngine.graphicEQ.revision, 1U, memory_order_relaxed);
	[self updateGraphicEQControls];
	[self savePreferences];
}

- (void)graphicEQPreviewViewDidClick:(GraphicEQPreviewView *)view {
	(void)view;
	[self showGraphicEQWindow:nil];
}

- (void)buildGraphicEQWindowIfNeeded {
	if (self.graphicEQWindow != nil) return;

	NSRect frame = NSMakeRect(0, 0, 960, 410);
	NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable;
	self.graphicEQWindow = [[NSWindow alloc] initWithContentRect:frame
													   styleMask:style
														 backing:NSBackingStoreBuffered
														   defer:NO];
	self.graphicEQWindow.title = @"31-Band Graphic EQ — 1/3-Octave Pre-Crossover Equalizer";
	self.graphicEQWindow.releasedWhenClosed = NO;
	self.graphicEQWindow.delegate = self;
	[self.graphicEQWindow center];

	NSView *cv = self.graphicEQWindow.contentView;

	self.graphicEQWindowEnabledCheckbox = [NSButton checkboxWithTitle:@"Enable 31-Band Graphic EQ"
															   target:self
															   action:@selector(graphicEQEnabledChanged:)];
	self.graphicEQWindowEnabledCheckbox.frame = NSMakeRect(18, 372, 210, 22);
	[cv addSubview:self.graphicEQWindowEnabledCheckbox];

	NSTextField *hintLabel = [self makeLabel:@"Tip: Cmd-Z undoes; Shift-Cmd-Z redoes. Double-click or Option-click a fader to reset it."
										bold:NO
										size:11
									   color:[NSColor secondaryLabelColor]];
	hintLabel.frame = NSMakeRect(240, 374, 540, 16);
	[cv addSubview:hintLabel];

	NSButton *flattenButton = [[NSButton alloc] initWithFrame:NSMakeRect(812, 369, 130, 28)];
	flattenButton.bezelStyle = NSBezelStyleRounded;
	flattenButton.title = @"Flatten All (0 dB)";
	flattenButton.target = self;
	flattenButton.action = @selector(flattenGraphicEQ:);
	[cv addSubview:flattenButton];

	self.graphicEQWindowPreviewView = [[GraphicEQPreviewView alloc] initWithFrame:NSMakeRect(18, 286, 924, 76)];
	self.graphicEQWindowPreviewView.clickable = NO;
	[cv addSubview:self.graphicEQWindowPreviewView];

	NSBox *faderBox = [[NSBox alloc] initWithFrame:NSMakeRect(18, 16, 924, 260)];
	faderBox.titlePosition = NSNoTitle;
	[cv addSubview:faderBox];

	NSMutableArray<NSSlider *> *sliders = [NSMutableArray arrayWithCapacity:kGraphicEQBandCount];
	NSMutableArray<NSTextField *> *valLabels = [NSMutableArray arrayWithCapacity:kGraphicEQBandCount];

	CGFloat leftMargin = 10.0;
	CGFloat colSpacing = (900.0 - leftMargin) / (CGFloat)kGraphicEQBandCount;

	for (int i = 0; i < kGraphicEQBandCount; ++i) {
		CGFloat x = leftMargin + (CGFloat)i * colSpacing;

		NSTextField *valLbl = [self makeLabel:@"0.0" bold:NO size:8.5 color:[NSColor labelColor]];
		valLbl.font = [NSFont monospacedDigitSystemFontOfSize:8.5 weight:NSFontWeightMedium];
		valLbl.alignment = NSTextAlignmentCenter;
		valLbl.frame = NSMakeRect(x - 2.0, 226.0, colSpacing + 4.0, 14.0);
		[faderBox.contentView addSubview:valLbl];
		[valLabels addObject:valLbl];

		GraphicEQBandSlider *slider = [[GraphicEQBandSlider alloc] initWithFrame:NSMakeRect(x + (colSpacing - 20.0) * 0.5, 32.0, 20.0, 190.0)];
		slider.vertical = YES;
		slider.minValue = kGraphicEQMinDb;
		slider.maxValue = kGraphicEQMaxDb;
		slider.floatValue = 0.0f;
		slider.numberOfTickMarks = 5;
		slider.continuous = YES;
		slider.tag = i;
		slider.target = self;
		slider.action = @selector(graphicEQSliderChanged:);
		slider.toolTip = [NSString stringWithFormat:@"%@ Hz Band (-12 to +12 dB)", kGraphicEQBandLabels[i]];
		[faderBox.contentView addSubview:slider];
		[sliders addObject:slider];

		NSTextField *freqLbl = [self makeLabel:kGraphicEQBandLabels[i] bold:YES size:8.0 color:[NSColor secondaryLabelColor]];
		freqLbl.font = [NSFont monospacedDigitSystemFontOfSize:7.5 weight:NSFontWeightSemibold];
		freqLbl.alignment = NSTextAlignmentCenter;
		freqLbl.frame = NSMakeRect(x - 3.0, 8.0, colSpacing + 6.0, 14.0);
		[faderBox.contentView addSubview:freqLbl];
	}

	self.graphicEQSliders = sliders;
	self.graphicEQValueLabels = valLabels;
}

- (void)showGraphicEQWindow:(id)sender {
	(void)sender;
	[self buildGraphicEQWindowIfNeeded];
	[self updateGraphicEQControls];
	[self.graphicEQWindow makeKeyAndOrderFront:nil];
	[NSApp activateIgnoringOtherApps:YES];
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
	NSColor *emphasisColor = enabled ? [NSColor labelColor] : [NSColor secondaryLabelColor];
	for (NSTextField *label in self.crossoverPrimaryLabels) {
		label.textColor = emphasisColor;
	}
	for (NSView *control in @[self.linkedCrossoverFrequencyLabel, self.linkedCrossoverSlopeLabel,
							  self.crossoverFrequencyLabelA, self.crossoverFrequencyLabelB,
							  self.crossoverSlopeLabelA, self.crossoverSlopeLabelB]) {
		if ([control isKindOfClass:[NSTextField class]]) {
			((NSTextField *)control).textColor = emphasisColor;
		}
	}
	self.crossoverLinkedCheckbox.state = linked ? NSControlStateValueOn : NSControlStateValueOff;
	self.crossoverLinkedCheckbox.enabled = enabled;

	self.linkedCrossoverFrequencySlider.doubleValue = CrossoverFrequencyToSliderValue(linkedFrequency);
	self.linkedCrossoverFrequencyLabel.stringValue = [NSString stringWithFormat:@"%.0f Hz", linkedFrequency];
	int linkedSlope = ClampCrossoverSlope(atomic_load_explicit(&gEngine.crossover.linkedSlopeDb, memory_order_relaxed));
	self.linkedCrossoverSlopeSlider.integerValue = linkedSlope;
	self.linkedCrossoverSlopeLabel.stringValue = [NSString stringWithFormat:@"%d dB/oct", linkedSlope];
	[self.linkedCrossoverAssignmentPopup selectItemWithTag:
		atomic_load_explicit(&gEngine.crossover.linkedOutputAHighPass, memory_order_relaxed) ? 1 : 0];

	[self.crossoverModePopupA selectItemWithTag:atomic_load_explicit(&gEngine.crossover.modeA, memory_order_relaxed)];
	[self.crossoverModePopupB selectItemWithTag:atomic_load_explicit(&gEngine.crossover.modeB, memory_order_relaxed)];
	self.crossoverFrequencySliderA.doubleValue = CrossoverFrequencyToSliderValue(frequencyA);
	self.crossoverFrequencySliderB.doubleValue = CrossoverFrequencyToSliderValue(frequencyB);
	self.crossoverFrequencyLabelA.stringValue = [NSString stringWithFormat:@"%.0f Hz", frequencyA];
	self.crossoverFrequencyLabelB.stringValue = [NSString stringWithFormat:@"%.0f Hz", frequencyB];
	int slopeA = ClampCrossoverSlope(atomic_load_explicit(&gEngine.crossover.slopeDbA, memory_order_relaxed));
	int slopeB = ClampCrossoverSlope(atomic_load_explicit(&gEngine.crossover.slopeDbB, memory_order_relaxed));
	self.crossoverSlopeSliderA.integerValue = slopeA;
	self.crossoverSlopeSliderB.integerValue = slopeB;
	self.crossoverSlopeLabelA.stringValue = [NSString stringWithFormat:@"%d dB/oct", slopeA];
	self.crossoverSlopeLabelB.stringValue = [NSString stringWithFormat:@"%d dB/oct", slopeB];

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
	[self.crossoverResponseView setNeedsDisplay:YES];
}

- (void)crossoverEnabledChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Toggle Crossover"];
	atomic_store_explicit(&gEngine.crossover.enabled,
						  self.crossoverEnabledCheckbox.state == NSControlStateValueOn,
						  memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)crossoverLinkedChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Crossover Link"];
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
	[self registerUndoForCurrentStateWithActionName:@"Change Crossover Frequency"];
	double frequency = CrossoverSliderValueToFrequency(self.linkedCrossoverFrequencySlider.doubleValue);
	atomic_store_explicit(&gEngine.crossover.linkedFrequency, frequency, memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)linkedCrossoverSlopeChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Crossover Slope"];
	atomic_store_explicit(&gEngine.crossover.linkedSlopeDb,
						  ClampCrossoverSlope((int)llround(self.linkedCrossoverSlopeSlider.doubleValue)),
						  memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)linkedCrossoverAssignmentChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Crossover Assignment"];
	atomic_store_explicit(&gEngine.crossover.linkedOutputAHighPass,
						  self.linkedCrossoverAssignmentPopup.selectedItem.tag == 1,
						  memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)crossoverModeAChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output A Crossover"];
	atomic_store_explicit(&gEngine.crossover.modeA, self.crossoverModePopupA.selectedItem.tag, memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)crossoverModeBChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output B Crossover"];
	atomic_store_explicit(&gEngine.crossover.modeB, self.crossoverModePopupB.selectedItem.tag, memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)crossoverFrequencyAChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output A Crossover"];
	atomic_store_explicit(&gEngine.crossover.frequencyA,
						  CrossoverSliderValueToFrequency(self.crossoverFrequencySliderA.doubleValue),
						  memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)crossoverFrequencyBChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output B Crossover"];
	atomic_store_explicit(&gEngine.crossover.frequencyB,
						  CrossoverSliderValueToFrequency(self.crossoverFrequencySliderB.doubleValue),
						  memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)crossoverSlopeAChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output A Crossover"];
	atomic_store_explicit(&gEngine.crossover.slopeDbA,
						  ClampCrossoverSlope((int)llround(self.crossoverSlopeSliderA.doubleValue)),
						  memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)crossoverSlopeBChanged:(id)sender {
	(void)sender;
	[self registerUndoForCurrentStateWithActionName:@"Change Output B Crossover"];
	atomic_store_explicit(&gEngine.crossover.slopeDbB,
						  ClampCrossoverSlope((int)llround(self.crossoverSlopeSliderB.doubleValue)),
						  memory_order_relaxed);
	[self updateCrossoverControls];
	[self savePreferences];
}

- (void)crossoverResponseView:(CrossoverResponseView *)view
	   didDragFrequency:(double)frequency
			 forOutputA:(BOOL)outputA {
	(void)view;
	[self registerUndoForCurrentStateWithActionName:@"Change Crossover Frequency"];
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
		self.balanceValueLabel.stringValue = @"A:100%  B:100%";
	} else if (b < 0.0f) {
		float pctB = sinf((float)M_PI_2 * (1.0f + b)) * 100.0f;
		self.balanceValueLabel.stringValue = [NSString stringWithFormat:@"A:100%%  B:%.0f%%", pctB];
	} else {
		float pctA = cosf((float)M_PI_2 * b) * 100.0f;
		self.balanceValueLabel.stringValue = [NSString stringWithFormat:@"A:%.0f%%  B:100%%", pctA];
	}

	self.trimValueLabelA.stringValue = [NSString stringWithFormat:@"%+.1f", gEngine.trimDbA];
	self.trimValueLabelB.stringValue = [NSString stringWithFormat:@"%+.1f", gEngine.trimDbB];

	self.delayValueLabelA.stringValue = [NSString stringWithFormat:@"%.1f ms", atomic_load(&gEngine.pathA.targetDelayMs)];
	self.delayValueLabelB.stringValue = [NSString stringWithFormat:@"%.1f ms", atomic_load(&gEngine.pathB.targetDelayMs)];
}

- (void)updateTelemetryUI {
	if (!self.window.isVisible || self.window.isMiniaturized || [NSApp isHidden]) return;

	[self.inputMeter setPeakLevelL:atomic_load_explicit(&gEngine.peakLevelL, memory_order_relaxed)
						peakLevelR:atomic_load_explicit(&gEngine.peakLevelR, memory_order_relaxed)
							  clipL:atomic_load_explicit(&gEngine.clippedL, memory_order_relaxed)
							  clipR:atomic_load_explicit(&gEngine.clippedR, memory_order_relaxed)];
	[self.meterA setPeakLevelL:atomic_load_explicit(&gEngine.pathA.peakLevelL, memory_order_relaxed)
					peakLevelR:atomic_load_explicit(&gEngine.pathA.peakLevelR, memory_order_relaxed)
						  clipL:atomic_load_explicit(&gEngine.pathA.clippedL, memory_order_relaxed)
						  clipR:atomic_load_explicit(&gEngine.pathA.clippedR, memory_order_relaxed)];
	[self.meterB setPeakLevelL:atomic_load_explicit(&gEngine.pathB.peakLevelL, memory_order_relaxed)
					peakLevelR:atomic_load_explicit(&gEngine.pathB.peakLevelR, memory_order_relaxed)
						  clipL:atomic_load_explicit(&gEngine.pathB.clippedL, memory_order_relaxed)
						  clipR:atomic_load_explicit(&gEngine.pathB.clippedR, memory_order_relaxed)];

	double fallbackRate = atomic_load_explicit(&gEngine.virtualSampleRate, memory_order_relaxed);
	if (!isfinite(fallbackRate) || fallbackRate < 8000.0) fallbackRate = 48000.0;
	double graphRateA = atomic_load_explicit(&gEngine.pathA.outputSampleRate, memory_order_relaxed);
	double graphRateB = atomic_load_explicit(&gEngine.pathB.outputSampleRate, memory_order_relaxed);
	if (!isfinite(graphRateA) || graphRateA < 8000.0) graphRateA = fallbackRate;
	if (!isfinite(graphRateB) || graphRateB < 8000.0) graphRateB = fallbackRate;
	if (fabs(graphRateA - self.displayedCrossoverSampleRateA) > 1.0 ||
		fabs(graphRateB - self.displayedCrossoverSampleRateB) > 1.0) {
		self.displayedCrossoverSampleRateA = graphRateA;
		self.displayedCrossoverSampleRateB = graphRateB;
		[self.crossoverResponseView setNeedsDisplay:YES];
	}

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
