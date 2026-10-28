/*
 * DualOutputPlugIn.c
 * Core Audio HAL AudioServerPlugIn virtual stereo device ("Dual Output")
 * Features:
 *   - Perceptual Cubic Volume Curve (s^3 / 60*log10(s) dB) for natural macOS Volume Keys
 *   - Per-Sample Smoothed Volume & Equal-Power Stereo Pan (System Preferences Balance)
 * Targets macOS Monterey 12.7.x (x86_64)
 */

#include <CoreAudio/AudioServerPlugIn.h>
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define kDeviceName                  "Dual Output"
#define kManufacturerName            "DualOutput Audio"
#define kDeviceUID                   "com.dualoutput.driver.device"
#define kModelUID                    "com.dualoutput.driver.model"

#define kChannelCount                2
#define kBytesPerSample              4
#define kBytesPerFrame               (kChannelCount * kBytesPerSample)
#define kRingBufferFrames            16384U
#define kRingBufferMask              (kRingBufferFrames - 1U)
#define kSafetyOffsetFrames          128U

enum {
    kObjectID_PlugIn                 = kAudioObjectPlugInObject, /* 1 */
    kObjectID_Device                 = 2,
    kObjectID_Stream_Output          = 3,
    kObjectID_Stream_Input           = 4,
    kObjectID_Volume_Output_Master   = 5,
    kObjectID_Volume_Output_Left     = 6,
    kObjectID_Volume_Output_Right    = 7,
    kObjectID_StereoPan_Output       = 8,
    kObjectID_Mute_Output            = 9
};

static const Float64 kSupportedSampleRates[] = {
    44100.0,
    48000.0,
    88200.0,
    96000.0
};
#define kSupportedSampleRatesCount (sizeof(kSupportedSampleRates) / sizeof(kSupportedSampleRates[0]))

static pthread_mutex_t              gStateMutex = PTHREAD_MUTEX_INITIALIZER;
static AudioServerPlugInHostRef     gPlugInHost = NULL;
static _Atomic UInt32               gRefCount = 0;

static Float64                      gSampleRate = 48000.0;
static UInt32                       gIOClientCount = 0;
static Boolean                      gStreamOutputActive = true;
static Boolean                      gStreamInputActive = true;

/* Volume, Per-Channel Balance, Stereo Pan & Mute Controls */
static _Atomic Float32              gVolumeMasterScalar = 1.0f;
static _Atomic Float32              gVolumeLeftScalar   = 1.0f;
static _Atomic Float32              gVolumeRightScalar  = 1.0f;
static _Atomic Float32              gStereoPanValue     = 0.5f; /* 0.0 = Left, 0.5 = Center, 1.0 = Right */
static _Atomic UInt32               gMuteState          = 0;

/* Real-time per-sample smoothed gains to prevent zipper noise on volume key presses */
static Float32                      gSmoothedGainL      = 1.0f;
static Float32                      gSmoothedGainR      = 1.0f;

/* Timing state for GetZeroTimeStamp */
static Float64                      gHostTicksPerFrame = 0.0;
static UInt64                       gNumberTimeStamps = 0;
static Float64                      gAnchorSampleTime = 0.0;
static UInt64                       gAnchorHostTime = 0;
static UInt64                       gTimelineSeed = 1;

/* Pre-allocated real-time loopback ring buffer */
static float                        gRingBuffer[kRingBufferFrames * kChannelCount];
static _Atomic int64_t              gLastWrittenSampleEnd = -1;

static void RecalculateClockParameters(void) {
    struct mach_timebase_info timebase;
    mach_timebase_info(&timebase);
    Float64 clockFrequency = ((Float64)timebase.denom / (Float64)timebase.numer) * 1000000000.0;
    gHostTicksPerFrame = clockFrequency / gSampleRate;
}

/*
 * Perceptual Cubic Volume Mapping:
 *   Amplitude = s^3
 *   dB        = 20 * log10(s^3) = 60 * log10(s)
 *   At s = 0.5 (50% on macOS Volume HUD), level is -18.06 dB (0.125 amplitude)
 *   instead of -6.02 dB, matching human logarithmic loudness perception.
 */
static inline Float32 ScalarToAmplitude(Float32 scalar) {
    if (scalar <= 0.0001f) {
        return 0.0f;
    }
    if (scalar >= 1.0f) {
        return 1.0f;
    }
    return scalar * scalar * scalar;
}

static Float32 ScalarToDecibels(Float32 scalar) {
    if (scalar <= 0.0001f) {
        return -96.0f;
    }
    if (scalar >= 1.0f) {
        return 0.0f;
    }
    Float32 db = 60.0f * log10f(scalar);
    if (db < -96.0f) db = -96.0f;
    if (db > 0.0f) db = 0.0f;
    return db;
}

static Float32 DecibelsToScalar(Float32 db) {
    if (db <= -96.0f) {
        return 0.0f;
    }
    if (db >= 0.0f) {
        return 1.0f;
    }
    return powf(10.0f, db / 60.0f);
}

static _Atomic Float32 *GetVolumeAtomicForObject(AudioObjectID inObjectID) {
    switch (inObjectID) {
        case kObjectID_Volume_Output_Master: return &gVolumeMasterScalar;
        case kObjectID_Volume_Output_Left:   return &gVolumeLeftScalar;
        case kObjectID_Volume_Output_Right:  return &gVolumeRightScalar;
        default:                             return NULL;
    }
}

static UInt32 GetControlElementForVolumeObject(AudioObjectID inObjectID) {
    switch (inObjectID) {
        case kObjectID_Volume_Output_Master: return kAudioObjectPropertyElementMain; /* 0 */
        case kObjectID_Volume_Output_Left:   return 1;
        case kObjectID_Volume_Output_Right:  return 2;
        default:                             return kAudioObjectPropertyElementMain;
    }
}

/* Forward declarations */
static HRESULT DualOutput_QueryInterface(void *inDriver, REFIID inUUID, LPVOID *outInterface);
static ULONG   DualOutput_AddRef(void *inDriver);
static ULONG   DualOutput_Release(void *inDriver);
static OSStatus DualOutput_Initialize(AudioServerPlugInDriverRef inDriver, AudioServerPlugInHostRef inHost);
static OSStatus DualOutput_CreateDevice(AudioServerPlugInDriverRef inDriver, CFDictionaryRef inDescription, const AudioServerPlugInClientInfo *inClientInfo, AudioObjectID *outDeviceObjectID);
static OSStatus DualOutput_DestroyDevice(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID);
static OSStatus DualOutput_AddDeviceClient(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, const AudioServerPlugInClientInfo *inClientInfo);
static OSStatus DualOutput_RemoveDeviceClient(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, const AudioServerPlugInClientInfo *inClientInfo);
static OSStatus DualOutput_PerformDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt64 inChangeAction, void *inChangeInfo);
static OSStatus DualOutput_AbortDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt64 inChangeAction, void *inChangeInfo);
static Boolean  DualOutput_HasProperty(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress);
static OSStatus DualOutput_IsPropertySettable(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress, Boolean *outIsSettable);
static OSStatus DualOutput_GetPropertyDataSize(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress, UInt32 inQualifierDataSize, const void *inQualifierData, UInt32 *outDataSize);
static OSStatus DualOutput_GetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress, UInt32 inQualifierDataSize, const void *inQualifierData, UInt32 inDataSize, UInt32 *outDataSize, void *outData);
static OSStatus DualOutput_SetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress, UInt32 inQualifierDataSize, const void *inQualifierData, UInt32 inDataSize, const void *inData);
static OSStatus DualOutput_StartIO(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID);
static OSStatus DualOutput_StopIO(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID);
static OSStatus DualOutput_GetZeroTimeStamp(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, Float64 *outSampleTime, UInt64 *outHostTime, UInt64 *outSeed);
static OSStatus DualOutput_WillDoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, Boolean *outWillDo, Boolean *outWillDoInPlace);
static OSStatus DualOutput_BeginIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo *inIOCycleInfo);
static OSStatus DualOutput_DoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, AudioObjectID inStreamObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo *inIOCycleInfo, void *ioMainBuffer, void *ioSecondaryBuffer);
static OSStatus DualOutput_EndIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo *inIOCycleInfo);

static AudioServerPlugInDriverInterface gDriverInterface = {
    NULL,
    DualOutput_QueryInterface,
    DualOutput_AddRef,
    DualOutput_Release,
    DualOutput_Initialize,
    DualOutput_CreateDevice,
    DualOutput_DestroyDevice,
    DualOutput_AddDeviceClient,
    DualOutput_RemoveDeviceClient,
    DualOutput_PerformDeviceConfigurationChange,
    DualOutput_AbortDeviceConfigurationChange,
    DualOutput_HasProperty,
    DualOutput_IsPropertySettable,
    DualOutput_GetPropertyDataSize,
    DualOutput_GetPropertyData,
    DualOutput_SetPropertyData,
    DualOutput_StartIO,
    DualOutput_StopIO,
    DualOutput_GetZeroTimeStamp,
    DualOutput_WillDoIOOperation,
    DualOutput_BeginIOOperation,
    DualOutput_DoIOOperation,
    DualOutput_EndIOOperation
};

static AudioServerPlugInDriverInterface *gDriverInterfacePtr = &gDriverInterface;
static AudioServerPlugInDriverRef gDriverRef = &gDriverInterfacePtr;

__attribute__((visibility("default")))
void *DualOutputPlugIn_Create(CFAllocatorRef inAllocator, CFUUIDRef inRequestedTypeUUID) {
    (void)inAllocator;
    if (CFEqual(inRequestedTypeUUID, kAudioServerPlugInTypeUUID)) {
        DualOutput_AddRef(gDriverRef);
        return gDriverRef;
    }
    return NULL;
}

static HRESULT DualOutput_QueryInterface(void *inDriver, REFIID inUUID, LPVOID *outInterface) {
    if (inDriver != gDriverRef || outInterface == NULL) {
        return kAudioHardwareBadObjectError;
    }
    CFUUIDRef requestedUUID = CFUUIDCreateFromUUIDBytes(NULL, inUUID);
    if (requestedUUID == NULL) {
        return kAudioHardwareIllegalOperationError;
    }
    bool matches = CFEqual(requestedUUID, IUnknownUUID) ||
                   CFEqual(requestedUUID, kAudioServerPlugInDriverInterfaceUUID);
    CFRelease(requestedUUID);
    if (!matches) {
        return E_NOINTERFACE;
    }
    DualOutput_AddRef(gDriverRef);
    *outInterface = gDriverRef;
    return S_OK;
}

static ULONG DualOutput_AddRef(void *inDriver) {
    if (inDriver != gDriverRef) {
        return 0;
    }
    return atomic_fetch_add(&gRefCount, 1) + 1;
}

static ULONG DualOutput_Release(void *inDriver) {
    if (inDriver != gDriverRef) {
        return 0;
    }
    UInt32 prev = atomic_fetch_sub(&gRefCount, 1);
    return (prev > 0) ? (prev - 1) : 0;
}

static OSStatus DualOutput_Initialize(AudioServerPlugInDriverRef inDriver, AudioServerPlugInHostRef inHost) {
    if (inDriver != gDriverRef || inHost == NULL) {
        return kAudioHardwareBadObjectError;
    }
    pthread_mutex_lock(&gStateMutex);
    gPlugInHost = inHost;
    RecalculateClockParameters();
    memset(gRingBuffer, 0, sizeof(gRingBuffer));
    atomic_store(&gLastWrittenSampleEnd, -1);
    pthread_mutex_unlock(&gStateMutex);
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_CreateDevice(AudioServerPlugInDriverRef inDriver, CFDictionaryRef inDescription, const AudioServerPlugInClientInfo *inClientInfo, AudioObjectID *outDeviceObjectID) {
    (void)inDriver; (void)inDescription; (void)inClientInfo; (void)outDeviceObjectID;
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus DualOutput_DestroyDevice(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID) {
    (void)inDriver; (void)inDeviceObjectID;
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus DualOutput_AddDeviceClient(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, const AudioServerPlugInClientInfo *inClientInfo) {
    (void)inClientInfo;
    if (inDriver != gDriverRef || inDeviceObjectID != kObjectID_Device) {
        return kAudioHardwareBadObjectError;
    }
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_RemoveDeviceClient(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, const AudioServerPlugInClientInfo *inClientInfo) {
    (void)inClientInfo;
    if (inDriver != gDriverRef || inDeviceObjectID != kObjectID_Device) {
        return kAudioHardwareBadObjectError;
    }
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_PerformDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt64 inChangeAction, void *inChangeInfo) {
    (void)inChangeInfo;
    if (inDriver != gDriverRef || inDeviceObjectID != kObjectID_Device) {
        return kAudioHardwareBadObjectError;
    }
    pthread_mutex_lock(&gStateMutex);
    Float64 requestedRate = (Float64)inChangeAction;
    for (size_t i = 0; i < kSupportedSampleRatesCount; ++i) {
        if (fabs(requestedRate - kSupportedSampleRates[i]) < 1.0) {
            gSampleRate = kSupportedSampleRates[i];
            RecalculateClockParameters();
            gNumberTimeStamps = 0;
            gAnchorSampleTime = 0.0;
            gAnchorHostTime = mach_absolute_time();
            gTimelineSeed += 1;
            memset(gRingBuffer, 0, sizeof(gRingBuffer));
            atomic_store(&gLastWrittenSampleEnd, -1);
            break;
        }
    }
    pthread_mutex_unlock(&gStateMutex);
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_AbortDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt64 inChangeAction, void *inChangeInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inChangeAction; (void)inChangeInfo;
    return kAudioHardwareNoError;
}

static Boolean DualOutput_HasProperty(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress) {
    (void)inClientProcessID;
    if (inDriver != gDriverRef || inAddress == NULL) {
        return false;
    }
    switch (inObjectID) {
        case kObjectID_PlugIn:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioObjectPropertyManufacturer:
                case kAudioObjectPropertyOwnedObjects:
                case kAudioPlugInPropertyBoxList:
                case kAudioPlugInPropertyTranslateUIDToDevice:
                case kAudioPlugInPropertyDeviceList:
                case kAudioPlugInPropertyResourceBundle:
                    return true;
                default:
                    return false;
            }
        case kObjectID_Device:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioObjectPropertyName:
                case kAudioObjectPropertyManufacturer:
                case kAudioObjectPropertyOwnedObjects:
                case kAudioDevicePropertyDeviceUID:
                case kAudioDevicePropertyModelUID:
                case kAudioDevicePropertyTransportType:
                case kAudioDevicePropertyRelatedDevices:
                case kAudioDevicePropertyClockDomain:
                case kAudioDevicePropertyDeviceIsAlive:
                case kAudioDevicePropertyDeviceIsRunning:
                case kAudioDevicePropertyDeviceCanBeDefaultDevice:
                case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
                case kAudioDevicePropertyLatency:
                case kAudioDevicePropertyStreams:
                case kAudioObjectPropertyControlList:
                case kAudioDevicePropertySafetyOffset:
                case kAudioDevicePropertyNominalSampleRate:
                case kAudioDevicePropertyAvailableNominalSampleRates:
                case kAudioDevicePropertyIsHidden:
                case kAudioDevicePropertyPreferredChannelsForStereo:
                case kAudioDevicePropertyPreferredChannelLayout:
                case kAudioDevicePropertyZeroTimeStampPeriod:
                    return true;
                default:
                    return false;
            }
        case kObjectID_Stream_Output:
        case kObjectID_Stream_Input:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioStreamPropertyIsActive:
                case kAudioStreamPropertyDirection:
                case kAudioStreamPropertyTerminalType:
                case kAudioStreamPropertyStartingChannel:
                case kAudioStreamPropertyLatency:
                case kAudioStreamPropertyVirtualFormat:
                case kAudioStreamPropertyPhysicalFormat:
                case kAudioStreamPropertyAvailableVirtualFormats:
                case kAudioStreamPropertyAvailablePhysicalFormats:
                    return true;
                default:
                    return false;
            }
        case kObjectID_Volume_Output_Master:
        case kObjectID_Volume_Output_Left:
        case kObjectID_Volume_Output_Right:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioObjectPropertyOwnedObjects:
                case kAudioControlPropertyScope:
                case kAudioControlPropertyElement:
                case kAudioLevelControlPropertyScalarValue:
                case kAudioLevelControlPropertyDecibelValue:
                case kAudioLevelControlPropertyDecibelRange:
                    return true;
                default:
                    return false;
            }
        case kObjectID_StereoPan_Output:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioObjectPropertyOwnedObjects:
                case kAudioControlPropertyScope:
                case kAudioControlPropertyElement:
                case kAudioStereoPanControlPropertyValue:
                case kAudioStereoPanControlPropertyPanningChannels:
                    return true;
                default:
                    return false;
            }
        case kObjectID_Mute_Output:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioObjectPropertyOwnedObjects:
                case kAudioControlPropertyScope:
                case kAudioControlPropertyElement:
                case kAudioBooleanControlPropertyValue:
                    return true;
                default:
                    return false;
            }
        default:
            return false;
    }
}

static OSStatus DualOutput_IsPropertySettable(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress, Boolean *outIsSettable) {
    (void)inClientProcessID;
    if (inDriver != gDriverRef || inAddress == NULL || outIsSettable == NULL) {
        return kAudioHardwareBadObjectError;
    }
    if (!DualOutput_HasProperty(inDriver, inObjectID, inClientProcessID, inAddress)) {
        return kAudioHardwareUnknownPropertyError;
    }
    *outIsSettable = false;
    switch (inObjectID) {
        case kObjectID_Device:
            if (inAddress->mSelector == kAudioDevicePropertyNominalSampleRate) {
                *outIsSettable = true;
            }
            break;
        case kObjectID_Stream_Output:
        case kObjectID_Stream_Input:
            if (inAddress->mSelector == kAudioStreamPropertyIsActive ||
                inAddress->mSelector == kAudioStreamPropertyVirtualFormat ||
                inAddress->mSelector == kAudioStreamPropertyPhysicalFormat) {
                *outIsSettable = true;
            }
            break;
        case kObjectID_Volume_Output_Master:
        case kObjectID_Volume_Output_Left:
        case kObjectID_Volume_Output_Right:
            if (inAddress->mSelector == kAudioLevelControlPropertyScalarValue ||
                inAddress->mSelector == kAudioLevelControlPropertyDecibelValue) {
                *outIsSettable = true;
            }
            break;
        case kObjectID_StereoPan_Output:
            if (inAddress->mSelector == kAudioStereoPanControlPropertyValue) {
                *outIsSettable = true;
            }
            break;
        case kObjectID_Mute_Output:
            if (inAddress->mSelector == kAudioBooleanControlPropertyValue) {
                *outIsSettable = true;
            }
            break;
        default:
            break;
    }
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_GetPropertyDataSize(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress, UInt32 inQualifierDataSize, const void *inQualifierData, UInt32 *outDataSize) {
    (void)inQualifierDataSize; (void)inQualifierData;
    if (inDriver != gDriverRef || inAddress == NULL || outDataSize == NULL) {
        return kAudioHardwareBadObjectError;
    }
    if (!DualOutput_HasProperty(inDriver, inObjectID, inClientProcessID, inAddress)) {
        return kAudioHardwareUnknownPropertyError;
    }
    *outDataSize = 0;

    switch (inObjectID) {
        case kObjectID_PlugIn:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioObjectPropertyOwnedObjects:
                case kAudioPlugInPropertyTranslateUIDToDevice:
                case kAudioPlugInPropertyDeviceList:
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioPlugInPropertyBoxList:
                    *outDataSize = 0;
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyManufacturer:
                case kAudioPlugInPropertyResourceBundle:
                    *outDataSize = sizeof(CFStringRef);
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_Device:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioDevicePropertyRelatedDevices:
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyName:
                case kAudioObjectPropertyManufacturer:
                case kAudioDevicePropertyDeviceUID:
                case kAudioDevicePropertyModelUID:
                    *outDataSize = sizeof(CFStringRef);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects:
                    if (inAddress->mScope == kAudioObjectPropertyScopeGlobal) {
                        *outDataSize = 7 * sizeof(AudioObjectID);
                    } else if (inAddress->mScope == kAudioObjectPropertyScopeOutput) {
                        *outDataSize = 6 * sizeof(AudioObjectID);
                    } else if (inAddress->mScope == kAudioObjectPropertyScopeInput) {
                        *outDataSize = 1 * sizeof(AudioObjectID);
                    } else {
                        *outDataSize = 0;
                    }
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyStreams:
                    if (inAddress->mScope == kAudioObjectPropertyScopeGlobal) {
                        *outDataSize = 2 * sizeof(AudioObjectID);
                    } else if (inAddress->mScope == kAudioObjectPropertyScopeOutput ||
                               inAddress->mScope == kAudioObjectPropertyScopeInput) {
                        *outDataSize = 1 * sizeof(AudioObjectID);
                    } else {
                        *outDataSize = 0;
                    }
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyControlList:
                    *outDataSize = 5 * sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyTransportType:
                case kAudioDevicePropertyClockDomain:
                case kAudioDevicePropertyDeviceIsAlive:
                case kAudioDevicePropertyDeviceIsRunning:
                case kAudioDevicePropertyDeviceCanBeDefaultDevice:
                case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
                case kAudioDevicePropertyLatency:
                case kAudioDevicePropertySafetyOffset:
                case kAudioDevicePropertyIsHidden:
                case kAudioDevicePropertyZeroTimeStampPeriod:
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyNominalSampleRate:
                    *outDataSize = sizeof(Float64);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyAvailableNominalSampleRates:
                    *outDataSize = (UInt32)(kSupportedSampleRatesCount * sizeof(AudioValueRange));
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyPreferredChannelsForStereo:
                    *outDataSize = 2 * sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyPreferredChannelLayout:
                    *outDataSize = offsetof(AudioChannelLayout, mChannelDescriptions) + (kChannelCount * sizeof(AudioChannelDescription));
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_Stream_Output:
        case kObjectID_Stream_Input:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyIsActive:
                case kAudioStreamPropertyDirection:
                case kAudioStreamPropertyTerminalType:
                case kAudioStreamPropertyStartingChannel:
                case kAudioStreamPropertyLatency:
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyVirtualFormat:
                case kAudioStreamPropertyPhysicalFormat:
                    *outDataSize = sizeof(AudioStreamBasicDescription);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyAvailableVirtualFormats:
                case kAudioStreamPropertyAvailablePhysicalFormats:
                    *outDataSize = (UInt32)(kSupportedSampleRatesCount * sizeof(AudioStreamRangedDescription));
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_Volume_Output_Master:
        case kObjectID_Volume_Output_Left:
        case kObjectID_Volume_Output_Right:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioControlPropertyScope:
                case kAudioControlPropertyElement:
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects:
                    *outDataSize = 0;
                    return kAudioHardwareNoError;
                case kAudioLevelControlPropertyScalarValue:
                case kAudioLevelControlPropertyDecibelValue:
                    *outDataSize = sizeof(Float32);
                    return kAudioHardwareNoError;
                case kAudioLevelControlPropertyDecibelRange:
                    *outDataSize = sizeof(AudioValueRange);
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_StereoPan_Output:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioControlPropertyScope:
                case kAudioControlPropertyElement:
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects:
                    *outDataSize = 0;
                    return kAudioHardwareNoError;
                case kAudioStereoPanControlPropertyValue:
                    *outDataSize = sizeof(Float32);
                    return kAudioHardwareNoError;
                case kAudioStereoPanControlPropertyPanningChannels:
                    *outDataSize = 2 * sizeof(UInt32);
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_Mute_Output:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                case kAudioObjectPropertyClass:
                case kAudioObjectPropertyOwner:
                case kAudioControlPropertyScope:
                case kAudioControlPropertyElement:
                case kAudioBooleanControlPropertyValue:
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects:
                    *outDataSize = 0;
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        default:
            return kAudioHardwareBadObjectError;
    }
}

static void FillStreamFormat(AudioStreamBasicDescription *outFormat, Float64 sampleRate) {
    memset(outFormat, 0, sizeof(AudioStreamBasicDescription));
    outFormat->mSampleRate       = sampleRate;
    outFormat->mFormatID         = kAudioFormatLinearPCM;
    outFormat->mFormatFlags      = kAudioFormatFlagsNativeFloatPacked;
    outFormat->mBytesPerPacket   = kBytesPerFrame;
    outFormat->mFramesPerPacket  = 1;
    outFormat->mBytesPerFrame    = kBytesPerFrame;
    outFormat->mChannelsPerFrame = kChannelCount;
    outFormat->mBitsPerChannel   = 32;
}

static OSStatus DualOutput_GetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress, UInt32 inQualifierDataSize, const void *inQualifierData, UInt32 inDataSize, UInt32 *outDataSize, void *outData) {
    if (inDriver != gDriverRef || inAddress == NULL || outDataSize == NULL || outData == NULL) {
        return kAudioHardwareBadObjectError;
    }
    if (!DualOutput_HasProperty(inDriver, inObjectID, inClientProcessID, inAddress)) {
        return kAudioHardwareUnknownPropertyError;
    }

    switch (inObjectID) {
        case kObjectID_PlugIn:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioObjectClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioPlugInClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwner:
                    if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectID *)outData) = kAudioObjectUnknown;
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyManufacturer:
                    if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
                    *((CFStringRef *)outData) = CFSTR(kManufacturerName);
                    *outDataSize = sizeof(CFStringRef);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects:
                case kAudioPlugInPropertyDeviceList:
                    if (inDataSize < sizeof(AudioObjectID)) {
                        *outDataSize = 0;
                        return kAudioHardwareNoError;
                    }
                    *((AudioObjectID *)outData) = kObjectID_Device;
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioPlugInPropertyBoxList:
                    *outDataSize = 0;
                    return kAudioHardwareNoError;
                case kAudioPlugInPropertyTranslateUIDToDevice: {
                    if (inDataSize < sizeof(AudioObjectID) || inQualifierData == NULL || inQualifierDataSize < sizeof(CFStringRef)) {
                        return kAudioHardwareBadPropertySizeError;
                    }
                    CFStringRef uid = *((const CFStringRef *)inQualifierData);
                    if (uid != NULL && CFEqual(uid, CFSTR(kDeviceUID))) {
                        *((AudioObjectID *)outData) = kObjectID_Device;
                    } else {
                        *((AudioObjectID *)outData) = kAudioObjectUnknown;
                    }
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                }
                case kAudioPlugInPropertyResourceBundle:
                    if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
                    *((CFStringRef *)outData) = CFSTR("");
                    *outDataSize = sizeof(CFStringRef);
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_Device:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioObjectClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioDeviceClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwner:
                    if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectID *)outData) = kObjectID_PlugIn;
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyName:
                    if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
                    *((CFStringRef *)outData) = CFSTR(kDeviceName);
                    *outDataSize = sizeof(CFStringRef);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyManufacturer:
                    if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
                    *((CFStringRef *)outData) = CFSTR(kManufacturerName);
                    *outDataSize = sizeof(CFStringRef);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyDeviceUID:
                    if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
                    *((CFStringRef *)outData) = CFSTR(kDeviceUID);
                    *outDataSize = sizeof(CFStringRef);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyModelUID:
                    if (inDataSize < sizeof(CFStringRef)) return kAudioHardwareBadPropertySizeError;
                    *((CFStringRef *)outData) = CFSTR(kModelUID);
                    *outDataSize = sizeof(CFStringRef);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects: {
                    AudioObjectID *list = (AudioObjectID *)outData;
                    UInt32 capacity = inDataSize / sizeof(AudioObjectID);
                    UInt32 count = 0;
                    if (inAddress->mScope == kAudioObjectPropertyScopeGlobal) {
                        AudioObjectID allObjs[7] = {
                            kObjectID_Stream_Output,
                            kObjectID_Stream_Input,
                            kObjectID_Volume_Output_Master,
                            kObjectID_Volume_Output_Left,
                            kObjectID_Volume_Output_Right,
                            kObjectID_StereoPan_Output,
                            kObjectID_Mute_Output
                        };
                        count = (capacity < 7) ? capacity : 7;
                        memcpy(list, allObjs, count * sizeof(AudioObjectID));
                    } else if (inAddress->mScope == kAudioObjectPropertyScopeOutput) {
                        AudioObjectID outObjs[6] = {
                            kObjectID_Stream_Output,
                            kObjectID_Volume_Output_Master,
                            kObjectID_Volume_Output_Left,
                            kObjectID_Volume_Output_Right,
                            kObjectID_StereoPan_Output,
                            kObjectID_Mute_Output
                        };
                        count = (capacity < 6) ? capacity : 6;
                        memcpy(list, outObjs, count * sizeof(AudioObjectID));
                    } else if (inAddress->mScope == kAudioObjectPropertyScopeInput) {
                        if (capacity >= 1) {
                            list[0] = kObjectID_Stream_Input;
                            count = 1;
                        }
                    }
                    *outDataSize = count * sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                }
                case kAudioDevicePropertyStreams: {
                    AudioObjectID *list = (AudioObjectID *)outData;
                    UInt32 capacity = inDataSize / sizeof(AudioObjectID);
                    UInt32 count = 0;
                    if (inAddress->mScope == kAudioObjectPropertyScopeGlobal) {
                        AudioObjectID streams[2] = { kObjectID_Stream_Output, kObjectID_Stream_Input };
                        count = (capacity < 2) ? capacity : 2;
                        memcpy(list, streams, count * sizeof(AudioObjectID));
                    } else if (inAddress->mScope == kAudioObjectPropertyScopeOutput) {
                        if (capacity >= 1) {
                            list[0] = kObjectID_Stream_Output;
                            count = 1;
                        }
                    } else if (inAddress->mScope == kAudioObjectPropertyScopeInput) {
                        if (capacity >= 1) {
                            list[0] = kObjectID_Stream_Input;
                            count = 1;
                        }
                    }
                    *outDataSize = count * sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                }
                case kAudioObjectPropertyControlList: {
                    AudioObjectID *list = (AudioObjectID *)outData;
                    UInt32 capacity = inDataSize / sizeof(AudioObjectID);
                    AudioObjectID ctrls[5] = {
                        kObjectID_Volume_Output_Master,
                        kObjectID_Volume_Output_Left,
                        kObjectID_Volume_Output_Right,
                        kObjectID_StereoPan_Output,
                        kObjectID_Mute_Output
                    };
                    UInt32 count = (capacity < 5) ? capacity : 5;
                    memcpy(list, ctrls, count * sizeof(AudioObjectID));
                    *outDataSize = count * sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                }
                case kAudioDevicePropertyTransportType:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = kAudioDeviceTransportTypeVirtual;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyRelatedDevices:
                    if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectID *)outData) = kObjectID_Device;
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyClockDomain:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = 0;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyDeviceIsAlive:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = 1;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyDeviceIsRunning:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    pthread_mutex_lock(&gStateMutex);
                    *((UInt32 *)outData) = (gIOClientCount > 0) ? 1 : 0;
                    pthread_mutex_unlock(&gStateMutex);
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyDeviceCanBeDefaultDevice:
                case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = 1;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyLatency:
                case kAudioDevicePropertyIsHidden:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = 0;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertySafetyOffset:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = kSafetyOffsetFrames;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyZeroTimeStampPeriod:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = kRingBufferFrames;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyNominalSampleRate:
                    if (inDataSize < sizeof(Float64)) return kAudioHardwareBadPropertySizeError;
                    pthread_mutex_lock(&gStateMutex);
                    *((Float64 *)outData) = gSampleRate;
                    pthread_mutex_unlock(&gStateMutex);
                    *outDataSize = sizeof(Float64);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyAvailableNominalSampleRates: {
                    UInt32 capacity = inDataSize / sizeof(AudioValueRange);
                    UInt32 count = (capacity < kSupportedSampleRatesCount) ? capacity : (UInt32)kSupportedSampleRatesCount;
                    AudioValueRange *ranges = (AudioValueRange *)outData;
                    for (UInt32 i = 0; i < count; ++i) {
                        ranges[i].mMinimum = kSupportedSampleRates[i];
                        ranges[i].mMaximum = kSupportedSampleRates[i];
                    }
                    *outDataSize = count * sizeof(AudioValueRange);
                    return kAudioHardwareNoError;
                }
                case kAudioDevicePropertyPreferredChannelsForStereo:
                    if (inDataSize < 2 * sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    ((UInt32 *)outData)[0] = 1;
                    ((UInt32 *)outData)[1] = 2;
                    *outDataSize = 2 * sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioDevicePropertyPreferredChannelLayout: {
                    UInt32 requiredSize = offsetof(AudioChannelLayout, mChannelDescriptions) + (kChannelCount * sizeof(AudioChannelDescription));
                    if (inDataSize < requiredSize) return kAudioHardwareBadPropertySizeError;
                    AudioChannelLayout *layout = (AudioChannelLayout *)outData;
                    memset(layout, 0, requiredSize);
                    layout->mChannelLayoutTag = kAudioChannelLayoutTag_UseChannelDescriptions;
                    layout->mNumberChannelDescriptions = kChannelCount;
                    layout->mChannelDescriptions[0].mChannelLabel = kAudioChannelLabel_Left;
                    layout->mChannelDescriptions[1].mChannelLabel = kAudioChannelLabel_Right;
                    *outDataSize = requiredSize;
                    return kAudioHardwareNoError;
                }
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_Stream_Output:
        case kObjectID_Stream_Input:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioObjectClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioStreamClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwner:
                    if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectID *)outData) = kObjectID_Device;
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyIsActive:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    pthread_mutex_lock(&gStateMutex);
                    *((UInt32 *)outData) = (inObjectID == kObjectID_Stream_Output)
                                           ? (gStreamOutputActive ? 1 : 0)
                                           : (gStreamInputActive ? 1 : 0);
                    pthread_mutex_unlock(&gStateMutex);
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyDirection:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = (inObjectID == kObjectID_Stream_Output) ? 0 : 1;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyTerminalType:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = (inObjectID == kObjectID_Stream_Output)
                                           ? kAudioStreamTerminalTypeSpeaker
                                           : kAudioStreamTerminalTypeMicrophone;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyStartingChannel:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = 1;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyLatency:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = 0;
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyVirtualFormat:
                case kAudioStreamPropertyPhysicalFormat:
                    if (inDataSize < sizeof(AudioStreamBasicDescription)) return kAudioHardwareBadPropertySizeError;
                    pthread_mutex_lock(&gStateMutex);
                    FillStreamFormat((AudioStreamBasicDescription *)outData, gSampleRate);
                    pthread_mutex_unlock(&gStateMutex);
                    *outDataSize = sizeof(AudioStreamBasicDescription);
                    return kAudioHardwareNoError;
                case kAudioStreamPropertyAvailableVirtualFormats:
                case kAudioStreamPropertyAvailablePhysicalFormats: {
                    UInt32 capacity = inDataSize / sizeof(AudioStreamRangedDescription);
                    UInt32 count = (capacity < kSupportedSampleRatesCount) ? capacity : (UInt32)kSupportedSampleRatesCount;
                    AudioStreamRangedDescription *descs = (AudioStreamRangedDescription *)outData;
                    for (UInt32 i = 0; i < count; ++i) {
                        FillStreamFormat(&descs[i].mFormat, kSupportedSampleRates[i]);
                        descs[i].mSampleRateRange.mMinimum = kSupportedSampleRates[i];
                        descs[i].mSampleRateRange.mMaximum = kSupportedSampleRates[i];
                    }
                    *outDataSize = count * sizeof(AudioStreamRangedDescription);
                    return kAudioHardwareNoError;
                }
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_Volume_Output_Master:
        case kObjectID_Volume_Output_Left:
        case kObjectID_Volume_Output_Right: {
            _Atomic Float32 *volAtom = GetVolumeAtomicForObject(inObjectID);
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioLevelControlClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioVolumeControlClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwner:
                    if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectID *)outData) = kObjectID_Device;
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects:
                    *outDataSize = 0;
                    return kAudioHardwareNoError;
                case kAudioControlPropertyScope:
                    if (inDataSize < sizeof(AudioObjectPropertyScope)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectPropertyScope *)outData) = kAudioObjectPropertyScopeOutput;
                    *outDataSize = sizeof(AudioObjectPropertyScope);
                    return kAudioHardwareNoError;
                case kAudioControlPropertyElement:
                    if (inDataSize < sizeof(AudioObjectPropertyElement)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectPropertyElement *)outData) = GetControlElementForVolumeObject(inObjectID);
                    *outDataSize = sizeof(AudioObjectPropertyElement);
                    return kAudioHardwareNoError;
                case kAudioLevelControlPropertyScalarValue:
                    if (inDataSize < sizeof(Float32) || volAtom == NULL) return kAudioHardwareBadPropertySizeError;
                    *((Float32 *)outData) = atomic_load(volAtom);
                    *outDataSize = sizeof(Float32);
                    return kAudioHardwareNoError;
                case kAudioLevelControlPropertyDecibelValue:
                    if (inDataSize < sizeof(Float32) || volAtom == NULL) return kAudioHardwareBadPropertySizeError;
                    *((Float32 *)outData) = ScalarToDecibels(atomic_load(volAtom));
                    *outDataSize = sizeof(Float32);
                    return kAudioHardwareNoError;
                case kAudioLevelControlPropertyDecibelRange: {
                    if (inDataSize < sizeof(AudioValueRange)) return kAudioHardwareBadPropertySizeError;
                    AudioValueRange *range = (AudioValueRange *)outData;
                    range->mMinimum = -96.0;
                    range->mMaximum = 0.0;
                    *outDataSize = sizeof(AudioValueRange);
                    return kAudioHardwareNoError;
                }
                default:
                    return kAudioHardwareUnknownPropertyError;
            }
        }

        case kObjectID_StereoPan_Output:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioControlClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioStereoPanControlClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwner:
                    if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectID *)outData) = kObjectID_Device;
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects:
                    *outDataSize = 0;
                    return kAudioHardwareNoError;
                case kAudioControlPropertyScope:
                    if (inDataSize < sizeof(AudioObjectPropertyScope)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectPropertyScope *)outData) = kAudioObjectPropertyScopeOutput;
                    *outDataSize = sizeof(AudioObjectPropertyScope);
                    return kAudioHardwareNoError;
                case kAudioControlPropertyElement:
                    if (inDataSize < sizeof(AudioObjectPropertyElement)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectPropertyElement *)outData) = kAudioObjectPropertyElementMain;
                    *outDataSize = sizeof(AudioObjectPropertyElement);
                    return kAudioHardwareNoError;
                case kAudioStereoPanControlPropertyValue:
                    if (inDataSize < sizeof(Float32)) return kAudioHardwareBadPropertySizeError;
                    *((Float32 *)outData) = atomic_load(&gStereoPanValue);
                    *outDataSize = sizeof(Float32);
                    return kAudioHardwareNoError;
                case kAudioStereoPanControlPropertyPanningChannels:
                    if (inDataSize < 2 * sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    ((UInt32 *)outData)[0] = 1;
                    ((UInt32 *)outData)[1] = 2;
                    *outDataSize = 2 * sizeof(UInt32);
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        case kObjectID_Mute_Output:
            switch (inAddress->mSelector) {
                case kAudioObjectPropertyBaseClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioBooleanControlClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyClass:
                    if (inDataSize < sizeof(AudioClassID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioClassID *)outData) = kAudioMuteControlClassID;
                    *outDataSize = sizeof(AudioClassID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwner:
                    if (inDataSize < sizeof(AudioObjectID)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectID *)outData) = kObjectID_Device;
                    *outDataSize = sizeof(AudioObjectID);
                    return kAudioHardwareNoError;
                case kAudioObjectPropertyOwnedObjects:
                    *outDataSize = 0;
                    return kAudioHardwareNoError;
                case kAudioControlPropertyScope:
                    if (inDataSize < sizeof(AudioObjectPropertyScope)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectPropertyScope *)outData) = kAudioObjectPropertyScopeOutput;
                    *outDataSize = sizeof(AudioObjectPropertyScope);
                    return kAudioHardwareNoError;
                case kAudioControlPropertyElement:
                    if (inDataSize < sizeof(AudioObjectPropertyElement)) return kAudioHardwareBadPropertySizeError;
                    *((AudioObjectPropertyElement *)outData) = kAudioObjectPropertyElementMain;
                    *outDataSize = sizeof(AudioObjectPropertyElement);
                    return kAudioHardwareNoError;
                case kAudioBooleanControlPropertyValue:
                    if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                    *((UInt32 *)outData) = atomic_load(&gMuteState);
                    *outDataSize = sizeof(UInt32);
                    return kAudioHardwareNoError;
                default:
                    return kAudioHardwareUnknownPropertyError;
            }

        default:
            return kAudioHardwareBadObjectError;
    }
}

static OSStatus DualOutput_SetPropertyData(AudioServerPlugInDriverRef inDriver, AudioObjectID inObjectID, pid_t inClientProcessID, const AudioObjectPropertyAddress *inAddress, UInt32 inQualifierDataSize, const void *inQualifierData, UInt32 inDataSize, const void *inData) {
    (void)inQualifierDataSize; (void)inQualifierData;
    if (inDriver != gDriverRef || inAddress == NULL || inData == NULL) {
        return kAudioHardwareBadObjectError;
    }
    Boolean settable = false;
    OSStatus err = DualOutput_IsPropertySettable(inDriver, inObjectID, inClientProcessID, inAddress, &settable);
    if (err != kAudioHardwareNoError) return err;
    if (!settable) return kAudioHardwareIllegalOperationError;

    switch (inObjectID) {
        case kObjectID_Device:
            if (inAddress->mSelector == kAudioDevicePropertyNominalSampleRate) {
                if (inDataSize < sizeof(Float64)) return kAudioHardwareBadPropertySizeError;
                Float64 newRate = *((const Float64 *)inData);
                bool valid = false;
                for (size_t i = 0; i < kSupportedSampleRatesCount; ++i) {
                    if (fabs(newRate - kSupportedSampleRates[i]) < 1.0) {
                        newRate = kSupportedSampleRates[i];
                        valid = true;
                        break;
                    }
                }
                if (!valid) return kAudioHardwareIllegalOperationError;
                if (gPlugInHost != NULL) {
                    gPlugInHost->RequestDeviceConfigurationChange(gPlugInHost, kObjectID_Device, (UInt64)newRate, NULL);
                }
                return kAudioHardwareNoError;
            }
            return kAudioHardwareUnknownPropertyError;

        case kObjectID_Stream_Output:
        case kObjectID_Stream_Input:
            if (inAddress->mSelector == kAudioStreamPropertyIsActive) {
                if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                UInt32 active = *((const UInt32 *)inData);
                pthread_mutex_lock(&gStateMutex);
                if (inObjectID == kObjectID_Stream_Output) {
                    gStreamOutputActive = (active != 0);
                } else {
                    gStreamInputActive = (active != 0);
                }
                pthread_mutex_unlock(&gStateMutex);
                return kAudioHardwareNoError;
            } else if (inAddress->mSelector == kAudioStreamPropertyVirtualFormat ||
                       inAddress->mSelector == kAudioStreamPropertyPhysicalFormat) {
                if (inDataSize < sizeof(AudioStreamBasicDescription)) return kAudioHardwareBadPropertySizeError;
                const AudioStreamBasicDescription *asbd = (const AudioStreamBasicDescription *)inData;
                if (asbd->mFormatID != kAudioFormatLinearPCM ||
                    asbd->mChannelsPerFrame != kChannelCount ||
                    asbd->mBitsPerChannel != 32) {
                    return kAudioDeviceUnsupportedFormatError;
                }
                bool validRate = false;
                for (size_t i = 0; i < kSupportedSampleRatesCount; ++i) {
                    if (fabs(asbd->mSampleRate - kSupportedSampleRates[i]) < 1.0) {
                        validRate = true;
                        break;
                    }
                }
                if (!validRate) return kAudioDeviceUnsupportedFormatError;
                if (gPlugInHost != NULL) {
                    gPlugInHost->RequestDeviceConfigurationChange(gPlugInHost, kObjectID_Device, (UInt64)asbd->mSampleRate, NULL);
                }
                return kAudioHardwareNoError;
            }
            return kAudioHardwareUnknownPropertyError;

        case kObjectID_Volume_Output_Master:
        case kObjectID_Volume_Output_Left:
        case kObjectID_Volume_Output_Right: {
            _Atomic Float32 *volAtom = GetVolumeAtomicForObject(inObjectID);
            if (volAtom == NULL) return kAudioHardwareBadObjectError;
            if (inAddress->mSelector == kAudioLevelControlPropertyScalarValue) {
                if (inDataSize < sizeof(Float32)) return kAudioHardwareBadPropertySizeError;
                Float32 v = *((const Float32 *)inData);
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                atomic_store(volAtom, v);
                if (gPlugInHost != NULL) {
                    AudioObjectPropertyAddress changed[2] = {
                        { kAudioLevelControlPropertyScalarValue, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain },
                        { kAudioLevelControlPropertyDecibelValue, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain }
                    };
                    gPlugInHost->PropertiesChanged(gPlugInHost, inObjectID, 2, changed);
                }
                return kAudioHardwareNoError;
            } else if (inAddress->mSelector == kAudioLevelControlPropertyDecibelValue) {
                if (inDataSize < sizeof(Float32)) return kAudioHardwareBadPropertySizeError;
                Float32 db = *((const Float32 *)inData);
                atomic_store(volAtom, DecibelsToScalar(db));
                if (gPlugInHost != NULL) {
                    AudioObjectPropertyAddress changed[2] = {
                        { kAudioLevelControlPropertyScalarValue, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain },
                        { kAudioLevelControlPropertyDecibelValue, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain }
                    };
                    gPlugInHost->PropertiesChanged(gPlugInHost, inObjectID, 2, changed);
                }
                return kAudioHardwareNoError;
            }
            return kAudioHardwareUnknownPropertyError;
        }

        case kObjectID_StereoPan_Output:
            if (inAddress->mSelector == kAudioStereoPanControlPropertyValue) {
                if (inDataSize < sizeof(Float32)) return kAudioHardwareBadPropertySizeError;
                Float32 pan = *((const Float32 *)inData);
                if (pan < 0.0f) pan = 0.0f;
                if (pan > 1.0f) pan = 1.0f;
                atomic_store(&gStereoPanValue, pan);
                if (gPlugInHost != NULL) {
                    AudioObjectPropertyAddress changed = {
                        kAudioStereoPanControlPropertyValue, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain
                    };
                    gPlugInHost->PropertiesChanged(gPlugInHost, kObjectID_StereoPan_Output, 1, &changed);
                }
                return kAudioHardwareNoError;
            }
            return kAudioHardwareUnknownPropertyError;

        case kObjectID_Mute_Output:
            if (inAddress->mSelector == kAudioBooleanControlPropertyValue) {
                if (inDataSize < sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
                UInt32 m = (*((const UInt32 *)inData) != 0) ? 1 : 0;
                atomic_store(&gMuteState, m);
                if (gPlugInHost != NULL) {
                    AudioObjectPropertyAddress changed = {
                        kAudioBooleanControlPropertyValue, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain
                    };
                    gPlugInHost->PropertiesChanged(gPlugInHost, kObjectID_Mute_Output, 1, &changed);
                }
                return kAudioHardwareNoError;
            }
            return kAudioHardwareUnknownPropertyError;

        default:
            return kAudioHardwareBadObjectError;
    }
}

static OSStatus DualOutput_StartIO(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID) {
    (void)inClientID;
    if (inDriver != gDriverRef || inDeviceObjectID != kObjectID_Device) {
        return kAudioHardwareBadObjectError;
    }
    pthread_mutex_lock(&gStateMutex);
    if (gIOClientCount == 0) {
        RecalculateClockParameters();
        memset(gRingBuffer, 0, sizeof(gRingBuffer));
        atomic_store(&gLastWrittenSampleEnd, -1);
        gNumberTimeStamps = 0;
        gAnchorSampleTime = 0.0;
        gAnchorHostTime = mach_absolute_time();
    }
    gIOClientCount += 1;
    pthread_mutex_unlock(&gStateMutex);
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_StopIO(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID) {
    (void)inClientID;
    if (inDriver != gDriverRef || inDeviceObjectID != kObjectID_Device) {
        return kAudioHardwareBadObjectError;
    }
    pthread_mutex_lock(&gStateMutex);
    if (gIOClientCount > 0) {
        gIOClientCount -= 1;
    }
    if (gIOClientCount == 0) {
        memset(gRingBuffer, 0, sizeof(gRingBuffer));
        atomic_store(&gLastWrittenSampleEnd, -1);
    }
    pthread_mutex_unlock(&gStateMutex);
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_GetZeroTimeStamp(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, Float64 *outSampleTime, UInt64 *outHostTime, UInt64 *outSeed) {
    (void)inClientID;
    if (inDriver != gDriverRef || inDeviceObjectID != kObjectID_Device ||
        outSampleTime == NULL || outHostTime == NULL || outSeed == NULL) {
        return kAudioHardwareBadObjectError;
    }

    UInt64 currentHostTime = mach_absolute_time();
    Float64 ticksPerRing = gHostTicksPerFrame * (Float64)kRingBufferFrames;
    UInt64 nextPeriodHostTime = gAnchorHostTime + (UInt64)((Float64)(gNumberTimeStamps + 1) * ticksPerRing);

    if (currentHostTime >= nextPeriodHostTime) {
        UInt64 elapsedPeriods = (UInt64)(((Float64)(currentHostTime - gAnchorHostTime)) / ticksPerRing);
        gNumberTimeStamps = elapsedPeriods;
    }

    *outSampleTime = gAnchorSampleTime + (Float64)(gNumberTimeStamps * kRingBufferFrames);
    *outHostTime   = gAnchorHostTime + (UInt64)((Float64)gNumberTimeStamps * ticksPerRing);
    *outSeed       = gTimelineSeed;
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_WillDoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, Boolean *outWillDo, Boolean *outWillDoInPlace) {
    (void)inClientID;
    if (inDriver != gDriverRef || inDeviceObjectID != kObjectID_Device ||
        outWillDo == NULL || outWillDoInPlace == NULL) {
        return kAudioHardwareBadObjectError;
    }
    switch (inOperationID) {
        case kAudioServerPlugInIOOperationWriteMix:
        case kAudioServerPlugInIOOperationReadInput:
            *outWillDo = true;
            *outWillDoInPlace = true;
            break;
        default:
            *outWillDo = false;
            *outWillDoInPlace = true;
            break;
    }
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_BeginIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo *inIOCycleInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientID; (void)inOperationID; (void)inIOBufferFrameSize; (void)inIOCycleInfo;
    return kAudioHardwareNoError;
}

static OSStatus DualOutput_DoIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, AudioObjectID inStreamObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo *inIOCycleInfo, void *ioMainBuffer, void *ioSecondaryBuffer) {
    (void)inStreamObjectID; (void)inClientID; (void)ioSecondaryBuffer;
    if (inDriver != gDriverRef || inDeviceObjectID != kObjectID_Device ||
        inIOCycleInfo == NULL || ioMainBuffer == NULL) {
        return kAudioHardwareBadObjectError;
    }

    if (inOperationID == kAudioServerPlugInIOOperationWriteMix) {
        const float *src = (const float *)ioMainBuffer;
        int64_t startFrame = (int64_t)inIOCycleInfo->mOutputTime.mSampleTime;

        bool muted        = (atomic_load_explicit(&gMuteState, memory_order_relaxed) != 0);
        Float32 masterAmp = muted ? 0.0f : ScalarToAmplitude(atomic_load_explicit(&gVolumeMasterScalar, memory_order_relaxed));
        Float32 leftAmp   = ScalarToAmplitude(atomic_load_explicit(&gVolumeLeftScalar, memory_order_relaxed));
        Float32 rightAmp  = ScalarToAmplitude(atomic_load_explicit(&gVolumeRightScalar, memory_order_relaxed));
        Float32 pan       = atomic_load_explicit(&gStereoPanValue, memory_order_relaxed);

        /* Unity-center sinusoidal equal-power stereo pan law */
        Float32 panGainL  = (pan <= 0.5f) ? 1.0f : cosf((Float32)M_PI_2 * (pan - 0.5f) * 2.0f);
        Float32 panGainR  = (pan >= 0.5f) ? 1.0f : sinf((Float32)M_PI_2 * pan * 2.0f);

        Float32 targetGainL = masterAmp * leftAmp * panGainL;
        Float32 targetGainR = masterAmp * rightAmp * panGainR;
        const Float32 slew  = 0.005f; /* ~4 ms one-pole smoothing at 48 kHz */

        for (UInt32 i = 0; i < inIOBufferFrameSize; ++i) {
            gSmoothedGainL += slew * (targetGainL - gSmoothedGainL);
            gSmoothedGainR += slew * (targetGainR - gSmoothedGainR);
            uint32_t ringIdx = ((uint32_t)(startFrame + (int64_t)i)) & kRingBufferMask;
            gRingBuffer[ringIdx * 2]     = src[i * 2]     * gSmoothedGainL;
            gRingBuffer[ringIdx * 2 + 1] = src[i * 2 + 1] * gSmoothedGainR;
        }
        atomic_store_explicit(&gLastWrittenSampleEnd, startFrame + (int64_t)inIOBufferFrameSize, memory_order_release);
    } else if (inOperationID == kAudioServerPlugInIOOperationReadInput) {
        float *dst = (float *)ioMainBuffer;
        int64_t startFrame = (int64_t)inIOCycleInfo->mInputTime.mSampleTime;
        int64_t writtenEnd = atomic_load_explicit(&gLastWrittenSampleEnd, memory_order_acquire);

        if (writtenEnd < 0 || (startFrame + (int64_t)inIOBufferFrameSize) > (writtenEnd + (int64_t)kRingBufferFrames / 2)) {
            memset(dst, 0, (size_t)inIOBufferFrameSize * kBytesPerFrame);
        } else {
            for (UInt32 i = 0; i < inIOBufferFrameSize; ++i) {
                int64_t samplePos = startFrame + (int64_t)i;
                uint32_t ringIdx = ((uint32_t)samplePos) & kRingBufferMask;
                dst[i * 2]     = gRingBuffer[ringIdx * 2];
                dst[i * 2 + 1] = gRingBuffer[ringIdx * 2 + 1];
                gRingBuffer[ringIdx * 2]     = 0.0f;
                gRingBuffer[ringIdx * 2 + 1] = 0.0f;
            }
        }
    }

    return kAudioHardwareNoError;
}

static OSStatus DualOutput_EndIOOperation(AudioServerPlugInDriverRef inDriver, AudioObjectID inDeviceObjectID, UInt32 inClientID, UInt32 inOperationID, UInt32 inIOBufferFrameSize, const AudioServerPlugInIOCycleInfo *inIOCycleInfo) {
    (void)inDriver; (void)inDeviceObjectID; (void)inClientID; (void)inOperationID; (void)inIOBufferFrameSize; (void)inIOCycleInfo;
    return kAudioHardwareNoError;
}