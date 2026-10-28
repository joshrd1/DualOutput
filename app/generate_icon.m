/*
 * generate_icon.m
 * Renders all 10 macOS .iconset PNG resolutions using CoreGraphics / AppKit
 * for conversion into AppIcon.icns via /usr/bin/iconutil.
 */

#import <Cocoa/Cocoa.h>

static void WriteIconPNG(NSString *path, int pixelSize) {
    NSBitmapImageRep *rep = [[NSBitmapImageRep alloc]
        initWithBitmapDataPlanes:NULL
                      pixelsWide:pixelSize
                      pixelsHigh:pixelSize
                   bitsPerSample:8
                 samplesPerPixel:4
                        hasAlpha:YES
                        isPlanar:NO
                  colorSpaceName:NSCalibratedRGBColorSpace
                     bytesPerRow:0
                    bitsPerPixel:0];
    rep.size = NSMakeSize(pixelSize, pixelSize);

    [NSGraphicsContext saveGraphicsState];
    NSGraphicsContext *nsCtx = [NSGraphicsContext graphicsContextWithBitmapImageRep:rep];
    [NSGraphicsContext setCurrentContext:nsCtx];
    CGContextRef ctx = [nsCtx CGContext];

    /* Normalize drawing coordinate space to 1024 x 1024 */
    CGFloat scale = (CGFloat)pixelSize / 1024.0;
    CGContextScaleCTM(ctx, scale, scale);

    /* 1. macOS Monterey Rounded-Squircle App Icon Base (with outer margin) */
    NSRect bgRect = NSMakeRect(92.0, 92.0, 840.0, 840.0);
    NSBezierPath *squircle = [NSBezierPath bezierPathWithRoundedRect:bgRect xRadius:190.0 yRadius:190.0];

    /* Drop shadow behind squircle */
    [NSGraphicsContext saveGraphicsState];
    NSShadow *bgShadow = [[NSShadow alloc] init];
    bgShadow.shadowOffset = NSMakeSize(0.0, -18.0);
    bgShadow.shadowBlurRadius = 36.0;
    bgShadow.shadowColor = [NSColor colorWithCalibratedWhite:0.0 alpha:0.45];
    [bgShadow set];
    [[NSColor colorWithCalibratedRed:0.08 green:0.10 blue:0.15 alpha:1.0] setFill];
    [squircle fill];
    [NSGraphicsContext restoreGraphicsState];

    /* Clip subsequent background drawing to the squircle */
    [NSGraphicsContext saveGraphicsState];
    [squircle addClip];

    /* Deep slate-navy vertical gradient */
    NSColor *topBg = [NSColor colorWithCalibratedRed:0.14 green:0.18 blue:0.26 alpha:1.0];
    NSColor *botBg = [NSColor colorWithCalibratedRed:0.06 green:0.08 blue:0.12 alpha:1.0];
    NSGradient *bgGrad = [[NSGradient alloc] initWithStartingColor:topBg endingColor:botBg];
    [bgGrad drawInRect:bgRect angle:-90.0];

    /* Subtle radial cyan ambient glow in the center */
    NSColor *glowCenter = [NSColor colorWithCalibratedRed:0.05 green:0.55 blue:0.95 alpha:0.20];
    NSColor *glowEdge   = [NSColor colorWithCalibratedRed:0.05 green:0.55 blue:0.95 alpha:0.0];
    NSGradient *radialGlow = [[NSGradient alloc] initWithStartingColor:glowCenter endingColor:glowEdge];
    [radialGlow drawFromCenter:NSMakePoint(512.0, 512.0)
                        radius:20.0
                      toCenter:NSMakePoint(512.0, 512.0)
                        radius:420.0
                       options:0];

    /* Subtle inner border highlight */
    [[NSColor colorWithCalibratedWhite:1.0 alpha:0.12] setStroke];
    squircle.lineWidth = 10.0;
    [squircle stroke];

    /* 2. Draw Glowing Dual-Splitter Symbol */
    [NSGraphicsContext saveGraphicsState];
    NSShadow *symbolGlow = [[NSShadow alloc] init];
    symbolGlow.shadowOffset = NSMakeSize(0.0, -4.0);
    symbolGlow.shadowBlurRadius = 28.0;
    symbolGlow.shadowColor = [NSColor colorWithCalibratedRed:0.0 green:0.75 blue:1.0 alpha:0.55];
    [symbolGlow set];

    /* Main Y-splitter tracks */
    NSBezierPath *tracks = [NSBezierPath bezierPath];
    tracks.lineWidth = 60.0;
    tracks.lineCapStyle = NSLineCapStyleRound;
    tracks.lineJoinStyle = NSLineJoinStyleRound;

    /* Stem from left input node to split point */
    [tracks moveToPoint:NSMakePoint(270.0, 512.0)];
    [tracks lineToPoint:NSMakePoint(455.0, 512.0)];

    /* Upper branch (Output A) */
    [tracks moveToPoint:NSMakePoint(455.0, 512.0)];
    [tracks lineToPoint:NSMakePoint(600.0, 692.0)];
    [tracks lineToPoint:NSMakePoint(755.0, 692.0)];

    /* Lower branch (Output B) */
    [tracks moveToPoint:NSMakePoint(455.0, 512.0)];
    [tracks lineToPoint:NSMakePoint(600.0, 332.0)];
    [tracks lineToPoint:NSMakePoint(755.0, 332.0)];

    [[NSColor colorWithCalibratedRed:0.18 green:0.82 blue:1.0 alpha:1.0] setStroke];
    [tracks stroke];

    /* Output A & Output B Arrowheads */
    NSBezierPath *arrows = [NSBezierPath bezierPath];
    arrows.lineWidth = 56.0;
    arrows.lineCapStyle = NSLineCapStyleRound;
    arrows.lineJoinStyle = NSLineJoinStyleRound;

    /* Top arrowhead */
    [arrows moveToPoint:NSMakePoint(685.0, 772.0)];
    [arrows lineToPoint:NSMakePoint(785.0, 692.0)];
    [arrows lineToPoint:NSMakePoint(685.0, 612.0)];

    /* Bottom arrowhead */
    [arrows moveToPoint:NSMakePoint(685.0, 412.0)];
    [arrows lineToPoint:NSMakePoint(785.0, 332.0)];
    [arrows lineToPoint:NSMakePoint(685.0, 252.0)];

    [[NSColor colorWithCalibratedRed:0.92 green:0.98 blue:1.0 alpha:1.0] setStroke];
    [arrows stroke];

    /* Left Input Node (Outer White Ring + Cyan Core) */
    NSBezierPath *outerNode = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(190.0, 432.0, 160.0, 160.0)];
    [[NSColor colorWithCalibratedRed:0.95 green:0.99 blue:1.0 alpha:1.0] setFill];
    [outerNode fill];

    NSBezierPath *innerNode = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(232.0, 474.0, 76.0, 76.0)];
    [[NSColor colorWithCalibratedRed:0.08 green:0.55 blue:0.92 alpha:1.0] setFill];
    [innerNode fill];

    [NSGraphicsContext restoreGraphicsState];
    [NSGraphicsContext restoreGraphicsState];
    [NSGraphicsContext restoreGraphicsState];

    NSData *pngData = [rep representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
    [pngData writeToFile:path atomically:YES];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: generate_icon <output.iconset directory>\n");
            return 1;
        }
        NSString *iconsetDir = [NSString stringWithUTF8String:argv[1]];
        [[NSFileManager defaultManager] createDirectoryAtPath:iconsetDir
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:nil];

        struct { const char *name; int pixels; } specs[] = {
            { "icon_16x16.png",      16   },
            { "icon_16x16@2x.png",   32   },
            { "icon_32x32.png",      32   },
            { "icon_32x32@2x.png",   64   },
            { "icon_128x128.png",    128  },
            { "icon_128x128@2x.png", 256  },
            { "icon_256x256.png",    256  },
            { "icon_256x256@2x.png", 512  },
            { "icon_512x512.png",    512  },
            { "icon_512x512@2x.png", 1024 }
        };

        for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); ++i) {
            NSString *filePath = [iconsetDir stringByAppendingPathComponent:[NSString stringWithUTF8String:specs[i].name]];
            WriteIconPNG(filePath, specs[i].pixels);
        }
    }
    return 0;
}
