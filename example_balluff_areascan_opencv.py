#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_balluff_areascan_opencv.py

Description:
    This script interfaces with Balluff cameras using the mvIMPACT Acquire SDK.
    It captures a user-defined number of image frames, processes them (assuming 16-bit, 3-channel RGB),
    displays them using OpenCV, and saves them with timestamped filenames.

Author:
    Felip Marti 2025
    fmarti@swin.edu.au
    
Dependencies:
    - mvIMPACT Acquire SDK
    - NumPy
    - OpenCV (cv2)

Notes:
    - Only 16-bit, 3-channel BayerRG16 formatted images are supported.
    - Ensure camera pixel format is configured accordingly:
      GenICam > ImageFormatControl > PixelFormat = BayerRG16
"""

import sys
import ctypes
import numpy as np
import cv2
from datetime import datetime
from mvIMPACT import acquire
from mvIMPACT.Common import exampleHelper


def get_device():
    """Initialises the device manager and returns a connected device."""
    dev_mgr = acquire.DeviceManager()
    dev_mgr.updateDeviceList()

    if dev_mgr.deviceCount() == 0:
        print("No devices found. Please connect a camera.")
        exampleHelper.requestENTERFromUser()
        sys.exit(-1)

    p_dev = dev_mgr[0] if dev_mgr.deviceCount() == 1 else exampleHelper.getDeviceFromUserInput(dev_mgr)
    return dev_mgr, p_dev


def get_frame_count():
    """Prompts the user to input the number of frames to capture."""
    print("Please enter the number of frames to capture followed by [ENTER]: ", end='')
    frames = exampleHelper.getNumberFromUser()
    if frames < 1:
        print("Invalid input. Please capture at least one image.")
        sys.exit(-1)
    return frames


def handle_unsupported_format(info, warning_flag):
    """Handles unsupported image formats with a single warning."""
    if not warning_flag[0]:
        print("\nUnsupported image format detected.")
        print(info)
        print("This script only supports 16-bit, 3-channel RGB images.")
        print("Please adjust the camera configuration:")
        print("   → Setting > Camera > GenICam > ImageFormatControl > PixelFormat = BayerRG16")
        print("Check with Felip or Shuo for help.\n")
        warning_flag[0] = True
    else:
        print("Skipped unsupported frame (not 16-bit / 3-channel)")


def display_and_save_frame(p_request):
    """Processes, displays, and saves the image from the request with a timestamped filename."""
    image_size = p_request.imageSize.read()
    height = p_request.imageHeight.read()
    width = p_request.imageWidth.read()
    channels = p_request.imageChannelCount.read()
    bit_depth = p_request.imageChannelBitDepth.read()
    pixel_format = p_request.imagePixelFormat.read()

    cbuf = (ctypes.c_char * image_size).from_address(int(p_request.imageData.read()))
    dtype = np.uint16
    arr = np.frombuffer(cbuf, dtype=dtype)

    info_str = (f"Image Info — size: {image_size}, height: {height}, width: {width}, "
                f"channels: {channels}, bitDepth: {bit_depth}, dtype: {dtype}, "
                f"pixelFormat: {pixel_format}, buffer size: {arr.size}")

    expected_size = height * width * channels
    if arr.size != expected_size or channels != 3 or bit_depth != 16:
        return None, info_str

    arr = arr.reshape((height, width, channels))
    arr = cv2.convertScaleAbs(arr, alpha=(255.0 / 65535.0))
    img_cv = arr

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]  # up to milliseconds
    filename = f"frame_{timestamp}.png"

    cv2.imshow("Captured Frame", img_cv)
    cv2.imwrite(filename, img_cv)
    cv2.waitKey(1)

    return img_cv, None


def capture_frames(p_dev, frame_count):
    """Captures and processes the specified number of frames."""
    fi = acquire.FunctionInterface(p_dev)
    stats = acquire.Statistics(p_dev)

    while fi.imageRequestSingle() == acquire.DMR_NO_ERROR:
        print("Frame queued")

    exampleHelper.manuallyStartAcquisitionIfNeeded(p_dev, fi)
    previous_request = None
    unsupported_warning_flag = [False]

    for i in range(frame_count):
        request_nr = fi.imageRequestWaitFor(10000)
        if not fi.isRequestNrValid(request_nr):
            print(f"imageRequestWaitFor failed ({request_nr}, "
                  f"{acquire.ImpactAcquireException.getErrorCodeAsString(request_nr)})")
            continue

        p_request = fi.getRequest(request_nr)
        if p_request.isOK:
            if i % 100 == 0:
                print(f"Info from {p_dev.serial.read()}: "
                      f"{stats.framesPerSecond.name()}: {stats.framesPerSecond.readS()}, "
                      f"{stats.errorCount.name()}: {stats.errorCount.readS()}, "
                      f"{stats.captureTime_s.name()}: {stats.captureTime_s.readS()}")

            frame, warning_info = display_and_save_frame(p_request)
            if warning_info:
                handle_unsupported_format(warning_info, unsupported_warning_flag)

        if previous_request is not None:
            previous_request.unlock()
        previous_request = p_request
        fi.imageRequestSingle()

    exampleHelper.manuallyStopAcquisitionIfNeeded(p_dev, fi)
    cv2.destroyAllWindows()


def main():
    """Main entry point for the script."""
    dev_mgr, p_dev = get_device()
    if p_dev is None:
        exampleHelper.requestENTERFromUser()
        sys.exit(-1)

    p_dev.open()
    frame_count = get_frame_count()
    capture_frames(p_dev, frame_count)
    exampleHelper.requestENTERFromUser()


if __name__ == "__main__":
    main()


