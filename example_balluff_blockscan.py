#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_balluff_blockscan.py

Description:
    Acquires block scan images (line scan blocks) from a Balluff camera triggered by an encoder.
    Images are stitched into a complete 2D image and saved after user interruption.

Author:
    Felip Martí, 2025
    fmarti@swin.edu.au

Dependencies:
    - mvIMPACT Acquire SDK
    - NumPy
    - OpenCV (cv2)
"""

import sys
import ctypes
import numpy as np
import cv2
from datetime import datetime
import threading
from mvIMPACT import acquire
from mvIMPACT.Common import exampleHelper

stop_acquisition = False  # Global stop flag


def get_device():
    dev_mgr = acquire.DeviceManager()
    dev_mgr.updateDeviceList()
    if dev_mgr.deviceCount() == 0:
        print("No devices found. Please connect a camera.")
        exampleHelper.requestENTERFromUser()
        sys.exit(-1)
    return dev_mgr, dev_mgr[0] if dev_mgr.deviceCount() == 1 else exampleHelper.getDeviceFromUserInput(dev_mgr)


def wait_for_user_to_stop():
    global stop_acquisition
    input("Press ENTER to stop acquisition...\n")
    stop_acquisition = True


def extract_block(p_request):
    image_size = p_request.imageSize.read()
    height = p_request.imageHeight.read()
    width = p_request.imageWidth.read()
    channels = p_request.imageChannelCount.read()
    bit_depth = p_request.imageChannelBitDepth.read()

    dtype = np.uint16 if bit_depth > 8 else np.uint8
    cbuf = (ctypes.c_char * image_size).from_address(int(p_request.imageData.read()))
    arr = np.frombuffer(cbuf, dtype=dtype)

    if arr.size != height * width * channels:
        print("Skipping malformed frame.")
        return None

    arr = arr.reshape((height, width, channels if channels > 1 else 1))
    arr = cv2.convertScaleAbs(arr, alpha=(255.0 / 65535.0)) if dtype == np.uint16 else arr
    return arr


def capture_blocks(p_dev):
    fi = acquire.FunctionInterface(p_dev)
    stats = acquire.Statistics(p_dev)

    # Queue initial requests
    for _ in range(5):
        fi.imageRequestSingle()

    exampleHelper.manuallyStartAcquisitionIfNeeded(p_dev, fi)
    previous_request = None
    stitched_blocks = []

    global stop_acquisition
    while not stop_acquisition:
        request_nr = fi.imageRequestWaitFor(500)
        if not fi.isRequestNrValid(request_nr):
            continue

        p_request = fi.getRequest(request_nr)
        if p_request.isOK:
            block = extract_block(p_request)
            if block is not None:
                stitched_blocks.append(block)

                if len(stitched_blocks) % 20 == 0:
                    preview = np.vstack(stitched_blocks[-40:])
                    cv2.imshow("Live Stitch Preview", preview)
                    cv2.waitKey(1)

        if previous_request is not None:
            previous_request.unlock()
        previous_request = p_request
        fi.imageRequestSingle()

    exampleHelper.manuallyStopAcquisitionIfNeeded(p_dev, fi)
    cv2.destroyAllWindows()

    if stitched_blocks:
        final_image = np.vstack(stitched_blocks)
        filename = f"stitched_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        cv2.imwrite(filename, final_image)
        print(f"Saved stitched image as {filename}")
    else:
        print("No blocks acquired. Nothing to save.")


def main():
    dev_mgr, p_dev = get_device()
    if p_dev is None:
        exampleHelper.requestENTERFromUser()
        sys.exit(-1)

    p_dev.open()

    # Start user stop thread
    stop_thread = threading.Thread(target=wait_for_user_to_stop)
    stop_thread.start()

    capture_blocks(p_dev)
    exampleHelper.requestENTERFromUser()


if __name__ == "__main__":
    main()


