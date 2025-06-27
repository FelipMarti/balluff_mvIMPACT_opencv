#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_balluff_areascan_record_video.py

High-performance version:
- Buffers raw request objects during acquisition (minimal processing)
- Converts frames after capture for video export
- Reduced logging, increased request queue

Author: Felip Marti (2025)
"""

import sys
import ctypes
import numpy as np
import cv2
import time
import threading
from datetime import datetime
from mvIMPACT import acquire
from mvIMPACT.Common import exampleHelper

stop_recording = False

# Terminal colours for clearer status output
class TerminalColours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def wait_for_enter():
    global stop_recording
    input("\nRecording... Press ENTER to stop.\n")
    stop_recording = True

def handle_unsupported_format(info, warning_flag):
    """Warns once if unsupported format is encountered."""
    if not warning_flag[0]:
        print(f"\n{TerminalColours.WARNING}Unsupported image format detected.{TerminalColours.ENDC}")
        print(info)
        print("This script only supports 16-bit, 3-channel RGB images.")
        print("Please adjust the camera configuration:")
        print("   → Setting > Camera > GenICam > ImageFormatControl > PixelFormat = BayerRG16")
        print("Check with Felip or Shuo for help.\n")
        warning_flag[0] = True
    else:
        print(f"{TerminalColours.WARNING}Skipped unsupported frame (not 16-bit / 3-channel){TerminalColours.ENDC}")

def convert_frame(p_request, warning_flag):
    """Converts 16-bit 3-channel image buffer to 8-bit OpenCV image."""
    image_size = p_request.imageSize.read()
    height = p_request.imageHeight.read()
    width = p_request.imageWidth.read()
    channels = p_request.imageChannelCount.read()
    bit_depth = p_request.imageChannelBitDepth.read()

    if channels != 3 or bit_depth != 16:
        handle_unsupported_format(
            f"Format: {channels} channels, {bit_depth}-bit depth", warning_flag
        )
        return None

    cbuf = (ctypes.c_char * image_size).from_address(int(p_request.imageData.read()))
    arr = np.frombuffer(cbuf, dtype=np.uint16)

    if arr.size != height * width * channels:
        handle_unsupported_format("Invalid array size", warning_flag)
        return None

    arr = arr.reshape((height, width, channels))
    img_cv = cv2.convertScaleAbs(arr, alpha=(255.0 / 65535.0))
    return img_cv

def capture_and_buffer(p_dev):
    global stop_recording
    fi = acquire.FunctionInterface(p_dev)

    # Reset internal queues to avoid stale entries
    fi.imageRequestReset(0, 0)
    time.sleep(0.2)  # Allow hardware to stabilise

    MAX_REQUESTS = 20
    queued = 0
    for _ in range(MAX_REQUESTS):
        result = fi.imageRequestSingle()
        if result == acquire.DMR_NO_ERROR:
            queued += 1
        else:
            print(f"{TerminalColours.WARNING}Failed to queue frame: {result}{TerminalColours.ENDC}")

    print(f"{TerminalColours.OKBLUE}Queued {queued} initial requests.{TerminalColours.ENDC}")

    exampleHelper.manuallyStartAcquisitionIfNeeded(p_dev, fi)

    print(f"{TerminalColours.OKBLUE}Capturing raw requests into memory...{TerminalColours.ENDC}")
    threading.Thread(target=wait_for_enter, daemon=True).start()

    previous_request = None
    requests = []
    timestamps = []
    warning_flag = [False]  # Track unsupported format warning
    start_time = time.time()

    while not stop_recording:
        request_nr = fi.imageRequestWaitFor(100, 0)  # 100ms timeout
        if not fi.isRequestNrValid(request_nr):
            continue

        p_request = fi.getRequest(request_nr)
        if p_request.isOK:
            frame = convert_frame(p_request, warning_flag)
            if frame is not None:
                requests.append((frame, time.time()))
                if len(requests) % 10 == 0:
                    print(f"{TerminalColours.OKGREEN}Captured {len(requests)} frames{TerminalColours.ENDC}")
            p_request.unlock()
        else:
            print(f"{TerminalColours.FAIL}Received invalid frame.{TerminalColours.ENDC}")
            p_request.unlock()

        if fi.imageRequestSingle() != acquire.DMR_NO_ERROR:
            print(f"{TerminalColours.WARNING}Could not re-queue new request.{TerminalColours.ENDC}")

    exampleHelper.manuallyStopAcquisitionIfNeeded(p_dev, fi)
    end_time = time.time()
    print(f"{TerminalColours.OKBLUE}Capture complete. Now processing frames...{TerminalColours.ENDC}")

    if requests:
        frames = [frame for frame, _ in requests]
        cv2.destroyAllWindows()

        height, width = frames[0].shape[:2]
        size = (width, height)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = f"video_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        duration = end_time - start_time
        effective_fps = len(frames) / duration
        fps = max(effective_fps, 1.0)  # Avoid 0 FPS

        writer = cv2.VideoWriter(output_filename, fourcc, fps, size)
        for f in frames:
            writer.write(f)
        writer.release()

        print("\nRecording Summary")
        print("------------------")
        print(f"Output file:      {output_filename}")
        print(f"Total frames:     {len(frames)}")
        print(f"Duration (s):     {duration:.2f}")
        print(f"Effective FPS:    {effective_fps:.2f}\n")
    else:
        print(f"{TerminalColours.FAIL}No frames captured.{TerminalColours.ENDC}")

def main():
    print(f"{TerminalColours.HEADER}Starting memory-buffered video capture...{TerminalColours.ENDC}")
    try:
        dev_mgr = acquire.DeviceManager()
        dev_mgr.updateDeviceList()
        if dev_mgr.deviceCount() == 0:
            print(f"{TerminalColours.FAIL}No camera found.{TerminalColours.ENDC}")
            return

        p_dev = dev_mgr[0]
        p_dev.open()

        capture_and_buffer(p_dev)

    except Exception as e:
        print(f"{TerminalColours.FAIL}Error: {e}{TerminalColours.ENDC}")

if __name__ == "__main__":
    main()


