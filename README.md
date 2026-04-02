# gst-liveportrait

A high-performance GStreamer video filter plugin for real-time head reenactment using **LivePortrait** and **TensorRT 10**.

## Overview

`gst-liveportrait` is a C++ GStreamer plugin that allows you to animate a static source image using a driving video stream. It leverages NVIDIA TensorRT for ultra-fast inference, achieving real-time performance (~45 FPS) on modern GPUs.

This implementation is strictly based on the architecture and custom CUDA kernels defined in the [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) repository.

## Features

- **High Performance:** ~45 FPS (~22ms latency) on NVIDIA RTX A5000.
- **Asynchronous Processing:** Uses Pinned Memory (`cudaMallocHost`) and `cudaMemcpyAsync` with a dedicated `cudaStream_t` to eliminate PCIe bottlenecks.
- **Full Retargeting:** Includes specialized engines for Eye and Lip retargeting, ensuring realistic gaze and mouth movements.
- **Stitching Engine:** Integrated stitching to ensure natural head proportions and alignment.
- **Motion Smoothing:** Implements the One-Euro filter to eliminate facial jitter.
- **GStreamer Integration:** Subclasses `GstVideoFilter` for seamless integration into any GStreamer pipeline.

## Prerequisites

- **Environment:** Docker with NVIDIA GPU support (`nvidia-container-runtime`).
- **Base Image:** `ducksouplab/ducksoup:ducksoup_plugins_gst1.28.0`.
- **Dependencies:** TensorRT 10.x, CUDA 12.x, OpenCV 4.x, GStreamer 1.28.

## Installation & Build

The plugin is designed to be built inside the provided Docker environment.

### 1. Build the Docker Image
```bash
docker build -t gst-liveportrait-env .
```

### 2. Export TensorRT Engines
Follow the instructions in `INSTRUCTIONS.md` (Phase 1) to download ONNX models and export them to `.trt` files using the patched scripts.

### 3. Compile the Plugin
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

Load the plugin by adding its path to `GST_PLUGIN_PATH`.

### GStreamer Pipeline Example
```bash
GST_PLUGIN_PATH=./build gst-launch-1.0 \
    filesrc location=assets/video_example.mp4 ! \
    decodebin ! videoconvert ! \
    videocrop left=280 right=280 ! \
    videoscale ! video/x-raw,width=512,height=512,format=RGB ! \
    liveportrait config-path=./checkpoints source-image=assets/test_image.jpg ! \
    videoconvert ! x264enc ! mp4mux ! filesink location=output.mp4
```

### Plugin Properties
- `config-path`: Path to the directory containing the TensorRT engines (e.g., `./checkpoints`).
- `source-image`: Path to the static source image (e.g., `assets/test_image.jpg`).

## Architecture

- **`CudaMemoryManager`**: Manages managed and pinned memory for zero-copy-like performance between host and device.
- **`TRTWrapper`**: Encapsulates TensorRT engine loading, execution, and I/O binding management.
- **`LivePortraitPipeline`**: Orchestrates the 7+ engines (Appearance, Motion, Warping, Stitching, etc.) and implements the relative motion logic.
- **`image_proc.cu`**: Custom CUDA kernels for preprocessing (normalization, transpose), postprocessing, and keypoint transformations.

## Performance Profiling

The plugin includes built-in `cudaEvent` profiling. Typical results on an RTX A5000:
- **Total Latency:** ~22ms
- **Preprocessing:** ~0.2ms
- **Motion Extraction:** ~2.0ms
- **Warping (Core):** ~19.0ms
- **Postprocessing:** ~0.01ms

## Acknowledgments

- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) for the original algorithm and model architecture.
- [grid-sample3d-trt-plugin](https://github.com/SeanWangJS/grid-sample3d-trt-plugin) for the custom TensorRT plugin.

## License

This project is licensed under the LGPL (consistent with GStreamer plugin standards).
