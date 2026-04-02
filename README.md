# gst-liveportrait

> [!WARNING]  
> **Status: Experimental / Non-Functional**  
> This project is currently under active development and is not yet functional for general use. The C++ port of the LivePortrait logic is being refined.

A high-performance GStreamer video filter plugin for real-time head reenactment using **LivePortrait** and **TensorRT 10**.

## Overview

`gst-liveportrait` is a C++ GStreamer plugin designed to animate a static source image using a driving video stream. It leverages NVIDIA TensorRT for ultra-fast inference, aiming for real-time performance on modern GPUs.

This implementation is based on the architecture and custom CUDA kernels defined in the [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) repository.

## Features (In Development)

- **High Performance:** Targeted ~45 FPS on NVIDIA RTX A5000.
- **Asynchronous Processing:** Uses Pinned Memory (`cudaMallocHost`) and `cudaMemcpyAsync`.
- **Full Retargeting:** Integration of Eye and Lip retargeting engines.
- **Stitching Engine:** Integrated alignment correction.
- **Motion Smoothing:** Implementation of the One-Euro filter.

## Installation & Build

The plugin must be built within the provided Docker environment to ensure all dependencies (TensorRT 10, CUDA 12, GStreamer 1.28) are met.

### 1. Build the Docker Image
```bash
docker build -t gst-liveportrait-env .
```

### 2. Export TensorRT Engines
Follow the instructions in `INSTRUCTIONS.md` (Phase 1) inside the container to download ONNX models and export them to `.trt` files.

### 3. Compile the Plugin using Docker
Run the following command from the project root to compile the plugin:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace gst-liveportrait-env bash -c "mkdir -p build && cd build && cmake .. && make -j$(nproc)"
```

## Usage (Experimental)

Once built, you can test the plugin using `gst-launch-1.0` within the Docker container.

### GStreamer Pipeline Example
```bash
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace gst-liveportrait-env bash -c "\
    GST_PLUGIN_PATH=./build gst-launch-1.0 \
    filesrc location=assets/video_example.mp4 ! \
    decodebin ! videoconvert ! \
    videocrop left=280 right=280 ! \
    videoscale ! video/x-raw,width=512,height=512,format=RGB ! \
    liveportrait config-path=./checkpoints source-image=assets/test_image.jpg ! \
    videoconvert ! x264enc ! mp4mux ! filesink location=outputs/output.mp4"
```

## Architecture

- **`CudaMemoryManager`**: Manages managed and pinned memory for host-device transfers.
- **`TRTWrapper`**: Encapsulates TensorRT engine loading and execution.
- **`LivePortraitPipeline`**: Orchestrates the inference engines and relative motion logic.
- **`image_proc.cu`**: Custom CUDA kernels for image processing and keypoint math.

## Acknowledgments

- [FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) for the original algorithm.
- [grid-sample3d-trt-plugin](https://github.com/SeanWangJS/grid-sample3d-trt-plugin) for the custom TensorRT plugin.

## License

This project is licensed under the LGPL.
