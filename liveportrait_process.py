#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def run_command(cmd, verbose=False):
    if verbose:
        print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        raise e

def process_liveportrait(
    input_path,
    output_path,
    source_path,
    config_path,
    plugin_path="./build",
    crop_left=280,
    crop_right=280,
    enable_eye_retargeting=False,
    eyes_open_ratio=0.0,
    eye_retargeting_strength=1.0,
    gaze_x=0.0,
    gaze_y=0.0,
    enable_pose_offset=False,
    pose_pitch_offset=0.0,
    pose_yaw_offset=0.0,
    pose_roll_offset=0.0,
    side_by_side=False,
    docker_image="ducksouplab/liveportrait_gst:latest",
    verbose=False
):
    # Absolute paths setup
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    source_abs = os.path.abspath(source_path)
    config_abs = os.path.abspath(config_path)
    plugin_abs = os.path.abspath(plugin_path)
    
    input_dir = os.path.dirname(input_abs)
    input_file = os.path.basename(input_abs)
    output_dir = os.path.dirname(output_abs)
    output_file = os.path.basename(output_abs)
    source_dir = os.path.dirname(source_abs)
    source_file = os.path.basename(source_abs)
    
    docker_work = "/work"
    docker_source = "/source"
    docker_config = "/checkpoints"
    docker_plugin = "/plugin"
    
    eye_str = ""
    if enable_eye_retargeting:
        eye_str = (
            f"enable-eye-retargeting=true "
            f"eyes-open-ratio={eyes_open_ratio} "
            f"eye-retargeting-strength={eye_retargeting_strength} "
            f"gaze-x={gaze_x} gaze-y={gaze_y} "
        )
    
    pose_str = ""
    if enable_pose_offset:
        pose_str = (
            f"enable-pose-offset=true "
            f"pose-pitch-offset={pose_pitch_offset} "
            f"pose-yaw-offset={pose_yaw_offset} "
            f"pose-roll-offset={pose_roll_offset} "
        )

    if side_by_side:
        # Complex pipeline for side-by-side
        pipeline = (
            f"filesrc location={docker_work}/{input_file} ! decodebin name=dec "
            f"dec. ! queue ! videoconvert ! tee name=t "
            f"t. ! queue ! videocrop left={crop_left} right={crop_right} ! videoscale ! video/x-raw,width=512,height=512 ! mix.sink_0 "
            f"t. ! queue ! videocrop left={crop_left} right={crop_right} ! videoscale ! video/x-raw,width=512,height=512,format=RGB ! "
            f"liveportrait config-path={docker_config} source-image={docker_source}/{source_file} {eye_str}{pose_str}! "
            f"videoconvert ! mix.sink_1 "
            f"videomixer name=mix sink_1::xpos=512 ! videoconvert ! x264enc bitrate=2000 tune=zerolatency ! mux. "
            f"dec. ! queue ! audioconvert ! audioresample ! avenc_aac bitrate=192000 ! aacparse ! mux. "
            f"mp4mux name=mux faststart=true ! filesink location={docker_work}/{output_file}"
        )
    else:
        # Standard pipeline
        pipeline = (
            f"filesrc location={docker_work}/{input_file} ! decodebin name=dec "
            f"dec. ! queue ! videoconvert ! "
            f"videocrop left={crop_left} right={crop_right} ! "
            f"videoscale ! video/x-raw,width=512,height=512,format=RGB ! "
            f"liveportrait config-path={docker_config} source-image={docker_source}/{source_file} {eye_str}{pose_str}! "
            f"videoconvert ! x264enc bitrate=2000 tune=zerolatency ! mux. "
            f"dec. ! queue ! audioconvert ! audioresample ! avenc_aac bitrate=192000 ! aacparse ! mux. "
            f"mp4mux name=mux faststart=true ! filesink location={docker_work}/{output_file}"
        )

    docker_cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{input_dir}:{docker_work}",
        "-v", f"{source_dir}:{docker_source}",
        "-v", f"{config_abs}:{docker_config}",
        "-v", f"{plugin_abs}:{docker_plugin}",
        "-e", f"GST_PLUGIN_PATH={docker_plugin}",
    ]
    
    if os.getenv("GST_DEBUG"):
        docker_cmd.extend(["-e", f"GST_DEBUG={os.getenv('GST_DEBUG')}"])
    
    if output_dir != input_dir:
        docker_cmd.extend(["-v", f"{output_dir}:/output_dir"])
        pipeline = pipeline.replace(f"location={docker_work}/{output_file}", f"location=/output_dir/{output_file}")

    docker_cmd.extend([docker_image, "bash", "-c", f"gst-launch-1.0 -q {pipeline}"])

    run_command(docker_cmd, verbose)
    return True

def main():
    parser = argparse.ArgumentParser(description="Wrapper for LivePortrait GStreamer plugin via Docker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plugin-path", default="./build")
    parser.add_argument("--crop-left", type=int, default=280)
    parser.add_argument("--crop-right", type=int, default=280)
    parser.add_argument("--enable-eye-retargeting", action="store_true")
    parser.add_argument("--eyes-open-ratio", type=float, default=0.0)
    parser.add_argument("--eye-retargeting-strength", type=float, default=1.0)
    parser.add_argument("--gaze-x", type=float, default=0.0)
    parser.add_argument("--gaze-y", type=float, default=0.0)
    parser.add_argument("--enable-pose-offset", action="store_true")
    parser.add_argument("--pose-pitch-offset", type=float, default=0.0)
    parser.add_argument("--pose-yaw-offset", type=float, default=0.0)
    parser.add_argument("--pose-roll-offset", type=float, default=0.0)
    parser.add_argument("--side-by-side", action="store_true", help="Generate side-by-side comparison")
    parser.add_argument("--docker-image", default="ducksouplab/liveportrait_gst:latest")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    try:
        process_liveportrait(
            input_path=args.input, output_path=args.output, source_path=args.source, config_path=args.config,
            plugin_path=args.plugin_path, crop_left=args.crop_left, crop_right=args.crop_right,
            enable_eye_retargeting=args.enable_eye_retargeting, eyes_open_ratio=args.eyes_open_ratio,
            eye_retargeting_strength=args.eye_retargeting_strength, gaze_x=args.gaze_x, gaze_y=args.gaze_y,
            enable_pose_offset=args.enable_pose_offset, pose_pitch_offset=args.pose_pitch_offset,
            pose_yaw_offset=args.pose_yaw_offset, pose_roll_offset=args.pose_roll_offset,
            side_by_side=args.side_by_side,
            docker_image=args.docker_image, verbose=args.verbose
        )
        print(f"Success! Output saved to {args.output}")
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
