import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import math
import time
import sys
import os
import argparse

# Enable debug logging
os.environ["GST_DEBUG"] = "3,liveportrait:5"

Gst.init(None)

def run_nod_demo(source_image, output_file):
    # Pipeline components
    pipeline_str = (
        f"filesrc location=/work/video_example.mp4 ! decodebin name=dec "
        f"dec. ! queue ! videoconvert ! videocrop left=280 right=280 ! videoscale ! video/x-raw,width=512,height=512,format=RGB ! "
        f"liveportrait name=lp config-path=/checkpoints source-image={source_image} enable-pose-offset=true ! "
        f"videoconvert ! x264enc bitrate=2000 tune=zerolatency ! mux. "
        f"dec. ! queue ! audioconvert ! audioresample ! lamemp3enc ! mpegaudioparse ! mux. "
        f"qtmux name=mux ! filesink location={output_file}"
    )

    print(f"Running nod demo for {source_image} -> {output_file}")
    pipeline = Gst.parse_launch(pipeline_str)
    lp = pipeline.get_by_name("lp")
    
    if not lp:
        print("Error: Could not find liveportrait element 'lp'")
        return

    loop = GLib.MainLoop()

    # Modulation parameters - Subtle for realism
    start_time = time.time()
    amplitude = 0.2  # 0.2 radians ~ 11.5 degrees
    frequency = 1.0  # 1 nod per second

    def update_pose():
        elapsed = time.time() - start_time
        # Sine wave for nodding (pitch)
        pitch_offset = amplitude * math.sin(2 * math.pi * frequency * elapsed)
        lp.set_property("pose-pitch-offset", pitch_offset)
        return True # Continue timer

    # Update every 33ms (~30fps)
    timer_id = GLib.timeout_add(33, update_pose)

    # Bus listener for EOS
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err.message}")
            loop.quit()

    bus.connect("message", on_message)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)
    print(f"Done. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_nod_demo(args.source, args.output)
