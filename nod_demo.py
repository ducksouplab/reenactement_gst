import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import math
import time
import sys
import os

# Enable debug logging
os.environ["GST_DEBUG"] = "3,liveportrait:5"

Gst.init(None)

def run_nod_demo():
    # Pipeline components
    # Using Petter as source
    pipeline_str = (
        "filesrc location=/work/video_example.mp4 ! decodebin name=dec "
        "dec. ! queue ! videoconvert ! videocrop left=280 right=280 ! videoscale ! video/x-raw,width=512,height=512,format=RGB ! "
        "liveportrait name=lp config-path=/checkpoints source-image=/source/Petter-Johansson.jpg enable-pose-offset=true ! "
        "videoconvert ! x264enc bitrate=2000 tune=zerolatency ! mux. "
        "dec. ! queue ! audioconvert ! audioresample ! lamemp3enc ! mpegaudioparse ! mux. "
        "qtmux name=mux ! filesink location=/work/petter_nod_demo.mp4"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    lp = pipeline.get_by_name("lp")
    
    if not lp:
        print("Error: Could not find liveportrait element 'lp'")
        return

    loop = GLib.MainLoop()

    # Modulation parameters - EXTREME for visibility
    start_time = time.time()
    amplitude = 0.6  # 0.6 radians ~ 34 degrees
    frequency = 1.0  # 1 nod per second

    def update_pose():
        elapsed = time.time() - start_time
        # Sine wave for nodding (pitch)
        pitch_offset = amplitude * math.sin(2 * math.pi * frequency * elapsed)
        lp.set_property("pose-pitch-offset", pitch_offset)
        # Print to stdout so we can see it in logs
        print(f"Update: pitch_offset={pitch_offset:.4f}", flush=True)
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

    print("Starting nodding demonstration...")
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)
    print("Done. Saved to petter_nod_demo.mp4")

if __name__ == "__main__":
    run_nod_demo()
