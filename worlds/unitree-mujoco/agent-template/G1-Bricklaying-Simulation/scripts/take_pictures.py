"""
Capture RGB and depth images from RealSense camera on user input.
Stores images to src/bricklaying/assets/scene_measurements/
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import queue

from bricklaying.perception.realsense import RealSenseCamera

# Output directory
OUTPUT_DIR = Path("src/bricklaying/assets/scene_measurements")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Global queue to pass messages from the input thread to the main loop
input_queue = queue.Queue()

def input_thread():
    while True:
        cmd = input() # Blocks here waiting for Enter
        input_queue.put(cmd)
        if cmd == 'q':
            break


def save_capture(rgb: np.ndarray, depth: np.ndarray, index: int):
    """Save RGB and depth images with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save RGB
    rgb_path = OUTPUT_DIR / f"rgb_{timestamp}_{index:03d}.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    # Save depth as 16-bit PNG
    depth_path = OUTPUT_DIR / f"depth_{timestamp}_{index:03d}.png"
    cv2.imwrite(str(depth_path), depth)
    
    # Save depth visualization (colorized for viewing)
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=0.03),
        cv2.COLORMAP_JET
    )
    depth_vis_path = OUTPUT_DIR / f"depth_vis_{timestamp}_{index:03d}.png"
    cv2.imwrite(str(depth_vis_path), depth_colormap)
    
    print(f"  Saved: {rgb_path.name}, {depth_path.name}")
    return rgb_path, depth_path


def main():
    print("Starting RealSense camera...")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print("\nControls:")
    print("  SPACE - Capture image")
    print("  Q     - Quit")
    print()

    # Start the input thread
    t = threading.Thread(target=input_thread)
    t.daemon = True
    t.start()

    with RealSenseCamera() as camera:
        print("Camera ready! Press ENTER to capture, type 'q' then ENTER to quit.")
        capture_count = 0

        while True:
            rgb, depth = camera.get_frames()

            # Check if user typed something
            if not input_queue.empty():
                cmd = input_queue.get()
                if cmd == 'q':
                    break
                else:
                    # Capture on any other Enter press
                    print("Capturing...")
                    save_capture(rgb, depth, capture_count)
                    capture_count += 1
    
        cv2.destroyAllWindows()
        print(f"\nCaptured {capture_count} images total.")
        print(f"Saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()

