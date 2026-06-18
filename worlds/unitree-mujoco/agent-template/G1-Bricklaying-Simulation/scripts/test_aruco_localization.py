"""
ArUco table localization test.

Captures N frames, saves an annotated JPEG per frame, and appends pose
results to a log file.
"""
import os
import cv2

from bricklaying.perception.realsense import RealSenseCamera
from bricklaying.perception.aruco_localizer import ArucoLocalizer

N_FRAMES = 3
OUT_DIR = "eval_outputs"


os.makedirs(OUT_DIR, exist_ok=True)

with RealSenseCamera() as cam:
    intrinsics = cam.intrinsics
    print(f"Camera ready: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
          f"cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f}")
    print(f"Capturing {N_FRAMES} frames → {OUT_DIR}/\n")
    
    localizer = ArucoLocalizer(intrinsics)

    try:
        for i in range(N_FRAMES):
            rgb, _ = cam.get_frames()
            detections = localizer.detect(rgb)
            result = localizer.estimate_table_pose(detections)
            frame = localizer.annotate(rgb, detections, result)

            img_path = os.path.join(OUT_DIR, f"frame_{i:04d}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            ids_seen = [d.marker_id for d in detections]
            if result:
                T = result.T_camera_to_table
                msg = (f"frame {i:04d}  markers={result.n_markers_used}"
                       f"  reproj={result.reprojection_error:.2f}px"
                       f"  ids={ids_seen}\n"
                       f"{T}\n")
            else:
                msg = f"frame {i:04d}  no pose  ids={ids_seen}\n"

            print(msg, end="")

    except KeyboardInterrupt:
        print("\nStopped early.")

print(f"\nDone. Images and poses saved to {OUT_DIR}/")

