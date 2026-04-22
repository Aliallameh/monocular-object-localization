#!/usr/bin/env python3
"""
Visual Measurement Tool

Displays frames with the detected bbox center marked, to help you
identify the actual floor contact point.
"""

import cv2
import json
from pathlib import Path

def visualize_frame(frame_num, frame_id, u_center, v_bottom):
    """Display frame with detected center marked."""
    
    frame_path = f'frame_{frame_num:03d}.jpg'
    if not Path(frame_path).exists():
        print(f"❌ {frame_path} not found")
        return
    
    img = cv2.imread(frame_path)
    if img is None:
        print(f"❌ Could not read {frame_path}")
        return
    
    # Draw detected center
    # Horizontal line at u_center
    cv2.line(img, (int(u_center), 0), (int(u_center), img.shape[0]), 
             color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    
    # Vertical line at v_bottom
    cv2.line(img, (0, int(v_bottom)), (img.shape[1], int(v_bottom)), 
             color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    
    # Circle at intersection (detected bottom-center)
    cv2.circle(img, (int(u_center), int(v_bottom)), 15, 
               color=(0, 255, 255), thickness=2)
    
    # Add text labels
    cv2.putText(img, f'Frame {frame_id}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.putText(img, f'Detected bottom-center: ({u_center:.0f}, {v_bottom:.0f})', 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, 'Green lines & circle = detected center', (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(img, 'Find ACTUAL FLOOR CONTACT (where bin touches ground)', (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    
    # Save annotated frame
    output_path = f'frame_{frame_num:03d}_annotated.jpg'
    cv2.imwrite(output_path, img)
    print(f"✓ Annotated frame saved: {output_path}")
    
    # Also display
    cv2.imshow(f'Frame {frame_id} - Press ESC to close', img)
    print(f"  Press ESC or any key to close the window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Create and display annotated frames."""
    
    frame_data = [
        (1, 50, 977, 880),
        (2, 200, 587, 696),
        (3, 500, 470, 827),
    ]
    
    print("""
╔════════════════════════════════════════════════════════════╗
║           VISUAL MEASUREMENT TOOL                          ║
║                                                            ║
║  This will display frames with detected center marked:    ║
║  - Green cross-hairs = detected bbox bottom-center        ║
║  - Yellow circle = center point                           ║
║                                                            ║
║  Your task: Find where the bin ACTUALLY touches           ║
║  the ground (typically a few pixels away)                 ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    for frame_num, frame_id, u_center, v_bottom in frame_data:
        print(f"\n→ Processing frame {frame_id}...")
        visualize_frame(frame_num, frame_id, u_center, v_bottom)
    
    print("\n✓ All annotated frames created!")
    print("\nAnnotated frames saved as frame_001_annotated.jpg, etc.")
    print("Use these to measure the offset to the actual floor contact point.")

if __name__ == '__main__':
    main()
