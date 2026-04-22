#!/usr/bin/env python3
"""
Centroid Measurement Helper

This script helps you record the pixel measurements and automatically
converts them to world coordinates.

Usage:
    python centroid_measurement_helper.py
"""

import json
from pathlib import Path

def measure_frame(frame_id, frame_num):
    """Interactive measurement for a single frame."""
    print(f"\n{'='*60}")
    print(f"Frame {frame_id} (file: frame_{frame_num:03d}.jpg)")
    print(f"{'='*60}")
    
    # Detected centers
    detected = {
        50: (977, 880),
        200: (587, 696),
        500: (470, 827)
    }
    
    bbox_heights = {
        50: 402.0,
        200: 234.2,
        500: 416.1
    }
    
    u_detected, v_detected = detected[frame_id]
    h_px = bbox_heights[frame_id]
    
    print(f"\nDetected bottom-center: u = {u_detected} px, v = {v_detected} px")
    print(f"BBox height: {h_px} px")
    
    # Depth estimation
    f_y = 1402.5  # focal length from calib.json
    BIN_HEIGHT_M = 0.65
    z_distance = (f_y * BIN_HEIGHT_M) / h_px
    
    print(f"Estimated depth: z ≈ {z_distance:.2f} m")
    
    # Ask user for measurement
    print(f"\nOpen frame_{frame_num:03d}.jpg and locate the ACTUAL FLOOR CONTACT point")
    print(f"(where the bin physically touches the ground)")
    
    try:
        u_actual = float(input(f"Enter actual floor contact u coordinate (pixels): "))
    except ValueError:
        print("Invalid input. Skipping this frame.")
        return None
    
    # Calculate offset
    offset_px = u_actual - u_detected
    f_x = 1402.5  # focal length from calib.json
    offset_m = abs(offset_px) * (z_distance / f_x)
    
    print(f"\n✓ Offset: {offset_px:.0f} px = {offset_m:.4f} m")
    
    notes = input("Notes for this frame (optional): ").strip()
    
    return {
        "frame_id": frame_id,
        "u_center_detected_px": u_detected,
        "u_contact_actual_px": u_actual,
        "offset_pixel": offset_px,
        "offset_world_m": offset_m,
        "depth_estimated_m": z_distance,
        "notes": notes
    }

def main():
    """Run measurement collection."""
    print("""
╔════════════════════════════════════════════════════════════╗
║        CENTROID OFFSET MEASUREMENT HELPER                  ║
║                                                            ║
║  This will measure the pixel offset between:              ║
║  1. Detected bbox bottom-center (from detector)           ║
║  2. Actual floor contact point (visual measurement)       ║
║                                                            ║
║  Expected offset: 0.2–0.4 m (confirms hypothesis)         ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    frame_ids = [50, 200, 500]
    frame_nums = [1, 2, 3]  # Frame filenames: frame_001.jpg, frame_002.jpg, frame_003.jpg
    
    measurements = []
    
    for frame_id, frame_num in zip(frame_ids, frame_nums):
        result = measure_frame(frame_id, frame_num)
        if result:
            measurements.append(result)
    
    if not measurements:
        print("\n❌ No measurements recorded.")
        return
    
    # Calculate statistics
    mean_offset = sum(m["offset_world_m"] for m in measurements) / len(measurements)
    
    # Hypothesis test
    hypothesis_confirmed = 0.15 < mean_offset < 0.5
    hypothesis_result = "✅ CONFIRMED" if hypothesis_confirmed else "❌ REFUTED"
    
    # Save results
    output = {
        "hypothesis": "Bin centroid (visual center) != floor contact point",
        "expected_offset_m": 0.25,
        "notes": "Bin is ~0.4m diameter cylinder. Contact is on edge, not at center.",
        "measurements": measurements,
        "mean_offset_m": mean_offset,
        "hypothesis_test": f"Mean offset {mean_offset:.4f}m",
        "result": hypothesis_result
    }
    
    # Ensure experiments directory exists
    Path('experiments').mkdir(exist_ok=True)
    
    output_file = Path('experiments/centroid_validation.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Measurements recorded: {len(measurements)}")
    for m in measurements:
        print(f"  Frame {m['frame_id']:3d}: offset = {m['offset_world_m']:7.4f} m ({m['offset_pixel']:6.0f} px)")
    print(f"\nMean offset: {mean_offset:.4f} m")
    print(f"Hypothesis: {hypothesis_result}")
    print(f"\n✓ Results saved to: {output_file}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    if hypothesis_confirmed:
        print("✅ Hypothesis CONFIRMED: Centroid approximation is the error source")
        print("   Next: Apply offset correction to localizer.py")
    else:
        print("❌ Hypothesis REFUTED: Centroid approximation is NOT the source")
        print("   Next: Investigate alternative error sources")

if __name__ == '__main__':
    main()
