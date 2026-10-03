"""
Utility functions for detection processing.
Handles upscaling and coordinate transformation for bounding boxes.
"""
import numpy as np
from typing import Union, Tuple


def upscale_detections(
    tracks: np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    inplace: bool = False,
) -> np.ndarray:
    """
    Upscale detection bounding boxes from source resolution to target resolution.
    
    This is used to transform detections from event camera resolution (640x480)
    to RGB image resolution (e.g., 1440x1080 or other sizes).
    
    Args:
        tracks: Structured numpy array with fields ['x', 'y', 'w', 'h', ...]
                or regular array with shape (N, 5+) where columns are [x, y, w, h, ...]
        source_width: Original width (e.g., 640 for event camera)
        source_height: Original height (e.g., 480 for event camera)
        target_width: Target width (e.g., 1440 for RGB)
        target_height: Target height (e.g., 1080 for RGB)
        inplace: If True, modify tracks in place; otherwise create a copy
        
    Returns:
        Upscaled tracks with bounding boxes scaled to target resolution
        
    Example:
        >>> # Upscale from event camera (640x480) to RGB (1440x1080)
        >>> tracks_upscaled = upscale_detections(
        ...     tracks, 
        ...     source_width=640, 
        ...     source_height=480,
        ...     target_width=1440, 
        ...     target_height=1080
        ... )
    """
    if source_width == target_width and source_height == target_height:
        return tracks if inplace else tracks.copy()
    
    # Calculate scale factors
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    
    # Copy if not inplace
    if not inplace:
        tracks = tracks.copy()
    
    # Handle structured array (from DSEC dataset)
    if tracks.dtype.names is not None:
        if 'x' in tracks.dtype.names:
            tracks['x'] = tracks['x'] * scale_x
        if 'y' in tracks.dtype.names:
            tracks['y'] = tracks['y'] * scale_y
        if 'w' in tracks.dtype.names:
            tracks['w'] = tracks['w'] * scale_x
        if 'h' in tracks.dtype.names:
            tracks['h'] = tracks['h'] * scale_y
    else:
        # Handle regular array format [x, y, w, h, ...]
        if tracks.ndim == 2 and tracks.shape[1] >= 4:
            tracks[:, 0] *= scale_x  # x
            tracks[:, 1] *= scale_y  # y
            tracks[:, 2] *= scale_x  # w
            tracks[:, 3] *= scale_y  # h
        else:
            raise ValueError(f"Invalid tracks shape: {tracks.shape}")
    
    return tracks


def upscale_bboxes(
    bboxes: np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    format: str = 'xywh',
) -> np.ndarray:
    """
    Upscale bounding boxes from source resolution to target resolution.
    
    Args:
        bboxes: Array of shape (N, 4) with bounding boxes
        source_width: Original width
        source_height: Original height
        target_width: Target width
        target_height: Target height
        format: Bounding box format - 'xywh' (top-left x,y and width,height)
                or 'xyxy' (top-left and bottom-right corners)
        
    Returns:
        Upscaled bounding boxes of shape (N, 4)
    """
    if source_width == target_width and source_height == target_height:
        return bboxes.copy()
    
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    
    bboxes_scaled = bboxes.copy()
    
    if format == 'xywh':
        # [x, y, w, h] format
        bboxes_scaled[:, 0] *= scale_x  # x
        bboxes_scaled[:, 1] *= scale_y  # y
        bboxes_scaled[:, 2] *= scale_x  # w
        bboxes_scaled[:, 3] *= scale_y  # h
    elif format == 'xyxy':
        # [x1, y1, x2, y2] format
        bboxes_scaled[:, 0] *= scale_x  # x1
        bboxes_scaled[:, 1] *= scale_y  # y1
        bboxes_scaled[:, 2] *= scale_x  # x2
        bboxes_scaled[:, 3] *= scale_y  # y2
    else:
        raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")
    
    return bboxes_scaled


def get_scale_factors(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> Tuple[float, float]:
    """
    Get scale factors for converting from source to target resolution.
    
    Args:
        source_width: Original width
        source_height: Original height
        target_width: Target width
        target_height: Target height
        
    Returns:
        Tuple of (scale_x, scale_y)
    """
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return scale_x, scale_y


# Constants for DSEC dataset resolutions
EVENT_CAMERA_WIDTH = 640
EVENT_CAMERA_HEIGHT = 480

# Common RGB resolutions in DSEC
RGB_RESOLUTION_RECTIFIED = (1440, 1080)  # (width, height)
RGB_RESOLUTION_DISTORTED = (1440, 1080)  # (width, height)
