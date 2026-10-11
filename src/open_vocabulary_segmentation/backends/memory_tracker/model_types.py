from transformers import (
    EdgeTamVideoModel,
    Sam2VideoInferenceSession,
    Sam2VideoModel,
    Sam2VideoProcessor,
    Sam3TrackerVideoInferenceSession,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)
from transformers.models.edgetam_video.modeling_edgetam_video import EdgeTamVideoSegmentationOutput
from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoSegmentationOutput
from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import Sam3TrackerVideoSegmentationOutput

type MemoryTrackerModel = Sam2VideoModel | Sam3TrackerVideoModel | EdgeTamVideoModel
type MemoryTrackerProcessor = Sam2VideoProcessor | Sam3TrackerVideoProcessor
type MemoryInferenceSession = Sam2VideoInferenceSession | Sam3TrackerVideoInferenceSession
type MemoryTrackerOutput = (
    Sam2VideoSegmentationOutput | Sam3TrackerVideoSegmentationOutput | EdgeTamVideoSegmentationOutput
)
