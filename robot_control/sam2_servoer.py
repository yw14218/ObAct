import rclpy
import cv2
import torch
import numpy as np
import supervision as sv
from base_servoer import CartesianVisualServoer
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

# Enable CUDA optimizations
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load models
sam2_checkpoint = "/home/yilong/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# Initialize grounding DINO model
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

class SAM2VisualServoer(CartesianVisualServoer):
    def __init__(self, rgb_ref, seg_ref, text, use_depth=False, silent=False):
        super().__init__(use_depth=use_depth, silent=silent)
        self.text = text
        self.rgb_ref = rgb_ref
        self.seg_ref = seg_ref

    def track(self):
        live_rgb, live_depth = self.observe()

        if live_rgb is None:
            self.log_error("No RGB image received. Check camera and topics.")
            raise RuntimeError

        # Process image for detection
        inputs = processor(images=live_rgb, text=self.text, return_tensors="pt").to(device, non_blocking=True)

        with torch.inference_mode():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[live_rgb.shape[:2]]
        )

        # # Prompt SAM image predictor to get the mask
        # image_predictor.set_image(live_rgb)

        # # Get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()

        # masks, scores, logits = image_predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_boxes,
        #     multimask_output=False,
        # )
        num_boxes = input_boxes.shape[0]
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            class_id = np.zeros(num_boxes, dtype=np.int32) 
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=live_rgb.copy(), detections=detections)

        # Display the result
        cv2.imshow("Live Tracking with Mask", annotated_frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    rclpy.init()
    
    dir = "robot_control/mug"

    # Load reference images
    rgb_ref = cv2.imread(f"{dir}/demo_wrist_rgb.png")[..., ::-1]  # Convert BGR to RGB
    seg_ref = cv2.imread(f"{dir}/demo_wrist_mask.png", cv2.IMREAD_GRAYSCALE).astype(bool)

    # Initialize visual servoer
    sam2vs = SAM2VisualServoer(rgb_ref, seg_ref, text="scissor", use_depth=True, silent=False)

    try:
        while rclpy.ok():
            if not sam2vs.silent:
                print("Starting new observation cycle...")

            # Capture live frame
            live_rgb, live_depth = sam2vs.observe(timeout=2.0)
            if live_rgb is not None and not sam2vs.silent:
                print(f"Received RGB image with shape: {live_rgb.shape}")

            # Run tracking and visualization
            sam2vs.track()

    except KeyboardInterrupt:
        print("Program terminated by user")

    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()
