import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class SEGSBBoxVisualizer:
    """Draw bounding boxes from a ComfyUI SEGS input and create a text report."""

    COLORS = (
        (255, 70, 70),
        (70, 220, 100),
        (70, 150, 255),
        (255, 190, 50),
        (200, 100, 255),
        (50, 220, 220),
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "box_thickness": ("INT", {"default": 3, "min": 1, "max": 15}),
                "show_label": ("BOOLEAN", {"default": True}),
                "show_confidence": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("annotated_image", "object_list")
    FUNCTION = "draw_boxes"
    CATEGORY = "Impact/Visualization"

    @staticmethod
    def _get_segments(segs):
        """Accept standard Impact Pack SEGS: (image_shape, [SEG, ...])."""
        if segs is None:
            return []

        if isinstance(segs, dict):
            segments = segs.get("segments", segs.get("segs", []))
        elif isinstance(segs, tuple) and len(segs) == 2:
            segments = segs[1]
        else:
            segments = segs

        if segments is None:
            return []

        try:
            return list(segments)
        except TypeError:
            return []

    @staticmethod
    def _get_value(segment, key, default=None):
        if isinstance(segment, dict):
            return segment.get(key, default)
        return getattr(segment, key, default)

    @staticmethod
    def _to_float(value):
        """Safely convert scalar Tensor, NumPy, or Python values to float."""
        if value is None:
            return None
        try:
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    return None
                value = value.detach().cpu().reshape(-1)[0].item()
            else:
                value = np.asarray(value)
                if value.size == 0:
                    return None
                value = value.reshape(-1)[0].item()

            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError, IndexError, RuntimeError):
            return None

    @staticmethod
    def _get_bbox(segment):
        bbox = SEGSBBoxVisualizer._get_value(segment, "bbox")
        if bbox is None:
            return None
        try:
            if len(bbox) != 4:
                return None
            coords = [SEGSBBoxVisualizer._to_float(v) for v in bbox]
            if any(v is None for v in coords):
                return None
            return coords
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _load_font():
        try:
            return ImageFont.truetype("arial.ttf", 16)
        except OSError:
            return ImageFont.load_default()

    def draw_boxes(
        self, image, segs, box_thickness, show_label, show_confidence
    ):
        if not isinstance(image, torch.Tensor):
            raise TypeError("The image input must be a PyTorch Tensor.")
        if image.ndim != 4:
            raise ValueError("Expected IMAGE tensor with shape [B, H, W, C].")
        if image.shape[0] == 0:
            raise ValueError("The input image batch is empty.")

        segments = self._get_segments(segs)
        batch = image.detach().cpu().float().numpy()

        output_images = []
        object_descriptions = []
        font = self._load_font()

        for batch_index, img_array in enumerate(batch, start=1):
            img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array).convert("RGB")
            draw = ImageDraw.Draw(pil_image)
            width, height = pil_image.size
            object_number = 0

            for segment in segments:
                bbox = self._get_bbox(segment)
                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox

                # Support normalized coordinates in the [0, 1] range.
                if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                    x1 *= width
                    x2 *= width
                    y1 *= height
                    y2 *= height

                x1 = max(0, min(int(round(x1)), width - 1))
                y1 = max(0, min(int(round(y1)), height - 1))
                x2 = max(0, min(int(round(x2)), width - 1))
                y2 = max(0, min(int(round(y2)), height - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                object_number += 1
                label = str(self._get_value(segment, "label", "object"))
                confidence = self._to_float(
                    self._get_value(segment, "confidence")
                )

                color = self.COLORS[(object_number - 1) % len(self.COLORS)]
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=color,
                    width=box_thickness,
                )

                annotation = f"#{object_number}"
                if show_label:
                    annotation += f" {label}"
                if show_confidence and confidence is not None:
                    annotation += f" {confidence:.2f}"

                text_box = draw.textbbox((0, 0), annotation, font=font)
                text_width = text_box[2] - text_box[0]
                text_height = text_box[3] - text_box[1]

                text_left = x1
                text_top = max(0, y1 - text_height - 8)
                text_right = min(width - 1, text_left + text_width + 8)
                text_bottom = min(height - 1, text_top + text_height + 6)

                draw.rectangle(
                    [text_left, text_top, text_right, text_bottom],
                    fill=color,
                )
                draw.text(
                    (text_left + 4, text_top + 2),
                    annotation,
                    fill=(0, 0, 0),
                    font=font,
                )

                report_line = (
                    f"Image #{batch_index} | Object #{object_number}: {label}"
                )
                if confidence is not None:
                    report_line += f" | Confidence: {confidence:.4f}"
                report_line += (
                    f" | BBox: ({x1}, {y1}, {x2}, {y2})"
                )
                object_descriptions.append(report_line)

            output_images.append(
                np.asarray(pil_image, dtype=np.float32) / 255.0
            )

        output_tensor = torch.from_numpy(np.stack(output_images, axis=0))
        report = (
            "\n".join(object_descriptions)
            if object_descriptions
            else "No objects detected."
        )
        return (output_tensor, report)


NODE_CLASS_MAPPINGS = {
    "SEGSBBoxVisualizer": SEGSBBoxVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SEGSBBoxVisualizer": "SEGS BBox Visualizer"
}
