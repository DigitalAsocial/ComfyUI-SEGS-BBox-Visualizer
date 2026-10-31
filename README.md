# ComfyUI-SEGS-BBox-Visualizer

A lightweight custom node for ComfyUI that visualizes bounding boxes from an Impact Pack `SEGS` input.

## Features

- Draws bounding boxes over the input image.
- Optionally displays each object's label and confidence.
- Supports pixel coordinates and normalized bounding-box coordinates.
- Produces a text report containing object labels, confidence scores (when available), and bounding-box coordinates.
- Supports batched `IMAGE` inputs.

## Node

**Category:** `Impact/Visualization`  
**Display name:** `SEGS BBox Visualizer`

### Inputs

| Input | Type | Default | Description |
|---|---|---:|---|
| `image` | IMAGE | — | Image on which the boxes are drawn |
| `segs` | SEGS | — | Detection/segmentation data in SEGS format |
| `box_thickness` | INT | 3 | Bounding-box line thickness (1–15) |
| `show_label` | BOOLEAN | true | Show the incoming SEGS label |
| `show_confidence` | BOOLEAN | true | Show confidence when supplied by the incoming SEGS data |

### Outputs

| Output | Type | Description |
|---|---|---|
| `annotated_image` | IMAGE | Input image with bounding boxes and optional annotations |
| `object_list` | STRING | Text report with object labels, confidence values, and bounding-box coordinates |

## Installation

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
2. Install [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack), which provides the `SEGS` data used by this node.
3. Clone this repository into the `custom_nodes` directory:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/YOUR_GITHUB_USERNAME/ComfyUI-SEGS-BBox-Visualizer.git
   ```

4. Restart ComfyUI.

No additional Python packages are required beyond the dependencies normally available in ComfyUI.

## Important note about labels

This node is a **visualizer**, not an object-recognition model. It displays the labels and confidence values supplied in the incoming `SEGS` data; it does not verify, correct, or reinterpret them. If the upstream detector labels a dragon as a horse, this node will display the supplied label.

## License

MIT. See [LICENSE](LICENSE).
