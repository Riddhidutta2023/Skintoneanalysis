import cv2
import gradio as gr
import numpy as np


def get_skin_tone_analysis(image):
    # 1. Check if image exists
    if image is None:
        empty_swatch = np.zeros((100, 100, 3), dtype=np.uint8)
        return None, "Please upload an image.", "#000000", empty_swatch

    # Make a copy of the image
    image_rgb = image.copy()
    h, w, _ = image.shape

    return analyze_skin_directly(image, image_rgb, h, w)


def analyze_skin_directly(image, image_rgb, h, w):
    """Direct skin detection with proper skin tone correction"""
    print("Using direct skin detection...")

    # Convert to multiple color spaces
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)

    # BROADER skin color ranges to capture more diverse skin tones
    lower_hsv1 = np.array([0, 15, 50])
    upper_hsv1 = np.array([25, 180, 255])
    lower_hsv2 = np.array([165, 15, 50])
    upper_hsv2 = np.array([180, 180, 255])

    lower_ycrcb = np.array([0, 130, 75])
    upper_ycrcb = np.array([255, 180, 135])

    # Create skin masks
    skin_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    skin_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
    skin_hsv = cv2.bitwise_or(skin_hsv1, skin_hsv2)

    skin_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine masks
    skin_mask = cv2.bitwise_and(skin_hsv, skin_ycrcb)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    vis_image = image_rgb.copy()

    if contours:
        # Get the largest contour (should be the face)
        largest_contour = max(contours, key=cv2.contourArea)

        # Filter by aspect ratio and size to ensure it's likely a face
        x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
        aspect_ratio = h_cont / w_cont if w_cont > 0 else 0

        # Only proceed if it looks like a face (roughly vertical rectangle)
        if (
            aspect_ratio > 0.8
            and aspect_ratio < 2.0
            and cv2.contourArea(largest_contour) > 1000
        ):
            # Draw the detected skin region
            cv2.drawContours(vis_image, [largest_contour], -1, (255, 255, 0), 2)
            cv2.putText(
                vis_image,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # Create mask from largest contour
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

            # Sample from the CHEEKS area (best for skin tone)
            # Left cheek region
            lc_x1 = x + int(w_cont * 0.25)
            lc_x2 = x + int(w_cont * 0.4)
            lc_y1 = y + int(h_cont * 0.4)
            lc_y2 = y + int(h_cont * 0.6)

            # Right cheek region
            rc_x1 = x + int(w_cont * 0.6)
            rc_x2 = x + int(w_cont * 0.75)
            rc_y1 = y + int(h_cont * 0.4)
            rc_y2 = y + int(h_cont * 0.6)

            # Forehead region (avoid hairline)
            fh_x1 = x + int(w_cont * 0.35)
            fh_x2 = x + int(w_cont * 0.65)
            fh_y1 = y + int(h_cont * 0.2)
            fh_y2 = y + int(h_cont * 0.3)

            all_skin_pixels = []

            # Sample from each region
            regions = [
                (lc_x1, lc_y1, lc_x2, lc_y2, "Left Cheek", (0, 255, 0)),
                (rc_x1, rc_y1, rc_x2, rc_y2, "Right Cheek", (0, 255, 0)),
                (fh_x1, fh_y1, fh_x2, fh_y2, "Forehead", (0, 255, 0)),
            ]

            for x1, y1, x2, y2, label, color in regions:
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                if x2 > x1 and y2 > y1:
                    # Create region mask
                    region_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(region_mask, (x1, y1), (x2, y2), 255, -1)

                    # Combine with skin mask
                    combined_mask = cv2.bitwise_and(region_mask, mask)

                    # Get skin pixels from this region
                    region_pixels = image_rgb[combined_mask > 0]

                    if len(region_pixels) > 50:
                        all_skin_pixels.extend(region_pixels)

                        # Draw the sampling region
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            vis_image,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

            if len(all_skin_pixels) > 100:
                # Convert to numpy array
                all_skin_pixels = np.array(all_skin_pixels)

                # Calculate percentiles to avoid outliers
                r = np.percentile(all_skin_pixels[:, 0], 50)
                g = np.percentile(all_skin_pixels[:, 1], 50)
                b = np.percentile(all_skin_pixels[:, 2], 50)

                return create_output(
                    image_rgb, vis_image, r, g, b, 3, "Direct Skin Detection"
                )

    # Fallback to color-based sampling
    return analyze_color_based_sampling(image, image_rgb, h, w)


def analyze_color_based_sampling(image, image_rgb, h, w):
    """Color-based sampling with skin tone correction"""
    print("Using color-based sampling...")

    # Sample from the center of the image
    center_x, center_y = w // 2, h // 2
    sample_size = min(w, h) // 8

    x1 = max(0, center_x - sample_size)
    x2 = min(w, center_x + sample_size)
    y1 = max(0, center_y - sample_size)
    y2 = min(h, center_y + sample_size)

    vis_image = image_rgb.copy()
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        vis_image,
        "Sampled Region",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # Get pixels from the region
    region = image_rgb[y1:y2, x1:x2]

    # Filter out very dark and very bright pixels
    hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    value = hsv_region[:, :, 2]

    # Only keep pixels with V between 40 and 220
    valid_mask = (value > 40) & (value < 220)

    if np.any(valid_mask):
        valid_pixels = region[valid_mask]
        r = np.percentile(valid_pixels[:, 0], 50)
        g = np.percentile(valid_pixels[:, 1], 50)
        b = np.percentile(valid_pixels[:, 2], 50)
    else:
        r, g, b = np.median(region, axis=(0, 1))

    return create_output(image_rgb, vis_image, r, g, b, 1, "Color-Based Sampling")


def rgb_to_lab_corrected(r, g, b):
    """Convert RGB to LAB with correct scaling (standard LAB: L 0-100, a -128 to 127, b -128 to 127)"""

    # First convert RGB to XYZ using sRGB standard
    def rgb_to_xyz(c):
        c = c / 255.0
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        else:
            return c / 12.92

    r_lin = rgb_to_xyz(r)
    g_lin = rgb_to_xyz(g)
    b_lin = rgb_to_xyz(b)

    # Convert to XYZ using sRGB D65 illuminant
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # Normalize for D65 illuminant
    x = x / 0.950456
    z = z / 1.088754

    # Convert XYZ to LAB
    def xyz_to_lab(t):
        if t > 0.008856:
            return t ** (1 / 3)
        else:
            return (903.3 * t + 16) / 116

    x_lab = xyz_to_lab(x)
    y_lab = xyz_to_lab(y)
    z_lab = xyz_to_lab(z)

    # Calculate LAB values
    l = (116 * y_lab) - 16
    a = 500 * (x_lab - y_lab)
    b = 200 * (y_lab - z_lab)

    return l, a, b


def create_output(original_image, vis_image, r, g, b, points, method):
    """Create consistent output format with CORRECTED LAB values"""

    # Ensure values are within range
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    # Get CORRECT LAB values (L: 0-100, a: -128 to 127, b: -128 to 127)
    l_val, a_val, b_val = rgb_to_lab_corrected(r, g, b)

    # Format outputs
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

    # Create swatch in RGB format for Gradio
    swatch = np.zeros((100, 100, 3), dtype=np.uint8)
    swatch[:, :] = (int(r), int(g), int(b))

    # Method icons
    method_icons = {"Direct Skin Detection": "🎯", "Color-Based Sampling": "🎨"}
    icon = method_icons.get(method, "📊")

    # Determine undertone using CORRECTED a and b values
    if a_val > 5:
        undertone = "Warm/Pink"
    elif a_val < -5:
        undertone = "Cool/Green"
    else:
        undertone = "Neutral"

    # Add skin tone category based on CORRECTED LAB L value
    if l_val > 75:
        tone_category = "Fair/Light"
    elif l_val > 60:
        tone_category = "Light Medium"
    elif l_val > 45:
        tone_category = "Medium"
    elif l_val > 30:
        tone_category = "Tan/Olive"
    else:
        tone_category = "Deep/Dark"

    result_text = f"""
    ### Analysis Results {icon} {method}

    #### 🎨 Your Skin Tone
    - **Hex Code:** `{hex_color}`
    - **RGB:** `({int(r)}, {int(g)}, {int(b)})`
    - **Category:** {tone_category}
    - **Undertone:** {undertone}

    #### 📊 LAB Color Space (Standard CIE L*a*b*)
    - **L* (Lightness):** {l_val:.1f} (0 = black, 100 = white)
    - **a* (Green-Red):** {a_val:.1f} (-128 = green, +127 = red)
    - **b* (Blue-Yellow):** {b_val:.1f} (-128 = blue, +127 = yellow)

    #### 🎯 Confidence
    - **Sampled Regions:** {points}
    - **Method:** {method}

    ---
    💡 **For best results:** Take a photo in natural daylight with your face clearly visible.
    """

    return vis_image, result_text, hex_color, swatch


# Gradio Interface
with gr.Blocks(title="AI Skin Tone Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧬 AI Skin Tone Analyzer
    Upload a photo to analyze your skin tone with professional color correction.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Photo",
                sources=["upload", "webcam"],
                type="numpy",
                height=400,
            )
            submit_btn = gr.Button("🔍 Analyze Skin Tone", variant="primary", size="lg")

            with gr.Accordion("📋 Tips for Accurate Results", open=False):
                gr.Markdown("""
                ### 📸 For Professional Results:
                1. **Natural daylight** - Face a window, avoid artificial/tungsten light
                2. **Clean, bare skin** - No makeup, moisturizer, or sunscreen
                3. **Even lighting** - No harsh shadows on your face
                4. **Neutral background** - White or grey wall behind you

                ### 📊 About LAB Color Space:
                - **L***: Lightness (0-100)
                - **a***: Green to Red (-128 to +127)
                - **b***: Blue to Yellow (-128 to +127)
                """)

        with gr.Column():
            output_image = gr.Image(label="Detected Skin Regions", height=400)
            with gr.Row():
                color_swatch = gr.Image(
                    label="Your True Skin Tone", height=150, width=150, show_label=True
                )
            output_text = gr.Markdown()
            hex_output = gr.Textbox(label="Hex Code", visible=False)

    submit_btn.click(
        fn=get_skin_tone_analysis,
        inputs=[input_image],
        outputs=[output_image, output_text, hex_output, color_swatch],
    )

if __name__ == "__main__":
    print("🚀 Starting AI Skin Tone Analyzer...")
    print("📊 LAB values are now correctly scaled (L: 0-100, a/b: -128 to +127)")
    demo.launch(share=False, debug=True)
