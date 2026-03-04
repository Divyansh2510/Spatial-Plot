import cv2
import numpy as np
import json


def preprocess_image(image):
    """
    Detecting white / light colored plot regions using HSV mask
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define white/light color range
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological closing to clean small gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def detect_plots(binary_img, original_img):
    """
    Detecting enclosed plot regions and filter unwanted contours
    """
    contours, hierarchy = cv2.findContours(
        binary_img,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    plots = []
    h, w = binary_img.shape
    plot_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Remove very small regions (noise)
        if area < 2000:
            continue

        # Remove extremely large regions (background / roads)
        if area > 0.8 * (h * w):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Removing contours touching image boundary
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue

        # Approximate polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        # Compute centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        polygon_px = []
        polygon_norm = []

        for point in approx:
            px = float(point[0][0])
            py = float(point[0][1])

            polygon_px.append([px, py])
            polygon_norm.append([px / w, py / h])

        plot_info = {
            "id_auto": plot_id,
            "polygon_px": polygon_px,
            "polygon_norm": polygon_norm,
            "centroid_px": [cx, cy],
            "contour_area_px": float(area),
            "bbox_px": [x, y, bw, bh],
            "plot_number_info": None
        }

        plots.append(plot_info)
        plot_id += 1

        # Drawing green contour
        cv2.drawContours(original_img, [approx], -1, (0, 255, 0), 3)

    return original_img, plots


def generate_json(image_path, image, plots, output_path):
    h, w = image.shape[:2]

    data = {
        "image_path": image_path,
        "image_size": [w, h],
        "plots": plots
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    input_image_path = "C:/Users/hp/Desktop/Python/Spatial-Plot/Layout Plan Image 2.png"
    output_image_path = "output_visualized.jpg"
    output_json_path = "output_data.json"

    image = cv2.imread(input_image_path)

    if image is None:
        print("Error loading image.")
        return

    binary = preprocess_image(image)
    output_image, plots = detect_plots(binary, image.copy())

    cv2.imwrite(output_image_path, output_image)
    generate_json(input_image_path, image, plots, output_json_path)

    print("Processing complete!")
    print(f"Detected {len(plots)} plots.")
    print("Outputs saved.")

if __name__ == "__main__":
    main()
