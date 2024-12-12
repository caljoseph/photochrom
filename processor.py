import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt


class PhotochromImageProcessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def visualize_step(self, image, title, save_path=None):
        """Helper function to visualize intermediate steps"""
        plt.figure(figsize=(10, 10))
        if len(image.shape) == 2:  # Grayscale
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def detect_photo_boundaries(self, image, debug_path=None):
        """Detect actual photograph boundaries using thresholding approach"""
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Save original for debugging
        if debug_path:
            self.visualize_step(gray, "Original Grayscale", f"{debug_path}_gray.png")

        # Apply binary thresholding
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        if debug_path:
            self.visualize_step(binary, "Binary Threshold", f"{debug_path}_binary.png")

        # Find contours on the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found - using safe margins")
            height, width = gray.shape
            margin = min(width, height) // 20  # 5% margin
            return (margin, margin, width - 2 * margin, height - 2 * margin)

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add small margin
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2 * margin)
        h = min(gray.shape[0] - y, h + 2 * margin)

        if debug_path:
            debug_img = gray.copy()
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.visualize_step(debug_img, "Detected Boundaries", f"{debug_path}_boundaries.png")

        # Sanity check - if the detected area is too small, use safe margins
        min_area_ratio = 0.4  # detected area should be at least 40% of image
        image_area = gray.shape[0] * gray.shape[1]
        detected_area = w * h

        if detected_area / image_area < min_area_ratio:
            print("Detected area too small - using safe margins")
            height, width = gray.shape
            margin = min(width, height) // 20
            return (margin, margin, width - 2 * margin, height - 2 * margin)

        return (x, y, w, h)

    def align_images(self, bw_img, color_img, debug_path=None):
        """Align images using feature matching"""
        # Convert color image to grayscale for feature matching
        if len(color_img.shape) == 3:
            color_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        else:
            color_gray = color_img

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(bw_img, None)
        kp2, des2 = sift.detectAndCompute(color_gray, None)

        if des1 is None or des2 is None:
            print("No features detected in one or both images")
            return None

        if debug_path:
            # Draw keypoints
            bw_kp = cv2.drawKeypoints(bw_img, kp1, None)
            color_kp = cv2.drawKeypoints(color_gray, kp2, None)
            self.visualize_step(bw_kp, "BW Keypoints", f"{debug_path}_bw_keypoints.png")
            self.visualize_step(color_kp, "Color Keypoints", f"{debug_path}_color_keypoints.png")

        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            print(f"Error during feature matching: {str(e)}")
            return None

        # Store good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            print(f"Not enough good matches found: {len(good_matches)}")
            return None

        if debug_path:
            # Draw matches
            match_img = cv2.drawMatches(bw_img, kp1, color_gray, kp2, good_matches, None)
            self.visualize_step(match_img, "Matches", f"{debug_path}_matches.png")

        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("Could not find homography matrix")
            return None

        # Warp BW image to align with color image
        aligned_bw = cv2.warpPerspective(bw_img, H, (color_img.shape[1], color_img.shape[0]))

        if debug_path:
            self.visualize_step(aligned_bw, "Aligned Result", f"{debug_path}_aligned.png")

        return aligned_bw

    def align_and_crop_images(self, bw_img, color_img, debug_path=None):
        """Align and crop images with robust fallbacks"""
        try:
            # First try SIFT alignment
            aligned_bw = self.align_images(bw_img, color_img, debug_path)

            if aligned_bw is None:
                print("SIFT alignment failed - using original image")
                aligned_bw = bw_img

            # Detect boundaries for both images
            bw_rect = self.detect_photo_boundaries(aligned_bw, str(debug_path) + "_bw")
            color_rect = self.detect_photo_boundaries(color_img, str(debug_path) + "_color")

            # Crop images
            bw_cropped = aligned_bw[bw_rect[1]:bw_rect[1] + bw_rect[3],
                         bw_rect[0]:bw_rect[0] + bw_rect[2]]
            color_cropped = color_img[color_rect[1]:color_rect[1] + color_rect[3],
                            color_rect[0]:color_rect[0] + color_rect[2]]

            # Ensure both images are the same size
            target_size = self.target_size
            bw_final = cv2.resize(bw_cropped, target_size)
            color_final = cv2.resize(color_cropped, target_size)

            if debug_path:
                self.visualize_step(bw_final, "Final BW", f"{debug_path}_final_bw.png")
                self.visualize_step(color_final, "Final Color", f"{debug_path}_final_color.png")

            return bw_final, color_final

        except Exception as e:
            print(f"Error in alignment and cropping: {str(e)}")
            return None, None

    def process_image_pair(self, bw_path, color_path, output_dir):
        """Process a pair of images: align, remove borders, and resize"""
        print(f"\nProcessing pair: {bw_path}")

        # Create debug directory
        debug_dir = Path(output_dir) / "debug" / Path(bw_path).stem
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Read images
        bw_img = cv2.imread(str(bw_path), cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(str(color_path))

        if bw_img is None or color_img is None:
            print(f"Failed to load images: {bw_path}, {color_path}")
            return False

        # Save original images
        self.visualize_step(bw_img, "Original BW", str(debug_dir / "original_bw.png"))
        self.visualize_step(color_img, "Original Color", str(debug_dir / "original_color.png"))

        # Process the images
        bw_final, color_final = self.align_and_crop_images(bw_img, color_img, str(debug_dir))

        if bw_final is None or color_final is None:
            print("Failed to process images")
            return False

        # Save processed images
        base_name = Path(bw_path).stem.replace('_bw', '')
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_bw_processed.jpg"), bw_final)
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_color_processed.jpg"), color_final)

        return True

    def process_directory(self, input_dir, output_dir):
        """Process all image pairs in a directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all BW images and their color counterparts
        bw_images = list(input_path.glob("*_bw.jpg"))

        success_count = 0
        for bw_path in bw_images:
            color_path = bw_path.parent / f"{bw_path.stem.replace('_bw', '')}_color.jpg"
            if color_path.exists():
                if self.process_image_pair(bw_path, color_path, output_path):
                    success_count += 1
                    print(f"Successfully processed pair {success_count}")
            else:
                print(f"No matching color image for {bw_path}")

        print(f"Processing complete. Successfully processed {success_count} pairs.")


if __name__ == "__main__":
    processor = PhotochromImageProcessor(target_size=(512, 512))
    processor.process_directory(
        input_dir="scraped_images/real_pairs",
        output_dir="processed_images/real_pairs"
    )