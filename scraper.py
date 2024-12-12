import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin
import re
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from pathlib import Path


class PhotochromScraper:
    def __init__(self, base_url="https://www.loc.gov/pictures/search/", output_dir="scraped_images"):
        self.base_url = base_url
        self.session = requests.Session()
        self.output_dir = output_dir

        self.real_pairs_dir = Path(output_dir) / "real_pairs"
        self.synthetic_pairs_dir = Path(output_dir) / "synthetic_pairs"

        for dir_path in [self.real_pairs_dir, self.synthetic_pairs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_synthetic_bw(self, color_img):
        if isinstance(color_img, np.ndarray):
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.array(color_img.convert('L'))

        # Random parameters to destroy the bw version a bit
        contrast = np.random.uniform(0.6, 2.0)
        brightness = np.random.uniform(-15, 3)
        grain = np.random.uniform(0, 0.25)
        blur = np.random.uniform(0, .25)

        # Apply effects
        gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
        noise = np.random.normal(0, grain, gray.shape).astype(np.uint8)
        gray = cv2.add(gray, noise)
        gray = cv2.GaussianBlur(gray, (3, 3), blur)

        return gray

    def get_soup(self, url):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url)
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch {url}: {str(e)}")
                    return None
                time.sleep(2)
        return None

    def get_highest_quality_jpeg(self, links):
        max_size = 0
        best_link = None

        for link in links:
            size_match = re.search(r'JPEG \(([\d.]+)(mb|kb)\)', link.text, re.IGNORECASE)
            if size_match and 'href' in link.attrs:
                size = float(size_match.group(1))
                unit = size_match.group(2).lower()
                if unit == 'mb':
                    size *= 1024

                if size > max_size:
                    max_size = size
                    best_link = link['href']

        return best_link

    def process_item_page(self, url):
        soup = self.get_soup(url)
        if not soup:
            return None

        images = {}
        for preview_div in soup.find_all('p', class_='image_preview'):
            desc_span = preview_div.find_next_sibling('span', class_='quiet')
            if not desc_span:
                continue

            desc = desc_span.text.strip().lower()
            links_p = preview_div.find_next('p')
            if not links_p:
                continue

            download_links = links_p.find_all('a', href=re.compile(r'\.jpg$'))
            best_jpeg = self.get_highest_quality_jpeg(download_links)

            if best_jpeg:
                if 'digital file from' in desc and 'item' not in desc:
                    images['color'] = best_jpeg
                elif 'b&w film copy neg' in desc:
                    images['bw'] = best_jpeg

        return images if 'color' in images else None

    def download_and_save_images(self, image_urls, base_name):
        try:
            response = self.session.get(image_urls['color'])
            response.raise_for_status()
            color_img = Image.open(BytesIO(response.content))

            if 'bw' in image_urls:
                color_path = self.real_pairs_dir / f"{base_name}_color.jpg"
                bw_path = self.real_pairs_dir / f"{base_name}_bw.jpg"

                color_img.save(color_path)

                response = self.session.get(image_urls['bw'])
                Image.open(BytesIO(response.content)).save(bw_path)
                print(f"Saved real pair: {base_name}")
            else:
                synthetic_color_path = self.synthetic_pairs_dir / f"{base_name}_color.jpg"
                synthetic_bw_path = self.synthetic_pairs_dir / f"{base_name}_bw.jpg"

                color_img.save(synthetic_color_path)

                synthetic_bw = self.create_synthetic_bw(color_img)
                Image.fromarray(synthetic_bw).save(synthetic_bw_path)

            return True

        except Exception as e:
            print(f"Failed to process {base_name}: {str(e)}")
            return False

    def scrape_page(self, page_num):
        url = f"{self.base_url}?sp={page_num}&co=pgz&st=grid"
        soup = self.get_soup(url)
        if not soup:
            return False

        item_links = soup.find_all('a', href=re.compile(r'/pictures/collection/pgz/item/'))

        for idx, link in enumerate(item_links):
            item_url = urljoin(self.base_url, link['href'])
            print(f"Processing item {idx + 1}/{len(item_links)} on page {page_num}")

            image_urls = self.process_item_page(item_url)
            if image_urls:
                base_name = re.search(r'item/(\d+)', item_url).group(1)
                self.download_and_save_images(image_urls, base_name)

            time.sleep(1)

        return True

    def scrape_all(self, start_page=1, end_page=76):
        for page_num in range(start_page, end_page + 1):
            print(f"\nProcessing page {page_num}/{end_page}")
            if not self.scrape_page(page_num):
                print(f"Failed to process page {page_num}")
            time.sleep(2)

    def reprocess_synthetic_pairs(self, synthetic_pairs_dir):
        """
        Reprocesses all black and white images in the synthetic pairs directory
        using the existing color images.
        """
        synthetic_dir = Path(synthetic_pairs_dir)

        # Get all color images
        color_images = list(synthetic_dir.glob("*_color.jpg"))

        print(f"Found {len(color_images)} color images to process")

        for color_path in color_images:
            # Get corresponding bw path
            bw_path = color_path.parent / color_path.name.replace("_color.jpg", "_bw.jpg")

            try:
                # Load color image
                color_img = Image.open(color_path)

                # Create new synthetic bw
                synthetic_bw = self.create_synthetic_bw(color_img)

                # Save the new bw image, overwriting the old one
                Image.fromarray(synthetic_bw).save(bw_path)

                print(f"Reprocessed: {bw_path.name}")

            except Exception as e:
                print(f"Failed to process {color_path.name}: {str(e)}")


# if __name__ == "__main__":
#     scraper = PhotochromScraper()
#     scraper.scrape_all(start_page=10, end_page=76)

if __name__ == "__main__":
    scraper = PhotochromScraper()
    base_dir = Path("scraped_images")
    synthetic_pairs_dir = base_dir / "synthetic_pairs"

    print("Starting reprocessing of synthetic black and white images...")
    scraper.reprocess_synthetic_pairs(synthetic_pairs_dir)
    print("Finished reprocessing!")