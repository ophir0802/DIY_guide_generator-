"""
Part Locator Module

Uses Google Gemini 1.5 Flash API to detect parts/components in images and return bounding boxes
in normalized 0-1000 coordinate system.
"""
import os
import json
import logging
import re
from typing import List, Optional, Dict, Any
from io import BytesIO

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator, ValidationError
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions ---

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain additional content.
    
    This is useful when models don't support JSON mode and return JSON
    embedded in markdown or with explanatory text.
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Parsed JSON dict if found, None otherwise
    """
    if not text or not text.strip():
        return None
    
    # Try parsing as-is first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text (looking for {...})
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array in text (looking for [...])
    json_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


# --- Pydantic Models ---

class PartLocation(BaseModel):
    """
    Represents a detected part/component with its bounding box coordinates.
    
    Attributes:
        part_name: Name of the detected part/component
        bbox_2d: Bounding box coordinates in format [ymin, xmin, ymax, xmax]
                 All coordinates are normalized to 0-1000 range
    """
    part_name: str = Field(..., description="Name of the detected part/component")
    bbox_2d: List[int] = Field(
        ..., 
        min_length=4, 
        max_length=4,
        description="Bounding box coordinates [ymin, xmin, ymax, xmax] in 0-1000 range"
    )
    
    @field_validator('bbox_2d')
    @classmethod
    def validate_bbox(cls, v: List[int]) -> List[int]:
        """
        Validate bounding box coordinates.
        
        Ensures:
        - All values are in 0-1000 range
        - ymin < ymax (vertical ordering)
        - xmin < xmax (horizontal ordering)
        """
        if len(v) != 4:
            raise ValueError("bbox_2d must contain exactly 4 values")
        
        ymin, xmin, ymax, xmax = v
        
        # Check coordinate range (0-1000)
        if not all(0 <= coord <= 1000 for coord in v):
            raise ValueError("All bbox coordinates must be in 0-1000 range")
        
        # Check ordering: ymin must be less than ymax (top < bottom)
        if ymin >= ymax:
            raise ValueError(f"ymin ({ymin}) must be less than ymax ({ymax})")
        
        # Check ordering: xmin must be less than xmax (left < right)
        if xmin >= xmax:
            raise ValueError(f"xmin ({xmin}) must be less than xmax ({xmax})")
        
        return v


class DetectionResult(BaseModel):
    """
    Wrapper model for part detection results from Gemini API.
    
    Attributes:
        parts: List of detected parts with their bounding boxes
    """
    parts: List[PartLocation] = Field(
        default_factory=list,
        description="List of detected parts. Empty list if no parts found."
    )


# --- PartLocator Class ---

class PartLocator:
    """
    Part detection class using Google Gemini 1.5 Flash API.
    
    Analyzes images to locate parts/components and returns bounding boxes in normalized
    coordinate system (0-1000 range).
    """
    
    def __init__(self):
        """
        Initialize the PartLocator with Gemini API configuration.
        
        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please set it using: export GOOGLE_API_KEY='your-api-key'"
            )
        
        # Initialize the Gemini client
        self.client = genai.Client(api_key=api_key)
        
        # Model name - using gemma-3-12b-it (higher quota limits)
        # Note: Use full model path with "models/" prefix
        self.model_name = "models/gemma-3-12b-it"
        
        # Check if model supports JSON mode
        # Gemini models support JSON mode, but Gemma models don't
        self.supports_json_mode = "gemini" in self.model_name.lower()
        
        logging.info("PartLocator initialized successfully")
    
    def _clean_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively remove unsupported fields from JSON schema for Gemini API.
        
        Gemini API doesn't support fields like $defs, $schema, title that Pydantic generates.
        This function recursively removes them from the entire schema tree.
        
        Args:
            schema: JSON schema dictionary from Pydantic
            
        Returns:
            Cleaned schema dictionary without unsupported fields
        """
        if not isinstance(schema, dict):
            return schema
        
        # Create a copy to avoid modifying the original
        cleaned = {}
        
        # Fields to remove at any level
        unsupported_fields = {'$defs', '$schema', 'title', 'description', 'examples', 'default'}
        
        for key, value in schema.items():
            # Skip unsupported fields
            if key in unsupported_fields:
                continue
            
            # Recursively clean nested dictionaries
            if isinstance(value, dict):
                cleaned[key] = self._clean_json_schema(value)
            # Recursively clean lists of dictionaries
            elif isinstance(value, list):
                cleaned[key] = [
                    self._clean_json_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned[key] = value
        
        return cleaned
    
    def _download_image(self, image_url: str) -> Optional[Image.Image]:
        """
        Download an image from a URL and convert it to a PIL Image object.
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            PIL Image object if successful, None otherwise
            
        Note:
            Uses requests library following the same patterns as crawler.py
        """
        try:
            logging.info(f"Downloading image from: {image_url}")
            
            # Download image with timeout
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary (some images might be RGBA or other formats)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logging.info(f"Successfully downloaded image: {image.size[0]}x{image.size[1]} pixels")
            return image
            
        except requests.RequestException as e:
            logging.error(f"Error downloading image from {image_url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Error processing image from {image_url}: {e}")
            return None
    
    def locate_parts(
        self, 
        image_url: str, 
        part_list: List[str]
    ) -> List[PartLocation]:
        """
        Locate parts/components in an image using Gemini API.
        
        Args:
            image_url: URL of the image to analyze
            part_list: List of part/component names to search for (e.g., ["condenser coil", "freezer fan"])
            
        Returns:
            List of PartLocation objects containing detected parts and their bounding boxes.
            Returns empty list if no parts found or if an error occurs.
            
        Coordinate System:
            Bounding boxes are returned in [ymin, xmin, ymax, xmax] format with normalized
            coordinates in 0-1000 range:
            - ymin, ymax: Vertical coordinates (0 = top, 1000 = bottom)
            - xmin, xmax: Horizontal coordinates (0 = left, 1000 = right)
            - ymin < ymax and xmin < xmax are guaranteed by validation
        """
        if not part_list:
            logging.warning("Empty part_list provided, returning empty results")
            return []
        
        # Download image
        image = self._download_image(image_url)
        if not image:
            logging.error(f"Failed to download image from {image_url}")
            return []
        
        try:
            # Prepare the prompt
            # The prompt instructs Gemini to:
            # 1. Return bounding boxes in [ymin, xmin, ymax, xmax] format
            # 2. Use normalized coordinates 0-1000 range
            # 3. Return null or empty list if part not visible
            # 4. Be strictly factual (no hallucinations)
            part_list_str = ", ".join(part_list)
            prompt = f"""Analyze this image and locate the following parts/components: {part_list_str}

For each part/component that is VISIBLE in the image, provide:
- The exact part name (must match one from the list: {part_list_str})
- A bounding box in format [ymin, xmin, ymax, xmax] with normalized coordinates in 0-1000 range

IMPORTANT RULES:
1. Only return parts/components that are CLEARLY VISIBLE in the image
2. If a part is not visible, do NOT include it in the results
3. Use normalized coordinates where:
   - 0,0 is the top-left corner
   - 1000,1000 is the bottom-right corner
   - ymin < ymax (vertical: top < bottom)
   - xmin < xmax (horizontal: left < right)
4. Be strictly factual - only report parts you can actually see
5. If multiple instances of the same part exist, return all of them
6. Return an empty list if no parts from the list are visible
7. Focus on mechanical/electrical components, not tools being used

Coordinate system explanation:
- ymin: Top edge of bounding box (0-1000, where 0 is top of image)
- xmin: Left edge of bounding box (0-1000, where 0 is left of image)
- ymax: Bottom edge of bounding box (0-1000, must be > ymin)
- xmax: Right edge of bounding box (0-1000, must be > xmin)

Output must be in JSON format:
{{
  "parts": [
    {{
      "part_name": "part name from list",
      "bbox_2d": [ymin, xmin, ymax, xmax]
    }}
  ]
}}

If no parts are found, return: {{"parts": []}}
"""
            
            # Generate content with structured output
            logging.info(f"Analyzing image for parts: {part_list}")
            
            # Convert PIL Image to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Configure JSON mode only if model supports it
            config = types.GenerateContentConfig()
            if self.supports_json_mode:
                config.response_mime_type = "application/json"
                config.response_schema = DetectionResult  # Pass the Pydantic model class directly
            
            # Use the new API - pass Pydantic model directly instead of JSON schema
            # The new SDK can handle Pydantic models natively
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=img_bytes, mime_type='image/png')
                ],
                config=config
            )
            
            # Parse the JSON response (handles both JSON mode and plain text with JSON)
            result_dict = extract_json_from_text(response.text)
            if result_dict is None:
                logging.warning(f"No valid JSON found in response, returning empty results")
                return []
            
            # Validate using Pydantic
            detection_result = DetectionResult(**result_dict)
            
            # Extract and return part locations
            parts = detection_result.parts
            logging.info(f"Detected {len(parts)} part(s) in image")
            
            for part in parts:
                logging.info(
                    f"  - {part.part_name}: bbox={part.bbox_2d} "
                    f"(ymin={part.bbox_2d[0]}, xmin={part.bbox_2d[1]}, "
                    f"ymax={part.bbox_2d[2]}, xmax={part.bbox_2d[3]})"
                )
            
            return parts
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response from Gemini API: {e}")
            logging.debug(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
            return []
        except ValidationError as e:
            logging.error(f"Validation error when parsing Gemini response: {e}")
            return []
        except Exception as e:
            logging.error(f"Error during part detection: {e}", exc_info=True)
            return []
