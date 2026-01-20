"""
Tool Locator Module

Uses Google Gemini 1.5 Flash API to detect tools in images and return bounding boxes
in normalized 0-1000 coordinate system.
"""
import os
import json
import logging
from typing import List, Optional
from io import BytesIO

import requests
import google.generativeai as genai
from pydantic import BaseModel, Field, field_validator, ValidationError
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Pydantic Models ---

class ToolLocation(BaseModel):
    """
    Represents a detected tool with its bounding box coordinates.
    
    Attributes:
        tool_name: Name of the detected tool
        bbox_2d: Bounding box coordinates in format [ymin, xmin, ymax, xmax]
                 All coordinates are normalized to 0-1000 range
    """
    tool_name: str = Field(..., description="Name of the detected tool")
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
    Wrapper model for tool detection results from Gemini API.
    
    Attributes:
        tools: List of detected tools with their bounding boxes
    """
    tools: List[ToolLocation] = Field(
        default_factory=list,
        description="List of detected tools. Empty list if no tools found."
    )


# --- ToolLocator Class ---

class ToolLocator:
    """
    Tool detection class using Google Gemini 1.5 Flash API.
    
    Analyzes images to locate tools and returns bounding boxes in normalized
    coordinate system (0-1000 range).
    """
    
    def __init__(self):
        """
        Initialize the ToolLocator with Gemini API configuration.
        
        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Please set it using: export GOOGLE_API_KEY='your-api-key'"
            )
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize the model
        # Using gemini-1.5-flash as it's free tier and supports multimodal input
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        logging.info("ToolLocator initialized successfully")
    
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
    
    def locate_tools(
        self, 
        image_url: str, 
        tool_list: List[str]
    ) -> List[ToolLocation]:
        """
        Locate tools in an image using Gemini API.
        
        Args:
            image_url: URL of the image to analyze
            tool_list: List of tool names to search for (e.g., ["hammer", "drill"])
            
        Returns:
            List of ToolLocation objects containing detected tools and their bounding boxes.
            Returns empty list if no tools found or if an error occurs.
            
        Coordinate System:
            Bounding boxes are returned in [ymin, xmin, ymax, xmax] format with normalized
            coordinates in 0-1000 range:
            - ymin, ymax: Vertical coordinates (0 = top, 1000 = bottom)
            - xmin, xmax: Horizontal coordinates (0 = left, 1000 = right)
            - ymin < ymax and xmin < xmax are guaranteed by validation
        """
        if not tool_list:
            logging.warning("Empty tool_list provided, returning empty results")
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
            # 3. Return null or empty list if tool not visible
            # 4. Be strictly factual (no hallucinations)
            tool_list_str = ", ".join(tool_list)
            prompt = f"""Analyze this image and locate the following tools: {tool_list_str}

For each tool that is VISIBLE in the image, provide:
- The exact tool name (must match one from the list: {tool_list_str})
- A bounding box in format [ymin, xmin, ymax, xmax] with normalized coordinates in 0-1000 range

IMPORTANT RULES:
1. Only return tools that are CLEARLY VISIBLE in the image
2. If a tool is not visible, do NOT include it in the results
3. Use normalized coordinates where:
   - 0,0 is the top-left corner
   - 1000,1000 is the bottom-right corner
   - ymin < ymax (vertical: top < bottom)
   - xmin < xmax (horizontal: left < right)
4. Be strictly factual - only report tools you can actually see
5. If multiple instances of the same tool exist, return all of them
6. Return an empty list if no tools from the list are visible

Coordinate system explanation:
- ymin: Top edge of bounding box (0-1000, where 0 is top of image)
- xmin: Left edge of bounding box (0-1000, where 0 is left of image)
- ymax: Bottom edge of bounding box (0-1000, must be > ymin)
- xmax: Right edge of bounding box (0-1000, must be > xmin)
"""
            
            # Convert Pydantic model to JSON schema for Gemini structured output
            # Gemini expects a JSON schema format
            response_schema = DetectionResult.model_json_schema()
            
            # Generate content with structured output
            logging.info(f"Analyzing image for tools: {tool_list}")
            
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema
                )
            )
            
            # Parse the JSON response
            result_dict = json.loads(response.text)
            
            # Validate using Pydantic
            detection_result = DetectionResult(**result_dict)
            
            # Extract and return tool locations
            tools = detection_result.tools
            logging.info(f"Detected {len(tools)} tool(s) in image")
            
            for tool in tools:
                logging.info(
                    f"  - {tool.tool_name}: bbox={tool.bbox_2d} "
                    f"(ymin={tool.bbox_2d[0]}, xmin={tool.bbox_2d[1]}, "
                    f"ymax={tool.bbox_2d[2]}, xmax={tool.bbox_2d[3]})"
                )
            
            return tools
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response from Gemini API: {e}")
            logging.debug(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
            return []
        except ValidationError as e:
            logging.error(f"Validation error when parsing Gemini response: {e}")
            return []
        except Exception as e:
            logging.error(f"Error during tool detection: {e}", exc_info=True)
            return []

