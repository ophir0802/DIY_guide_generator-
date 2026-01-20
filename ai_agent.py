"""
Instruction Synthesis Agent

Transforms raw scraped DIY guides into standardized Action-Object sequences
using Google Gemini API and ToolLocator for geometric location inference.
"""
import os
import json
import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError

from tool_locator import ToolLocator
from crawler import Guide

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Pydantic Models for Standardized Format ---

class GeometricLocation(BaseModel):
    """Normalized geometric location (0-1 range)."""
    x: float = Field(ge=0.0, le=1.0, description="X coordinate (left edge)")
    y: float = Field(ge=0.0, le=1.0, description="Y coordinate (top edge)")
    w: float = Field(ge=0.0, le=1.0, description="Width")
    h: float = Field(ge=0.0, le=1.0, description="Height")


class StandardizedStep(BaseModel):
    """Standardized step format."""
    step_id: int = Field(..., description="Step number (1-indexed)")
    description: str = Field(..., description="Concise step description")
    actions: List[str] = Field(..., description="Array of discrete action strings")
    tool: str = Field(..., description="Single tool name (must be from toolbox)")
    part: str = Field(..., description="The part/object being worked on")
    geometric_location: GeometricLocation = Field(..., description="Normalized coordinates (0-1)")
    hands: int = Field(ge=1, le=2, description="Number of hands required (1 or 2)")


class GuideHeader(BaseModel):
    """Header section of standardized guide."""
    title: str = Field(..., description="Guide title")
    description: str = Field(..., description="Summary of the entire guide")
    toolbox: List[str] = Field(..., description="List of all tools needed (normalized from supplies)")


class StandardizedGuide(BaseModel):
    """Final standardized guide format matching exact specification."""
    header: GuideHeader
    steps: List[StandardizedStep]


# --- Instruction Synthesis Agent ---

class InstructionSynthesisAgent:
    """
    LLM-driven agent that transforms raw DIY guides into standardized format.
    
    Uses Google Gemini API for text processing and ToolLocator for geometric
    location inference from images.
    """
    
    def __init__(self):
        """
        Initialize the agent with Gemini API and ToolLocator.
        
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
        
        # Initialize the model (using gemini-1.5-flash for cost efficiency)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Initialize ToolLocator for geometric location inference
        self.tool_locator = ToolLocator()
        
        logging.info("InstructionSynthesisAgent initialized successfully")
    
    def _convert_bbox_format(self, bbox_2d: List[int]) -> GeometricLocation:
        """
        Convert bounding box from tool_locator format to geometric_location format.
        
        Args:
            bbox_2d: Bounding box in format [ymin, xmin, ymax, xmax] with coordinates 0-1000
            
        Returns:
            GeometricLocation with {x, y, w, h} in 0-1 range
        """
        if len(bbox_2d) != 4:
            raise ValueError(f"bbox_2d must have 4 elements, got {len(bbox_2d)}")
        
        ymin, xmin, ymax, xmax = bbox_2d
        
        # Convert from 0-1000 range to 0-1 range
        x = xmin / 1000.0
        y = ymin / 1000.0
        w = (xmax - xmin) / 1000.0
        h = (ymax - ymin) / 1000.0
        
        # Ensure values are within bounds
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        
        return GeometricLocation(x=x, y=y, w=w, h=h)
    
    def _get_geometric_location_from_image(
        self, 
        tool: str, 
        image_urls: List[str]
    ) -> GeometricLocation:
        """
        Get geometric location of tool in images using ToolLocator.
        
        Args:
            tool: Tool name to locate
            image_urls: List of image URLs to search
            
        Returns:
            GeometricLocation with normalized coordinates (0-1)
            Returns default center location if tool not found
        """
        if not image_urls:
            logging.warning(f"No images available for tool '{tool}', using default location")
            return GeometricLocation(x=0.5, y=0.5, w=0.1, h=0.1)
        
        # Try each image until tool is found
        for image_url in image_urls:
            try:
                tool_locations = self.tool_locator.locate_tools(image_url, [tool])
                
                if tool_locations:
                    # Use first match
                    bbox = tool_locations[0].bbox_2d
                    logging.info(f"Found tool '{tool}' in image {image_url}")
                    return self._convert_bbox_format(bbox)
                
            except Exception as e:
                logging.warning(f"Error locating tool '{tool}' in image {image_url}: {e}")
                continue
        
        # Tool not found in any image, use default center location
        logging.warning(f"Tool '{tool}' not found in any image, using default location")
        return GeometricLocation(x=0.5, y=0.5, w=0.1, h=0.1)
    
    def _normalize_toolbox(self, supplies: List[str]) -> List[str]:
        """
        Clean, deduplicate, and normalize tool names from supplies list.
        
        Args:
            supplies: Raw supplies list from scraped guide
            
        Returns:
            Normalized list of tool names
        """
        normalized = []
        seen = set()
        
        for supply in supplies:
            # Clean and normalize
            tool = supply.strip()
            if not tool:
                continue
            
            # Remove common prefixes/suffixes
            tool = tool.replace("Replacement ", "").replace(" replacement", "")
            tool = tool.replace("Clean ", "").replace(" clean", "")
            
            # Normalize case (title case)
            tool = tool.title()
            
            # Deduplicate
            tool_lower = tool.lower()
            if tool_lower not in seen:
                seen.add(tool_lower)
                normalized.append(tool)
        
        return normalized
    
    def _generate_header(
        self, 
        title: str, 
        steps: List[List[str]], 
        supplies: List[str]
    ) -> GuideHeader:
        """
        Generate header with description and normalized toolbox using LLM.
        
        Args:
            title: Guide title
            steps: List of [headline, content] tuples
            supplies: Raw supplies list
            
        Returns:
            GuideHeader with description and toolbox
        """
        # Normalize toolbox first
        toolbox = self._normalize_toolbox(supplies)
        
        # Prepare step summaries for context
        step_summaries = []
        for i, (headline, content) in enumerate(steps, 1):
            step_summaries.append(f"Step {i}: {headline}")
        
        steps_text = "\n".join(step_summaries)
        supplies_text = ", ".join(supplies)
        
        prompt = f"""You are a technical documentation expert. Generate a concise description and normalize the toolbox for this DIY guide.

Title: {title}

Steps:
{steps_text}

Raw Supplies List: {supplies_text}

Tasks:
1. Generate a concise description (1-2 sentences) that summarizes what this guide teaches/how to do.
2. Extract and normalize the toolbox - filter to actual tools only, remove duplicates, standardize names.

Output JSON format:
{{
    "description": "<1-2 sentence summary>",
    "toolbox": ["tool1", "tool2", ...]
}}

Rules:
- Description should be clear and concise
- Toolbox should only include actual tools (not materials like "replacement part" unless it's a tool)
- Normalize tool names (e.g., "Screwdrivers" -> "Screwdriver", "PH000 screwdriver" -> "PH000 Screwdriver")
- Remove duplicates
- Only include tools explicitly mentioned in supplies or steps
"""
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json"
                    )
                )
                
                result = json.loads(response.text)
                
                # Validate and use normalized toolbox from LLM, or fallback to our normalization
                if "toolbox" in result and isinstance(result["toolbox"], list):
                    toolbox = result["toolbox"]
                
                description = result.get("description", f"This guide teaches how to {title.lower()}.")
                
                return GuideHeader(
                    title=title,
                    description=description,
                    toolbox=toolbox
                )
                
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logging.error("Failed to parse JSON after retries, using fallback")
                    break
            except Exception as e:
                logging.warning(f"Error generating header (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logging.error("Failed after retries, using fallback")
                    break
        
        # Fallback
        return GuideHeader(
            title=title,
            description=f"This guide teaches how to {title.lower()}.",
            toolbox=toolbox
        )
    
    def _process_step(
        self,
        step_data: List[str],
        step_number: int,
        toolbox: List[str],
        image_urls: List[str]
    ) -> StandardizedStep:
        """
        Process a single step to extract structured data.
        
        Args:
            step_data: [headline, content] tuple
            step_number: Step number (1-indexed)
            toolbox: List of available tools
            image_urls: List of image URLs for geometric location inference
            
        Returns:
            StandardizedStep with all required fields
        """
        headline, content = step_data[0], step_data[1]
        
        toolbox_str = ", ".join(toolbox)
        
        prompt = f"""You are a technical instruction parser. Extract structured data from this DIY step.

Step Headline: {headline}
Step Content: {content}
Available Tools: {toolbox_str}

Extract the following information:
1. A concise description (1 sentence)
2. Break down the step into discrete action strings (array)
3. Identify the tool used (must be from available tools list)
4. Identify the part/object being worked on
5. Determine number of hands needed (1 or 2)

Output JSON format:
{{
    "description": "<concise step description>",
    "actions": ["<action 1>", "<action 2>", ...],
    "tool": "<tool name from available tools>",
    "part": "<part/object being worked on>",
    "hands": 1 or 2
}}

IMPORTANT RULES:
- Only use tools from the available tools list
- Break content into discrete, actionable strings
- Tool must match one from available tools (exact or close match)
- Part should be the specific component/object mentioned
- Hands: 1 for simple actions (press, turn, insert), 2 for complex actions (lift, hold, separate)
- Be strictly factual - no hallucinations
- If tool is not clearly mentioned, use the most likely tool from available tools
"""
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json"
                    )
                )
                
                result = json.loads(response.text)
                
                # Extract fields
                description = result.get("description", headline)
                actions = result.get("actions", [content])
                tool = result.get("tool", toolbox[0] if toolbox else "Unknown")
                part = result.get("part", "Unknown")
                hands = result.get("hands", 2)
                
                # Validate tool is in toolbox (fuzzy match)
                tool = self._match_tool_to_toolbox(tool, toolbox)
                
                # Get geometric location using ToolLocator
                geometric_location = self._get_geometric_location_from_image(tool, image_urls)
                
                return StandardizedStep(
                    step_id=step_number,
                    description=description,
                    actions=actions if isinstance(actions, list) else [actions],
                    tool=tool,
                    part=part,
                    geometric_location=geometric_location,
                    hands=hands
                )
                
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decode error for step {step_number} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logging.error(f"Failed to parse JSON for step {step_number} after retries, using fallback")
                    break
            except Exception as e:
                logging.warning(f"Error processing step {step_number} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logging.error(f"Failed to process step {step_number} after retries, using fallback")
                    break
        
        # Fallback
        tool = toolbox[0] if toolbox else "Unknown"
        geometric_location = self._get_geometric_location_from_image(tool, image_urls)
        
        return StandardizedStep(
            step_id=step_number,
            description=headline,
            actions=[content],
            tool=tool,
            part="Unknown",
            geometric_location=geometric_location,
            hands=2
        )
    
    def _match_tool_to_toolbox(self, tool: str, toolbox: List[str]) -> str:
        """
        Match tool name to closest match in toolbox.
        
        Args:
            tool: Tool name from LLM
            toolbox: List of available tools
            
        Returns:
            Matched tool name from toolbox
        """
        if not toolbox:
            return tool
        
        tool_lower = tool.lower()
        
        # Exact match
        for t in toolbox:
            if t.lower() == tool_lower:
                return t
        
        # Partial match
        for t in toolbox:
            if tool_lower in t.lower() or t.lower() in tool_lower:
                return t
        
        # Return first tool as fallback
        return toolbox[0]
    
    def synthesize_guide(self, raw_guide: dict) -> StandardizedGuide:
        """
        Process a raw guide and transform it into standardized format.
        
        Args:
            raw_guide: Raw guide dictionary matching Guide model
            
        Returns:
            StandardizedGuide with header and steps
        """
        # Validate input
        try:
            guide = Guide(**raw_guide)
        except ValidationError as e:
            raise ValueError(f"Invalid guide format: {e}")
        
        logging.info(f"Processing guide: {guide.title}")
        
        # Generate header
        header = self._generate_header(guide.title, guide.steps, guide.supplies)
        
        # Process each step
        standardized_steps = []
        for i, step_data in enumerate(guide.steps, 1):
            logging.info(f"Processing step {i}/{len(guide.steps)}")
            step = self._process_step(
                step_data,
                i,
                header.toolbox,
                guide.image_urls
            )
            standardized_steps.append(step)
            
            # Add delay to respect API rate limits
            time.sleep(1)
        
        return StandardizedGuide(header=header, steps=standardized_steps)


# --- Processing Functions ---

def process_single_file(
    input_path: str, 
    output_dir: str = "standardized_data"
) -> bool:
    """
    Process a single raw guide file and save standardized output.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Directory to save output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load input file
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_guide = json.load(f)
        
        # Process with agent
        agent = InstructionSynthesisAgent()
        standardized = agent.synthesize_guide(raw_guide)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        input_filename = Path(input_path).stem
        output_filename = f"standardized_{input_filename}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(standardized.model_dump(), f, indent=2, ensure_ascii=False)
        
        logging.info(f"Successfully processed: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}", exc_info=True)
        return False


def process_batch(
    input_dir: str = "raw_data", 
    output_dir: str = "standardized_data"
) -> Dict[str, bool]:
    """
    Process all JSON files in input directory.
    
    Args:
        input_dir: Directory containing raw guide JSON files
        output_dir: Directory to save standardized outputs
        
    Returns:
        Dictionary mapping input files to success status
    """
    results = {}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return results
    
    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        logging.warning(f"No JSON files found in {input_dir}")
        return results
    
    logging.info(f"Found {len(json_files)} files to process")
    
    for json_file in json_files:
        success = process_single_file(str(json_file), output_dir)
        results[str(json_file)] = success
    
    success_count = sum(1 for v in results.values() if v)
    logging.info(f"Batch processing complete: {success_count}/{len(results)} successful")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single file mode
        input_file = sys.argv[1]
        process_single_file(input_file)
    else:
        # Batch mode
        process_batch()
