"""
Instruction Synthesis Agent

Transforms raw scraped DIY guides into standardized Action-Object sequences
using Google Gemini API and PartLocator for geometric location inference.
"""
import os
import json
import logging
import time
import re
from typing import List, Optional, Dict, Any
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError as GenAIAPIError
from pydantic import BaseModel, Field, ValidationError

from part_locator import PartLocator
from crawler import Guide

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Custom Exceptions ---

class AgentAbortError(Exception):
    """Raised when the agent should abort due to too many consecutive failures."""
    pass


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


# --- Pydantic Models for Standardized Format ---

class GeometricLocation(BaseModel):
    """Normalized geometric location (0-1 range). Null values indicate tool not detected in image."""
    x: Optional[float] = Field(None, ge=0.0, le=1.0, description="X coordinate (left edge), null if tool not detected")
    y: Optional[float] = Field(None, ge=0.0, le=1.0, description="Y coordinate (top edge), null if tool not detected")
    w: Optional[float] = Field(None, ge=0.0, le=1.0, description="Width, null if tool not detected")
    h: Optional[float] = Field(None, ge=0.0, le=1.0, description="Height, null if tool not detected")


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
    
    Uses Google Gemini API for text processing and PartLocator for geometric
    location inference from images.
    """
    
    # Abort thresholds
    MAX_CONSECUTIVE_FAILURES = 5  # Abort after this many consecutive API failures
    MAX_TOTAL_FAILURES = 10  # Abort after this many total failures in a single guide
    SERVER_OVERLOAD_DELAY = 30  # Seconds to wait when server is overloaded (503)
    
    def __init__(self, skip_tool_location: bool = False):
        """
        Initialize the agent with Gemini API and PartLocator.
        
        Args:
            skip_tool_location: If True, skip part location detection and use null location values
        
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
        
        # Initialize PartLocator for geometric location inference
        self.part_locator = PartLocator()
        
        # Part location settings
        self.skip_tool_location = skip_tool_location
        
        # Failure tracking for abort mechanism
        self._consecutive_failures = 0
        self._total_failures = 0
        
        logging.info("InstructionSynthesisAgent initialized successfully")
    
    def _reset_failure_counts(self):
        """Reset failure counters (call at start of new guide processing)."""
        self._consecutive_failures = 0
        self._total_failures = 0
    
    def _record_success(self):
        """Record a successful API call."""
        self._consecutive_failures = 0
    
    def _record_failure(self, error: Exception) -> None:
        """
        Record an API failure and check if abort threshold is reached.
        
        Args:
            error: The exception that caused the failure
            
        Raises:
            AgentAbortError: If abort threshold is reached
        """
        self._consecutive_failures += 1
        self._total_failures += 1
        
        # Check for server overload (503) - needs longer wait
        is_server_overload = (
            isinstance(error, ServerError) or 
            (hasattr(error, 'args') and '503' in str(error))
        )
        
        if is_server_overload:
            logging.warning(
                f"Server overloaded (503). Consecutive failures: {self._consecutive_failures}/"
                f"{self.MAX_CONSECUTIVE_FAILURES}, Total: {self._total_failures}/{self.MAX_TOTAL_FAILURES}"
            )
        
        # Check abort thresholds
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            raise AgentAbortError(
                f"Aborting: {self._consecutive_failures} consecutive API failures. "
                f"The API appears to be unavailable. Please try again later."
            )
        
        if self._total_failures >= self.MAX_TOTAL_FAILURES:
            raise AgentAbortError(
                f"Aborting: {self._total_failures} total API failures in this guide. "
                f"Too many errors encountered. Please check your API key and try again later."
            )
    
    def _convert_bbox_format(self, bbox_2d: List[int]) -> GeometricLocation:
        """
        Convert bounding box from part_locator format to geometric_location format.
        
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
        part: str, 
        image_urls: List[str]
    ) -> GeometricLocation:
        """
        Get geometric location of part/component in images using PartLocator.
        
        Args:
            part: Part/component name to locate
            image_urls: List of image URLs to search
            
        Returns:
            GeometricLocation with normalized coordinates (0-1) if part found,
            or null values if part not detected in image
        """
        # Check if part location is disabled
        if self.skip_tool_location:
            logging.info(f"Part location disabled, returning null location for '{part}'")
            return GeometricLocation(x=None, y=None, w=None, h=None)
        
        if not image_urls:
            logging.warning(f"No images available for part '{part}', returning null location")
            return GeometricLocation(x=None, y=None, w=None, h=None)
        
        # Try each image until part is found
        for image_url in image_urls:
            try:
                part_locations = self.part_locator.locate_parts(image_url, [part])
                
                if part_locations:
                    # Use first match
                    bbox = part_locations[0].bbox_2d
                    logging.info(f"Found part '{part}' in image {image_url}")
                    return self._convert_bbox_format(bbox)
                
            except Exception as e:
                logging.warning(f"Error locating part '{part}' in image {image_url}: {e}")
                continue
        
        # Part not found in any image, return null location
        logging.warning(f"Part '{part}' not found in any image, returning null location")
        return GeometricLocation(x=None, y=None, w=None, h=None)
    
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
            
            # Remove common prefixes (only at start of string, case-insensitive)
            tool_lower = tool.lower()
            if tool_lower.startswith("replacement "):
                tool = tool[len("replacement "):].strip()
            elif tool_lower.startswith("clean ") and len(tool.split()) > 1:
                # Only remove "Clean " prefix if there are multiple words
                # This prevents removing "clean" from words like "cleaner"
                tool = tool[len("clean "):].strip()
            
            # Normalize case (title case)
            tool = tool.title()
            
            # Clean up any double spaces
            tool = " ".join(tool.split())
            
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
                # Configure JSON mode only if model supports it
                config = types.GenerateContentConfig()
                if self.supports_json_mode:
                    config.response_mime_type = "application/json"
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                
                # Parse JSON from response (handles both JSON mode and plain text with JSON)
                result = extract_json_from_text(response.text)
                if result is None:
                    raise json.JSONDecodeError("No valid JSON found in response", response.text, 0)
                
                # Validate and use normalized toolbox from LLM, or fallback to our normalization
                if "toolbox" in result and isinstance(result["toolbox"], list):
                    toolbox = result["toolbox"]
                
                description = result.get("description", f"This guide teaches how to {title.lower()}.")
                
                # Record success
                self._record_success()
                
                return GuideHeader(
                    title=title,
                    description=description,
                    toolbox=toolbox
                )
                
            except json.JSONDecodeError as e:
                response_preview = response.text[:200] if 'response' in locals() and response.text else "N/A"
                logging.warning(
                    f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Response preview: {response_preview}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logging.error("Failed to parse JSON after retries, using fallback")
                    break
            except (ServerError, GenAIAPIError) as e:
                # Record failure and check abort threshold
                self._record_failure(e)
                
                # Check if it's a server overload error (503)
                is_overload = '503' in str(e) or 'overloaded' in str(e).lower()
                delay = self.SERVER_OVERLOAD_DELAY if is_overload else retry_delay * (2 ** attempt)
                
                logging.warning(
                    f"API error generating header (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Waiting {delay}s before retry..."
                )
                
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    logging.error("Failed after retries, using fallback")
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
                # Configure JSON mode only if model supports it
                config = types.GenerateContentConfig()
                if self.supports_json_mode:
                    config.response_mime_type = "application/json"
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                
                # Parse JSON from response (handles both JSON mode and plain text with JSON)
                result = extract_json_from_text(response.text)
                if result is None:
                    raise json.JSONDecodeError("No valid JSON found in response", response.text, 0)
                
                # Extract fields
                description = result.get("description", headline)
                actions = result.get("actions", [content])
                tool = result.get("tool", toolbox[0] if toolbox else "Unknown")
                part = result.get("part", "Unknown")
                hands = result.get("hands", 2)
                
                # Validate tool is in toolbox (fuzzy match)
                tool = self._match_tool_to_toolbox(tool, toolbox)
                
                # Get geometric location using PartLocator (locate the PART, not the tool)
                geometric_location = self._get_geometric_location_from_image(part, image_urls)
                
                # Record success
                self._record_success()
                
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
                response_preview = response.text[:200] if 'response' in locals() and response.text else "N/A"
                logging.warning(
                    f"JSON decode error for step {step_number} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Response preview: {response_preview}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logging.error(f"Failed to parse JSON for step {step_number} after retries, using fallback")
                    break
            except (ServerError, GenAIAPIError) as e:
                # Record failure and check abort threshold
                self._record_failure(e)
                
                # Check if it's a server overload error (503)
                is_overload = '503' in str(e) or 'overloaded' in str(e).lower()
                delay = self.SERVER_OVERLOAD_DELAY if is_overload else retry_delay * (2 ** attempt)
                
                logging.warning(
                    f"API error processing step {step_number} (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Waiting {delay}s before retry..."
                )
                
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"Failed to process step {step_number} after retries, using fallback")
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
        geometric_location = self._get_geometric_location_from_image("Unknown", image_urls)
        
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
            
        Raises:
            AgentAbortError: If too many consecutive API failures occur
            ValueError: If guide format is invalid
        """
        # Reset failure counters for new guide
        self._reset_failure_counts()
        
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
    output_dir: str = "standardized_data",
    skip_tool_location: bool = False
) -> bool:
    """
    Process a single raw guide file and save standardized output.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Directory to save output
        skip_tool_location: If True, skip part location detection and use null location values
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load input file
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_guide = json.load(f)
        
        # Process with agent
        agent = InstructionSynthesisAgent(skip_tool_location=skip_tool_location)
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
    
    except AgentAbortError as e:
        logging.error(f"ABORTED: {e}")
        logging.error(
            "The API is experiencing issues. Please wait a few minutes and try again. "
            "If the problem persists, check your API key and quota."
        )
        return False
    
    except KeyboardInterrupt:
        logging.warning("Processing interrupted by user (Ctrl+C)")
        return False
        
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}", exc_info=True)
        return False


def process_batch(
    input_dir: str = "raw_data", 
    output_dir: str = "standardized_data",
    stop_on_abort: bool = True,
    skip_tool_location: bool = False
) -> Dict[str, bool]:
    """
    Process all JSON files in input directory.
    
    Args:
        input_dir: Directory containing raw guide JSON files
        output_dir: Directory to save standardized outputs
        stop_on_abort: If True, stop processing remaining files when API abort occurs
        skip_tool_location: If True, skip part location detection and use null location values
        
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
    
    aborted = False
    for i, json_file in enumerate(json_files, 1):
        if aborted and stop_on_abort:
            logging.warning(f"Skipping remaining {len(json_files) - i + 1} files due to previous abort")
            for remaining_file in json_files[i-1:]:
                results[str(remaining_file)] = False
            break
            
        logging.info(f"Processing file {i}/{len(json_files)}: {json_file.name}")
        
        try:
            success = process_single_file(str(json_file), output_dir, skip_tool_location)
            results[str(json_file)] = success
        except KeyboardInterrupt:
            logging.warning("Batch processing interrupted by user (Ctrl+C)")
            results[str(json_file)] = False
            # Mark remaining as not processed
            for remaining_file in json_files[i:]:
                results[str(remaining_file)] = False
            break
        
        # Check if the last file processing failed due to abort-like conditions
        # (process_single_file returns False but doesn't raise)
        if not success:
            # Add a small delay before next file to be nice to the API
            time.sleep(2)
    
    success_count = sum(1 for v in results.values() if v)
    logging.info(f"Batch processing complete: {success_count}/{len(results)} successful")
    
    return results


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process DIY guides with AI to create standardized output'
    )
    parser.add_argument(
        'input_file', 
        nargs='?', 
        help='Single file to process (if omitted, processes all files in raw_data/)'
    )
    parser.add_argument(
        '--skip-tool-location', 
        action='store_true', 
        help='Skip part location detection and use null location values (saves API calls)'
    )
    args = parser.parse_args()
    
    if args.input_file:
        # Single file mode
        process_single_file(args.input_file, skip_tool_location=args.skip_tool_location)
    else:
        # Batch mode
        process_batch(skip_tool_location=args.skip_tool_location)
