"""
Tests for Instruction Synthesis Agent.

Tests the transformation of raw DIY guides into standardized format.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_agent import (
    InstructionSynthesisAgent,
    GeometricLocation,
    StandardizedStep,
    GuideHeader,
    StandardizedGuide,
    process_single_file,
    process_batch
)


class TestGeometricLocation:
    """Tests for GeometricLocation model."""
    
    def test_valid_location(self):
        """Test creating valid geometric location."""
        loc = GeometricLocation(x=0.5, y=0.3, w=0.2, h=0.1)
        assert loc.x == 0.5
        assert loc.y == 0.3
        assert loc.w == 0.2
        assert loc.h == 0.1
    
    def test_boundary_values(self):
        """Test boundary values (0.0 and 1.0)."""
        loc_min = GeometricLocation(x=0.0, y=0.0, w=0.0, h=0.0)
        loc_max = GeometricLocation(x=1.0, y=1.0, w=1.0, h=1.0)
        assert loc_min.x == 0.0
        assert loc_max.x == 1.0
    
    def test_invalid_values(self):
        """Test that invalid values raise ValidationError."""
        with pytest.raises(ValidationError):
            GeometricLocation(x=1.5, y=0.5, w=0.2, h=0.1)
        
        with pytest.raises(ValidationError):
            GeometricLocation(x=-0.1, y=0.5, w=0.2, h=0.1)


class TestCoordinateConversion:
    """Tests for coordinate conversion utilities."""
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    def test_convert_bbox_format(self):
        """Test conversion from tool_locator format to geometric_location."""
        agent = InstructionSynthesisAgent()
        
        # Test conversion: [ymin, xmin, ymax, xmax] (0-1000) -> {x, y, w, h} (0-1)
        bbox = [100, 200, 300, 400]  # ymin=100, xmin=200, ymax=300, xmax=400
        result = agent._convert_bbox_format(bbox)
        
        assert result.x == 0.2  # 200/1000
        assert result.y == 0.1  # 100/1000
        assert result.w == 0.2  # (400-200)/1000
        assert result.h == 0.2  # (300-100)/1000
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    def test_convert_bbox_boundaries(self):
        """Test conversion at boundaries."""
        agent = InstructionSynthesisAgent()
        
        # Top-left corner
        bbox = [0, 0, 100, 100]
        result = agent._convert_bbox_format(bbox)
        assert result.x == 0.0
        assert result.y == 0.0
        
        # Bottom-right corner
        bbox = [900, 900, 1000, 1000]
        result = agent._convert_bbox_format(bbox)
        assert result.x == 0.9
        assert result.y == 0.9


class TestToolboxNormalization:
    """Tests for toolbox normalization."""
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    def test_normalize_toolbox(self):
        """Test toolbox normalization."""
        agent = InstructionSynthesisAgent()
        
        supplies = [
            "Screwdrivers",
            "screwdriver",
            "PH000 screwdriver",
            "Replacement condenser coil",
            "Vacuum cleaner",
            "  Brush  ",
            ""
        ]
        
        normalized = agent._normalize_toolbox(supplies)
        
        # Should remove duplicates and normalize
        assert "Screwdriver" in normalized or "Screwdrivers" in normalized
        assert "Vacuum Cleaner" in normalized
        assert "Brush" in normalized
        assert "" not in normalized
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    def test_match_tool_to_toolbox(self):
        """Test tool matching to toolbox."""
        agent = InstructionSynthesisAgent()
        
        toolbox = ["Screwdriver", "Vacuum Cleaner", "Brush"]
        
        # Exact match
        assert agent._match_tool_to_toolbox("Screwdriver", toolbox) == "Screwdriver"
        
        # Case insensitive
        assert agent._match_tool_to_toolbox("screwdriver", toolbox) == "Screwdriver"
        
        # Partial match
        assert agent._match_tool_to_toolbox("PH000 Screwdriver", toolbox) == "Screwdriver"
        
        # No match - returns first tool
        assert agent._match_tool_to_toolbox("Unknown Tool", toolbox) == "Screwdriver"


class TestGeometricLocationInference:
    """Tests for geometric location inference."""
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    @patch('ai_agent.ToolLocator')
    def test_get_geometric_location_found(self, mock_tool_locator_class):
        """Test getting geometric location when tool is found."""
        # Mock ToolLocator
        mock_tool_locator = Mock()
        mock_tool_location = Mock()
        mock_tool_location.bbox_2d = [100, 200, 300, 400]
        mock_tool_locator.locate_tools.return_value = [mock_tool_location]
        mock_tool_locator_class.return_value = mock_tool_locator
        
        agent = InstructionSynthesisAgent()
        agent.tool_locator = mock_tool_locator
        
        image_urls = ["https://example.com/image.jpg"]
        result = agent._get_geometric_location_from_image("Screwdriver", image_urls)
        
        assert result.x == 0.2
        assert result.y == 0.1
        assert result.w == 0.2
        assert result.h == 0.2
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    @patch('ai_agent.ToolLocator')
    def test_get_geometric_location_not_found(self, mock_tool_locator_class):
        """Test getting default location when tool is not found."""
        # Mock ToolLocator
        mock_tool_locator = Mock()
        mock_tool_locator.locate_tools.return_value = []
        mock_tool_locator_class.return_value = mock_tool_locator
        
        agent = InstructionSynthesisAgent()
        agent.tool_locator = mock_tool_locator
        
        image_urls = ["https://example.com/image.jpg"]
        result = agent._get_geometric_location_from_image("Unknown Tool", image_urls)
        
        # Should return default center location
        assert result.x == 0.5
        assert result.y == 0.5
        assert result.w == 0.1
        assert result.h == 0.1
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    def test_get_geometric_location_no_images(self):
        """Test getting default location when no images available."""
        agent = InstructionSynthesisAgent()
        
        result = agent._get_geometric_location_from_image("Screwdriver", [])
        
        # Should return default center location
        assert result.x == 0.5
        assert result.y == 0.5
        assert result.w == 0.1
        assert result.h == 0.1


class TestStepProcessing:
    """Tests for step processing."""
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    @patch('ai_agent.ToolLocator')
    @patch('ai_agent.genai.Client')
    def test_process_step_success(self, mock_client_class, mock_tool_locator_class):
        """Test successful step processing."""
        # Mock LLM response
        mock_client = Mock()
        mock_models = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "description": "Remove the screws",
            "actions": ["Locate the screws", "Remove the screws"],
            "tool": "Screwdriver",
            "part": "Bottom screws",
            "hands": 2
        })
        mock_models.generate_content.return_value = mock_response
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client
        
        # Mock ToolLocator
        mock_tool_locator = Mock()
        mock_tool_location = Mock()
        mock_tool_location.bbox_2d = [100, 200, 300, 400]
        mock_tool_locator.locate_tools.return_value = [mock_tool_location]
        mock_tool_locator_class.return_value = mock_tool_locator
        
        agent = InstructionSynthesisAgent()
        agent.client = mock_client
        agent.tool_locator = mock_tool_locator
        
        step_data = ["Step 1 - Remove Screws", "Locate and remove the bottom screws."]
        toolbox = ["Screwdriver", "Wrench"]
        image_urls = ["https://example.com/image.jpg"]
        
        result = agent._process_step(step_data, 1, toolbox, image_urls)
        
        assert result.step_id == 1
        assert result.description == "Remove the screws"
        assert len(result.actions) == 2
        assert result.tool == "Screwdriver"
        assert result.part == "Bottom screws"
        assert result.hands == 2


class TestGuideSynthesis:
    """Tests for full guide synthesis."""
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'})
    @patch('ai_agent.ToolLocator')
    @patch('ai_agent.genai.Client')
    def test_synthesize_guide(self, mock_client_class, mock_tool_locator_class):
        """Test synthesizing a complete guide."""
        # Mock LLM responses
        mock_client = Mock()
        mock_models = Mock()
        
        # Header response
        header_response = Mock()
        header_response.text = json.dumps({
            "description": "This guide teaches how to clean a freezer.",
            "toolbox": ["Screwdriver", "Vacuum Cleaner"]
        })
        
        # Step response
        step_response = Mock()
        step_response.text = json.dumps({
            "description": "Remove the screws",
            "actions": ["Locate the screws", "Remove the screws"],
            "tool": "Screwdriver",
            "part": "Bottom screws",
            "hands": 2
        })
        
        mock_models.generate_content.side_effect = [header_response, step_response]
        mock_client.models = mock_models
        mock_client_class.return_value = mock_client
        
        # Mock ToolLocator
        mock_tool_locator = Mock()
        mock_tool_location = Mock()
        mock_tool_location.bbox_2d = [100, 200, 300, 400]
        mock_tool_locator.locate_tools.return_value = [mock_tool_location]
        mock_tool_locator_class.return_value = mock_tool_locator
        
        agent = InstructionSynthesisAgent()
        agent.client = mock_client
        agent.tool_locator = mock_tool_locator
        
        raw_guide = {
            "title": "How to Clean Freezer",
            "author": "Test Author",
            "additional_text_boxes": [],
            "supplies": ["Screwdriver", "Vacuum cleaner"],
            "steps": [
                ["Step 1 - Remove Screws", "Locate and remove the bottom screws."]
            ],
            "image_urls": ["https://example.com/image.jpg"],
            "url": "https://example.com/guide"
        }
        
        result = agent.synthesize_guide(raw_guide)
        
        assert isinstance(result, StandardizedGuide)
        assert result.header.title == "How to Clean Freezer"
        assert len(result.steps) == 1
        assert result.steps[0].step_id == 1


class TestFileProcessing:
    """Tests for file processing functions."""
    
    @patch('ai_agent.InstructionSynthesisAgent')
    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    @patch('pathlib.Path')
    def test_process_single_file_success(
        self, mock_path, mock_makedirs, mock_open, mock_agent_class
    ):
        """Test successful single file processing."""
        # Mock file operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = '{"title": "Test Guide", "author": "Test", "additional_text_boxes": [], "supplies": [], "steps": [], "image_urls": [], "url": "https://test.com"}'
        mock_open.return_value = mock_file
        
        # Mock path
        mock_path_instance = Mock()
        mock_path_instance.stem = "test_guide"
        mock_path.return_value = mock_path_instance
        
        # Mock agent
        mock_agent = Mock()
        mock_standardized = Mock()
        mock_standardized.model_dump.return_value = {"header": {}, "steps": []}
        mock_agent.synthesize_guide.return_value = mock_standardized
        mock_agent_class.return_value = mock_agent
        
        result = process_single_file("test.json", "output")
        
        assert result is True
        mock_agent.synthesize_guide.assert_called_once()
    
    @patch('builtins.open', create=True)
    def test_process_single_file_error(self, mock_open):
        """Test file processing with error."""
        mock_open.side_effect = FileNotFoundError("File not found")
        
        result = process_single_file("nonexistent.json", "output")
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
