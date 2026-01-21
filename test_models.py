"""
Model Testing Script

Tests different Gemini models to find which works best for the DIY guide generation use case.
Tests for:
- Availability
- Speed
- Quality of JSON responses
- Rate limiting behavior
"""
import os
import time
import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types

# Test prompt for consistency
TEST_PROMPT = """You are a technical documentation expert. Generate a concise description for this DIY guide.

Title: How to Replace a Light Switch

Steps:
Step 1: Turn off the power
Step 2: Remove the old switch
Step 3: Install the new switch
Step 4: Test the switch

Raw Supplies List: Screwdriver, Voltage Tester, Wire Stripper, New Light Switch

Output JSON format:
{
    "description": "<1-2 sentence summary>",
    "toolbox": ["tool1", "tool2", ...]
}

Rules:
- Description should be clear and concise
- Toolbox should only include actual tools
- Normalize tool names
"""

# Models to test (add or remove based on availability)
# Note: Use full model path with "models/" prefix
MODELS_TO_TEST = [
    "models/gemma-3-12b-it",  # Currently used - higher quota limits
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-2.0-flash-exp",
    "models/gemini-2.0-flash",
    "models/gemini-exp-1206",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
]


class ModelTester:
    """Test different Gemini models for suitability."""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        self.client = genai.Client(api_key=api_key)
        self.results: List[Dict[str, Any]] = []
    
    def list_available_models(self) -> List[str]:
        """List all available models for the API key."""
        print("=" * 70)
        print("LISTING AVAILABLE MODELS")
        print("=" * 70)
        
        available = []
        try:
            for model in self.client.models.list():
                model_name = model.name
                
                # Skip embedding and image generation models
                if any(skip in model_name.lower() for skip in [
                    'embedding', 'imagen', 'veo', 'aqa', 'gecko'
                ]):
                    continue
                
                # Only include gemini models (most likely to work)
                if 'gemini' in model_name.lower() or 'gemma' in model_name.lower():
                    available.append(model_name)
                    print(f"  {model_name}")
        
        except Exception as e:
            print(f"Error listing models: {e}")
        
        print(f"\nTotal models found: {len(available)}")
        print("=" * 70)
        return available
    
    def test_model(self, model_name: str) -> Dict[str, Any]:
        """Test a single model."""
        print(f"\n{'='*70}")
        print(f"TESTING: {model_name}")
        print(f"{'='*70}")
        
        result = {
            "model": model_name,
            "available": False,
            "response_time": None,
            "success": False,
            "error": None,
            "response_quality": None,
            "json_valid": False
        }
        
        try:
            start_time = time.time()
            
            response = self.client.models.generate_content(
                model=model_name,
                contents=TEST_PROMPT,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            response_time = time.time() - start_time
            result["response_time"] = round(response_time, 2)
            result["available"] = True
            
            # Try to parse JSON
            try:
                parsed = json.loads(response.text)
                result["json_valid"] = True
                result["success"] = True
                
                # Check quality
                has_description = "description" in parsed
                has_toolbox = "toolbox" in parsed
                toolbox_valid = isinstance(parsed.get("toolbox"), list) if has_toolbox else False
                
                quality_score = sum([has_description, has_toolbox, toolbox_valid])
                result["response_quality"] = f"{quality_score}/3"
                
                print(f"✓ Success!")
                print(f"  Response time: {response_time:.2f}s")
                print(f"  Quality: {quality_score}/3")
                print(f"  Description: {parsed.get('description', 'N/A')[:60]}...")
                print(f"  Toolbox: {parsed.get('toolbox', [])}")
                
            except json.JSONDecodeError as e:
                result["error"] = f"Invalid JSON: {str(e)}"
                print(f"✗ Invalid JSON response")
                print(f"  Response: {response.text[:100]}...")
        
        except Exception as e:
            result["error"] = str(e)
            error_msg = str(e)
            
            if "503" in error_msg or "overloaded" in error_msg.lower():
                print(f"✗ Server overloaded (503)")
            elif "404" in error_msg or "not found" in error_msg.lower():
                print(f"✗ Model not found (404)")
                result["available"] = False
            elif "429" in error_msg or "quota" in error_msg.lower():
                print(f"✗ Rate limit exceeded (429)")
            else:
                print(f"✗ Error: {error_msg}")
        
        return result
    
    def test_all_models(self, model_list: Optional[List[str]] = None):
        """Test all models in the list."""
        if model_list is None:
            model_list = MODELS_TO_TEST
        
        print(f"\n{'='*70}")
        print(f"TESTING {len(model_list)} MODELS")
        print(f"{'='*70}")
        
        for i, model_name in enumerate(model_list, 1):
            print(f"\n[{i}/{len(model_list)}] ", end="")
            result = self.test_model(model_name)
            self.results.append(result)
            
            # Small delay between tests to be nice to the API
            if i < len(model_list):
                time.sleep(2)
    
    def print_summary(self):
        """Print summary of all test results."""
        print(f"\n\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}\n")
        
        if not self.results:
            print("No models were tested.")
            print("\nTIP: Try option 1 to test predefined models or option 3 to enter model names manually.")
            print("=" * 70)
            return
        
        successful = [r for r in self.results if r["success"]]
        available = [r for r in self.results if r["available"]]
        
        print(f"Total models tested: {len(self.results)}")
        print(f"Available models: {len(available)}")
        print(f"Successful responses: {len(successful)}")
        print(f"Success rate: {len(successful)/len(self.results)*100:.1f}%\n")
        
        if successful:
            print("RECOMMENDED MODELS (sorted by speed):")
            print("-" * 70)
            successful.sort(key=lambda x: x["response_time"] or 999)
            
            for i, result in enumerate(successful, 1):
                print(f"{i}. {result['model']:<35} "
                      f"Speed: {result['response_time']}s  "
                      f"Quality: {result['response_quality']}")
            
            fastest = successful[0]
            print(f"\n✓ FASTEST MODEL: {fastest['model']} ({fastest['response_time']}s)")
        
        print("\n" + "=" * 70)
        
        # Failed models
        failed = [r for r in self.results if not r["success"]]
        if failed:
            print("\nFAILED MODELS:")
            print("-" * 70)
            for result in failed:
                error_summary = result["error"][:50] if result["error"] else "Unknown"
                print(f"✗ {result['model']:<35} {error_summary}")
        
        print("\n" + "=" * 70)
    
    def save_results(self, filename: str = "model_test_results.json"):
        """Save test results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {filename}")


def main():
    """Main function to run model tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "GEMINI MODEL TESTING TOOL")
    print("=" * 70)
    
    tester = ModelTester()
    
    # First, list all available models
    available_models = tester.list_available_models()
    
    print("\n" + "=" * 70)
    print("Choose an option:")
    print("1. Test all predefined models")
    print("2. Test only available models from the list above")
    print("3. Enter custom model names to test")
    print("=" * 70)
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice == "2":
        # Filter MODELS_TO_TEST to only include available ones
        models_to_test = [m for m in MODELS_TO_TEST if m in available_models]
        if not models_to_test:
            print("\nNo predefined models are available. Testing all available models...")
            models_to_test = available_models
    elif choice == "3":
        custom = input("Enter model names (comma-separated): ").strip()
        models_to_test = [m.strip() for m in custom.split(",")]
    else:
        models_to_test = MODELS_TO_TEST
    
    # Run tests
    tester.test_all_models(models_to_test)
    
    # Print summary
    tester.print_summary()
    
    # Save results
    tester.save_results()
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
