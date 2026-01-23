# System Prompt Templates for Instruction Synthesis Agent

This document contains the prompt engineering templates used by the Instruction Synthesis Agent to transform raw DIY guides into standardized Action-Object sequences.

## Header Generation Prompt

**Purpose:** Generate a concise description and normalize the toolbox from raw supplies.

**Input:**
- Title: Guide title
- Steps: List of step headlines
- Raw Supplies: List of supplies from scraped data

**Output Format:**
```json
{
    "description": "<1-2 sentence summary>",
    "toolbox": ["tool1", "tool2", ...]
}
```

**Prompt Template:**
```
You are a technical documentation expert. Generate a concise description and normalize the toolbox for this DIY guide.

Title: {title}

Steps:
{step_summaries}

Raw Supplies List: {supplies_list}

Tasks:
1. Generate a concise description (1-2 sentences) that summarizes what this guide teaches/how to do.
2. Extract and normalize the toolbox - filter to actual tools only, remove duplicates, standardize names.

Output JSON format:
{
    "description": "<1-2 sentence summary>",
    "toolbox": ["tool1", "tool2", ...]
}

Rules:
- Description should be clear and concise
- Toolbox should only include actual tools (not materials like "replacement part" unless it's a tool)
- Normalize tool names (e.g., "Screwdrivers" -> "Screwdriver", "PH000 screwdriver" -> "PH000 Screwdriver")
- Remove duplicates
- Only include tools explicitly mentioned in supplies or steps
```

## Step Processing Prompt

**Purpose:** Extract structured data from a single step (description, actions, tool, part, hands).

**Input:**
- Step Headline: Step title/headline
- Step Content: Full step text content
- Available Tools: List of tools from normalized toolbox

**Output Format:**
```json
{
    "description": "<concise step description>",
    "actions": ["<action 1>", "<action 2>", ...],
    "tool": "<tool name from available tools>",
    "part": "<part/object being worked on>",
    "hands": 1 or 2
}
```

**Prompt Template:**
```
You are a technical instruction parser. Extract structured data from this DIY step.

Step Headline: {headline}
Step Content: {content}
Available Tools: {toolbox_list}

Extract the following information:
1. A concise description (1 sentence)
2. Break down the step into discrete action strings (array)
3. Identify the tool used (must be from available tools list)
4. Identify the part/object being worked on
5. Determine number of hands needed (1 or 2)

Output JSON format:
{
    "description": "<concise step description>",
    "actions": ["<action 1>", "<action 2>", ...],
    "tool": "<tool name from available tools>",
    "part": "<part/object being worked on>",
    "hands": 1 or 2
}

IMPORTANT RULES:
- Only use tools from the available tools list
- Break content into discrete, actionable strings
- Tool must match one from available tools (exact or close match)
- Part should be the specific component/object mentioned
- Hands: 1 for simple actions (press, turn, insert), 2 for complex actions (lift, hold, separate)
- Be strictly factual - no hallucinations
- If tool is not clearly mentioned, use the most likely tool from available tools
```

## Key Constraints

1. **No Hallucinations:** Only extract information explicitly present in the input text
2. **Tool Validation:** Tools must come from the provided toolbox list
3. **Action Granularity:** Break steps into discrete, actionable sub-steps
4. **Part Identification:** Extract specific components/objects being worked on
5. **Hands Count:** Infer from action complexity (1 for simple, 2 for complex)

## Geometric Location

Geometric locations are determined separately using `part_locator.py`:
- PartLocator analyzes images to find part/component locations
- Coordinates are converted from [ymin, xmin, ymax, xmax] (0-1000) to {x, y, w, h} (0-1)
- If part not found in images, null values are returned (x=None, y=None, w=None, h=None)
