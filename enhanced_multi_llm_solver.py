import streamlit as st
import numpy as np
from PIL import Image
import os
import re
import subprocess
import sys
import json
from io import StringIO

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Enhanced Multi-LLM DSA Solver",
    page_icon="🚀",
    layout="wide"
)

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Enhanced Tesseract Setup for Streamlit Cloud
try:
    import pytesseract
    
    # Streamlit Cloud and Linux systems typically have tesseract in /usr/bin/
    tesseract_paths = [
        '/usr/bin/tesseract',                    # Linux/Streamlit Cloud (primary)
        '/usr/local/bin/tesseract',              # macOS/Linux alternative
        'tesseract',                             # System PATH (try this first for Streamlit Cloud)
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',        # Windows
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',  # Windows 32-bit
    ]
    
    tesseract_found = False
    tesseract_error = ""
    
    # First, try the default system installation (most likely to work on Streamlit Cloud)
    try:
        pytesseract.pytesseract.tesseract_cmd = 'tesseract'
        version = pytesseract.get_tesseract_version()
        tesseract_found = True
        tesseract_error = f"Found Tesseract version: {version}"
    except Exception as e:
        tesseract_error = f"Default tesseract failed: {str(e)}"
        
        # If default fails, try specific paths
        for path in tesseract_paths:
            try:
                if path != "tesseract":
                    pytesseract.pytesseract.tesseract_cmd = path
                
                # Test if tesseract is working
                version = pytesseract.get_tesseract_version()
                tesseract_found = True
                tesseract_error = f"Found Tesseract at {path}, version: {version}"
                break
            except Exception as e:
                tesseract_error += f" | {path}: {str(e)}"
                continue
    
    TESSERACT_AVAILABLE = tesseract_found
    
except ImportError as e:
    TESSERACT_AVAILABLE = False
    tesseract_error = f"pytesseract package not installed: {str(e)}"

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
    AZURE_AI_AVAILABLE = True
except ImportError:
    AZURE_AI_AVAILABLE = False

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

# Configuration for Streamlit Cloud Deployment
try:
    # Get configuration from Streamlit secrets
    AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
    INFERENCE_ENDPOINT = st.secrets["INFERENCE_ENDPOINT"]
    LLAMA_MODEL = st.secrets.get("LLAMA_MODEL", "Meta-Llama-3.1-405B-Instruct")
    CODESTRAL_MODEL = st.secrets.get("CODESTRAL_MODEL", "Codestral-2501")
    DEEPSEEK_MODEL = st.secrets.get("DEEPSEEK_MODEL", "DeepSeek-R1-0528")

    # Azure OpenAI Configuration
    GPT5_ENDPOINT = st.secrets["GPT5_ENDPOINT"]
    GPT5_API_KEY = st.secrets["GPT5_API_KEY"]
    GPT5_API_VERSION = st.secrets.get("GPT5_API_VERSION", "2024-12-01-preview")

    # GPT-4.1 Configuration
    GPT41_ENDPOINT = st.secrets["GPT41_ENDPOINT"]
    GPT41_API_KEY = st.secrets["GPT41_API_KEY"]
    GPT41_API_VERSION = st.secrets.get("GPT41_API_VERSION", "2024-12-01-preview")
    
except Exception as e:
    st.error(f"⚠️ Configuration Error: {str(e)}")
    st.error("Please ensure all required secrets are configured in Streamlit Cloud.")
    st.stop()

# Available models
GPT5_MODEL_NAME = "gpt-5"
GPT5_DEPLOYMENT = "gpt-5"
GPT41_MODEL_NAME = "gpt-4.1"
GPT41_DEPLOYMENT = "gpt-4.1"

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

@st.cache_resource
def get_inference_client():
    """Initialize Azure AI Inference client"""
    if not AZURE_AI_AVAILABLE:
        return None
    try:
        client = ChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_API_KEY)
        )
        return client
    except Exception as e:
        st.error(f"Inference client error: {str(e)}")
        return None

@st.cache_resource
def get_gpt5_client():
    """Initialize GPT-5 Azure OpenAI client"""
    if not AZURE_OPENAI_AVAILABLE:
        return None
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_version=GPT5_API_VERSION,
            azure_endpoint=GPT5_ENDPOINT,
            api_key=GPT5_API_KEY,
        )
        return client
    except Exception as e:
        st.error(f"GPT-5 client error: {str(e)}")
        return None

@st.cache_resource
def get_gpt41_client():
    """Initialize GPT-4.1 Azure OpenAI client"""
    if not AZURE_OPENAI_AVAILABLE:
        return None
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_version=GPT41_API_VERSION,
            azure_endpoint=GPT41_ENDPOINT,
            api_key=GPT41_API_KEY,
        )
        return client
    except Exception as e:
        st.error(f"GPT-4.1 client error: {str(e)}")
        return None

def call_ai_model(prompt, model_name, system_message="You are an expert AI assistant.", max_tokens=4000):
    """Unified function to call any AI model"""
    
    # Handle GPT-5 through its specific client
    if model_name == GPT5_MODEL_NAME:
        client = get_gpt5_client()
        if not client:
            return f"Failed to initialize GPT-5 client for {model_name}"
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=GPT5_DEPLOYMENT
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling {model_name}: {str(e)}"
    
    # Handle GPT-4.1 through its specific client
    if model_name == GPT41_MODEL_NAME:
        client = get_gpt41_client()
        if not client:
            return f"Failed to initialize GPT-4.1 client for {model_name}"
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=GPT41_DEPLOYMENT,
                max_tokens=13107,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling {model_name}: {str(e)}"
    
    # Handle Azure AI Inference models (Llama, Codestral, DeepSeek)
    if model_name in [LLAMA_MODEL, CODESTRAL_MODEL, DEEPSEEK_MODEL]:
        client = get_inference_client()
        if not client:
            return f"Failed to initialize AI client for {model_name}"
        
        try:
            response = client.complete(
                messages=[
                    SystemMessage(content=system_message),
                    UserMessage(content=prompt)
                ],
                model=model_name,
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling {model_name}: {str(e)}"
    
    return f"Unknown model: {model_name}"

def extract_text_from_image(image):
    """Extract text from image using OCR with graceful fallback"""
    if not TESSERACT_AVAILABLE:
        return """
⚠️ OCR CURRENTLY UNAVAILABLE
        
Tesseract OCR is not installed on this Streamlit Cloud instance.

WORKAROUND: 
1. Use manual input below (fully functional!)
2. Copy/paste the problem text from the image
3. Or take a screenshot and type the problem manually

Your Enhanced Multi-LLM DSA Solver works perfectly with manual input!
All AI models and advanced features are available.
        """
        
    try:
        if CV2_AVAILABLE:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_img = thresh
        else:
            processed_img = np.array(image.convert('L'))
        
        # Try multiple OCR configurations for better results
        configs = [
            r'--oem 3 --psm 6',  # Default config
            r'--oem 3 --psm 13', # Raw line. Treat the image as a single text line
            r'--oem 3 --psm 8',  # Single word
            r'--oem 3 --psm 7',  # Single text line
        ]
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(processed_img, config=config)
                if text.strip():
                    return text.strip()
            except Exception:
                continue
                
        return "❌ No text detected in image. Please try manual input."
        
    except Exception as e:
        return f"❌ OCR Error: {str(e)}. Please use manual input instead."

def extract_code_from_response(response_text, language="Python"):
    """Extract code from response based on the programming language"""
    language_patterns = {
        "Python": [r'```python\n(.*?)\n```', r'```py\n(.*?)\n```'],
        "Java": [r'```java\n(.*?)\n```'],
        "C++": [r'```cpp\n(.*?)\n```', r'```c\+\+\n(.*?)\n```'],
        "Go": [r'```go\n(.*?)\n```', r'```golang\n(.*?)\n```'],
        "C#": [r'```csharp\n(.*?)\n```', r'```cs\n(.*?)\n```', r'```c#\n(.*?)\n```']
    }
    
    # Try language-specific patterns first
    if language in language_patterns:
        for pattern in language_patterns[language]:
            code_blocks = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if code_blocks:
                return code_blocks[0]
    
    # Fallback to generic code blocks
    code_blocks = re.findall(r'```\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    return response_text

def extract_python_code(response_text):
    """Extract Python code from response (backward compatibility)"""
    return extract_code_from_response(response_text, "Python")

def filter_model_response(response_text, model_name):
    """Filter out any unwanted parts from model responses"""
    # Currently only handles DeepSeek R1 thinking parts, but can be extended for other models
    if "DeepSeek-R1" not in model_name:
        return response_text
    
    # Remove thinking tags and content (case insensitive)
    filtered_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any remaining thinking markers
    filtered_text = re.sub(r'<thinking>.*?</thinking>', '', filtered_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove any other thinking patterns
    filtered_text = re.sub(r'<Think>.*?</Think>', '', filtered_text, flags=re.DOTALL)
    
    # Clean up extra whitespace
    filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text.strip())
    
    return filtered_text

def extract_code_with_fallback(response_text, model_name, language="Python"):
    """Extract code with fallback LLM if extraction has issues"""
    # First try normal extraction
    code = extract_code_from_response(response_text, language)
    
    # If no code found, try using another LLM to extract
    if not code:
        extraction_prompt = f"""
        Extract ONLY the {language} code from this response. Remove any thinking parts or explanations.
        Return ONLY the clean {language} code that can be executed directly.
        
        Response to extract from:
        {response_text}
        
        Provide only the {language} code, nothing else.
        """
        
        # Try GPT-5, then GPT-4.1, then other models for extraction
        for extraction_model in [GPT5_MODEL_NAME, GPT41_MODEL_NAME, LLAMA_MODEL, CODESTRAL_MODEL]:
            try:
                extracted_response = call_ai_model(
                    extraction_prompt, 
                    extraction_model, 
                    f"You are a code extraction specialist. Return only clean {language} code.",
                    max_tokens=4000
                )
                if extracted_response and not str(extracted_response).startswith("Error calling"):
                    extracted_code = extract_code_from_response(extracted_response, language)
                    if extracted_code:
                        return extracted_code
            except:
                continue
    
    return code

def run_code(code, language="Python"):
    """Execute code in the specified programming language and capture output"""
    try:
        import tempfile
        import uuid
        
        # Define file extensions and execution commands for different languages
        language_config = {
            "Python": {
                "extension": ".py",
                "command": [sys.executable],
                "compile": False
            },
            "Java": {
                "extension": ".java",
                "command": ["java"],
                "compile": True,
                "compile_cmd": ["javac"]
            },
            "C++": {
                "extension": ".cpp",
                "command": ["./temp_solution"],
                "compile": True,
                "compile_cmd": ["g++", "-o", "temp_solution"]
            },
            "Go": {
                "extension": ".go",
                "command": ["go", "run"],
                "compile": False
            },
            "C#": {
                "extension": ".cs",
                "command": ["dotnet", "run"],
                "compile": False
            }
        }
        
        if language not in language_config:
            return "", f"Unsupported language: {language}", 1
            
        config = language_config[language]
        temp_file = f'temp_solution_{uuid.uuid4().hex[:8]}{config["extension"]}'
        
        # Handle Java class name extraction
        if language == "Java":
            # Extract class name from code
            import re
            class_match = re.search(r'public\s+class\s+(\w+)', code)
            if class_match:
                class_name = class_match.group(1)
                temp_file = f'{class_name}.java'
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Compile if needed
        command = config["command"] + [temp_file]  # Default command
        
        if config.get("compile", False):
            if language == "C++":
                compile_result = subprocess.run(
                    config["compile_cmd"] + [temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8'
                )
                if compile_result.returncode != 0:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    return "", f"Compilation error: {compile_result.stderr}", 1
                command = config["command"]  # Use compiled executable
                    
            elif language == "Java":
                compile_result = subprocess.run(
                    config["compile_cmd"] + [temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8'
                )
                if compile_result.returncode != 0:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    return "", f"Compilation error: {compile_result.stderr}", 1
                
                # For Java, run the compiled class
                class_name = temp_file.replace('.java', '')
                command = config["command"] + [class_name]
        
        # Execute the code
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8'
        )
        
        # Clean up files
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if language == "C++" and os.path.exists("temp_solution"):
            os.remove("temp_solution")
        if language == "Java":
            class_file = temp_file.replace('.java', '.class')
            if os.path.exists(class_file):
                os.remove(class_file)
        
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Execution timed out (30 seconds)", 1
    except FileNotFoundError as e:
        return "", f"Compiler/Runtime not found: {str(e)}. Please ensure {language} is installed.", 1
    except Exception as e:
        return "", str(e), 1

def run_python_code(code):
    """Execute Python code and capture output (backward compatibility)"""
    return run_code(code, "Python")

def deep_problem_analysis(problem_text, model_name):
    """Deep analysis of the problem using specified model"""
    system_message = "You are an expert competitive programming analyst. Provide detailed, structured analysis."
    
    prompt = f"""
    Perform a DEEP ANALYSIS of this DSA problem. Be extremely thorough.

    PROBLEM:
    {problem_text}

    Provide comprehensive analysis in this exact JSON format:
    {{
        "problem_type": "Specific problem category (e.g., Dynamic Programming, Graph, Array, etc.)",
        "difficulty_level": "Estimated difficulty (Easy/Medium/Hard)",
        "key_insights": [
            "Critical insight 1",
            "Critical insight 2",
            "Critical insight 3"
        ],
        "algorithm_approaches": [
            {{
                "name": "Approach name",
                "time_complexity": "O(n) notation",
                "space_complexity": "O(n) notation",
                "pros": "Advantages",
                "cons": "Disadvantages"
            }}
        ],
        "edge_cases": [
            "Specific edge case 1",
            "Specific edge case 2",
            "Specific edge case 3"
        ],
        "input_constraints": {{
            "size_limits": "Array/string size limits",
            "value_ranges": "Number value ranges",
            "special_conditions": "Any special input conditions"
        }},
        "optimal_approach": {{
            "algorithm": "Best algorithm name",
            "reason": "Why this approach is optimal",
            "implementation_hints": [
                "Hint 1 for implementation",
                "Hint 2 for implementation"
            ]
        }}
    }}

    Be specific and actionable in your analysis.
    """
    
    return call_ai_model(prompt, model_name, system_message, max_tokens=6000)

def generate_solution_with_model(problem_text, analysis, model_name, iteration_num, language="Python"):
    """Generate solution using specific model with iteration-aware prompting"""
    
    system_messages = {
        GPT41_MODEL_NAME: f"You are GPT-4.1, an expert competitive programming assistant focused on correctness and optimal solutions in {language}.",
        GPT5_MODEL_NAME: f"You are GPT-5, an advanced expert competitive programming assistant with superior analytical capabilities and {language} solution generation.",
        LLAMA_MODEL: f"You are Meta-Llama-3.1-405B, an expert algorithm designer focused on correctness and efficiency in {language}.",
        CODESTRAL_MODEL: f"You are Codestral-2501, specialized in generating clean, bug-free {language} code for competitive programming.",
        DEEPSEEK_MODEL: f"You are DeepSeek-R1, an advanced reasoning model with superior problem-solving capabilities in {language}."
    }
    
    iteration_context = {
        1: "This is the FIRST iteration. Focus on creating a solid, well-tested foundation.",
        2: "This is the SECOND iteration. The previous attempt likely had basic issues. Focus on correctness.",
        3: "This is the THIRD iteration. Previous attempts failed. Be more careful with edge cases and algorithm logic.",
        4: "This is FOURTH iteration. Multiple failures occurred. Completely rethink the approach.",
        5: "This is FIFTH+ iteration. Previous approaches failed significantly. Use a different algorithmic strategy."
    }
    
    # Language-specific code structure templates
    code_templates = {
        "Python": """
```python
def solve(params):
    # Your solution logic here
    return result

# Test cases
def test():
    tests = [
        # 10 test cases with expected results
    ]
    f = 0
    for i, (inp, exp) in enumerate(tests):
        res = solve(*inp) if isinstance(inp, tuple) else solve(inp)
        if res == exp:
            print(f"Test {{i+1}}: PASS")
        else:
            print(f"Test {{i+1}}: FAIL Expected={{exp}} Got={{res}}")
            f += 1
    print(f"Failed: {{f}}")

test()
```""",
        
        "Java": """
```java
public class Solution {
    public static ReturnType solve(params) {
        // Your solution logic here
        return result;
    }
    
    public static void main(String[] args) {
        // Test cases
        Object[][] tests = {
            // 10 test cases with expected results
        };
        int f = 0;
        for (int i = 0; i < tests.length; i++) {
            Object result = solve(/* extract params from tests[i] */);
            Object expected = /* extract expected from tests[i] */;
            if (result.equals(expected)) {
                System.out.println("Test " + (i+1) + ": PASS");
            } else {
                System.out.println("Test " + (i+1) + ": FAIL Expected=" + expected + " Got=" + result);
                f++;
            }
        }
        System.out.println("Failed: " + f);
    }
}
```""",
        
        "C++": """
```cpp
#include <iostream>
#include <vector>
// Add other necessary headers

ReturnType solve(params) {
    // Your solution logic here
    return result;
}

int main() {
    // Test cases
    vector<pair<InputType, ExpectedType>> tests = {
        // 10 test cases with expected results
    };
    int f = 0;
    for (int i = 0; i < tests.size(); i++) {
        auto result = solve(tests[i].first);
        auto expected = tests[i].second;
        if (result == expected) {
            cout << "Test " << (i+1) << ": PASS" << endl;
        } else {
            cout << "Test " << (i+1) << ": FAIL Expected=" << expected << " Got=" << result << endl;
            f++;
        }
    }
    cout << "Failed: " << f << endl;
    return 0;
}
```""",
        
        "Go": """
```go
package main

import "fmt"

func solve(params) ReturnType {
    // Your solution logic here
    return result
}

func main() {
    // Test cases
    tests := []struct {
        input    InputType
        expected ExpectedType
    }{
        // 10 test cases with expected results
    }
    
    f := 0
    for i, test := range tests {
        result := solve(test.input)
        if result == test.expected {
            fmt.Printf("Test %d: PASS\\n", i+1)
        } else {
            fmt.Printf("Test %d: FAIL Expected=%v Got=%v\\n", i+1, test.expected, result)
            f++
        }
    }
    fmt.Printf("Failed: %d\\n", f)
}
```""",
        
        "C#": """
```csharp
using System;

public class Solution 
{
    public static ReturnType Solve(params)
    {
        // Your solution logic here
        return result;
    }
    
    public static void Main()
    {
        // Test cases
        var tests = new (InputType input, ExpectedType expected)[]
        {
            // 10 test cases with expected results
        };
        
        int f = 0;
        for (int i = 0; i < tests.Length; i++)
        {
            var result = Solve(tests[i].input);
            var expected = tests[i].expected;
            if (result.Equals(expected))
            {
                Console.WriteLine($"Test {i+1}: PASS");
            }
            else
            {
                Console.WriteLine($"Test {i+1}: FAIL Expected={expected} Got={result}");
                f++;
            }
        }
        Console.WriteLine($"Failed: {f}");
    }
}
```"""
    }
    
    context = iteration_context.get(iteration_num, iteration_context[5])
    
    prompt = f"""
    {context}

    PROBLEM:
    {problem_text}

    ANALYSIS:
    {analysis}

    CRITICAL REQUIREMENTS:
    1. Generate COMPLETE {language} code that ACTUALLY WORKS
    2. Include exactly 10 comprehensive test cases covering ALL edge cases
    3. Use appropriate variable naming conventions for {language}
    4. NO comments in the code whatsoever
    5. Each test case must print "Test X: PASS" or "Test X: FAIL" with expected vs actual
    6. Include final summary: "Failed: X" or "All tests passed"
    7. Handle ALL edge cases from the analysis
    8. Make sure the algorithm is CORRECT, not just syntactically valid
    9. Follow {language} best practices and conventions

    CODE STRUCTURE ({language}):
    {code_templates.get(language, code_templates["Python"])}

    Focus on algorithmic correctness. Make it WORK, not just compile.
    Use {language} syntax and idioms properly.
    """
    
    return call_ai_model(prompt, model_name, system_messages.get(model_name, f"You are an expert {language} programmer."), max_tokens=8000)

def analyze_failure_patterns(iteration_history):
    """Analyze failure patterns across iterations to provide strategic insights"""
    if not iteration_history:
        return "No iteration history available"
    
    analysis = "FAILURE PATTERN ANALYSIS:\n"
    analysis += f"Total iterations attempted: {len(iteration_history)}\n\n"
    
    # Analyze test case progression
    test_progressions = []
    for hist in iteration_history:
        stdout = hist.get('stdout', '')
        if stdout:
            passed = len([line for line in stdout.split('\n') if 'PASS' in line and 'Test' in line])
            failed = len([line for line in stdout.split('\n') if 'FAIL' in line and 'Test' in line])
            test_progressions.append((passed, failed))
    
    if test_progressions:
        analysis += "TEST CASE PROGRESSION:\n"
        for i, (passed, failed) in enumerate(test_progressions):
            analysis += f"Iteration {i+1}: {passed} passed, {failed} failed\n"
        
        # Check if we're making progress
        if len(test_progressions) > 1:
            first_passed = test_progressions[0][0]
            last_passed = test_progressions[-1][0]
            if last_passed <= first_passed:
                analysis += "\n⚠️ WARNING: No improvement in test cases. Algorithm approach may be fundamentally wrong.\n"
            else:
                analysis += f"\n✅ PROGRESS: Improved from {first_passed} to {last_passed} passing tests.\n"
    
    # Analyze error patterns
    error_patterns = {}
    for hist in iteration_history:
        stderr = hist.get('stderr', '')
        if stderr:
            if 'SyntaxError' in stderr:
                error_patterns['syntax'] = error_patterns.get('syntax', 0) + 1
            elif 'TypeError' in stderr:
                error_patterns['type'] = error_patterns.get('type', 0) + 1
            elif 'IndexError' in stderr:
                error_patterns['index'] = error_patterns.get('index', 0) + 1
            elif 'KeyError' in stderr:
                error_patterns['key'] = error_patterns.get('key', 0) + 1
    
    if error_patterns:
        analysis += "\nERROR PATTERNS:\n"
        for error_type, count in error_patterns.items():
            analysis += f"- {error_type} errors: {count} times\n"
    
    # Strategic recommendations
    analysis += "\nSTRATEGIC RECOMMENDATIONS:\n"
    if len(iteration_history) >= 3:
        if any('syntax' in hist.get('stderr', '').lower() for hist in iteration_history[-2:]):
            analysis += "- Focus on syntax correctness, use simpler code structure\n"
        
        recent_outputs = [hist.get('stdout', '') for hist in iteration_history[-2:]]
        if all('FAIL' in output for output in recent_outputs if output):
            analysis += "- Algorithm logic is consistently wrong, try different approach\n"
            analysis += "- Consider: 1) Different data structure, 2) Different algorithm, 3) Reverse thinking\n"
    
    return analysis

def choose_next_model(iteration_num, previous_models_used):
    """Strategically choose the next model based on iteration and previous usage"""
    # All available working models
    all_models = [GPT5_MODEL_NAME, GPT41_MODEL_NAME, LLAMA_MODEL, CODESTRAL_MODEL, DEEPSEEK_MODEL]
    
    model_strategy = {
        1: GPT5_MODEL_NAME,      # Start with GPT-5 for superior capabilities
        2: GPT41_MODEL_NAME,     # Fallback to GPT-4.1 for reliability
        3: LLAMA_MODEL,          # Switch to Llama for different approach
        4: CODESTRAL_MODEL,      # Codestral for code generation
        5: DEEPSEEK_MODEL,       # DeepSeek for advanced reasoning
        6: GPT5_MODEL_NAME,      # Back to GPT-5 with different strategy
        7: GPT41_MODEL_NAME,     # GPT-4.1 again
        8: LLAMA_MODEL,          # Llama again with more focus
    }
    
    # For iterations beyond 8, cycle through all models
    if iteration_num > 8:
        return all_models[(iteration_num - 1) % len(all_models)]
    
    return model_strategy.get(iteration_num, GPT5_MODEL_NAME)

def enhanced_solve_with_multi_llm(problem_text, max_iterations=10, language="Python"):
    """Enhanced solving with strategic multi-LLM approach and deep analysis"""
    
    # Language mapping for syntax highlighting
    language_mapping = {
        "Python": "python",
        "Java": "java", 
        "C++": "cpp",
        "Go": "go",
        "C#": "csharp"
    }
    
    # Step 1: Deep Problem Analysis
    st.subheader("🧠 Deep Problem Analysis")
    analysis_progress = st.progress(0)
    
    with st.spinner("Performing deep problem analysis..."):
        # Use GPT-5 first, then GPT-4.1 as fallback for analysis
        analysis_model = None
        analysis_result = None
        
        # Try GPT-5 first, then fallback to other models
        analysis_models = [GPT5_MODEL_NAME, GPT41_MODEL_NAME, LLAMA_MODEL]
        for model in analysis_models:
            analysis_result = deep_problem_analysis(problem_text, model)
            if analysis_result and not str(analysis_result).startswith("Error calling"):
                analysis_model = model
                break
        
        if not analysis_result or str(analysis_result).startswith("Error calling"):
            st.error("Failed to perform problem analysis")
            return
    
    analysis_progress.progress(1.0)
    
    # Display analysis
    st.success(f"✅ Analysis completed using {analysis_model}")
    with st.expander("📊 Detailed Problem Analysis", expanded=True):
        st.text_area("Analysis Result", analysis_result, height=400)
    
    # Step 2: Multi-LLM Iterative Solution Generation
    st.subheader("🚀 Multi-LLM Solution Generation")
    
    iteration_history = []
    current_code = None
    models_used = []
    
    # Create columns for real-time progress tracking
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("**🎯 Strategy Panel**")
        strategy_container = st.empty()
        model_container = st.empty()
        progress_container = st.empty()
    
    for iteration in range(1, max_iterations + 1):
        with col1:
            st.markdown(f"### 🔄 Iteration {iteration}")
            
            # Choose model strategically
            chosen_model = choose_next_model(iteration, models_used)
            models_used.append(chosen_model)
            
            # Update strategy panel
            with col2:
                strategy_container.markdown(f"**Current Strategy:**\nIteration {iteration}/{max_iterations}")
                model_container.markdown(f"**Using Model:**\n{chosen_model}")
                progress_container.progress(iteration / max_iterations)
            
            st.info(f"🤖 Using {chosen_model} for iteration {iteration}")
            
            # Generate solution with the chosen model
            with st.spinner(f"Generating solution with {chosen_model}..."):
                solution_response = generate_solution_with_model(
                    problem_text, analysis_result, chosen_model, iteration, language
                )
            
            if str(solution_response).startswith("Error calling"):
                st.error(f"Model call failed: {solution_response}")
                continue
            
            # Extract code from response
            filtered_response = filter_model_response(solution_response, chosen_model)
            current_code = extract_code_with_fallback(filtered_response, chosen_model, language)
            
            if not current_code:
                st.error("No code extracted from response")
                continue
            
            # Display code with language-specific syntax highlighting
            language_mapping = {
                "Python": "python",
                "Java": "java", 
                "C++": "cpp",
                "Go": "go",
                "C#": "csharp"
            }
            st.code(current_code, language=language_mapping.get(language, "python"), line_numbers=True)
            
            # Execute and analyze
            with st.spinner("Testing solution..."):
                stdout, stderr, returncode = run_code(current_code, language)
            
            # Store iteration results
            iteration_data = {
                'iteration': iteration,
                'model': chosen_model,
                'code': current_code,
                'stdout': stdout,
                'stderr': stderr,
                'returncode': returncode,
                'response': filtered_response  # Store filtered response instead of raw
            }
            iteration_history.append(iteration_data)
            
            # Display results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown("**📤 Output:**")
                if stdout:
                    st.text_area(f"Output {iteration}", stdout, height=200, key=f"out_{iteration}")
                    
                    # Check for success
                    failure_indicators = ["FAIL", "failed", "ERROR", "Exception"]
                    success_indicators = ["Failed: 0", "All tests passed"]
                    
                    has_failures = any(fail in stdout for fail in failure_indicators)
                    has_success = any(success in stdout for success in success_indicators)
                    
                    if has_success and not has_failures:
                        st.success("🎉 All test cases passed!")
                        
                        # Final solution display
                        st.subheader("🏆 FINAL SOLUTION")
                        st.code(current_code, language=language_mapping.get(language, "python"), line_numbers=True)
                        
                        # Set appropriate file extension
                        file_extensions = {
                            "Python": ".py",
                            "Java": ".java",
                            "C++": ".cpp",
                            "Go": ".go",
                            "C#": ".cs"
                        }
                        file_ext = file_extensions.get(language, ".py")
                        
                        st.download_button(
                            label="📥 Download Solution",
                            data=current_code,
                            file_name=f"enhanced_dsa_solution{file_ext}",
                            mime="text/plain"
                        )
                        
                        # Success metrics
                        st.subheader("📊 Success Metrics")
                        st.metric("Iterations Used", iteration)
                        st.metric("Models Used", len(set(models_used)))
                        st.metric("Success Rate", f"{(1/iteration)*100:.1f}%")
                        
                        return
                    elif has_failures:
                        st.warning(f"⚠️ Test failures detected in iteration {iteration}")
                else:
                    st.info("No output generated")
            
            with result_col2:
                st.markdown("**❌ Errors:**")
                if stderr:
                    st.text_area(f"Errors {iteration}", stderr, height=200, key=f"err_{iteration}")
                else:
                    st.success("No errors!")
            
            # Deep analysis of failures
            if iteration > 1:
                with st.expander(f"🔍 Failure Analysis - Iteration {iteration}"):
                    failure_analysis = analyze_failure_patterns(iteration_history)
                    st.text(failure_analysis)
    
    # Final summary if max iterations reached
    st.subheader("📋 Final Summary")
    st.error(f"❌ Maximum iterations ({max_iterations}) reached without complete solution")
    
    # Show progression metrics
    if iteration_history:
        st.markdown("**📈 Progression Analysis:**")
        
        models_tried = list(set([h['model'] for h in iteration_history]))
        test_progression = []
        
        for hist in iteration_history:
            stdout = hist.get('stdout', '')
            if stdout:
                passed = len([line for line in stdout.split('\n') if 'PASS' in line and 'Test' in line])
                test_progression.append(passed)
            else:
                test_progression.append(0)
        
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("Models Tried", len(models_tried))
        with col_y:
            st.metric("Best Test Score", max(test_progression) if test_progression else 0)
        with col_z:
            st.metric("Final Test Score", test_progression[-1] if test_progression else 0)
        
        # Show best code if any tests passed
        if test_progression and max(test_progression) > 0:
            best_iteration = test_progression.index(max(test_progression))
            best_code = iteration_history[best_iteration]['code']
            
            st.subheader("🥈 Best Attempt")
            st.info(f"Iteration {best_iteration + 1} with {max(test_progression)} passing tests")
            st.code(best_code, language=language_mapping.get(language, "python"))
            
            # Set appropriate file extension for best attempt
            file_extensions = {
                "Python": ".py",
                "Java": ".java", 
                "C++": ".cpp",
                "Go": ".go",
                "C#": ".cs"
            }
            file_ext = file_extensions.get(language, ".py")
            
            st.download_button(
                label="📥 Download Best Attempt",
                data=best_code,
                file_name=f"best_attempt_solution{file_ext}",
                mime="text/plain"
            )

def main():
    st.title("🚀 Enhanced Multi-LLM DSA Solver with GPT-5")
    st.markdown("**Advanced AI Pipeline:** GPT-5 Analysis + Strategic Multi-Model Approach + Multi-Language Support")
    st.info("💻 **Supported Languages:** Python • Java • C++ • Go • C# | Choose your preferred language in the sidebar!")
    
    # Enhanced System status with troubleshooting info
    with st.expander("🔧 System Status & Troubleshooting"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if TESSERACT_AVAILABLE:
                st.success("✅ Tesseract OCR")
                st.caption(f"Debug: {tesseract_error}")
            else:
                st.warning("⚠️ OCR Unavailable")
                st.info("💡 **Solution:** Use manual input below - works perfectly!")
                with st.expander("🔍 OCR Troubleshooting (Optional)"):
                    st.markdown("""
                    **Tesseract isn't installed on this Streamlit Cloud instance.**
                    
                    **Quick Fix Options:**
                    1. **Use Manual Input** ← Recommended! Works perfectly
                    2. Try redeploying the app (may fix package installation)
                    3. Contact Streamlit Cloud support if OCR is critical
                    
                    **Your app works fully without OCR!**
                    All AI models and DSA solving features are available.
                    """)
                    st.code(f"Debug info: {tesseract_error[:200]}...", language="text")
                with st.expander("� Tesseract Fix Steps"):
                    st.markdown("""
                    **For Streamlit Cloud:**
                    1. Ensure `packages.txt` contains:
                       ```
                       tesseract-ocr
                       tesseract-ocr-eng
                       ```
                    2. Check deployment logs for package installation errors
                    3. Try redeploying the app
                    4. If still fails, OCR will be disabled but manual input works
                    """)
        with col2:
            st.success("✅ OpenCV" if CV2_AVAILABLE else "⚠️ Basic Processing")
        with col3:
            if AZURE_AI_AVAILABLE:
                st.success("✅ Azure AI")
            else:
                st.error("❌ Azure AI")
                st.info("📝 **Fix:** Check requirements_enhanced.txt has azure-ai-inference")
        with col4:
            client = get_inference_client()
            if client:
                st.success("✅ Models Ready")
            else:
                st.error("❌ Models Failed")
                st.info("📝 **Fix:** Add Azure credentials to Streamlit Cloud Secrets")
        
        # Troubleshooting section
        if not TESSERACT_AVAILABLE or not AZURE_AI_AVAILABLE or not get_inference_client():
            st.markdown("### 🆘 **Troubleshooting for Streamlit Cloud:**")
            
            if not TESSERACT_AVAILABLE:
                st.markdown("""
                **❌ Tesseract OCR Issue:**
                1. Ensure `packages.txt` is in your repository root
                2. Content should be:
                   ```
                   tesseract-ocr
                   tesseract-ocr-eng
                   libtesseract-dev
                   libleptonica-dev
                   ```
                3. Redeploy the app after adding the file
                """)
            
            if not AZURE_AI_AVAILABLE:
                st.markdown("""
                **❌ Azure AI Issue:**
                1. Check `requirements_enhanced.txt` includes:
                   ```
                   azure-ai-inference>=1.0.0b4
                   azure-core>=1.29.0
                   azure-identity>=1.15.0
                   ```
                2. Redeploy after updating requirements
                """)
            
            if not get_inference_client():
                st.markdown("""
                **❌ Models Failed:**
                1. Go to Streamlit Cloud app settings → Secrets
                2. Add these secrets:
                   ```toml
                   AZURE_API_KEY = "your-azure-api-key-here"
                   INFERENCE_ENDPOINT = "https://your-endpoint.services.ai.azure.com/models"
                   LLAMA_MODEL = "Meta-Llama-3.1-405B-Instruct"
                   CODESTRAL_MODEL = "Codestral-2501"
                   DEEPSEEK_MODEL = "DeepSeek-R1-0528"
                   GPT5_ENDPOINT = "https://your-gpt5-endpoint.openai.azure.com/"
                   GPT5_API_KEY = "your-gpt5-api-key-here"
                   GPT5_API_VERSION = "2024-12-01-preview"
                   GPT41_ENDPOINT = "https://your-gpt41-endpoint.cognitiveservices.azure.com/"
                   GPT41_API_KEY = "your-gpt41-api-key-here"
                   GPT41_API_VERSION = "2024-12-01-preview"
                   ```
                3. Restart the app
                """)
        
        # Show required secrets configuration
        if not AZURE_AI_AVAILABLE or not get_inference_client():
            st.markdown("### 📋 **Required Streamlit Cloud Secrets Configuration:**")
            st.code("""
# Add these to your Streamlit Cloud app secrets:
AZURE_API_KEY = "your-azure-inference-api-key"
INFERENCE_ENDPOINT = "https://your-endpoint.services.ai.azure.com/models"
GPT5_ENDPOINT = "https://your-gpt5-endpoint.openai.azure.com/"
GPT5_API_KEY = "your-gpt5-api-key"
GPT41_ENDPOINT = "https://your-gpt41-endpoint.cognitiveservices.azure.com/"
GPT41_API_KEY = "your-gpt41-api-key"

# Optional (have defaults):
LLAMA_MODEL = "Meta-Llama-3.1-405B-Instruct"
CODESTRAL_MODEL = "Codestral-2501" 
DEEPSEEK_MODEL = "DeepSeek-R1-0528"
GPT5_API_VERSION = "2024-12-01-preview"
GPT41_API_VERSION = "2024-12-01-preview"
            """, language="toml")
    
    # Model availability check - only run once per session
    if AZURE_OPENAI_AVAILABLE and (get_gpt5_client() or get_gpt41_client()):
        if 'model_status_checked' not in st.session_state:
            st.subheader("🤖 Model Status Check")
            col1, col2, col3 = st.columns(3)
            
            test_models = [GPT5_MODEL_NAME, GPT41_MODEL_NAME, LLAMA_MODEL]
            for i, model in enumerate(test_models):
                with [col1, col2, col3][i]:
                    with st.spinner(f"Testing {model}..."):
                        test_result = call_ai_model("Hello", model, "You are a test assistant", max_tokens=10)
                        if test_result and not str(test_result).startswith("🔴") and not str(test_result).startswith("Error"):
                            st.success(f"✅ {model}")
                        else:
                            st.error(f"❌ {model}")
                            st.caption(f"Error: {str(test_result)[:100]}...")
            
            # Mark that status check has been completed
            st.session_state['model_status_checked'] = True
            
        else:
            # Show a simple status indicator without re-testing
            st.subheader("🤖 Model Status")
            st.info("✅ Models initialized successfully. Status checked at page load.")
        
        # Show additional available models
        with st.expander("🔍 Available Models"):
            st.success("✅ GPT-5: Advanced problem analysis and solution generation")
            st.success("✅ GPT-4.1: Reliable coding and problem solving")
            st.info("✅ Meta-Llama-3.1-405B: Open source reasoning model")
            st.info("✅ Codestral-2501: Code generation specialist")
            st.info("✅ DeepSeek-R1-0528: Advanced reasoning model")
    
    elif AZURE_OPENAI_AVAILABLE:
        st.warning("⚠️ Azure OpenAI available but client initialization failed - check credentials")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_iterations = st.slider(
            "Max Iterations", 
            min_value=3, 
            max_value=20, 
            value=10,
            help="Maximum number of solution attempts with different models"
        )
        
        # Programming Language Selection
        st.header("💻 Programming Language")
        programming_language = st.selectbox(
            "Select Programming Language:",
            options=["Python", "Java", "C++", "Go", "C#"],
            index=0,  # Default to Python
            help="Choose the programming language for code generation"
        )
        
        st.header("🎯 Strategy Options")
        show_detailed = st.checkbox("Show Detailed Analysis", value=True)
        show_code_progression = st.checkbox("Show Code Progression", value=True)
        
        st.header("📝 Manual Input")
        manual_text = st.text_area(
            "Enter DSA Problem:",
            placeholder="Type or paste your problem here...",
            height=150
        )
        
        if manual_text and st.button("🚀 Solve with Enhanced Multi-LLM"):
            st.session_state['solve_enhanced'] = manual_text
            st.session_state['programming_language'] = programming_language
    
    # Check for manual solve
    if 'solve_enhanced' in st.session_state:
        problem_text = st.session_state['solve_enhanced']
        selected_language = st.session_state.get('programming_language', 'Python')
        del st.session_state['solve_enhanced']
        if 'programming_language' in st.session_state:
            del st.session_state['programming_language']
        
        st.subheader("📝 Problem Input")
        st.text_area("Problem Text", problem_text, height=150)
        st.info(f"🔧 Selected Language: **{selected_language}**")
        
        enhanced_solve_with_multi_llm(problem_text, max_iterations, selected_language)
        return
    
    # OCR File uploader
    st.subheader("📸 Upload Problem Images")
    uploaded_files = st.file_uploader(
        "Upload problem images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.subheader("📸 Uploaded Images")
        
        cols = st.columns(min(len(uploaded_files), 3))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image {i+1}", use_column_width=True)
        
        if st.button("🔍 Extract Text and Solve with Enhanced Multi-LLM", type="primary"):
            all_text = ""
            
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                
                with st.spinner(f"Extracting text from Image {i+1}..."):
                    text = extract_text_from_image(image)
                
                if text and not text.startswith("OCR_PLACEHOLDER"):
                    st.success(f"✅ Text extracted from Image {i+1}")
                    with st.expander(f"📝 Text from Image {i+1}"):
                        st.text(text)
                    all_text += f"\n\nImage {i+1}:\n{text}"
                else:
                    st.warning(f"⚠️ No text found in Image {i+1}")
            
            if all_text.strip():
                st.subheader("🔗 Combined Problem Text")
                st.text_area("Extracted Text", all_text, height=200)
                
                # Use the same language selection as in sidebar, default to Python
                try:
                    current_language = programming_language if programming_language else "Python"
                except NameError:
                    current_language = "Python"
                
                enhanced_solve_with_multi_llm(all_text, max_iterations, current_language)
            else:
                st.error("❌ No text extracted. Use manual input instead.")

if __name__ == "__main__":
    main()
