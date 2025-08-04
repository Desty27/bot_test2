import streamlit as st
import numpy as np
from PIL import Image
import os
import re
import subprocess
import sys
import json
from io import StringIO

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        'tesseract'
    ]
    
    tesseract_found = False
    for path in tesseract_paths:
        try:
            if path != "tesseract":
                pytesseract.pytesseract.tesseract_cmd = path
            pytesseract.get_tesseract_version()
            tesseract_found = True
            break
        except Exception:
            continue
    
    TESSERACT_AVAILABLE = tesseract_found
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
    AZURE_AI_AVAILABLE = True
except ImportError:
    AZURE_AI_AVAILABLE = False

# Azure Configuration
try:
    AZURE_API_KEY = st.secrets.get("AZURE_API_KEY", "EvC0NHV0wNV8WXaWuJ0YRkOMMcMsPzUjA7cCsQlO6n0TpUpqVRFeJQQJ99BGAC5T7U2XJ3w3AAAAACOGPQyF")
    INFERENCE_ENDPOINT = st.secrets.get("INFERENCE_ENDPOINT", "https://i-avi-mcu2s38r-francecentral.services.ai.azure.com/models")
    LLAMA_MODEL = st.secrets.get("LLAMA_MODEL", "Meta-Llama-3.1-405B-Instruct")
    CODESTRAL_MODEL = st.secrets.get("CODESTRAL_MODEL", "Codestral-2501")
    DEEPSEEK_MODEL = st.secrets.get("DEEPSEEK_MODEL", "DeepSeek-R1-0528")
    GPT4_MODELS = ["gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-4"]
except:
    AZURE_API_KEY = "EvC0NHV0wNV8WXaWuJ0YRkOMMcMsPzUjA7cCsQlO6n0TpUpqVRFeJQQJ99BGAC5T7U2XJ3w3AAAAACOGPQyF"
    INFERENCE_ENDPOINT = "https://i-avi-mcu2s38r-francecentral.services.ai.azure.com/models"
    LLAMA_MODEL = "Meta-Llama-3.1-405B-Instruct"
    CODESTRAL_MODEL = "Codestral-2501"
    DEEPSEEK_MODEL = "DeepSeek-R1-0528"
    GPT4_MODELS = ["gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-4"]

# Set page config
st.set_page_config(
    page_title="Enhanced Multi-LLM DSA Solver",
    page_icon="🚀",
    layout="wide"
)

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

def call_ai_model(prompt, model_name, system_message="You are an expert AI assistant.", max_tokens=4000):
    """Unified function to call any AI model"""
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

def extract_text_from_image(image):
    """Extract text from image using OCR"""
    if not TESSERACT_AVAILABLE:
        return "OCR_PLACEHOLDER_TEXT: Tesseract OCR not available. Please use manual input."
        
    try:
        if CV2_AVAILABLE:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_img = thresh
        else:
            processed_img = np.array(image.convert('L'))
        
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        return text.strip() if text.strip() else "No text detected in image"
    except Exception as e:
        return f"OCR Error: {str(e)}. Please use manual input instead."

def extract_python_code(response_text):
    """Extract Python code from response"""
    code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    code_blocks = re.findall(r'```\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    return response_text

def filter_deepseek_response(response_text, model_name):
    """Filter out thinking part from DeepSeek R1 responses"""
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

def extract_code_with_fallback(response_text, model_name):
    """Extract code with fallback LLM if DeepSeek has issues"""
    # First try normal extraction
    code = extract_python_code(response_text)
    
    # If no code found and it's DeepSeek, try using another LLM to extract
    if not code and "DeepSeek-R1" in model_name:
        extraction_prompt = f"""
        Extract ONLY the Python code from this response. Remove any thinking parts or explanations.
        Return ONLY the clean Python code that can be executed directly.
        
        Response to extract from:
        {response_text}
        
        Provide only the Python code, nothing else.
        """
        
        # Try GPT-4.1 for extraction
        for extraction_model in ["gpt-4.1", LLAMA_MODEL, CODESTRAL_MODEL]:
            try:
                extracted_response = call_ai_model(
                    extraction_prompt, 
                    extraction_model, 
                    "You are a code extraction specialist. Return only clean Python code.",
                    max_tokens=4000
                )
                if not extracted_response.startswith("Error calling"):
                    extracted_code = extract_python_code(extracted_response)
                    if extracted_code:
                        return extracted_code
            except:
                continue
    
    return code

def run_python_code(code):
    """Execute Python code and capture output"""
    try:
        import tempfile
        import uuid
        temp_file = f'temp_solution_{uuid.uuid4().hex[:8]}.py'
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8'
        )
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Execution timed out (30 seconds)", 1
    except Exception as e:
        return "", str(e), 1

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

def generate_solution_with_model(problem_text, analysis, model_name, iteration_num):
    """Generate solution using specific model with iteration-aware prompting"""
    
    # Handle combined DeepSeek + Codestral approach
    if model_name == "deepseek+codestral":
        return generate_combined_deepseek_codestral_solution(problem_text, analysis, iteration_num)
    
    system_messages = {
        "gpt-4.1": "You are GPT-4.1, an expert competitive programming assistant focused on correctness and optimal solutions.",
        LLAMA_MODEL: "You are Meta-Llama-3.1-405B, an expert algorithm designer focused on correctness and efficiency.",
        CODESTRAL_MODEL: "You are Codestral-2501, specialized in generating clean, bug-free Python code for competitive programming.",
        DEEPSEEK_MODEL: "You are DeepSeek-R1, an expert code reviewer and optimizer with deep understanding of algorithmic correctness."
    }
    
    iteration_context = {
        1: "This is the FIRST iteration. Focus on creating a solid, well-tested foundation.",
        2: "This is the SECOND iteration. The previous attempt likely had basic issues. Focus on correctness.",
        3: "This is the THIRD iteration. Previous attempts failed. Be more careful with edge cases and algorithm logic.",
        4: "This is FOURTH iteration. Multiple failures occurred. Completely rethink the approach.",
        5: "This is FIFTH+ iteration. Previous approaches failed significantly. Use a different algorithmic strategy."
    }
    
    context = iteration_context.get(iteration_num, iteration_context[5])
    
    prompt = f"""
    {context}

    PROBLEM:
    {problem_text}

    ANALYSIS:
    {analysis}

    CRITICAL REQUIREMENTS:
    1. Generate COMPLETE Python code that ACTUALLY WORKS
    2. Include exactly 10 comprehensive test cases covering ALL edge cases
    3. Use shorter variable names (a, b, c, i, j, k, n, m, etc.)
    4. NO comments in the code whatsoever
    5. Each test case must print "Test X: PASS" or "Test X: FAIL" with expected vs actual
    6. Include final summary: "Failed: X" or "All tests passed"
    7. Handle ALL edge cases from the analysis
    8. Make sure the algorithm is CORRECT, not just syntactically valid

    CODE STRUCTURE:
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
    ```

    Focus on algorithmic correctness. Make it WORK, not just compile.
    """
    
    return call_ai_model(prompt, model_name, system_messages.get(model_name, "You are an expert programmer."), max_tokens=8000)

def generate_combined_deepseek_codestral_solution(problem_text, analysis, iteration_num):
    """Combined approach: DeepSeek for analysis/approach + Codestral for implementation"""
    
    iteration_context = {
        1: "This is the FIRST iteration. Focus on creating a solid, well-tested foundation.",
        2: "This is the SECOND iteration. The previous attempt likely had basic issues. Focus on correctness.",
        3: "This is the THIRD iteration. Previous attempts failed. Be more careful with edge cases and algorithm logic.",
        4: "This is FOURTH iteration. Multiple failures occurred. Completely rethink the approach.",
        5: "This is FIFTH+ iteration. Previous approaches failed significantly. Use a different algorithmic strategy."
    }
    
    context = iteration_context.get(iteration_num, iteration_context[5])
    
    # Step 1: Get DeepSeek's algorithmic approach and strategy
    deepseek_prompt = f"""
    {context}

    PROBLEM:
    {problem_text}

    INITIAL ANALYSIS:
    {analysis}

    As DeepSeek-R1, provide a DETAILED ALGORITHMIC STRATEGY for solving this problem:

    1. APPROACH SELECTION: Which algorithm/technique is most suitable and why?
    2. STEP-BY-STEP ALGORITHM: Detailed pseudocode or algorithm steps
    3. EDGE CASES: What specific edge cases need handling?
    4. IMPLEMENTATION HINTS: Key implementation details and potential pitfalls
    5. TEST STRATEGY: What test cases would validate the solution?
    6. OPTIMIZATION OPPORTUNITIES: Performance considerations

    Provide a comprehensive strategy that a coding specialist can implement directly.
    Focus on CORRECTNESS and COMPLETENESS over brevity.
    """
    
    # Get DeepSeek's strategic analysis
    deepseek_response = call_ai_model(
        deepseek_prompt, 
        DEEPSEEK_MODEL, 
        "You are DeepSeek-R1, an expert algorithmic strategist. Provide detailed, actionable implementation strategies.",
        max_tokens=6000
    )
    
    if deepseek_response.startswith("Error calling"):
        return f"DeepSeek analysis failed: {deepseek_response}"
    
    # Filter out thinking parts from DeepSeek
    filtered_deepseek = filter_deepseek_response(deepseek_response, DEEPSEEK_MODEL)
    
    # Step 2: Pass DeepSeek's strategy to Codestral for implementation
    codestral_prompt = f"""
    {context}

    ORIGINAL PROBLEM:
    {problem_text}

    DEEPSEEK'S ALGORITHMIC STRATEGY:
    {filtered_deepseek}

    As Codestral-2501, implement the strategy provided by DeepSeek into WORKING Python code:

    CRITICAL IMPLEMENTATION REQUIREMENTS:
    1. Follow DeepSeek's algorithmic approach EXACTLY
    2. Generate COMPLETE Python code that ACTUALLY WORKS
    3. Include exactly 10 comprehensive test cases covering ALL edge cases mentioned by DeepSeek
    4. Use shorter variable names (a, b, c, i, j, k, n, m, etc.)
    5. NO comments in the code whatsoever
    6. Each test case must print "Test X: PASS" or "Test X: FAIL" with expected vs actual
    7. Include final summary: "Failed: X" or "All tests passed"
    8. Handle ALL edge cases identified by DeepSeek
    9. Implement any optimizations suggested by DeepSeek

    CODE STRUCTURE:
    ```python
    def solve(params):
        # Implement DeepSeek's algorithm here
        return result

    def test():
        tests = [
            # 10 test cases based on DeepSeek's test strategy
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
    ```

    Focus on translating DeepSeek's strategy into CORRECT, WORKING code.
    """
    
    # Get Codestral's implementation
    codestral_response = call_ai_model(
        codestral_prompt,
        CODESTRAL_MODEL,
        "You are Codestral-2501, an expert code implementation specialist. Convert algorithmic strategies into perfect working code.",
        max_tokens=8000
    )
    
    if codestral_response.startswith("Error calling"):
        return f"Codestral implementation failed: {codestral_response}"
    
    # Combine both responses for full context
    combined_response = f"""
DEEPSEEK ALGORITHMIC STRATEGY:
{filtered_deepseek}

---

CODESTRAL IMPLEMENTATION:
{codestral_response}
"""
    
    return combined_response

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
    model_strategy = {
        1: "gpt-4.1",        # Start with GPT-4.1 for strong foundation
        2: LLAMA_MODEL,      # Switch to Llama for algorithm design
        3: "deepseek+codestral",  # Combined DeepSeek analysis + Codestral code
        4: "gpt-4.1",        # Back to GPT-4.1 with different approach
        5: LLAMA_MODEL,      # Llama again with more focus
        6: "deepseek+codestral",  # Combined approach again
    }
    
    # For iterations beyond 6, cycle through models including combined approach
    if iteration_num > 6:
        models = ["gpt-4.1", LLAMA_MODEL, "deepseek+codestral"]
        return models[(iteration_num - 1) % len(models)]
    
    return model_strategy.get(iteration_num, LLAMA_MODEL)

def enhanced_solve_with_multi_llm(problem_text, max_iterations=10):
    """Enhanced solving with strategic multi-LLM approach and deep analysis"""
    
    # Step 1: Deep Problem Analysis
    st.subheader("🧠 Deep Problem Analysis")
    analysis_progress = st.progress(0)
    
    with st.spinner("Performing deep problem analysis..."):
        # Use the most capable model for analysis
        analysis_model = None
        analysis_result = None
        
        for model in GPT4_MODELS + [LLAMA_MODEL]:
            analysis_result = deep_problem_analysis(problem_text, model)
            if not analysis_result.startswith("Error calling"):
                analysis_model = model
                break
        
        if not analysis_result or analysis_result.startswith("Error calling"):
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
            
            # Generate solution
            if chosen_model == "deepseek+codestral":
                with st.spinner(f"Generating solution with DeepSeek strategy + Codestral implementation..."):
                    solution_response = generate_solution_with_model(
                        problem_text, analysis_result, chosen_model, iteration
                    )
            else:
                with st.spinner(f"Generating solution with {chosen_model}..."):
                    solution_response = generate_solution_with_model(
                        problem_text, analysis_result, chosen_model, iteration
                    )
            
            if solution_response.startswith("Error calling"):
                st.error(f"Model call failed: {solution_response}")
                continue
            
            # Filter DeepSeek R1 thinking parts and extract code
            if chosen_model == "deepseek+codestral":
                # For combined approach, extract code from Codestral part
                filtered_response = solution_response
                current_code = extract_python_code(solution_response)
            else:
                filtered_response = filter_deepseek_response(solution_response, chosen_model)
                current_code = extract_code_with_fallback(filtered_response, chosen_model)
            
            if not current_code:
                st.error("No code extracted from response")
                continue
            
            st.code(current_code, language='python', line_numbers=True)
            
            # Execute and analyze
            with st.spinner("Testing solution..."):
                stdout, stderr, returncode = run_python_code(current_code)
            
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
                        st.code(current_code, language='python', line_numbers=True)
                        
                        st.download_button(
                            label="📥 Download Solution",
                            data=current_code,
                            file_name="enhanced_dsa_solution.py",
                            mime="text/python"
                        )
                        
                        # Success metrics
                        st.subheader("📊 Success Metrics")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Iterations Used", iteration)
                        with col_b:
                            st.metric("Models Used", len(set(models_used)))
                        with col_c:
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
            st.code(best_code, language='python')
            
            st.download_button(
                label="📥 Download Best Attempt",
                data=best_code,
                file_name="best_attempt_solution.py",
                mime="text/python"
            )

def main():
    st.title("🚀 Enhanced Multi-LLM DSA Solver")
    st.markdown("**Advanced AI Pipeline:** Strategic Multi-Model Approach with Deep Analysis")
    
    # System status
    with st.expander("🔧 System Status"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success("✅ Tesseract OCR" if TESSERACT_AVAILABLE else "❌ Tesseract OCR")
        with col2:
            st.success("✅ OpenCV" if CV2_AVAILABLE else "⚠️ Basic Processing")
        with col3:
            st.success("✅ Azure AI" if AZURE_AI_AVAILABLE else "❌ Azure AI")
        with col4:
            client = get_inference_client()
            st.success("✅ Models Ready" if client else "❌ Models Failed")
    
    # Model availability check
    if AZURE_AI_AVAILABLE:
        st.subheader("🤖 Available Models")
        col1, col2, col3 = st.columns(3)
        
        test_models = [LLAMA_MODEL, CODESTRAL_MODEL, DEEPSEEK_MODEL]
        for i, model in enumerate(test_models):
            with [col1, col2, col3][i]:
                test_result = call_ai_model("Test", model, "Test", max_tokens=10)
                if not test_result.startswith("Error"):
                    st.success(f"✅ {model.split('-')[0]}")
                else:
                    st.error(f"❌ {model.split('-')[0]}")
    
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
    
    # Check for manual solve
    if 'solve_enhanced' in st.session_state:
        problem_text = st.session_state['solve_enhanced']
        del st.session_state['solve_enhanced']
        
        st.subheader("📝 Problem Input")
        st.text_area("Problem Text", problem_text, height=150)
        
        enhanced_solve_with_multi_llm(problem_text, max_iterations)
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
                enhanced_solve_with_multi_llm(all_text, max_iterations)
            else:
                st.error("❌ No text extracted. Use manual input instead.")

if __name__ == "__main__":
    main()
