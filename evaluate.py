import os
import torch
import textwrap
import traceback
import re
import subprocess
import unittest

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class CodeGenerator:
    def __init__(self, model_path="./models/final_checkpoint.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large-ntp-py")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-large-ntp-py")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # Sanity check
            test_output = self.generate_code("print('hello')")
            if not test_output or "placeholder" in test_output:
                raise RuntimeError("Model failed basic generation test")

        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def generate_code(self, prompt, max_length=1024, temperature=0.7):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    early_stopping=True,
                    num_beams=3
                )
            code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if not code.strip():
                raise ValueError("Empty code generated")
            return code
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return "def placeholder():\n    raise NotImplementedError('Code generation failed')"

    def generate_tests(self, function_description):
        prompt =  f"""
You are an expert Python developer. Your task is to write a complete unit test suite using the `unittest` module for the following function description:

{function_description}

Requirements:
- Use the built-in `unittest` framework.
- Create a test class that inherits from `unittest.TestCase`.
- Write multiple test methods starting with `test_`.
- Use clear and appropriate assertions.
- Ensure that the test file can be executed directly with `if __name__ == '__main__': unittest.main()`.

Only output valid Python code.
"""
        tests = self.generate_code(prompt, max_length=1024, temperature=0.7)
      

    @staticmethod
    def execute_code(code, test_code=""):
        safe_globals = {
            '__builtins__': {
                'print': print,
                'range': range,
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'isinstance': isinstance,
                'int': int,
                        'assert': AssertionError,
                'unittest': unittest
            }
        }
        try:
            full_code = CodeGenerator._prepare_code(code, test_code)
            compile(full_code, '<string>', 'exec')
            exec(full_code, safe_globals)
            return {"status": "success", "output": None}
        except SyntaxError as e:
            return {
                "status": "syntax_error",
                "error": f"Line {e.lineno}: {str(e)}",
                "traceback": traceback.format_exc()
            }
        except AssertionError as e:
            return {
                "status": "test_failure", 
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        except Exception as e:
            return {
                "status": "runtime_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    @staticmethod
    def _prepare_code(main_code, test_code):
        main_code = textwrap.dedent(main_code).strip()
        test_code = textwrap.dedent(test_code).strip()
        if not re.match(r'(def\s+\w+\(|class\s+\w+)', main_code):
            main_code = f"def generated_function():\n{textwrap.indent(main_code, '    ')}"
        if "except" not in main_code and "try" not in main_code:
            main_code = main_code.replace(
                "def generated_function():\n", 
                "def generated_function():\n    try:\n"
            ) + "\n    except Exception as e:\n        raise RuntimeError(f'Execution failed: {str(e)}')"

        if not test_code or "unittest" not in test_code:
            test_code = """
import unittest

class TestGeneratedFunction(unittest.TestCase):
    def test_basic_functionality(self):
        try:
            result = generated_function()
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Function execution failed: {str(e)}")

if __name__ == '__main__':
    unittest.main()
            """
        return f"{main_code}\n\n{test_code}"

    @staticmethod
    def save_to_file(content, filename):
        try:
            with open(filename, "w") as f:
                f.write(content)
            print(f"[✓] Saved to {filename}")
        except Exception as e:
            print(f"[✗] Failed to save {filename}: {str(e)}")

    @staticmethod
    def run_static_analysis(filename):
        print(f"\n=== Static Analysis for {filename} ===")

        def run_tool(tool, args):
            print(f"\n[{tool}]")
            try:
                subprocess.run(args, check=False)
            except Exception as e:
                print(f"{tool} failed: {e}")

        run_tool("Bandit", ["bandit", "-r", filename])
        run_tool("Pylint", ["pylint", filename])
        run_tool("Radon", ["radon", "cc", filename, "-a"])


def main():
    try:
        generator = CodeGenerator()

        print("\n=== Function Generator ===")
        prompt = input("Enter a function description (e.g., 'calculate factorial'): ")

        # Generate Code
        print("\n[+] Generating code...")
        code = generator.generate_code(prompt)
        print("\n--- Generated Code ---\n", code)

        # Save original
        raw_code_file = "raw_generated_code.py"
        generator.save_to_file(code, raw_code_file)

        # Edit
        if input("\nEdit the code? (y/n): ").lower() == 'y':
            print("\nEnter new code (Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            code = "\n".join(lines)
            print("\n--- Modified Code ---\n", code)

        code_file = "generated_code.py"
        generator.save_to_file(code, code_file)

        # Generate Tests
        print("\n[+] Generating tests...")
        tests = generator.generate_tests(prompt)
        print("\n--- Generated Tests ---\n", tests)

        # Save test before edit
        raw_test_file = "raw_tests_code.py"
        generator.save_to_file(tests, raw_test_file)

        # Edit tests
        if input("\nEdit the tests? (y/n): ").lower() == 'y':
            print("\nEnter new test code (Enter twice to finish):")
            test_lines = []
            while True:
                line = input()
                if line == "":
                    break
                test_lines.append(line)
            tests = "\n".join(test_lines)
            print("\n--- Modified Tests ---\n", tests)

        test_file = "tests_code.py"
        generator.save_to_file(tests, test_file)

        # Execute
        print("\n[+] Executing code...")
        result = generator.execute_code(code, tests)
        print("\n--- Execution Result ---")
        print("Status:", result["status"])
        if result["status"] != "success":
            print("Error:", result["error"])
            print("Traceback:", result["traceback"])

        # Run linters
        generator.run_static_analysis(code_file)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
