import os
import subprocess
import sys


start = 0
end = 5000  

missing_files = set()

# Get API key from environment variable or use a default for testing
api_key = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY")

for i in range(start, end):
    filename = f"{i}_plans.txt"
    if not os.path.isfile(filename):
        missing_files.add(i)
        # Run the plan generation script with the current index
        subprocess.run([
            sys.executable, 
            "../generate_plans_from_LLM/generate_apps_plan.py",
            "--test_path", "../data",
            "--save_path", "../plans",
            "--start", str(i),
            "--end", str(i + 1),
            "--api_key", "api_key",
        ])

if missing_files:
    print(len(missing_files))
else:
    print("All files are present.")
