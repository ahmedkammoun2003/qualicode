import os
import json
import glob
import torch
import time
import aiohttp
import asyncio
from tqdm import tqdm

async def fetch(session, url, payload, problem_id, save_path):
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=3000000)) as response:
            async for line in response.content:
                data = json.loads(line.decode("utf-8"))
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                    print(content, end="", flush=True)
                    formatted_paragraph = content.replace('\n', r'\n')
                    formatted_paragraph = formatted_paragraph.replace("\\", r"\\")
                    formatted_paragraph = formatted_paragraph.replace('"', r'\"')                    
                    # Save response to a file
                    with open(os.path.join(save_path, f"{problem_id}_plans.txt"), 'a', encoding='utf-8') as f:
                        f.write(content + "\n")  # Append content to the file
    except asyncio.TimeoutError:
        print("Request timed out")

async def process_problem(session, problem, args):
    start_time = time.time()
    prob_path = os.path.join(problem)
    print(f"\nProcessing problem: {prob_path}\n")

    problem_id = int(problem.split('\\')[-1])
    if os.path.exists(os.path.join(args.save_path, f"{problem_id}.txt")):
        return
    
    code_path = os.path.join(prob_path, "solutions.json")
    if not os.path.exists(code_path):
        return
    
    with open(code_path, 'r', encoding='utf-8') as f:
        code_list = json.load(f)
    min_code = min(code_list, key=lambda x: len(x))
    
    question_file_path = os.path.join(prob_path, "question.txt")
    with open(question_file_path, 'r', encoding='utf-8') as f:
        prompt_plan = f.read()
    
    input_text = "code:\n" + min_code + "Write a step-by-step solution plan following the above code:\n"
    
    payload = {
        "model": "deepseek-r1:32b",
        "reasoning_effort": 0,
        "narrative_style": "concise",
        "messages": [
            {"role": "system", "content": "You are an assistant that helps programmers understand the code step by step in short responses."},
            {"role": "user", "content": prompt_plan + input_text + "make your response short straight to the point and only use the given code do not generate code or think about another algorithm"}
        ],
        "temperature": 0.2,
        "stream": False
    }
    
    await fetch(session, args.OLLAMA_API_URL, payload, problem_id, args.save_path)
    
    elapsed_time = time.time() - start_time
    print(f"\nTime taken to process problem {problem_id}: {elapsed_time:.2f} seconds")

async def main(args):
    original_problems = glob.glob(args.test_path + '\\*')
    problems = sorted(original_problems)
    
    if args.start > len(problems) or args.start < 0:
        print(f"Start index {args.start} > number of problems {len(problems)}")
        return
    
    start = args.start
    end = len(problems) if args.end is None or args.end > len(problems) else args.end
    problems = problems[start:end]
    
    print(f"Original problems found: {original_problems}")
    print(f"Sorted problems: {problems}")
    
    async with aiohttp.ClientSession() as session:
        for i, problem in enumerate(problems, 1):
            print(f"Processing {i}/{len(problems)}...")
            await process_problem(session, problem, args)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Use Ollama to generate plan.")
    parser.add_argument("--test_path", default="", type=str, help='Path to test samples')
    parser.add_argument("--save_path", default="", type=str, help='Path to save plans')
    parser.add_argument("-s", "--start", default=0, type=int, help='Start index of test samples')
    parser.add_argument("-e", "--end", default=5000, type=int, help='End index of test samples')
    parser.add_argument("-m", "--model", default="llama2", type=str, help='Ollama model name')
    parser.add_argument("-d", "--delay", default=1, type=int, help='Delay between requests in seconds')
    parser.add_argument("-u", "--OLLAMA_API_URL", default="http://localhost:11434/api/chat", type=str, help='Ollama API URL')
    
    args = parser.parse_args()
    
    asyncio.run(main(args))
