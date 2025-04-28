import os
import json
import glob
import time
import asyncio
from tqdm import tqdm
from openai import OpenAI

async def fetch(client, problem_id, prompt_plan, min_code, save_path):
    try:
        input_text = "code:\n" + min_code + "Write a step-by-step solution plan following the above code:\n"
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:11434",  # Replace with your site URL
            },
            model="google/gemini-2.0-flash-thinking-exp:free",
            messages=[
                {"role": "system", "content": "You are an assistant that helps programmers understand the code step by step in short responses.You always respond in english."},
                {"role": "user", "content": prompt_plan + input_text + "make your response short straight to the point and only use the given code do not generate code or think about another algorithm"}
            ],
            temperature=0.2
        )
        
        content = completion.choices[0].message.content
        if content:  # Check if content is not empty/null
            print(content, end="", flush=True)
            # Save response to a file
            with open(os.path.join(save_path, f"{problem_id}_plans.txt"), 'a', encoding='utf-8') as f:
                f.write(content + "\n")  # Append content to the file
    except Exception as e:
        print(f"Error processing problem {problem_id}: {str(e)}")

async def process_problem(client, problem, args):
    start_time = time.time()
    prob_path = problem
    print(f"\nProcessing problem: {prob_path}\n")

    problem_id = int(os.path.basename(problem))
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
    
    await fetch(client, problem_id, prompt_plan, min_code, args.save_path)
    
    elapsed_time = time.time() - start_time
    print(f"\nTime taken to process problem {problem_id}: {elapsed_time:.2f} seconds")

async def main(args):
    if os.name == 'nt':  # Windows specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    original_problems = glob.glob(os.path.join(args.test_path, '*'))
    problems = sorted(original_problems)
    
    if args.start > len(problems) or args.start < 0:
        print(f"Start index {args.start} > number of problems {len(problems)}")
        return
    
    start = args.start
    end = len(problems) if args.end is None or args.end > len(problems) else args.end
    problems = problems[start:end]
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=args.api_key,
    )
    
    for i, problem in enumerate(problems, 1):
        print(f"Processing {i}/{len(problems)}...")
        await process_problem(client, problem, args)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Use OpenRouter API to generate plan.")
    parser.add_argument("--test_path", default="", type=str, help='Path to test samples')
    parser.add_argument("--save_path", default="", type=str, help='Path to save plans')
    parser.add_argument("-s", "--start", default=0, type=int, help='Start index of test samples')
    parser.add_argument("-e", "--end", default=5000, type=int, help='End index of test samples')
    parser.add_argument("-m", "--model", default="deepseek/deepseek-coder-7b-instruct", type=str, help='OpenRouter model name')
    parser.add_argument("-d", "--delay", default=1, type=int, help='Delay between requests in seconds')
    parser.add_argument("--api_key", required=True, type=str, help='OpenRouter API key')
    
    args = parser.parse_args()
    
    asyncio.run(main(args))