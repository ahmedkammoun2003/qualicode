import os  # This helps our program talk to the computer’s file system. Like finding your toy box, opening it, and putting toys (files) in or taking them out.
import json  # This helps us read and write files full of structured information (like a list of your favorite toys in a box called .json).
import glob  # This lets us look around and find lots of files that match a pattern. Like saying “find all the red toy cars in the room!”
import torch  # This is usually used to make AI models smart. We bring it to the party but we don’t really use it in this code. It’s like an extra toy no one plays with.
import time  # This tells the program what time it is and helps it measure how long things take. Like setting a timer when we wait for cookies to bake. 🍪
import aiohttp  # This helps us talk to websites (like sending a letter and getting a response, but with the internet).
import asyncio  # This lets our program do many things at once! Like listening to a story while playing with blocks.
from tqdm import tqdm  # This gives us a cool progress bar, but we’re not playing with it right now.

# 🎯 This is our helper function. It talks to the internet and waits for it to talk back.
async def fetch(session, url, payload, problem_id, save_path):
    try:
        # 🚀 We send our request to the internet with a package full of info (payload).
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=3000000)) as response:
            # 📬 We wait for messages to come back slowly, one piece at a time.
            async for line in response.content:
                # 🧩 We turn the internet's words (text) into something our program can understand (a Python dictionary).
                data = json.loads(line.decode("utf-8"))
                
                # 💌 We check: is there a message? Is there any useful content?
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]  # 📝 Get the message text!
                    print(content, end="", flush=True)  # 📣 Show it on the screen immediately, so we’re not waiting.

                    # 🛠️ We clean the message to make it safe to write to a file.
                    formatted_paragraph = content.replace('\n', r'\n')  # Pretend new lines are plain text
                    formatted_paragraph = formatted_paragraph.replace("\\", r"\\")  # Make backslashes safe
                    formatted_paragraph = formatted_paragraph.replace('"', r'\"')  # Make double quotes safe
                    
                    # 📁 Now we open a file with the name based on the problem ID and add our content there.
                    with open(os.path.join(save_path, f"{problem_id}_plans.txt"), 'a', encoding='utf-8') as f:
                        f.write(content + "\n")  # 🖊️ We write the message down like taking notes.

    except asyncio.TimeoutError:  # 🕓 If we wait too long and get no answer...
        print("Request timed out")  # 📢 We tell ourselves, “it took too long!”

# 🧠 This part looks at one problem (like one puzzle), reads its files, and asks for help to solve it step by step.
async def process_problem(session, problem, args):
    start_time = time.time()  # ⏱️ Start the timer to see how long it takes
    prob_path = problem  # 🗂️ This is the folder with our problem inside
    print(f"\nProcessing problem: {prob_path}\n")  # 📣 Tell us what problem we're working on

    problem_id = int(os.path.basename(problem))  # 🏷️ Grab the number at the end of the folder name, like Problem #42
    
    # 🛑 If we already did this problem before (file exists), we skip it!
    if os.path.exists(os.path.join(args.save_path, f"{problem_id}.txt")):
        return
    
    code_path = os.path.join(prob_path, "solutions.json")  # 📦 This is the file where all our possible solutions are stored
    if not os.path.exists(code_path):  # 😢 If there’s no code there, we can’t work with it
        return
    
    with open(code_path, 'r', encoding='utf-8') as f:
        code_list = json.load(f)  # 📖 Open the box and see all the solutions inside
    min_code = min(code_list, key=lambda x: len(x))  # 🧠 Pick the shortest code (the easiest one to understand)

    question_file_path = os.path.join(prob_path, "question.txt")  # 📃 This file has the question or problem text
    with open(question_file_path, 'r', encoding='utf-8') as f:
        prompt_plan = f.read()  # 🗣️ We read the question like reading instructions

    # 🧩 Now we make the final input that we’ll send to the AI model
    input_text = "code:\n" + min_code + "Write a step-by-step solution plan following the above code:\n"
    
    payload = {
        "model": "deepseek-r1:7b",  # 🤖 This is the name of the smart assistant we’re asking
        "reasoning_effort": 0,  # 🛋️ This tells it not to think too hard (keep it simple)
        "narrative_style": "concise",  # 🗨️ We want short and straight answers
        "messages": [  # 💌 This is like a conversation
            {"role": "system", "content": "You are an assistant that helps programmers understand the code step by step in short responses.You always respond in english."},
            {"role": "user", "content": prompt_plan + input_text + "make your response short straight to the point and only use the given code do not generate code or think about another algorithm"}
        ],
        "temperature": 0.2,  # ❄️ Keep the answers focused and not too random
        "stream": False  # 🚫 Don’t send the answer in small pieces, give it all at once
    }

    # 📞 Now we call our fetch function to talk to the model and get the plan
    await fetch(session, args.OLLAMA_API_URL, payload, problem_id, args.save_path)
    
    # ⏱️ Check how much time we spent solving this one
    elapsed_time = time.time() - start_time
    print(f"\nTime taken to process problem {problem_id}: {elapsed_time:.2f} seconds")

# 🧩 This is where we go through many problems one by one
async def main(args):
    if os.name == 'nt':  # 🪟 Special fix for Windows computers so they don’t break when doing many things at once
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    original_problems = glob.glob(os.path.join(args.test_path, '*'))  # 🔍 Look for all problem folders
    problems = sorted(original_problems)  # 📚 Put them in order
    
    # 😵 If the user gave us a bad starting number, we stop!
    if args.start > len(problems) or args.start < 0:
        print(f"Start index {args.start} > number of problems {len(problems)}")
        return

    # 📏 Decide which problems we’ll do: from start to end
    start = args.start
    end = len(problems) if args.end is None or args.end > len(problems) else args.end
    problems = problems[start:end]  # 🎯 Cut the list to just the ones we want
    
    async with aiohttp.ClientSession() as session:  # 🌐 Open our internet session
        for i, problem in enumerate(problems, 1):  # 🔁 Go through each problem one by one
            print(f"Processing {i}/{len(problems)}...")  # 📣 Show our progress
            await process_problem(session, problem, args)  # 🧠 Work on the problem

# 🏁 This is where everything starts running
if __name__ == "__main__":
    import argparse  # 🎙️ This helps us take commands from the user

    parser = argparse.ArgumentParser(description="Use Ollama to generate plan.")  # 📜 Set up how we talk to the script
    
    parser.add_argument("--test_path", default="", type=str, help='Path to test samples')  # 📁 Where are the problems?
    parser.add_argument("--save_path", default="", type=str, help='Path to save plans')  # 📂 Where do we save the answers?
    parser.add_argument("-s", "--start", default=0, type=int, help='Start index of test samples')  # 🏁 Where should we start?
    parser.add_argument("-e", "--end", default=5000, type=int, help='End index of test samples')  # 🛑 Where should we stop?
    parser.add_argument("-m", "--model", default="llama2", type=str, help='Ollama model name')  # 🧠 Which AI brain to use?
    parser.add_argument("-d", "--delay", default=1, type=int, help='Delay between requests in seconds')  # ⏲️ Wait time between calls
    parser.add_argument("-u", "--OLLAMA_API_URL", default="http://localhost:11434/api/chat", type=str, help='Ollama API URL')  # 🌐 Where is our smart model?

    args = parser.parse_args()  # 🛠️ Collect all the user’s input settings
    asyncio.run(main(args))  # 🧃 Let’s start the main show!
