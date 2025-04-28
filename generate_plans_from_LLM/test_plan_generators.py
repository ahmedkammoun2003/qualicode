import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from generate_apps_plan import process_problem, fetch
from generate_apps_plan_ollama import process_problem as process_problem_ollama

class TestPlanGenerators(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_problem_dir = os.path.join(self.temp_dir, "test_problem")
        os.makedirs(self.test_problem_dir)
        
        # Create mock files
        self.create_mock_files()
        
        # Mock arguments
        self.args = MagicMock()
        self.args.test_path = self.temp_dir
        self.args.save_path = self.temp_dir
        self.args.start = 0
        self.args.end = 1
        self.args.api_key = "test_key"
        
    def create_mock_files(self):
        # Create mock solutions.json
        solutions = ["def test(): return True"]
        with open(os.path.join(self.test_problem_dir, "solutions.json"), 'w') as f:
            json.dump(solutions, f)
            
        # Create mock question.txt
        with open(os.path.join(self.test_problem_dir, "question.txt"), 'w') as f:
            f.write("Test question")

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('generate_apps_plan.OpenAI')
    async def test_openai_plan_generation(self, mock_openai):
        # Mock OpenAI response
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Test plan"))]
        mock_openai.return_value.chat.completions.create.return_value = mock_completion
        
        # Test process_problem
        await process_problem(mock_openai(), self.test_problem_dir, self.args)
        
        # Check if plan file was created
        plan_file = os.path.join(self.temp_dir, "0_plans.txt")
        self.assertTrue(os.path.exists(plan_file))
        
    @patch('aiohttp.ClientSession')
    async def test_ollama_plan_generation(self, mock_session):
        # Mock aiohttp response
        mock_response = MagicMock()
        mock_response.content.__aiter__.return_value = [
            b'{"message": {"content": "Test plan"}}'
        ]
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        # Test process_problem_ollama
        await process_problem_ollama(mock_session, self.test_problem_dir, self.args)
        
        # Check if plan file was created
        plan_file = os.path.join(self.temp_dir, "0_plans.txt")
        self.assertTrue(os.path.exists(plan_file))

if __name__ == '__main__':
    unittest.main()