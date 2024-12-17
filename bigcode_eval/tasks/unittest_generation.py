"""Unit Test Generation Task
Tests the ability of language models to generate comprehensive unit tests
for given code while maintaining proper testing practices and coverage.
"""

from bigcode_eval.base import Task

class UnitTestGeneration(Task):
    DATASET_PATH = None
    
    def __init__(self):
        super().__init__(
            stop_words=["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"],
            requires_execution=True
        )
        
        self.test_data = [{
            "prompt": """Create comprehensive unit tests for this Python class. Include tests for all methods and edge cases.
Use unittest framework and follow proper testing practices. Test initialization, deposits, withdrawals, insufficient funds, and transaction history.

class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance
        self.transactions = []
        
    def deposit(self, amount):
        self.balance += amount
        self.transactions.append(f"Deposited {amount}")
        
    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds!")
            return False
        self.balance -= amount
        self.transactions.append(f"Withdrew {amount}")
        return True
        
    def check_balance(self):
        return self.balance
        
    def display_transactions(self):
        for transaction in self.transactions:
            print(transaction)""",
            
            "reference": """
import unittest
from io import StringIO
import sys

class TestBankAccount(unittest.TestCase):
    def setUp(self):
        # Initialize with an account number and a balance of 1000
        self.bank_account = BankAccount("12345", 1000)

    def test_initial_balance(self):
        # Test that the initial balance is set correctly
        self.assertEqual(self.bank_account.check_balance(), 1000)

    def test_deposit(self):
        # Test that deposit increases the balance correctly
        self.bank_account.deposit(500)
        self.assertEqual(self.bank_account.check_balance(), 1500)

    def test_withdraw(self):
        # Test that withdraw decreases the balance correctly
        success = self.bank_account.withdraw(200)
        self.assertTrue(success)
        self.assertEqual(self.bank_account.check_balance(), 800)

    def test_withdraw_insufficient_funds(self):
        # Test that withdraw does not allow withdrawing more than the balance
        success = self.bank_account.withdraw(2000)
        self.assertFalse(success)
        self.assertEqual(self.bank_account.check_balance(), 1000)

    def test_display_transactions(self):
        # Test that transactions are correctly displayed
        # Capture the print output
        captured_output = StringIO()
        sys.stdout = captured_output
        self.bank_account.deposit(500)
        self.bank_account.withdraw(300)
        self.bank_account.display_transactions()
        sys.stdout = sys.__stdout__  # Reset redirect
        # Check that the correct transaction messages are printed
        output = captured_output.getvalue().splitlines()
        self.assertIn("Deposited 500", output)
        self.assertIn("Withdrew 300", output)

    def test_multiple_transactions(self):
        # Test that multiple transactions are correctly processed
        self.bank_account.deposit(1000)
        self.bank_account.withdraw(300)
        self.bank_account.deposit(200)
        self.assertEqual(self.bank_account.check_balance(), 1900)
        self.assertEqual(len(self.bank_account.transactions), 3)

    def tearDown(self):
        # Clean up after tests
        self.bank_account = None

if __name__ == '__main__':
    unittest.main()""",
            
            "test": """
def check(candidate):
    # Test class structure
    class_elements = [
        "class TestBankAccount(unittest.TestCase)",
        "def setUp(self)",
        "def tearDown(self)",
        "if __name__ == '__main__':",
        "unittest.main()"
    ]
    
    # Test method definitions
    test_methods = [
        "test_initial_balance",
        "test_deposit",
        "test_withdraw",
        "test_withdraw_insufficient_funds",
        "test_display_transactions",
        "test_multiple_transactions"
    ]
    
    # Test imports
    required_imports = [
        "import unittest",
        "from io import StringIO",
        "import sys"
    ]
    
    # Test assertions
    assertions = [
        "assertEqual",
        "assertTrue",
        "assertFalse",
        "assertIn"
    ]
    
    # Test proper testing practices
    testing_practices = [
        "self.bank_account = BankAccount",
        "captured_output = StringIO()",
        "sys.stdout = captured_output",
        "sys.stdout = sys.__stdout__"
    ]
    
    # Calculate scores
    all_checks = {
        "structure": class_elements,
        "methods": test_methods,
        "imports": required_imports,
        "assertions": assertions,
        "practices": testing_practices
    }
    
    scores = {}
    total_checks = 0
    total_passed = 0
    
    for category, patterns in all_checks.items():
        passed = sum(1 for pattern in patterns if pattern in candidate)
        total = len(patterns)
        scores[category] = {
            "passed": passed,
            "total": total,
            "score": round((passed / total) * 100, 2)
        }
        total_checks += total
        total_passed += passed
    
    overall_score = round((total_passed / total_checks) * 100, 2)
    
    return {
        "unittest_score": overall_score,
        "details": {
            category: f"{info['passed']}/{info['total']} ({info['score']}%)"
            for category, info in scores.items()
        }
    }
"""
        }]

    def get_dataset(self):
        return self.test_data

    def get_prompt(self, doc):
        template = f"""<|start_of_role|>user<|end_of_role|>{doc["prompt"]}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""
        print(f"Prompt passed to model: {template}")
        return template

    def get_reference(self, doc):
        return doc["reference"]

    def postprocess_generation(self, generation, idx):
        prompt = self.get_prompt(self.test_data[idx])
        generation = generation[len(prompt):].strip()
        
        generation = generation.replace("<|end_of_text|>", "")
        generation = generation.replace("<|end_of_role|>", "")
        generation = generation.replace("```python", "")
        generation = generation.replace("```", "")
        
        if "import unittest" in generation:
            start_idx = generation.find("import unittest")
            generation = generation[start_idx:]
        elif "class TestBankAccount" in generation:
            start_idx = generation.find("class TestBankAccount")
            generation = generation[start_idx:]
            
        print(f"Code generated by model: {generation}")
        return generation.strip()

    def process_results(self, generations, references):
        """Evaluate the generated unit tests"""
        for i, (gen, ref) in enumerate(zip(generations, references)):
            print(f"Generation {i}: {gen}")
            print(f"Reference {i}: {ref}")

        try:
            # Execute the check function with the generated code
            exec(self.test_data[0]["test"])
            local_vars = locals()
            check_function = local_vars['check']
            result = check_function(generations[0][0])
            
            return result
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return {
                "unittest_score": 0.0,
                "error": str(e)
            }