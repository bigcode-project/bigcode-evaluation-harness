"""Documentation Generation Task
Tests the ability of language models to generate comprehensive docstrings 
for Python code following Google-style documentation standards.
"""

from bigcode_eval.base import Task

class DocumentationGeneration(Task):
    DATASET_PATH = None
    
    def __init__(self):
        super().__init__(
            stop_words=["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"],
            requires_execution=True
        )
        
        self.test_data = [{
            "prompt": """Add comprehensive documentation comments (docstrings) to this Python class.
Include class-level documentation explaining the purpose, attributes, and methods.
Add method-level documentation with Args, Returns, and Updates where applicable.
Follow Google-style docstring format.

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
            
            "reference": """class BankAccount:
    \"\"\"
    A class to represent a bank account.

    Attributes:
        account_number (str): The unique account number for the bank account.
        balance (float): The current balance of the bank account.
        transactions (list): A list that stores transaction records.

    Methods:
        __init__(account_number, balance=0):
            Initializes a new bank account with the given account number and optional initial balance.
        
        deposit(amount):
            Deposits a specified amount into the account, updating the balance and adding the transaction to the transaction history.
        
        withdraw(amount):
            Withdraws a specified amount from the account if sufficient balance exists. If not, prints an error message.
        
        check_balance():
            Returns the current balance of the account.
        
        display_transactions():
            Prints out a list of all transactions that have been made for this account.
    \"\"\"
    
    def __init__(self, account_number, balance=0):
        \"\"\"
        Initializes a new bank account.

        Args:
            account_number (str): The unique account number for the bank account.
            balance (float, optional): The initial balance for the bank account. Default is 0.
        \"\"\"
        self.account_number = account_number
        self.balance = balance
        self.transactions = []

    def deposit(self, amount):
        \"\"\"
        Deposits a specified amount into the bank account.

        Args:
            amount (float): The amount to deposit into the account.

        Updates:
            - Increases the account balance by the deposit amount.
            - Adds a transaction entry to the transaction history.
        \"\"\"
        self.balance += amount
        self.transactions.append(f"Deposited {amount}")

    def withdraw(self, amount):
        \"\"\"
        Withdraws a specified amount from the bank account if sufficient funds are available.

        Args:
            amount (float): The amount to withdraw from the account.

        Returns:
            bool: True if withdrawal is successful, False if there are insufficient funds.

        Updates:
            - Decreases the account balance by the withdrawal amount if successful.
            - Adds a transaction entry to the transaction history.
        \"\"\"
        if amount > self.balance:
            print("Insufficient funds!")
            return False
        self.balance -= amount
        self.transactions.append(f"Withdrew {amount}")
        return True

    def check_balance(self):
        \"\"\"
        Returns the current balance of the bank account.

        Returns:
            float: The current balance of the bank account.
        \"\"\"
        return self.balance

    def display_transactions(self):
        \"\"\"
        Prints out all the transactions that have been made for this bank account.
        This method does not return anything. It simply prints transaction details.
        \"\"\"
        for transaction in self.transactions:
            print(transaction)""",
            
            "test": """
def check(candidate):
    # Test class-level docstring components
    class_doc_elements = [
        "A class to represent a bank account",
        "Attributes:",
        "account_number (str)",
        "balance (float)",
        "transactions (list)",
        "Methods:",
        "__init__",
        "deposit",
        "withdraw",
        "check_balance",
        "display_transactions"
    ]
    
    # Test method docstring components for __init__
    init_doc_elements = [
        "Initializes a new bank account",
        "Args:",
        "account_number (str)",
        "balance (float, optional)",
        "Default is 0"
    ]
    
    # Test method docstring components for deposit
    deposit_doc_elements = [
        "Deposits a specified amount",
        "Args:",
        "amount (float)",
        "Updates:",
        "Increases the account balance",
        "transaction entry"
    ]
    
    # Test method docstring components for withdraw
    withdraw_doc_elements = [
        "Withdraws a specified amount",
        "Args:",
        "amount (float)",
        "Returns:",
        "bool: True if withdrawal is successful",
        "Updates:",
        "Decreases the account balance"
    ]
    
    # Test method docstring components for other methods
    other_doc_elements = [
        "Returns the current balance",
        "Returns float:",
        "Prints out all the transactions",
        "prints transaction details"
    ]
    
    # Calculate scores
    all_checks = {
        "class_documentation": class_doc_elements,
        "init_method": init_doc_elements,
        "deposit_method": deposit_doc_elements,
        "withdraw_method": withdraw_doc_elements,
        "other_methods": other_doc_elements
    }
    
    scores = {}
    total_checks = 0
    total_passed = 0
    
    for category, patterns in all_checks.items():
        passed = sum(1 for pattern in patterns if pattern.lower() in candidate.lower())
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
        "documentation_score": overall_score,
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
        
        if "class BankAccount:" in generation:
            start_idx = generation.find("class BankAccount:")
            generation = generation[start_idx:]
            
        print(f"Code generated by model: {generation}")
        return generation.strip()

    def process_results(self, generations, references):
        """Evaluate the generated documentation"""
        for i, (gen, ref) in enumerate(zip(generations, references)):
            print(f"Generation {i}: {gen}")
            print(f"Reference {i}: {ref}")

        try:
            exec(self.test_data[0]["test"])
            local_vars = locals()
            check_function = local_vars['check']
            result = check_function(generations[0][0])
            
            return result
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return {
                "documentation_score": 0.0,
                "error": str(e)
            }