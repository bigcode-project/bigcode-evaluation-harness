"""Bug Fixing Task
Tests the ability of language models to identify and fix bugs in Python code
while maintaining the original functionality.
"""

from bigcode_eval.base import Task

class BugFixing(Task):
    DATASET_PATH = None
    
    def __init__(self):
        super().__init__(
            stop_words=["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"],
            requires_execution=True
        )
        
        self.test_data = [{
            "prompt": """Fix all bugs in this Python code.

class BankAccount:
    def __init__(self, account_number balance=0):
        self.account_number = account_number
        self.balance = balance
        self.transactions = []

    def deposit(self, amount):
        self.balance + amount
        self.transactions.append(f"Deposited {amount}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds!)
            return False
        self.balance -= amount
        self.transactions.append(f"Withdrew {amount}")
        return True

    def check_balance(self):
        return self.balance

    def display_transactions():
        for transaction in self.transactions:
            print(transaction)


class BankSystem:
    def __initialize__(self):
        self.accounts = {}

    def create_account(self, account_number):
        if account_number not in self.accounts:
            self.accounts[account_number] = BankAccount(account_number)
            return True
        else:
            print("Account already exists!")
            return False

    def get_account(self, account_number):
        return self.accounts.get(account_number)
        print("Not reachable statement")


def main():
    bank_system = BankSystem()

    while rue:
        print("\\nBank Account Management System")
        print("1. Create new account")
        print("2. Deposit funds")
        print("3. Withdraw funds")
        print("4. Check balance")
        print("5. Display transaction history")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            account_number = input("Enter new account number: ")
            bank_system.create_account(account_number)
        elif choice == "2":
            account_number = input("Enter account number: ")
            amount = float(input("Enter deposit amount: "))
            account = bank_system.get_account(account_number)
            if account:
                account.deposit(amount)
        el choice == "3":
            account_number = input("Enter account number: ")
            amount = float(input("Enter withdrawal amount: "))
            account = bank_system.get_account(account_number)
            if account and account.withdraw(amount):
                print("Withdrawal successful!")
        elif choice == "4":
            account_number = input("Enter account number: ")
            account = bank_system.get_account(account_number)
            if account:
                print(f"Account balance: {account.checkbalance()}")
        elif choice == "5"::
            account_number = input("Enter account number: ")
            account = bank_system.get_account(account_number)
            if account:
                account.display_transactions()
        elif choice == "6":
            break
        else:
            system.printf("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
    maain""",
            
            "reference": """class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self.balance = balance
        self.transactions = []

    def deposit(self, amount):
        self.balance += amount  # Corrected: should be += to add to the balance
        self.transactions.append(f"Deposited {amount}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds!")  # Fixed missing quotation mark
            return False
        self.balance -= amount
        self.transactions.append(f"Withdrew {amount}")
        return True

    def check_balance(self):
        return self.balance

    def display_transactions(self):
        for transaction in self.transactions:
            print(transaction)


class BankSystem:
    def __init__(self):
        self.accounts = {}

    def create_account(self, account_number):
        if account_number not in self.accounts:
            self.accounts[account_number] = BankAccount(account_number)
            return True
        else:
            print("Account already exists!")
            return False

    def get_account(self, account_number):
        return self.accounts.get(account_number)


def main():
    bank_system = BankSystem()

    while True:
        print("\\nBank Account Management System")
        print("1. Create new account")
        print("2. Deposit funds")
        print("3. Withdraw funds")
        print("4. Check balance")
        print("5. Display transaction history")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            account_number = input("Enter new account number: ")
            bank_system.create_account(account_number)
        elif choice == "2":
            account_number = input("Enter account number: ")
            amount = float(input("Enter deposit amount: "))
            account = bank_system.get_account(account_number)
            if account:
                account.deposit(amount)
        elif choice == "3":
            account_number = input("Enter account number: ")
            amount = float(input("Enter withdrawal amount: "))
            account = bank_system.get_account(account_number)
            if account and account.withdraw(amount):
                print("Withdrawal successful!")
        elif choice == "4":
            account_number = input("Enter account number: ")
            account = bank_system.get_account(account_number)
            if account:
                print(f"Account balance: {account.check_balance()}")
        elif choice == "5":
            account_number = input("Enter account number: ")
            account = bank_system.get_account(account_number)
            if account:
                account.display_transactions()
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()""",
            
            "test": """
def check(candidate):
    try:
        # Check for syntax fixes
        syntax_fixes = [
            ('def __init__(self, account_number balance=0)', 'def __init__(self, account_number, balance=0)'),
            ('self.balance + amount', 'self.balance += amount'),
            ('display_transactions():', 'display_transactions(self):'),
            ('def __initialize__', 'def __init__'),
            ('el choice', 'elif choice'),
            ('elif choice == "5":', 'elif choice == "5":'),
            ('while rue:', 'while True:'),
            ('account.checkbalance()', 'account.check_balance()'),
            ('system.printf', 'print'),
            ('print("Not reachable statement")', '')
        ]
        
        syntax_score = sum(1 for old, new in syntax_fixes 
                         if old not in candidate and new in candidate)
        
        # Check for structure
        structure_checks = [
            'class BankAccount:',
            'class BankSystem:',
            'def main():',
            'def __init__',
            'def deposit',
            'def withdraw',
            'def check_balance',
            'def display_transactions',
            'self.accounts = {}',
            'self.transactions = []',
            'if __name__ == "__main__"'
        ]
        
        structure_score = sum(1 for check in structure_checks 
                            if check in candidate)
        
        # Check for functionality
        functionality_checks = [
            'self.balance += amount',
            'self.transactions.append',
            'if amount > self.balance:',
            'return self.accounts.get(account_number)',
            'bank_system = BankSystem()',
            'input(',
            'float(input(',
            'print(f"Account balance:'
        ]
        
        functionality_score = sum(1 for check in functionality_checks 
                                if check in candidate)
        
        # Calculate total score
        total_checks = len(syntax_fixes) + len(structure_checks) + len(functionality_checks)
        total_passed = syntax_score + structure_score + functionality_score
        
        return {
            'bug_fixing_score': round((total_passed / total_checks) * 100, 2),
            'details': {
                'syntax_fixes': f'{syntax_score}/{len(syntax_fixes)}',
                'code_structure': f'{structure_score}/{len(structure_checks)}',
                'functionality': f'{functionality_score}/{len(functionality_checks)}'
            }
        }
    except Exception as e:
        print(f'Evaluation error: {str(e)}')
        return {
            'bug_fixing_score': 0.0,
            'error': str(e)
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
        """Evaluate the bug fixes"""
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
                "bug_fixing_score": 0.0,
                "error": str(e)
            }