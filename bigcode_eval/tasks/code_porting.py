"""Code Porting Evaluation Task
Tests the ability of language models to port code between programming languages
while maintaining functionality and class structures.
"""

from bigcode_eval.base import Task

class CodePorting(Task):
    DATASET_PATH = None
    
    def __init__(self):
        super().__init__(
            # Update stop words to handle Java-specific tokens but avoid markdown
            stop_words=["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"],
            requires_execution=True
        )
        
        # Create the test dataset with the complete problem
        self.test_data = [{
            "prompt": """Port this Python code to Java. Include all necessary imports at the beginning of the file.

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
            
            "reference": """import java.util.ArrayList;
import java.util.List;

public class BankAccount {
    private String accountNumber;
    private double balance;
    private List<String> transactions;

    public BankAccount(String accountNumber) {
        this.accountNumber = accountNumber;
        this.balance = 0;
        this.transactions = new ArrayList<>();
    }

    public void deposit(double amount) {
        this.balance += amount;
        transactions.add("Deposited " + amount);
    }

    public boolean withdraw(double amount) {
        if (amount > this.balance) {
            System.out.println("Insufficient funds!");
            return false;
        }
        this.balance -= amount;
        transactions.add("Withdrew " + amount);
        return true;
    }

    public double checkBalance() {
        return this.balance;
    }

    public void displayTransactions() {
        for (String transaction : transactions) {
            System.out.println(transaction);
        }
    }
}""",
            
            "test": """
def check(candidate):
    # Test imports
    assert "import java.util.ArrayList;" in candidate, "Missing ArrayList import"
    assert "import java.util.List;" in candidate, "Missing List import"
    
    # Test class declaration and fields
    assert "public class BankAccount" in candidate, "Invalid class declaration"
    assert "private String accountNumber;" in candidate, "Missing accountNumber field"
    assert "private double balance;" in candidate, "Missing balance field"
    assert "private List<String> transactions;" in candidate, "Missing transactions field"
    
    # Test constructor
    assert "public BankAccount(String accountNumber)" in candidate, "Invalid constructor signature"
    assert "this.accountNumber = accountNumber;" in candidate, "Missing accountNumber initialization"
    assert "this.balance = 0;" in candidate, "Missing balance initialization"
    assert "this.transactions = new ArrayList<>();" in candidate, "Missing transactions initialization"
    
    # Test methods
    assert "public void deposit(double amount)" in candidate, "Invalid deposit method signature"
    assert "public boolean withdraw(double amount)" in candidate, "Invalid withdraw method signature"
    assert "public double checkBalance()" in candidate, "Invalid checkBalance method signature"
    assert "public void displayTransactions()" in candidate, "Invalid displayTransactions method signature"
    
    # Test method contents
    assert "this.balance += amount;" in candidate, "Invalid deposit logic"
    assert "if (amount > this.balance)" in candidate, "Invalid withdrawal check"
    assert "return this.balance;" in candidate, "Invalid balance return"
    assert "System.out.println" in candidate, "Missing System.out.println"
""",
            "entry_point": "port_to_java"
        }]

    def get_dataset(self):
        return self.test_data

    def get_prompt(self, doc):
        template = f"""<|start_of_role|>user<|end_of_role|>{doc["prompt"]}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""
        print(f"Prompt passed to model: {template}")
        return template

    def get_reference(self, doc):
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def postprocess_generation(self, generation, idx):
        # Remove prompt
        prompt = self.get_prompt(self.test_data[idx])
        generation = generation[len(prompt):].strip()
        
        # Remove any template tokens and markdown
        generation = generation.replace("<|end_of_text|>", "")
        generation = generation.replace("<|end_of_role|>", "")
        generation = generation.replace("```java", "")
        generation = generation.replace("```", "")
        
        # Clean up any explanatory text
        if "import" in generation:
            start_idx = generation.find("import")
            generation = generation[start_idx:]
        elif "public class" in generation:
            start_idx = generation.find("public class")
            generation = generation[start_idx:]
            
        print(f"Code generated by model: {generation}")
        return generation.strip()

    def process_results(self, generations, references):
        """Evaluate the generated Java code against the reference implementation"""
        for i, (gen, ref) in enumerate(zip(generations, references)):
            print(f"Generation {i}: {gen}")
            print(f"Reference {i}: {ref}")

        def evaluate_porting(generation, reference_code):
            try:
                code = generation[0]  # Get the generated code
                total_checks = 0
                passed_checks = 0
                
                # Check imports
                if "import java.util.ArrayList;" in code:
                    passed_checks += 1
                if "import java.util.List;" in code:
                    passed_checks += 1
                total_checks += 2
                
                # Check class structure
                structural_elements = [
                    "public class BankAccount",
                    "private String accountNumber;",
                    "private double balance;",
                    "private List<String> transactions;"
                ]
                for element in structural_elements:
                    if element in code:
                        passed_checks += 1
                total_checks += len(structural_elements)
                
                # Check method signatures
                method_signatures = [
                    "public BankAccount(String accountNumber",
                    "public void deposit(double amount)",
                    "public boolean withdraw(double amount)",
                    "public double checkBalance()",
                    "public void displayTransactions()"
                ]
                for signature in method_signatures:
                    if signature in code:
                        passed_checks += 1
                total_checks += len(method_signatures)
                
                # Check implementation details
                implementation_details = [
                    "this.accountNumber = accountNumber;",
                    "this.balance = 0;",
                    "this.transactions = new ArrayList<>();",
                    "this.balance += amount;",
                    "this.transactions.add",
                    "System.out.println",
                    "if (amount > this.balance)",
                    "return this.balance;"
                ]
                for detail in implementation_details:
                    if detail in code:
                        passed_checks += 1
                total_checks += len(implementation_details)
                
                # Calculate similarity score
                similarity_score = passed_checks / total_checks if total_checks > 0 else 0
                
                return {
                    "similarity_score": round(similarity_score * 100, 2),
                    "checks_passed": passed_checks,
                    "total_checks": total_checks,
                    "details": {
                        "imports": "imports_verified" if passed_checks >= 2 else "imports_missing",
                        "structure": f"structure_score: {passed_checks}/{total_checks}",
                        "implementation": f"implementation_score: {passed_checks}/{total_checks}"
                    }
                }
                
            except Exception as e:
                print(f"Evaluation error: {str(e)}")
                return {"similarity_score": 0.0, "error": str(e)}

        eval_result = evaluate_porting(generations[0], self.test_data[0]["reference"])
        
        return {
            "similarity_score": eval_result["similarity_score"],
            "evaluation_details": eval_result["details"] if "details" in eval_result else {}
        }