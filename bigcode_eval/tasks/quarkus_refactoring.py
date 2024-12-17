"""Code Refactoring to Quarkus Task 
Tests the ability of language models to refactor traditional Java code into Quarkus-based REST applications
while maintaining functionality and adding proper REST endpoints.
"""

from bigcode_eval.base import Task

class QuarkusRefactoring(Task):
    DATASET_PATH = None
    
    def __init__(self):
        super().__init__(
            stop_words=["<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"],
            requires_execution=True
        )
        
        self.test_data = [{
            "prompt": """Refactor this Java code into a Quarkus application with REST endpoints. Include all necessary imports and annotations.
Add proper REST endpoints for all operations (create account, deposit, withdraw, check balance, view transactions).
Use appropriate HTTP methods and status codes.

import java.util.ArrayList;
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
            
            "reference": """import javax.enterprise.context.ApplicationScoped;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.List;

@Path("/bankaccount")
@ApplicationScoped
public class BankAccountResource {
    private List<BankAccount> bankAccounts = new ArrayList<>();

    @POST
    @Path("/create")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response createAccount(BankAccount account) {
        bankAccounts.add(account);
        return Response.status(Response.Status.CREATED).entity(account).build();
    }

    @GET
    @Path("/{accountNumber}")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getAccount(@PathParam("accountNumber") String accountNumber) {
        BankAccount account = findAccount(accountNumber);
        if (account != null) {
            return Response.ok(account).build();
        }
        return Response.status(Response.Status.NOT_FOUND).build();
    }

    @POST
    @Path("/{accountNumber}/deposit")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response deposit(@PathParam("accountNumber") String accountNumber, double amount) {
        BankAccount account = findAccount(accountNumber);
        if (account == null) {
            return Response.status(Response.Status.NOT_FOUND).build();
        }
        account.deposit(amount);
        return Response.ok(account).build();
    }

    @POST
    @Path("/{accountNumber}/withdraw")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response withdraw(@PathParam("accountNumber") String accountNumber, double amount) {
        BankAccount account = findAccount(accountNumber);
        if (account == null) {
            return Response.status(Response.Status.NOT_FOUND).build();
        }
        if (!account.withdraw(amount)) {
            return Response.status(Response.Status.BAD_REQUEST).entity("Insufficient funds").build();
        }
        return Response.ok(account).build();
    }

    @GET
    @Path("/{accountNumber}/balance")
    @Produces(MediaType.APPLICATION_JSON)
    public Response checkBalance(@PathParam("accountNumber") String accountNumber) {
        BankAccount account = findAccount(accountNumber);
        if (account != null) {
            return Response.ok(account.checkBalance()).build();
        }
        return Response.status(Response.Status.NOT_FOUND).build();
    }

    @GET
    @Path("/{accountNumber}/transactions")
    @Produces(MediaType.APPLICATION_JSON)
    public Response displayTransactions(@PathParam("accountNumber") String accountNumber) {
        BankAccount account = findAccount(accountNumber);
        if (account != null) {
            return Response.ok(account.getTransactions()).build();
        }
        return Response.status(Response.Status.NOT_FOUND).build();
    }

    private BankAccount findAccount(String accountNumber) {
        return bankAccounts.stream()
                .filter(account -> account.getAccountNumber().equals(accountNumber))
                .findFirst()
                .orElse(null);
    }
}

@ApplicationScoped
class BankAccount {
    private String accountNumber;
    private double balance;
    private List<String> transactions;

    public BankAccount() {
        // Default constructor needed for JAX-RS
    }

    public BankAccount(String accountNumber) {
        this.accountNumber = accountNumber;
        this.balance = 0;
        this.transactions = new ArrayList<>();
    }

    public String getAccountNumber() {
        return accountNumber;
    }

    public double getBalance() {
        return balance;
    }

    public List<String> getTransactions() {
        return transactions;
    }

    public void deposit(double amount) {
        this.balance += amount;
        transactions.add("Deposited " + amount);
    }

    public boolean withdraw(double amount) {
        if (amount > this.balance) {
            return false;
        }
        this.balance -= amount;
        transactions.add("Withdrew " + amount);
        return true;
    }

    public double checkBalance() {
        return this.balance;
    }
}"""
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
        generation = generation.replace("```java", "")
        generation = generation.replace("```", "")
        
        if "import" in generation:
            start_idx = generation.find("import")
            generation = generation[start_idx:]
        elif "@Path" in generation:
            start_idx = generation.find("@Path")
            generation = generation[start_idx:]
            
        print(f"Code generated by model: {generation}")
        return generation.strip()

    def process_results(self, generations, references):
        """Evaluate the generated Quarkus code"""
        for i, (gen, ref) in enumerate(zip(generations, references)):
            print(f"Generation {i}: {gen}")
            print(f"Reference {i}: {ref}")

        def evaluate_porting(generation):
            try:
                code = generation[0]  # Get the generated code
                checks = {
                    "imports": [
                        "javax.enterprise.context.ApplicationScoped",
                        "javax.ws.rs",
                        "javax.ws.rs.core.MediaType",
                        "javax.ws.rs.core.Response"
                    ],
                    "annotations": [
                        "@Path(\"/bankaccount\")",
                        "@ApplicationScoped",
                        "@POST",
                        "@GET",
                        "@Consumes(MediaType.APPLICATION_JSON)",
                        "@Produces(MediaType.APPLICATION_JSON)"
                    ],
                    "endpoints": [
                        "@Path(\"/create\")",
                        "@Path(\"/{accountNumber}\")",
                        "@Path(\"/{accountNumber}/deposit\")",
                        "@Path(\"/{accountNumber}/withdraw\")",
                        "@Path(\"/{accountNumber}/balance\")",
                        "@Path(\"/{accountNumber}/transactions\")"
                    ],
                    "responses": [
                        "Response.status(Response.Status.CREATED)",
                        "Response.status(Response.Status.NOT_FOUND)",
                        "Response.status(Response.Status.BAD_REQUEST)",
                        "Response.ok(",
                        "return Response"
                    ],
                    "methods": [
                        "@PathParam(\"accountNumber\")",
                        "findAccount(String accountNumber)",
                        "public Response createAccount(BankAccount account)",
                        "public Response deposit(",
                        "public Response withdraw(",
                        "public Response checkBalance(",
                        "public Response displayTransactions("
                    ],
                    "bankaccount": [
                        "public String getAccountNumber()",
                        "public double getBalance()",
                        "public List<String> getTransactions()",
                        "public BankAccount()"
                    ]
                }
                
                scores = {}
                total_checks = 0
                total_passed = 0
                
                for category, patterns in checks.items():
                    passed = sum(1 for pattern in patterns if pattern in code)
                    total = len(patterns)
                    scores[category] = {
                        "passed": passed,
                        "total": total,
                        "score": round((passed / total) * 100, 2) if total > 0 else 0
                    }
                    total_checks += total
                    total_passed += passed
                
                overall_score = round((total_passed / total_checks) * 100, 2) if total_checks > 0 else 0
                
                return {
                    "refactoring_score": overall_score,
                    "details": {
                        category: f"{info['passed']}/{info['total']} ({info['score']}%)"
                        for category, info in scores.items()
                    }
                }
                
            except Exception as e:
                print(f"Evaluation error: {str(e)}")
                return {
                    "refactoring_score": 0.0,
                    "error": str(e)
                }

        result = evaluate_porting(generations[0])
        return result