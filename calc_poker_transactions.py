from itertools import permutations
import random

def calculate_poker_transactions(start_blinds, end_blinds):
    assert len(start_blinds) == len(end_blinds), "The number of players must be the same"
    assert sum(start_blinds) == sum(end_blinds), "The total number of blinds must be the same but the sum of start_blinds is {} and the sum of end_blinds is {}".format(sum(start_blinds), sum(end_blinds))
    # Calculate the net gain/loss for each player
    net_blinds = [end - start for start, end in zip(start_blinds, end_blinds)]
    
    # Separate the players into creditors and debtors
    creditors = [(i, net) for i, net in enumerate(net_blinds) if net > 0]
    debtors = [(i, -net) for i, net in enumerate(net_blinds) if net < 0]
    
    transactions = []
    
    # Minimize transactions by matching creditors and debtors
    i, j = 0, 0
    while i < len(creditors) and j < len(debtors):
        creditor_index, credit_amount = creditors[i]
        debtor_index, debt_amount = debtors[j]
        
        # Determine the transaction amount
        transaction_amount = min(credit_amount, debt_amount)
        
        # Record the transaction
        transactions.append((debtor_index, creditor_index, transaction_amount))
        
        # Update the remaining amounts
        creditors[i] = (creditor_index, credit_amount - transaction_amount)
        debtors[j] = (debtor_index, debt_amount - transaction_amount)
        
        # Move to the next creditor or debtor if fully settled
        if creditors[i][1] == 0:
            i += 1
        if debtors[j][1] == 0:
            j += 1
    
    return transactions

def calculate_optimal_settlements(initial_blinds, final_blinds):
    # Step 1: Calculate the net balance for each player
    net_balances = [final - initial for initial, final in zip(initial_blinds, final_blinds)]

    # Step 2: Separate debtors (negative balances) and creditors (positive balances)
    debtors = []
    creditors = []

    for i, balance in enumerate(net_balances):
        if balance < 0:
            debtors.append((i, -balance))  # store the index and the absolute value of the debt
        elif balance > 0:
            creditors.append((i, balance))  # store the index and the amount owed to the creditor

    # Step 3: Perform settlements between debtors and creditors
    settlements = []
    
    # Recursive helper to settle debts optimally
    def settle(debtors, creditors, settlements):
        # Base case: if no debtors or creditors remain
        if not debtors or not creditors:
            return

        debtor_idx, debt = debtors[0]
        creditor_idx, credit = creditors[0]

        # Find the minimum amount to settle between this debtor and creditor
        settlement_amount = min(debt, credit)
        settlements.append((debtor_idx, creditor_idx, settlement_amount))

        # Update the remaining debt and credit after the transaction
        if debt > credit:
            # Debtor still owes money, so reduce their debt and remove this creditor
            settle([(debtor_idx, debt - settlement_amount)] + debtors[1:], creditors[1:], settlements)
        elif credit > debt:
            # Creditor still needs to be paid, so reduce their credit and remove this debtor
            settle(debtors[1:], [(creditor_idx, credit - settlement_amount)] + creditors[1:], settlements)
        else:
            # Exact match, remove both the debtor and creditor
            settle(debtors[1:], creditors[1:], settlements)

    # Step 4: Call the recursive settlement function
    settle(debtors, creditors, settlements)
    
    return settlements


def calculate_poker_transactions_optimal(start_blinds, end_blinds, max_num_transactions=None):
    """ To find the settlement with the smallest number of transactions, we can just try every order of the players.
    """
    best_transactions = None
    best_num_transactions = float('inf')
    num_transactions = {}
    order_permutations = permutations(range(len(start_blinds)))
    i = 0
    for p in order_permutations:
        transactions = calculate_poker_transactions([start_blinds[i] for i in p], [end_blinds[i] for i in p])
        # Store the number of transactions for this order
        num_transactions.setdefault(len(transactions), 0)
        num_transactions[len(transactions)] += 1
        if len(transactions) < best_num_transactions:
            best_transactions = transactions
            best_num_transactions = len(transactions)
        i += 1
        if max_num_transactions is not None and i >= max_num_transactions:
            break
    print(num_transactions)
    return best_transactions

def create_random_case():
    num_players = random.randint(2, 10)
    start_blinds = [random.randint(0, 1000) for _ in range(num_players)]
    total_start_blinds = sum(start_blinds)
    end_blinds = [random.randint(0, 1000) for _ in range(num_players)]
    total_end_blinds = sum(end_blinds)
    
    # Adjust end_blinds to match the sum of start_blinds
    adjustment_factor = total_start_blinds / total_end_blinds
    end_blinds = [int(blind * adjustment_factor) for blind in end_blinds]
    
    # Adjust the last element to ensure the sums are exactly equal
    end_blinds[-1] += total_start_blinds - sum(end_blinds)
    
    return start_blinds, end_blinds

def check_correct_settlements(start_blinds, end_blinds, transactions):
    # Check that the total blinds are conserved
    assert sum(start_blinds) == sum(end_blinds), "The total number of blinds must be the same but the sum of start_blinds is {} and the sum of end_blinds is {}".format(sum(start_blinds), sum(end_blinds))
    print(f"Cheking whether the transactions are valid: {start_blinds} -> {end_blinds}")
    # Check that the transactions are valid
    for debtor, creditor, amount in transactions:
        print(f"Processing transaction: {debtor} -> {creditor} ({amount})")
        assert 0 <= debtor < len(start_blinds), "Invalid debtor index"
        assert 0 <= creditor < len(start_blinds), "Invalid creditor index"
        assert amount > 0, "Transaction amount must be positive"
        assert start_blinds[debtor] >= amount, "Debtor does not have enough blinds to pay"
        assert end_blinds[creditor] >= amount, "Creditor does not have enough space to receive the blinds"
        end_blinds[debtor] -= amount
        end_blinds[creditor] += amount
    
    print("All checks passed")
    
    return True

# Example usage
start_blinds = [100, 100, 200, 100, 100, 100]
end_blinds = [42, 82, 176.5, 0, 118, 281.5]
start_blinds, end_blinds = create_random_case()
transactions1 = calculate_optimal_settlements(start_blinds, end_blinds)
transactions2 = calculate_poker_transactions(start_blinds, end_blinds)
transactions = transactions1
check_correct_settlements(start_blinds, end_blinds, transactions1)

# Check the number of transactions
print(f"Number of transactions (optimal): {len(transactions1)}")
print(f"Number of transactions (normal): {len(transactions2)}")
for debtor, creditor, amount in transactions:
    print(f"Player {debtor} pays {amount} blinds to Player {creditor}")