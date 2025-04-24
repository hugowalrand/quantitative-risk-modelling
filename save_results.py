import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from investment_sim import InvestmentSimulation

# Initialize and run simulation
sim = InvestmentSimulation()
print(f"\nInitial loan amount: €{sim.loan_amount:,.2f}")
print(f"Deferred amount after {sim.deferral_months} months: €{sim.deferred_amount:,.2f}")
print(f"Monthly payment during repayment period: €{sim.monthly_payment:,.2f}")

results = sim.run_full_simulation()

# Save results to CSV
results.to_csv('../output/results_summary.csv')

print(f"\nSimulation complete. Results saved to 'output/results_summary.csv'")
print("\nDetailed Results:")
print(results.round(2).to_string())
