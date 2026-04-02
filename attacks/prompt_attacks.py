import pandas as pd
from tqdm import tqdm

from evaluator import (
    DEFAULT_REFUSAL_SUGGESTION,
    evaluate_model_safety_summary,
    passed_single_turn_safety,
)

def run_red_team_test(dataframe, chatbot):
    results = []
    
    print(f"Starting Red Teaming test on {len(dataframe)} prompts...")
    
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        user_input = row['input']
        expected_refusal = row['target']
        

        actual_response = chatbot.chat(user_input) 
        
       
        # Treat expected_refusal as a suggestion, not an exact match.
        is_safe = passed_single_turn_safety(
            response_text=actual_response,
            refusal_suggestion=str(expected_refusal or DEFAULT_REFUSAL_SUGGESTION),
        )
        
        results.append({
            "input": user_input,
            "expected_refusal": expected_refusal,
            "actual_response": actual_response,
            "passed_safety": is_safe
        })
    
    return pd.DataFrame(results)

def evaluate_model_safety(results_df):
    summary = evaluate_model_safety_summary(results_df)

    print(f"\n--- Test Summary ---")
    print(f"Total Prompts: {summary['total_attempts']}")
    print(f"Successful Jailbreaks: {summary['successful_jailbreaks']}")
    print(f"Penetration Rate: {summary['penetration_rate']:.2f}%")

    # --- Export Failed Cases ---
    successful_jailbreaks = results_df[results_df['passed_safety'] == False]
    if not successful_jailbreaks.empty:
        print("Warning: Jailbreaks detected. Saving to 'failed_cases.csv'...")
        successful_jailbreaks.to_csv("attacks/reports/failed_cases.csv", index=False)
        
        successful_jailbreaks[['input', 'actual_response']].head()
    else:
        print("No jailbreaks detected.")
        
def check_response_safety(bot_response, expected_refusal):
    # Backwards compatible shim for older notebook code.
    return passed_single_turn_safety(
        response_text=str(bot_response or ""),
        refusal_suggestion=str(expected_refusal or DEFAULT_REFUSAL_SUGGESTION),
    )