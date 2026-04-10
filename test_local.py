import sys
import os

# Ensure the script can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CodeDebugAction
from server.code_debug_env_environment import CodeDebugEnvironment

def run_local_test():
    print("⚙️ Initializing CodeDebugEnvironment...")
    env = CodeDebugEnvironment()
    
    # 1. Start a new episode
    obs = env.reset()
    print("\n🎯 NEW PROBLEM RECEIVED!")
    print("\n[Buggy Code]")
    print("---------------------------------")
    print(obs.buggy_code.strip())
    print("---------------------------------\n")
    
    # 2. Get the correct solution from the hidden state (just for testing!)
    correct_solution = env.state.correct_solution
    
    print("🤖 Agent action: Submitting the fixed code...")
    print("---------------------------------")
    print(correct_solution.strip())
    print("---------------------------------\n")
    
    # 3. Take a step with the correct action
    action = CodeDebugAction(fixed_code=correct_solution)
    next_obs = env.step(action)
    
    # 4. Check the reward and results
    print("🔍 ENVIRONMENT RESPONSE:")
    print(f"Test Execution: {next_obs.test_results}")
    print(f"Hint Output:    {next_obs.hint}")
    print(f"Total Reward:   {next_obs.reward}")
    print(f"Episode Done:   {next_obs.done}")
    
    if next_obs.reward == 1.0:
        print("\n✅ LOCAL TEST PASSED! The environment correctly grades the fixed code.")
    else:
        print("\n❌ LOCAL TEST FAILED!")

if __name__ == "__main__":
    run_local_test()
