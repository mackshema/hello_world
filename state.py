import uuid
import re

# Gemini 2.0 Flash pricing (as of mid-2026)
PRICING_INPUT_PER_1M = 0.075  # USD per 1M input tokens
PRICING_OUTPUT_PER_1M = 0.30  # USD per 1M output tokens

class RunState:
    """Tracks agent competitor state, task metrics, and estimated credit usage."""
    def __init__(self):
        self.run_id          = str(uuid.uuid4())
        self.agent_id        = ""
        self.task_id         = ""
        self.current_level   = 1
        self.total_score     = 0
        self.tasks_attempted = 0
        self.level_history   = []
        
        # Credit and token tracking
        self.input_tokens    = 0
        self.output_tokens   = 0
        self.estimated_cost   = 0.0  # Cumulative cost in USD

    def record_tokens(self, input_tks: int, output_tks: int):
        """Records token usage and increments cumulative estimated cost."""
        self.input_tokens += input_tks
        self.output_tokens += output_tks
        
        # Calculate cost
        cost_in = (input_tks / 1_000_000) * PRICING_INPUT_PER_1M
        cost_out = (output_tks / 1_000_000) * PRICING_OUTPUT_PER_1M
        self.estimated_cost += (cost_in + cost_out)

    def estimate_and_record_tokens(self, text: str, is_input: bool = True):
        """Estimates token count based on string length and records it.
        1 token is roughly 4 characters in English text.
        """
        estimated_tks = max(1, len(text) // 4)
        if is_input:
            self.record_tokens(estimated_tks, 0)
        else:
            self.record_tokens(0, estimated_tks)

    def record(self, level: int, title: str, score: int, levelled_up: bool):
        """Records the result of a submitted task and updates level progression."""
        self.tasks_attempted += 1
        self.total_score     += score
        if levelled_up: 
            self.current_level = level + 1
            
        self.level_history.append({
            "level": level, 
            "task": title,
            "score": score, 
            "up": levelled_up
        })
        icon = "✓" if levelled_up else ("~" if score >= 70 else "✗")
        
        print("\n" + "="*50)
        print(f"  [RunState] Task Attempt Complete!")
        print(f"  Result: {icon} Level {level} -> Score: {score}/100")
        print(f"  Total Attempted: {self.tasks_attempted} | Total Score: {self.total_score}")
        print(f"  Competitor Current Level: {self.current_level}")
        print(f"  Estimated Credits Used: {self.input_tokens} Input, {self.output_tokens} Output")
        print(f"  Estimated Cost: ${self.estimated_cost:.6f} USD")
        print("="*50 + "\n")
