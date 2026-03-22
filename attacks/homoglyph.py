import random
import logging
import homoglyphs as hg

logger = logging.getLogger(__name__)

class HomoglyphTool:
    def __init__(self, replace_percentage: float = 0.2):
        """
        replace_percentage: How much of the prompt to mutate (0.2 = 20% of letters)
        """
        self.name = "Homoglyph Generator"
        self.replace_percentage = replace_percentage
        # Initialize the homoglyph generator
        self.hg_generator = hg.Homoglyphs(languages={'en'}, strategy=hg.STRATEGY_LOAD)

    def _get_letter_length(self, prompt: str) -> int:
        return sum(1 for char in prompt if char.isalpha())

    def apply(self, prompt: str) -> str:
        """Applies homoglyph substitution to a percentage of the prompt."""
        length = self._get_letter_length(prompt)
        
        # If there are no letters, just return the prompt
        if length == 0:
            return prompt

        enum_prompt = list(enumerate(prompt))
        prompt_chars = list(prompt) # Convert string to list of chars so we can modify it
        
        # Filter to only alphabetic characters
        filtered = [item for item in enum_prompt if item[1].isalpha()]

        # Calculate how many letters to replace based on our percentage
        num_to_replace = int(length * self.replace_percentage)
        letters_to_replace = random.sample(filtered, min(num_to_replace, len(filtered)))

        for index, letter in letters_to_replace:
            try:
                # Get look-alike characters
                combinations = self.hg_generator.get_combinations(letter)
                if combinations:
                    # Swap the original letter with a random look-alike
                    prompt_chars[index] = random.choice(combinations)
            except Exception as e:
                logger.error(f"Cannot get homoglyph for {letter}: {e}")
                continue

        # Rejoin the characters back into a single string
        return "".join(prompt_chars)