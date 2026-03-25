import math
import random
import string
import logging
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

logger = logging.getLogger(__name__)

class CharSwapTool:
    def __init__(self, word_swap_ratio: float = 0.2):
        self.name = "Character Swap"
        self.word_swap_ratio = word_swap_ratio
        self.detokenizer = TreebankWordDetokenizer()

    def apply(self, prompt: str) -> str:
        """Applies character swapping to a percentage of words in the prompt."""
        try:
            word_list = word_tokenize(prompt)
            word_list_len = len(word_list)
            
            # If the prompt is empty, return it as-is
            if word_list_len == 0:
                return prompt

            # Calculate how many words to perturb based on the ratio
            num_perturb_words = math.ceil(word_list_len * self.word_swap_ratio)
            
            # Get random indices to apply the swap (ensuring we don't sample out of bounds)
            indices_to_sample = min(num_perturb_words, word_list_len)
            random_words_idx = random.sample(range(word_list_len), indices_to_sample)

            for idx in random_words_idx:
                word = word_list[idx]
                
                # Only swap if it's not punctuation and is long enough to have internal characters
                if word not in string.punctuation and len(word) > 3:
                    # Pick a random index inside the word (avoiding the very first character)
                    idx1 = random.randint(1, len(word) - 2)
                    
                    # Convert string to list to mutate characters
                    idx_elements = list(word)
                    
                    # Swap the character with the one immediately next to it
                    idx_elements[idx1], idx_elements[idx1 + 1] = (
                        idx_elements[idx1 + 1],
                        idx_elements[idx1],
                    )
                    
                    # Rejoin the mutated word and put it back in the list
                    word_list[idx] = "".join(idx_elements)

            # Detokenize back into a single string prompt
            new_prompt = self.detokenizer.detokenize(word_list)
            return new_prompt

        except Exception as e:
            logger.error(f"[{self.name}Tool] Error augmenting prompt: {e}")
            return prompt