import math
import random
import string
import logging
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

logger = logging.getLogger(__name__)

class InsertPunctuationTool:
    def __init__(self, word_swap_ratio: float = 0.2):
        self.name = "Insert Punctuation"
        self.word_swap_ratio = word_swap_ratio
        self.detokenizer = TreebankWordDetokenizer()

    def apply(self, prompt: str) -> str:
        """Applies random punctuation insertion to a percentage of words in the prompt."""
        try:
            word_list = word_tokenize(prompt)
            word_list_len = len(word_list)
            
            # If the prompt is empty, return it as-is
            if word_list_len == 0:
                return prompt

            # Calculate how many words to perturb based on the ratio
            num_perturb_words = math.ceil(word_list_len * self.word_swap_ratio)
            
            # The space of characters we wish to insert (punctuation + space)
            dec_space = string.punctuation + " "
            
            # Pick ONE random punctuation mark to use for this entire prompt iteration
            chosen_dec = random.choice(dec_space)
            
            # Get random indices to apply the insertion
            indices_to_sample = min(num_perturb_words, word_list_len)
            random_words_idx = random.sample(range(word_list_len), indices_to_sample)

            for idx in random_words_idx:
                # Only insert if the word itself isn't just a punctuation mark
                if word_list[idx] not in dec_space:
                    word_list[idx] = chosen_dec + word_list[idx]

            # Detokenize back into a single string prompt
            new_prompt = self.detokenizer.detokenize(word_list)
            return new_prompt

        except Exception as e:
            logger.error(f"[{self.name}Tool] Error augmenting prompt: {e}")
            return prompt