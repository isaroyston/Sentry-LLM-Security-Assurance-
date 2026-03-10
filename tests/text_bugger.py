import logging
from textattack.augmentation import Augmenter
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

logger = logging.getLogger(__name__)

class TextBuggerTool:
    def __init__(
        self, 
        word_swap_ratio: float = 0.6, 
        top_k: int = 20
    ):
        self.name = "TextBugger"
        
        # 1. Initialize the 5 composite attack strategies
        transformation = CompositeTransformation(
            [
                # Insert spaces into words
                WordSwapRandomCharacterInsertion(
                    random_one=True, letters_to_insert=" ", skip_first_char=True, skip_last_char=True
                ),
                # Delete random characters (typos)
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # Swap adjacent characters (fast typing errors)
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # Inject visually similar characters (homoglyphs)
                WordSwapHomoglyphSwap(),
                # Swap words for synonyms using GloVe embeddings
                WordSwapEmbedding(max_candidates=top_k),
            ]
        )
        
        # 2. Build constraints (USE has been removed for maximum speed)
        constraints = [
            RepeatModification(), 
            StopwordModification()
        ]
        
        # 3. Initialize the Augmenter
        self.augmenter = Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=word_swap_ratio,
            transformations_per_example=1, # Generate exactly one mutated prompt per turn
        )

    def apply(self, prompt: str) -> str:
        """Applies TextBugger augmentation to the prompt."""
        try:
            results = self.augmenter.augment(prompt)
            if results:
                return results[0] # Return the mutated string
            return prompt
        except Exception as e:
            logger.error(f"[{self.name}Tool] Error augmenting prompt: {e}")
            return prompt