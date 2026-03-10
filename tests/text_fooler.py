import os
import logging
import time
import tensorflow as tf
# Suppress TensorFlow logging before importing textattack
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("textattack").setLevel(logging.INFO)

from textattack.augmentation import Augmenter
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding

logger = logging.getLogger(__name__)

class TextFoolerTool:
    def __init__(
        self, 
        word_swap_ratio: float = 0.2, 
        cosine_sim: float = 0.5, 
        max_candidates: int = 50
    ):
        self.name = "TextFooler"
        
        # Initialize Transformation
        transformation = WordSwapEmbedding(max_candidates=max_candidates)
        
        # Initialize Stopwords (TextFooler defaults)
        stopwords = {
            "a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
        }
        
        # Build constraints
        constraints = [
            RepeatModification(),
            StopwordModification(stopwords=stopwords),
            WordEmbeddingDistance(min_cos_sim=cosine_sim),
            PartOfSpeech(allow_verb_noun_swap=True)
        ]
        
        # Initialize the Augmenter (We set transformations_per_example to 1 because we only need one mutated string per turn)
        self.augmenter = Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=word_swap_ratio,
            transformations_per_example=1,
        )

    def apply(self, prompt: str) -> str:
        """Applies TextFooler augmentation to the prompt."""
        try:
            # textattack expects a string and returns a list of augmented strings
            results = self.augmenter.augment(prompt)
            if results:
                return results[0] # Return the first (and only) generated mutation
            return prompt
        except Exception as e:
            # TextAttack can sometimes throw errors on very short prompts or edge cases.
            # If it fails, fail gracefully by returning the original prompt.
            logger.error(f"[TextFoolerTool] Error augmenting prompt: {e}")
            return prompt