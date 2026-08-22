# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
import string
from typing import List, Optional, Tuple

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the polarity of the word right after them,
# e.g. "not happy" should count against the score, not for it.
NEGATION_WORDS = {"not", "no", "never"}

# Collapses 3+ repeated characters down to 2, e.g. "soooo" -> "soo".
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

# Simple text emoticons like ":)", ":-(", "8-D", ":/" so punctuation
# stripping doesn't tear them apart.
EMOTICON_PATTERN = re.compile(r"^[:;=8][\-o\*']?[)\(dDpP/\\|\]\[]+$")


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Strips whitespace, lowercases, splits on spaces, then for each
        resulting token:
          - Collapses runs of 3+ repeated characters ("soooo" -> "soo")
          - Strips surrounding punctuation ("day!" -> "day"), unless the
            token is a text emoticon like ":)" or ":-(", which is left
            intact. Unicode emoji (e.g. "🥲") aren't ASCII punctuation, so
            they're naturally left intact too.
        """
        cleaned = text.strip().lower()
        raw_tokens = cleaned.split()

        tokens: List[str] = []
        for raw_token in raw_tokens:
            token = REPEATED_CHAR_PATTERN.sub(r"\1\1", raw_token)

            if not EMOTICON_PATTERN.match(token):
                token = token.strip(string.punctuation)

            if token:
                tokens.append(token)

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score, negative words decrease it.
        As a modeling improvement, simple negation is handled: a positive
        or negative word immediately preceded by "not", "no", or "never"
        has its effect on the score flipped (e.g. "not happy" counts as
        negative, "never bad" counts as positive).
        """
        score, _, _ = self._score_tokens(self.preprocess(text))
        return score

    def _score_tokens(self, tokens: List[str]) -> Tuple[int, List[str], List[str]]:
        """
        Shared scoring pass used by both score_text() and explain(), so
        the two never disagree about how a piece of text was scored.

        Returns (score, positive_hits, negative_hits). A word that was
        flipped by a preceding negation word is recorded as "not <word>"
        in whichever hit list matches its *effective* polarity.
        """
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negate_next = False

        for token in tokens:
            if token in NEGATION_WORDS:
                negate_next = True
                continue

            if token in self.positive_words:
                if negate_next:
                    score -= 1
                    negative_hits.append(f"not {token}")
                else:
                    score += 1
                    positive_hits.append(token)
            elif token in self.negative_words:
                if negate_next:
                    score += 1
                    positive_hits.append(f"not {token}")
                else:
                    score -= 1
                    negative_hits.append(token)

            negate_next = False

        return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The mapping is:
          - score > 0                              -> "positive"
          - score < 0                               -> "negative"
          - score == 0, with no sentiment words hit -> "neutral"
          - score == 0, with hits on both sides     -> "mixed"

        A score of 0 doesn't always mean the text is neutral: "proud but
        stressed" and "This is fine" both net to 0, but the first has
        real, conflicting signal while the second has none at all. Only
        the former should count as "mixed".
        """
        score, positive_hits, negative_hits = self._score_tokens(self.preprocess(text))

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"

        if positive_hits and negative_hits:
            return "mixed"

        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        Example explanation:
          'Score = 2 (positive: [\'love\', \'great\'], negative: [])'

        Uses the same scoring pass as score_text(), so a negated word
        like "not happy" shows up under "negative" instead of silently
        being counted as positive.
        """
        score, positive_hits, negative_hits = self._score_tokens(self.preprocess(text))

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
