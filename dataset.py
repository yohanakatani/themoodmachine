"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "hopeful",
    "proud",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
]

# Additional posts covering slang, emojis, sarcasm, and mixed feelings.
SAMPLE_POSTS += [
    "no cap this is the best day ever",
    "I'm lowkey stressed but kind of proud of myself",
    "I absolutely love getting stuck in traffic",
    "ugh today was rough but at least it's over now 🥲",
    "so happy right now 😂😂😂",
    "highkey excited for the concert tonight",
    "feeling kinda meh about this whole thing :/",
    "not bad at all, honestly kinda great",
]

TRUE_LABELS += [
    "positive",  # "no cap this is the best day ever"
    "mixed",     # "I'm lowkey stressed but kind of proud of myself"
    "negative",  # "I absolutely love getting stuck in traffic" (sarcasm)
    "mixed",     # "ugh today was rough but at least it's over now 🥲"
    "positive",  # "so happy right now 😂😂😂"
    "positive",  # "highkey excited for the concert tonight"
    "neutral",   # "feeling kinda meh about this whole thing :/"
    "positive",  # "not bad at all, honestly kinda great"
]

# More posts, skewed toward negative/neutral/mixed to balance out the
# positive-heavy label distribution from the batch above.
SAMPLE_POSTS += [
    "I hate mornings but I love coffee",
    "This job is so boring, I can't take it anymore",
    "Ehh, nothing special happened today",
    "I'm never going back there, it was awful",
    "highkey done with this awful week 💀",
    "Not gonna lie, this is actually pretty good",
    "I guess it was fine, nothing to write home about",
    "lol i'm dead this is hilarious 😂",
]

TRUE_LABELS += [
    "mixed",     # "I hate mornings but I love coffee"
    "negative",  # "This job is so boring, I can't take it anymore"
    "neutral",   # "Ehh, nothing special happened today"
    "negative",  # "I'm never going back there, it was awful"
    "negative",  # "highkey done with this awful week 💀"
    "positive",  # "Not gonna lie, this is actually pretty good"
    "neutral",   # "I guess it was fine, nothing to write home about"
    "positive",  # "lol i'm dead this is hilarious 😂" (slang, no sentiment words)
]
