# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

I compared both models on the same dataset.

## 1. Model Overview

**Model type:**
I compared both models: the rule based `MoodAnalyzer` and the scikit-learn `LogisticRegression` model from `ml_experiments.py`.

**Intended purpose:**
Classify a short piece of text (a social-media-style post) as one of four moods: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**
- **Rule based:** Tokenize the text, then add +1 to a score for each word found in `POSITIVE_WORDS` and -1 for each word found in `NEGATIVE_WORDS`. A word immediately preceded by "not", "no", or "never" has its effect flipped. The final score is converted to a label using thresholds (`score >= 1` → positive, `score <= -1` → negative), with two tie-breaking rules for a score of 0: if the text hit both a positive and a negative word, label it "mixed"; if it hit neither, label it "neutral".
- **ML:** `CountVectorizer` turns each post into a bag-of-words vector (a count of which words appear), and `LogisticRegression` is fit on those vectors against `TRUE_LABELS`.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` in `dataset.py` has 22 posts. The starter file shipped with 6; I added 16 more across two rounds covering slang, emoji, sarcasm, and mixed-feeling posts, as called for by the TODO comment in `dataset.py`. Label breakdown across all 22: 8 positive, 6 negative, 4 mixed, 4 neutral.

**Labeling process:**
For the posts I added, I labeled each by how a person would plainly read it: `"no cap this is the best day ever"` → positive, `"highkey excited for the concert tonight"` → positive, `"feeling kinda meh about this whole thing :/"` → neutral. Some were deliberately hard to label even for a person: `"I'm lowkey stressed but kind of proud of myself"` and `"I hate mornings but I love coffee"` (real conflicting feelings → mixed), and `"I absolutely love getting stuck in traffic"` (sarcasm — the literal words are positive but the intended meaning is negative → I labeled it negative based on intended meaning, not literal words).

**Important characteristics of your dataset:**
- Contains slang ("no cap", "lowkey", "highkey", "lol i'm dead")
- Contains emoji and text emoticons (🥲, 😂, 💀, :/)
- Contains sarcasm (e.g. "I absolutely love getting stuck in traffic")
- 4 of the 22 posts are labeled "mixed"

**Possible issues with the dataset:**
It's still small (22 posts) and I labeled all of it myself, so there's no second opinion on the ambiguous cases (the sarcastic post and the "mixed" posts in particular). The second round of additions was deliberately weighted toward negative/neutral/mixed to reduce the original skew toward positive, but positive (8) is still the largest single category. Both models are evaluated on this same small set, so the numbers below describe behavior on these 22 posts specifically, not general accuracy.

## 3. How the Rule Based Model Works

**Your scoring rules:**
- `POSITIVE_WORDS` (12 words): happy, great, good, love, excited, awesome, fun, chill, relaxed, amazing, hopeful, proud
- `NEGATIVE_WORDS` (10 words): sad, bad, terrible, awful, angry, upset, tired, stressed, hate, boring
- Negation: "not", "no", or "never" immediately before a word flips its contribution to the score (e.g. "not happy" scores -1, "never bad" scores +1).
- Preprocessing removes punctuation attached to a word (`"day!"` → `"day"`) and collapses repeated characters (`"soooo"` → `"soo"`), but leaves text emoticons like `":)"` and unicode emoji intact.
- Label thresholds default to `score >= 1` for positive and `score <= -1` for negative; both are configurable constructor arguments (`positive_threshold`, `negative_threshold`).
- A score of exactly 0 is labeled "mixed" if the text hit at least one positive word and one negative word (they canceled out), or "neutral" if it hit neither.

**Strengths of this approach:**
On this dataset it correctly labels every post whose sentiment word is explicitly in the word lists, including the negated case ("I am not happy about this" → negative) and both "mixed" posts that have one word from each list ("Feeling tired but kind of hopeful", "I'm lowkey stressed but kind of proud of myself").

**Weaknesses of this approach:**
It missed 4 of the 22 posts in the current run (accuracy 0.82):
- `"no cap this is the best day ever"` → predicted neutral, true positive. None of "no cap", "best", or "ever" are in `POSITIVE_WORDS`, so slang the word lists don't cover gets no score at all. (Note: "no cap" also contains the negation word "no", but there's no positive/negative word directly after it to flip.)
- `"I absolutely love getting stuck in traffic"` → predicted positive, true negative. The model matches "love" and scores it positive; it has no way to detect that the sentence is sarcastic.
- `"ugh today was rough but at least it's over now 🥲"` → predicted neutral, true mixed. "rough" isn't in `NEGATIVE_WORDS` and the emoji isn't scored at all, so nothing is detected and it falls back to neutral rather than mixed.
- `"lol i'm dead this is hilarious 😂"` → predicted neutral, true positive. Same underlying issue as "no cap": "dead" and "hilarious" aren't in `POSITIVE_WORDS`/`NEGATIVE_WORDS`, and "dead" here is slang for finding something funny, not literally negative.

## 4. How the ML Model Works

**Features used:**
Bag of words using `CountVectorizer` (a count of which exact words appear in each post — no notion of word meaning or word order).

**Training data:**
Trained on all 22 posts in `SAMPLE_POSTS` with their `TRUE_LABELS`.

**Training behavior:**
`evaluate_on_dataset` in `ml_experiments.py` evaluates the model on the same 22 posts it was trained on, so the 1.00 accuracy below is training accuracy (how well it memorized the training set), not a measure of how it would perform on new, unseen posts. This held at 1.00 both before and after the second round of 8 posts was added, so adding more (still tiny) training data didn't reveal any change in training accuracy.

**Strengths and weaknesses:**
Strength: because it's fit directly on the labels, it got every sarcastic and slang post right (`"I absolutely love getting stuck in traffic"` → negative, `"lol i'm dead this is hilarious 😂"` → positive) and all four "mixed" posts right — cases the rule based model can't handle by word-list lookup alone. Weakness: this is exactly what training accuracy can't tell you — the model may have just memorized that these particular sentences map to these particular labels rather than learning anything general about sarcasm or slang. With only 22 training examples, it has no real ability to generalize, and there's no held-out test set in this project to check that.

## 5. Evaluation

**How you evaluated the model:**
Ran `python main.py` and `python ml_experiments.py`, which each print a predicted-vs-true comparison for all 22 posts in `SAMPLE_POSTS` and a final accuracy score.

- Rule based accuracy: **0.82** (18/22 correct)
- ML accuracy: **1.00** (22/22 correct, but this is training accuracy — see Section 4)

**Examples of correct predictions:**
- `"Today was a terrible day"` → both models predicted negative (true: negative). "terrible" is in `NEGATIVE_WORDS`, an unambiguous case for both.
- `"I am not happy about this"` → both models predicted negative (true: negative). The rule based model gets this right specifically because of its negation handling.
- `"I hate mornings but I love coffee"` → both models predicted mixed (true: mixed). For the rule based model, "hate" (-1) and "love" (+1) cancel to a score of 0 with hits on both sides.

**Examples of incorrect predictions:**
- Rule based only, wrong: `"I absolutely love getting stuck in traffic"` → predicted positive, true negative (sarcasm; ML got this one right).
- Rule based only, wrong: `"no cap this is the best day ever"` and `"lol i'm dead this is hilarious 😂"` → both predicted neutral, true positive (slang not in the word list; ML got both right).
- Both models struggle in the same underlying way when a word isn't recognized: the rule based model scores it 0 by lookup, and the ML model has no signal from that specific word unless it saw it in training — the difference here is the ML model saw all 22 posts during training, so it isn't a fair test of which model handles unseen slang better.

## 6. Limitations

- The dataset is small (22 posts), and I'm the only one who labeled it, so ambiguous cases (sarcasm, mixed feelings) reflect one person's judgment.
- The rule based model can only ever score words that are literally in `POSITIVE_WORDS`/`NEGATIVE_WORDS` (22 words combined), so slang, misspellings, and any word outside that list is invisible to it.
- The rule based model has no way to detect sarcasm — it scores words at face value.
- The ML model's 1.00 accuracy is training accuracy on the exact same 22 posts it trained on; this project doesn't have a separate held-out test set, so I can't say how it performs on posts it hasn't seen.
- Neither model was tested on longer text; all `SAMPLE_POSTS` are single short sentences.

## 7. Ethical Considerations

- Misclassifying a message expressing real distress as "neutral" or "positive" (as the rule based model did with "no cap this is the best day ever" resolving to neutral due to missing slang) could be genuinely harmful if a tool like this were used to flag people who need support — a missed signal is worse than a false alarm here.
- Both models were built entirely from words and posts I chose myself; slang or expressions from language communities I'm not familiar with are likely underrepresented in `POSITIVE_WORDS`/`NEGATIVE_WORDS` and in `SAMPLE_POSTS`, which could make the models systematically less accurate for those groups.
- Analyzing personal messages for mood at all raises privacy considerations that are out of scope for a lab exercise but would matter in any real deployment.

## 8. Ideas for Improvement

- Keep adding labeled data, especially more "negative", "mixed", and "neutral" examples: a second round of additions already narrowed the skew toward "positive" from 6/14 (43%) to 8/22 (36%), but positive is still the largest single category.
- Have someone else independently label the ambiguous posts (sarcasm, mixed feelings) and compare against my labels.
- Add a real held-out test set for the ML model instead of only measuring training accuracy.
- Expand `POSITIVE_WORDS`/`NEGATIVE_WORDS` to cover the slang that currently causes rule based misses (e.g. "no cap", "rough", "dead"/"hilarious").
- Give emoji and emoticons actual sentiment scores instead of preserving them as tokens that are never matched against anything.
- Try `TfidfVectorizer` instead of `CountVectorizer` for the ML model, though with only 22 training examples this likely wouldn't be a meaningful test either way.
