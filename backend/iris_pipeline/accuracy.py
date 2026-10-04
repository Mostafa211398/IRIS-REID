import unicodedata


def normalize(text):
    aliases = {"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}
    return "".join(str(unicodedata.digit(c)) if c.isdecimal() else c
                   for c in (aliases.get(char, char) for char in unicodedata.normalize("NFC", text)) if not c.isspace())


def edit_distance(left, right):
    previous = list(range(len(right)+1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]


class Accumulator:
    def __init__(self):
        self.count = self.exact = self.errors = self.characters = 0
        self.similarity = 0.0

    def add(self, truth, prediction):
        truth, prediction = normalize(truth), normalize(prediction)
        distance = edit_distance(truth, prediction)
        self.count += 1
        self.exact += truth == prediction
        self.errors += distance
        self.characters += len(truth)
        self.similarity += 1-distance/max(len(truth), len(prediction), 1)

    def result(self):
        return {"samples": self.count, "exact_matches": self.exact, "exact_match_accuracy": self.exact/self.count if self.count else 0,
                "character_errors": self.errors, "characters": self.characters, "character_error_rate": self.errors/self.characters if self.characters else 0,
                "normalized_edit_similarity": self.similarity/self.count if self.count else 0,
                "normalization": "NFC; remove whitespace; normalize decimal digits and Arabic aliases to export vocabulary"}


def metrics(samples, variant):
    accumulator = Accumulator()
    for sample in samples:
        accumulator.add(sample["truth"], sample[variant]["raw_text"])
    return accumulator.result()
