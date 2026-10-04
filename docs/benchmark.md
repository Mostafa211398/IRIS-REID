# Accuracy comparisons

Create a UTF-8 JSON manifest. Image paths can be absolute or relative to the manifest. Each image must already be a plate crop. Labels follow the training archive's **visual left-to-right character order**: digits, then Arabic letters in their physical order in the crop.

```json
[
  {"path": "plate-01.png", "truth": "123بسم"},
  {"path": "camera-b/plate-02.png", "truth": "456دم"}
]
```

If an existing label uses logical Arabic order, reverse its letters for the manifest. For example, logical Arabic `مسب` becomes visual sequence `بسم`. Do not reverse the digits. Unicode display can reorder Arabic on screen; the stored string/codepoint order determines the label.

Enter the manifest's full local path on the Accuracy screen. Both OCR readings use the same original crop; one passes through the enhancement stage. Raw predictions, confidences, label strings, and model hashes are retained. Corrections do not change accuracy scores.

Scoring applies NFC Unicode normalization, removes whitespace, converts decimal digits to ASCII, and maps Arabic variants to the supplied model vocabulary: `أ/إ/آ → ا`, `ى → ي`, and `ة → ه`. Normalized exact match counts equal strings. Character error rate is the total Levenshtein edit distance divided by the total number of ground-truth characters; it can exceed 100% for excessive insertions. Normalized edit similarity averages `1 - distance / max(truth length, prediction length, 1)` across samples. These rules are recorded with each metrics result.

The supplied archive reports approximately 45.1% exact match and 0.8474 edit similarity on 164 held-out images. These reported training measurements are separate from this application's comparisons and do not establish accuracy on client footage.
