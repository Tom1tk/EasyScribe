# Test audio

Large and generated test recordings are not stored in `tests/data`.

`selftest_speech.wav` (23.3 s, generated English speech, mono 16 kHz
16-bit PCM) and its golden transcript `selftest_speech.txt` live in the
tracked `assets/` folder instead. They contain no personal data, are copied
into the release payload as `assets/selftest_speech.*`, and let the packed
`--selftest` time a real speech decode against the expected v3.2.3 output.

Other `tests/data/*.wav` files are local-only and ignored by git.
