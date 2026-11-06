# Selftest speech

`selftest_speech.wav` is the project's generated 23.3-second English test
recording (mono, 16 kHz, 16-bit PCM), copied from the v3.2.3 test suite. It
contains no personal data. `selftest_speech.txt` is the expected plain-text
transcript produced by the v3.2.3 whisper-cli flags (beam size 5). The release
payload includes both as `assets/selftest_speech.wav` and
`assets/selftest_speech.txt` so the packed `--selftest` times a real speech
decode and checks the golden text.
