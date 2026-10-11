Put API keys here (this directory is git-ignored; nothing in the release contains a key).

    gemini_api_key.json   {"api_key": "..."}   (only needed for Gemini API-key access; the
                                                pipeline normally uses Vertex AI ADC instead)
    openai_api_key.json   {"api_key": "..."}   (GPT-5.4 zero-shot tests / --model GPT)

Environment variables take precedence: GEMINI_API_KEY, OPENAI_API_KEY. Gemini calls go through
Vertex AI: export GOOGLE_CLOUD_PROJECT=<project-id> and authenticate with
`gcloud auth application-default login` (or GOOGLE_APPLICATION_CREDENTIALS=<service-account.json>).
