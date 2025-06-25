# Vid Slide Generator

This Streamlit application generates short video ads by combining AI-generated images, text-to-speech audio and animated captions.

## Requirements

Install system and Python dependencies:

```bash
# Optional system packages
xargs -a packages.txt sudo apt install -y

# Python requirements
pip install -r requirements.txt
```

## Running the App

Set your API keys and S3 details using Streamlit secrets or environment variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `S3_BUCKET_NAME`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Launch the app with:

```bash
streamlit run app.py
```

You can upload a CSV describing topics and voices or enter tasks manually. Generated videos are uploaded to your configured S3 bucket.
