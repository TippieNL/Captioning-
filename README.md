# JoyCaption Beta One Local App

Local-first image captioning web app powered exclusively by JoyCaption Beta One.

## Features
- **Backend**: FastAPI + Transformers (AutoProcessor + LlavaForConditionalGeneration).
- **Frontend**: Single-page HTML + Tailwind CDN.
- **Modes**: Descriptive, word-count limited, length+tone, straightforward caption, tags, and creative prompt.
- **API**: `/health`, `/api/caption`, `/api/caption/batch`.
- **Local-only**: Model loaded once and reused in-memory.

## Model Guardrail
Only `fancyfeast/llama-joycaption-beta-one-hf-llava` is supported. Any other model name is rejected.

## Project Structure
```
backend/
  app/
    main.py
    prompts.py
frontend/
  index.html
requirements.txt
run.sh
```

## Setup
```bash
./run.sh
```

Open `frontend/index.html` in a browser and make sure the backend is running at `http://localhost:8000`.

### GPU Notes
- CUDA is preferred; CPU is allowed with a warning in the backend logs.
- GPU uses bfloat16 for model weights and pixel values where applicable.

## API
### Health
```bash
curl http://localhost:8000/health
```

### Single caption
```bash
curl -X POST http://localhost:8000/api/caption \
  -F image=@/path/to/image.jpg \
  -F mode=descriptive_long
```

### Word count mode
```bash
curl -X POST http://localhost:8000/api/caption \
  -F image=@/path/to/image.jpg \
  -F mode=descriptive_word_count \
  -F word_count=120
```

### Batch captions
```bash
curl -X POST http://localhost:8000/api/caption/batch \
  -F images=@/path/to/one.jpg \
  -F images=@/path/to/two.jpg \
  -F mode=straightforward
```

## Frontend Usage
1. Drag and drop an image or click to upload.
2. Choose a caption mode and optional tone/length/word count.
3. Click **Generate caption**.
4. Copy or download the output.
