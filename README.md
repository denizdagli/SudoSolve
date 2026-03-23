<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/7823730b-af84-4ef8-a005-4aba79b17339

## Run Locally

**Prerequisites:** Node.js, Python 3.10+

### 1. Frontend Setup (React)
1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

### 2. Backend Setup (Python)
The backend uses FastAPI and Gemini AI for Sudoku recognition.
1. Go to the API directory:
   `cd api`
2. Activate the virtual environment:
   `source venv/bin/activate`
3. Install dependencies (if not already done):
   `pip install -r requirements.txt`
4. Run the backend server:
   `python3 index.py` (veya `fastapi dev index.py`)

# SudoSolve
