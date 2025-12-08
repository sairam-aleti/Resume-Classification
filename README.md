# Resume Intelligence Platform

A simple tool to classify resumes and rank candidates based on job requirements. Built with Python and Streamlit.

## What It Does

- **Upload & Classify Resumes**: Paste a resume or upload a PDF/TXT file to see what role it matches
- **Find Best Candidates**: Post a job description and get a ranked list of matching candidates
- **Keep Track of Candidates**: Manage all resumes in one place with MongoDB
- **No Duplicates**: Automatically catches duplicate candidate entries
- **See the Results**: Beautiful, easy-to-read interface

## What You Need

- Python 3.10 or higher
- MongoDB account (it's free)
- A few Python packages (we'll install them)

## Getting Started

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd resume-classification
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup MongoDB
- Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- Create a free account and cluster
- Get your connection string

### 5. Create `.env` File
Create a file named `.env` in the project folder:
```env
MONGO_URI=<your_mongodb_connection_string>
MONGO_DB=resume_classifier
MONGO_COLLECTION=candidates
```

Replace `<your_mongodb_connection_string>` with your actual MongoDB Atlas connection string from the MongoDB Dashboard.

### 6. Run Locally
```bash
streamlit run resume_classification.py
```

Open http://localhost:8501 in your browser and start using it!

## How It Works

**Tab 1 - Classify Resume:**
- Paste a resume or upload a file
- Click "Classify" 
- See what role the system thinks it matches
- Option to add to your candidate pool

**Tab 2 - Rank Candidates:**
- Paste a job description
- See which candidates match best
- Download results as CSV or JSON

**Tab 3 - Candidate Pool:**
- Search through all your candidates
- Filter by role
- Export data

## Deploy It Online

### Using Streamlit Cloud (Free & Easy)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and connect your repo
4. Select `resume_classification.py` as the main file
5. Go to settings → Secrets and add:
   ```
   MONGO_URI = "your_connection_string"
   MONGO_DB = "resume_classifier"
   MONGO_COLLECTION = "candidates"
   ```
6. Click "Deploy"

That's it! Your app is live.

### Using Docker (For Other Platforms)

Build and run with Docker:
```bash
docker build -t resume-app .
docker run -p 8501:8501 resume-app
```

Then deploy to Heroku, AWS, Google Cloud, or wherever you want.

## Files in This Project

```
resume-classification/
├── resume_classification.py    # The main app
├── data.py                      # Database stuff
├── best_role_classifier.joblib  # The trained model
├── requirements.txt             # Python packages
├── .env                         # Your secrets (don't share!)
└── README.md                    # This file
```

## Important Stuff

- **Keep `.env` secret** - Never put it on GitHub
- **MongoDB IP** - Whitelist your server's IP in MongoDB Atlas
- **Test First** - Always test locally before deploying
- **Model Size** - The classifier file is pretty small, works fast

## Roles It Recognizes

- Software Engineer
- Data Scientist
- ML Engineer
- Backend / Full-Stack Developer
- DevOps / Cloud Engineer

## Issues?

- Check that MongoDB is working (test connection)
- Make sure all packages installed correctly
- Try running locally first to debug

That's all you need to know! Enjoy. 🚀
