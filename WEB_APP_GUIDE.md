# Web Application Guide

## Overview

The Log Analyzer now includes a modern web-based chat interface similar to ChatGPT and DeepSeek, allowing you to upload log files or paste log content and receive instant AI-powered analysis.

## Features

- 🎨 **Modern Chat Interface**: Clean, ChatGPT-style UI
- 📎 **File Upload**: Drag-and-drop or click to upload log files
- 💬 **Text Input**: Paste log content directly in the chat
- 🤖 **AI Analysis**: Real-time analysis using ML models
- 🎯 **Root Cause Detection**: Identifies primary failures
- ⚡ **Instant Results**: Fast analysis and formatted responses

## Installation

### 1. Install Dependencies

```bash
# macOS/Linux
pip3 install -r requirements.txt

# Windows
pip install -r requirements.txt
```

### 2. Ensure Models are Trained

Make sure you've trained the ML models first:

```bash
# macOS/Linux
python3 scripts/train_ml_models.py --train-all

# Windows
python scripts/train_ml_models.py --train-all
```

## Running the Web Application

### macOS/Linux

```bash
./run_web.sh
```

Or directly:

```bash
python3 app.py
```

### Windows

```batch
run_web.bat
```

Or directly:

```batch
python app.py
```

## Usage

1. **Start the Server**: Run the application using the commands above
2. **Open Browser**: Navigate to `http://localhost:5000`
3. **Upload or Paste Logs**:
   - Click "📎 Upload Log File" to upload a file
   - Or paste log content in the text area
4. **Get Analysis**: Click "Send" or press Enter
5. **View Results**: Analysis results appear in the chat interface

## API Endpoints

### POST `/api/analyze`
Analyze log content from text input.

**Request:**
```json
{
  "log_content": "your log content here..."
}
```

**Response:**
```json
{
  "success": true,
  "response": "formatted markdown response",
  "data": { /* full analysis results */ }
}
```

### POST `/api/upload`
Upload and analyze a log file.

**Request:** Multipart form data with `file` field

**Response:**
```json
{
  "success": true,
  "filename": "server.log",
  "response": "formatted markdown response",
  "data": { /* full analysis results */ }
}
```

### GET `/api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "analyzer_loaded": true
}
```

## Configuration

### Port
Change the port by setting the `PORT` environment variable:

```bash
PORT=8080 python3 app.py
```

### Debug Mode
Enable debug mode:

```bash
DEBUG=true python3 app.py
```

### File Size Limit
Default max file size is 50MB. Modify in `app.py`:

```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

## Troubleshooting

### Port Already in Use
If port 5000 is already in use:

```bash
PORT=8080 python3 app.py
```

### Models Not Loading
Ensure models are trained and `data/models/ensemble_config.json` exists:

```bash
python3 scripts/train_ml_models.py --train-all
```

### Flask Not Found
Install Flask dependencies:

```bash
pip3 install flask flask-cors werkzeug
```

## Architecture

- **Backend**: Flask (Python)
- **Frontend**: HTML/CSS/JavaScript
- **Analysis Engine**: Existing LogAnalyzer with ML models
- **API**: RESTful endpoints for analysis

## File Structure

```
web/
├── templates/
│   └── index.html          # Main chat interface
└── static/
    ├── css/
    │   └── style.css       # Styling
    └── js/
        └── app.js          # Frontend logic

app.py                      # Flask application
run_web.sh                  # macOS/Linux launcher
run_web.bat                 # Windows launcher
```

## Customization

### Styling
Modify `web/static/css/style.css` to customize the appearance.

### Response Format
Modify `format_analysis_response()` in `app.py` to change how results are formatted.

### Chat Behavior
Modify `web/static/js/app.js` to customize chat interactions.

## Security Notes

- File uploads are limited to `.txt`, `.log`, and `.json` files
- Uploaded files are stored temporarily and deleted after analysis
- Maximum file size is 50MB by default
- CORS is enabled for cross-origin requests (modify if needed)

## Next Steps

- Add authentication if needed
- Implement chat history persistence
- Add export functionality for analysis results
- Integrate with external logging systems

