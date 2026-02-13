# 🧬 AI Skin Tone Analyzer

An intelligent skin tone analysis tool that uses computer vision to detect and analyze skin tones from photos. Built with OpenCV, Gradio, and Python.

## ✨ Features

- 🎯 **Smart Face Detection** - Automatically detects faces and samples skin from forehead, cheeks, and nose
- 🎨 **Accurate Color Analysis** - Returns HEX, RGB, and LAB color values
- 🔍 **Skin Undertone Detection** - Identifies warm, cool, or neutral undertones
- 📊 **Real-time Visualization** - Shows exactly which regions are being sampled
- 💻 **Web Interface** - Easy-to-use Gradio interface

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager (recommended) or pip

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/skin-tone-analysis.git
cd skin-tone-analysis
\`\`\`
2.Install with UV(recommended):
\`\`\`bash
uv venv --python 3.12
uv pip install -e .

\`\`\`

3.Run the app

\`\`\`bash

uv run python main.py
# or
python main.py

\`\`\`

📸 How to Use

    Upload a clear photo of a face (well-lit, facing forward)

    Click "Analyze Skin Tone"

    View your results:

        🎨 HEX and RGB color codes

        📊 LAB color space values

        🏷️ Skin tone category

        🌡️ Undertone detection

🛠️ Technology Stack

    OpenCV - Face detection and image processing

    Gradio - Web interface

    NumPy - Numerical operations

    Python 3.12+ - Core language
    
📁 Project Structure

skin-tone-analysis/
├── main.py              # Main application
├── pyproject.toml       # Project dependencies
├── README.md           # Documentation
└── LICENSE            # MIT License

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments

    OpenCV team for the amazing computer vision library

    Gradio team for the easy-to-use web interface framework