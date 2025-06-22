# Fourier Series Decomposition Visualization

This repository contains a Python script that **animates the decomposition of a square wave** into a sum of its harmonic sine components. The project provides a visual and intuitive understanding of how the Fourier series works to approximate periodic signals.

---

## 📌 Project Overview

The **Fourier series** allows any periodic function to be represented as a sum of sine and cosine functions. In this project:

- A **square wave** is approximated using its Fourier series.
- Higher-order harmonics are added one by one to show convergence.
- The **Gibbs phenomenon** (overshoot near discontinuities) is clearly observed.

---

## 🎞️ Features

- 🎥 **Animated Visualization**: Displays the construction of a square wave from sine wave components.
- 🔁 **Progressive Approximation**: Watch how accuracy improves with more harmonics.
- 💾 **Auto Save**: Animation is saved as `.gif` or `.mp4` in the same directory.
- ⚙️ **Easily Customizable**: Tweak the number of harmonics, animation speed, and wave period.

---

## 🛠️ How to Run

### ✅ Prerequisites

- **Python 3.x**
- **Python packages**:
  - `numpy`
  - `matplotlib`

- **Optional (for saving animations)**:
  - `ImageMagick` (GIF support)
  - `FFmpeg` (MP4 support)
  - Or fallback: `Pillow` (usually installed with `matplotlib`)

### 📦 Installation

Install Python packages:

```bash
pip install matplotlib numpy
Install an animation writer (if needed):

Windows:
Download ImageMagick

Download FFmpeg

Make sure they are added to your system PATH.

macOS (with Homebrew):
bash
Copy
Edit
brew install imagemagick ffmpeg
Ubuntu/Debian:
bash
Copy
Edit
sudo apt-get install imagemagick ffmpeg
▶️ Run the Script
Run the script from the terminal:

bash
Copy
Edit
python Fourier_Series_Decomposition.py
A plot window will appear with the animation.

The script saves the animation as fourier_square_wave.gif or fourier_square_wave.mp4.

📁 Project Structure
bash
Copy
Edit
Fourier_Series_Decomposition/
├── Fourier_Series_Decomposition.py    # Main animation script
├── README.md                          # This file
└── fourier_square_wave.gif / .mp4     # Output animation after running
⚙️ Customization
You can edit the following variables at the top of Fourier_Series_Decomposition.py:

Variable	Description
MAX_HARMONIC_TERMS	Number of harmonics (odd terms only)
ANIMATION_FRAMES	Frame count (controls animation speed/smoothness)
PERIOD / L_HALF_PERIOD	Period of the square wave

To visualize other periodic functions:
Define a new waveform function (e.g., triangle_wave, sawtooth_wave, etc.)

Write a corresponding Fourier series function to compute its coefficients (a₀, aₙ, bₙ)

🤝 Contributing
Contributions are welcome!

Add support for new waveforms

Improve animation features

Optimize performance

Feel free to fork the repo and open a pull request.

📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.

👩‍💻 Credits
Developed by Irfana
📧 irfanaaslam69@gmail.com
Inspired by the beauty and utility of Fourier analysis in signal processing.
