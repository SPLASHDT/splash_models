# splash_sourcode
Source code for SPLASH DT: digital approaches to predict wave hazards. 

Original code to create a digital twin demonstrator of wave overtopping for Dawlish and Penzance (UK). The ultimate aim is to transform weather and climate research and improve operational hazard management to increase UK resilience.

This repository includes:
- training and testing main script
- Random Forest (RF) models
- Master scripts to run the models.

Input Features: 
- Significant wave height
- Mean period
- Wave direction
- Wind speed
- Wind direction
- Freeboard
  
Target features:
- Overtopping binary (1/0)
- Overtopping frequency (number of overtopping events per 10 min window)

  
NE/Z503423/1

Copyright (c) 2025, SPLASH team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify and/or merge copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
