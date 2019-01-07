# Real-time Sign Language Interpreter

We are a team of volunteers from PyTorch Scholarship Challenge from Facebook hosted by Udacity creating awesome Deep Learning models.

### Dataset
[Sign Language Dataset](https://www.kaggle.com/kumawatmanish/deep-learning-sign-language-dataset/data)

### File Description
- main.py
    - Entry point of the application for detecting sign language gesture
- collect_data.py
    - Entry point of the application for creating new sign language data

### Setup
Run the following command to install all required python packages
```console
pip install -r requirements.txt
```

### How to Detect Gesture
Run the following command to install all required python packages
```console
python main.py
```

##### Instructions
- Place your hand on the screen around the green boxes as shown on the figure below
- Press 'z' button to start hand detection
- Press 'z' button once again to stop hand detection
- Press ESC button to exit the application

![Hand](https://res.cloudinary.com/practicaldev/image/fetch/s--iIEtBPzW--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_auto%2Cw_880/https://thepracticaldev.s3.amazonaws.com/i/aaijxoqwmrkyx8epxq4t.png "Hand Histogram")

### How to Add new image data
See the sign language number gesture in https://github.com/ardamavi/Sign-Language-Digits-Dataset

Run the following command to install all required python packages
```console
python collect_data.py
```

##### Instructions
- Press 'z' on your keyboard to start detecting hand
- Press 0-9 on your keyboard to change active label
- Press 'c' on your keyboard to capture a single data
- Press 's' on you keyboard to save all the captured data into file, the file will be located in the current application as X_{datetime}.npy and Y_{datetime}.npy
- Press 'd' on your keyboard to delete last recorded data
- Press ESC button to exit the application

Watch the example of adding new image data on https://youtu.be/lEJL5Xflwjo

### References
[1] [Kaggle Example Script](https://www.kaggle.com/kumawatmanish/deep-learning-sign-language-dataset/code)

[2] [Finger Detection](https://github.com/amarlearning/Finger-Detection-and-Tracking)

[3] [Hand Detection](https://github.com/sashagaz/Hand_Detection)

### Notes
We also developed a sign language interpretation android application. Please check our repository on https://github.com/forfireonly/SignLanguage

### More about our project
- [Pitch Deck](https://goo.gl/Wr9qCF)

- [Landing Page](https://mariannajan.github.io/funtorchSignLanguage/)

- [Introduction Video](https://goo.gl/Rz96JJ)

- [Demo Desktop Application](https://youtu.be/AvHXbdcuj8M)

- [MacOS Application](https://goo.gl/XaJgSu)

- [Windows Application](http://bit.ly/funtorch_win64)

- [Mobile Application](https://goo.gl/9oHJhG)
