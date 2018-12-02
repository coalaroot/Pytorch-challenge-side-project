### SUMMARY  
Virtual Meetup 3 - Sunday, 2 December 2018  

### ATTENDEES:  
@z0k, @gopalkrizna, @Anna po, @Samuel Cahyawijaya  

### HIGHLIGHTS OF DISCUSSION:
1. We discussed Flask as a possible framework for implementing the Sign
   Language web app.  
2. There are a number of possible approaches to consider for an end-to-end
   pipeline. We considered two possibilities:   
    - Use unsupervised techniques with OpenCV to create masks of the hand,
      which are then fed into a neural network for classification.  
    - Use an object detection model such as YOLO or SSD to locate a hand, crop
      the image and then feed into a classifier.  
3. We agreed that we'll build a simple neural network prototype before experimenting with (extensive) preprocessing.   
4. The first task is to build a classifier based on this Kaggle kernel: https://www.kaggle.com/kumawatmanish/deep-learning-sign-language-dataset/notebook
 Everyone is welcome to build their own classifier in PyTorch as practice.  
5. The next meetup will be on Wednesday, 5 December at 17:00 UTC.  

