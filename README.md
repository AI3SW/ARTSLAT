A Real-Time Sign LAnguage Translator (ARTSLAT)
===
---

ARTSLAT is a real-time sign letter translator that uses Google's Mediapipe hand alongside a small pre-trained model to 
create a lightweight translator that uses a webcam to detect the correct ASL sign letter.

## **Usage**

### _Javascript_

For either the translator or the trainer module, run the relevant html file under /JS_Soln. For the source code of https://bryanongse.github.io/ARTSLAT/ go to
JS_Soln/website_html.

### _Python_
``` bash
$ pip install -r requirements.txt
$ python coordinate_model.py # Pure Translation
$ python coordinate_model_sentence.py # Inteprets letters in a sentence
```


## **Module Implementation**

### **Translator Module**
```model_coordinate.py / translator.html```

The Mediapipe Hands module detects and tracks instances of hands, them in a coordinate form. In ARTSLAT, with this 
[Dataset](https://www.kaggle.com/grassknoted/asl-alphabet), coordinates of the hands for each image were extracted and normalised before training a model on the data.

#### Normalisation
1. **_Changing the origin_**
    * The wrist joint is set as the origin
    * Thus the hand does not need to be localised to an area in the image
2. **_Scaling every coordinate_**
    * For every set of data, the coordinates are scaled so that the distance between the
    carpometacarpal joint and the wrist is exactly 1 unit long
    * Thus in the final product, hand sizes and its distance to the camera is inconsequential
    

The model is expected to recognise the relationship between the different joints, and extrapolate the data to output a letter. The model in particular was designed to be lightweight with only 2 layers but still achieved a weighted f1-score of **0.95**.

### **Trainer Module**
```trainer.html```

A trainer to teach the ASL letters was created by superimposing a letter (guide hand) onto the user's hand, by tracking the user's wrist joint and placing the guide hand at the point. However, it was sometimes difficult for users to visualise the proper sign as some signs needed 3D representation.

Thus, an AR representation of the signed letter was created so that, by just rotating one's hand, one could see a 3d representation of the signed letter. However, to track rotation, the guide hand had to undergo a transformation.

#### Coordinate Transformation
![planenormalmthd](https://i.imgur.com/vKjryCh.gif)
1. **_Assume the palm as a flat 2D plane in a 3D space_**
    * Take 2 vectors, both starting at the wrist to the metacarpophalangeal 
      joints of the index and pinky finger
    * The plane is defined as a 2D plane containing these 2 vectors
    
2. **_Find the normal vectors_**
    * Cross product of the index vector to the pinky vector
   
3. **_Find the rotation matrix between the normal vector_**
    * Differences in normal vector indicate how much the 2d plane has to be rotated by to fit the other 2d plane

4. **_Apply the rotation matrix_**

This allows the signed letter to follow the user's palm to see the correct shapes in a pseudo-augmented-reality representation.


## Limitations

* The Dataset the model trained on could be better, and seems to conflict with some variants
of the ASL sign letters. _(See letter 'T')_
  * Data scrubbing needs to be done to increase accuracy
  
* The trainer module has difficulties when the user's hand is flipped
   * Conditional smoothing should be implemented to reduce inaccuracies

* Certain letters are difficult to display due to the letters requiring x or y axis rotation
of the hand _(See letter 'P')_