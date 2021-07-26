A Real-Time Sign LAnguage Translator (ARTSLAT)
===
---

ARTSLAT is a real-time sign letter translator that uses Google's Mediapipe hand alongside a small pre-trained model to 
create a lightweight translator that uses a webcam to detect the correct ASL sign letter.

## **Usage**

### _Javascript_

Download and run the ???? .html file located under the **_JS_Soln_** directory.

### _Python_
``` bash
$ pip install -r requirements.txt
$ python coordinate_model.py # Pure Translation
$ python coordinate_model_sentence.py # Inteprets letters in a sentence
```


## **Module Implementation**

### **Translator Module**
```model_coordinate.py / translator.html```

The Mediapipe Hands module detects and tracks instances of hands, 
returning them in a pictorial or coordinate form. In ARTSLAT, by using this particular
[Dataset](https://www.kaggle.com/grassknoted/asl-alphabet)
, I extracted the coordinates of the hands for each image and applied a form of normalisation
before training a model on the data.

#### Normalisation
1. **_Changing the origin_**
    * The wrist joint is set as the origin
    * Thus the hand does not need to be localised to an area in the image
2. **_Scaling every coordinate_**
    * For every set of data, the coordinates are scaled so that the distance between the
    carpometacarpal joint and the wrist is exactly 1 unit long
    * Thus in the final product, hand sizes and its distance to the camera is inconsequential
    

The model is expected to then recognise the relationship between the different joints, and
extrapolate that data to output a given image. The model in particular was designed to be lightweight
with only 2 layers, a dense layer and a softmax layer. The weighted average of the model gives an f1-score
of **0.95**.

### **Trainer Module**
```trainer.html```

Besides a translator, a trainer was envisioned to help teach the ASL letters. To do this, an signed letter
is super imposed onto the users hand, by tracking the user's wrist joint and
placing the alphabet on their joint. However, while this worked on a 2d image, it was sometimes difficult
for users to visualise the proper sign.

Thus, an attempt was made to map the signed letter onto the user's palm so that, by rotating their hand,
one could see a 3d representation of the signed letter. To track rotation, a form of tranformation of the signed letter
was required.

#### Coordinate Transformation
![planenormalmthd](https://i.imgur.com/vKjryCh.gif)
1. **_Assume the palm as a flat 2d plane in a 3d space_**
    * Let the origin be located at the wrist joint
    * Take 2 vectors, both starting at the wrist to the metacarpophalangeal 
      joints of the index and pinky finger
    * The plane is defined as a 2d plane containing these 2 vectors
    
2. **_Find the normal vectors_**
    * A simple representation of a plane is through its normal vector
    * Done by the cross product of the index vector to the pinky vector

3. **_Find the rotation matrix between the normal vector_**
    * Differences in normal vector indicate how much the 2d plane has to be rotated by to fit the other 2d plane

4. **_Apply the rotation matrix_**

This allows the signed letter to follow the user's palm to see the correct shapes in a pseudo-augmented-reality representation.

