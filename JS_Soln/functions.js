// Normalise takes the coordinates of the first hand and normalises it so the model recognises it
function Normalise(lms, height, width){
var normalised_landmarks = new Array(42);

var basex = lms[0][0].x * width;
var basey = lms[0][0].y * height;
var thumbx = lms[0][1].x * width;
var thumby = lms[0][1].y * height;

var scale_factor = (((thumbx - basex) ** 2) + ((thumby - basey) ** 2)) ** 0.5;

for (let index = 0; index < lms[0].length; index++) {
  normalised_landmarks[index*2] = ( (lms[0][index].x * width) - basex) / scale_factor;
  normalised_landmarks[index*2+1] = ( (lms[0][index].y * height) - basey) / scale_factor;
}

return normalised_landmarks;
}


// Rotation Matrix returns the rotation matrix to turn a norm of a 3d
// base vector to the norm of a 3d destination vector
function rotationmatrix(base,dest){
//please neaten tq
const a = math.divide(base,math.norm(base))
const b = math.divide(dest, math.norm(dest))
const v = math.cross(a,b)
const c = math.dot(a,b)
const s = math.norm(v)
const eye = math.identity(3,3)
const kmat = math.matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
const rotation_matrix = math.add(eye,kmat,math.multiply(math.multiply(kmat,kmat), ((1 - c) / (s**2)) ) )

return rotation_matrix
}


// UnNormalise attempts to map a set of predetermined alphabet coordinate to the user's own hand coordinates
// This is primarily done through the plane-normal method
function UnNormalise(lms, height, width, alphabet_coord){
//FIRST HAND ONLY
var unnormalised_landmarks = new Array(21);

var basex = lms[0][0].x;
var basey = lms[0][0].y;
var basez = lms[0][0].z;
var thumbx = lms[0][5].x;
var thumby = lms[0][5].y;
var thumbz = lms[0][5].z;

//const normalised_lms = Normalise2(lms, height, width) //ONLY TO CALC ROTATION MATRIX
const scale_factor = ( ((thumbx-basex)**2) + ((thumby-basey)**2) + ((thumbz-basez)**2))**0.5;

//NORMAL VECTOR CALCULATION
const base_vector05 = [alphabet_coord[15+1],alphabet_coord[15+2],alphabet_coord[15+3] ] //+1 cos the first item in alpha coord is the letter
const base_vector017 = [alphabet_coord[51+1],alphabet_coord[51+2],alphabet_coord[51+3] ]
const base_normalvector = math.cross(base_vector05, base_vector017)

const dest_vector05 = [(thumbx-basex), (thumby-basey), (thumbz-basez) *2 ] // *2 as responsiveness increases with a larger z value
const dest_vector017 = [(lms[0][17].x-basex), (lms[0][17].y-basey), (lms[0][17].z-basez) *2 ]
const dest_normalvector = math.cross(dest_vector05,dest_vector017)

const RM = rotationmatrix(base_normalvector,dest_normalvector)

//USING OG LMS TO SCALE TO CORRECT WIDTH
for (let index = 0; index < lms[0].length; index++) {

  const x = alphabet_coord[index*3 + 1];
  const y = alphabet_coord[index*3 + 2];
  const z = alphabet_coord[index*3 + 3];

  var vector = math.matrix([[x],[y],[z]])
  var finalvector = math.multiply(RM, vector)

  var tempJson = {
    "x": finalvector.subset(math.index(0,0)) * scale_factor + lms[0][0].x,
    "y": finalvector.subset(math.index(1,0)) * scale_factor + lms[0][0].y,
    "z": (finalvector.subset(math.index(2,0)) * scale_factor + lms[0][0].z),
  };
  unnormalised_landmarks[index] = tempJson;
}

return unnormalised_landmarks;
}


// Returns index of max value in an array
function indexOfMax(arr) {
if (arr.length === 0) {
    return -1;
}

var max = arr[0];
var maxIndex = 0;

for (var i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
        maxIndex = i;
        max = arr[i];
    }
}

return maxIndex;
}


// Given an array, returns the average of column number == num
function Average(arr,counted,num){
    var avg = 0
    for (var l=0; l < counted; l++){
      avg = avg + arr[l][num]
    }
    avg = avg/counted
    return avg
  }
