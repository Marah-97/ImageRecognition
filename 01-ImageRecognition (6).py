import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from tensorflow.keras.applications import mobilenet_v2

# Set up figure
fig, ax = plt.subplots()
vid = plt.imshow(np.ones((224,224,3)))
lbl = plt.text(5, 20, "Loading...", size=20)
lbl.set_bbox({'facecolor':'white', 'alpha':0.5})
ax.set_axis_off()
plt.ion()
plt.show()

# Set up neural network
model = mobilenet_v2.MobileNetV2(weights='imagenet')

# Set up video capture
cap = cv2.VideoCapture(0)
fig.canvas.mpl_connect('close_event', lambda evt : cap.release())

# Function to call for each frame
def update(i):
    # Capture frame from webcam
    ret, frame_bgr = cap.read()
    assert ret
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)    
    min_dim = np.min(frame.shape[0:2])
    frame = cv2.resize(frame[0:min_dim, 0:min_dim, :], (224, 224))
    vid.set_data(frame)

    # Update classification
    processed_image = mobilenet_v2.preprocess_input(np.expand_dims(frame, axis=0))
    predictions = model.predict(processed_image)
    label = mobilenet_v2.decode_predictions(predictions)    
    lbl.set_text(label[0][0][1])
    
    return vid, lbl


# Run animation loop
ani = FuncAnimation(fig, update, blit=True, interval=100)
