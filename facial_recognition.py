import cv2 as cv

# Load pre-trained face detector (Haar Cascade)
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(frame):
    """Detects faces in the given frame and returns their coordinates."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

if __name__ == "__main__":
    # Test the face detection independently
    video_capture = cv.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        faces = detect_face(frame)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()
