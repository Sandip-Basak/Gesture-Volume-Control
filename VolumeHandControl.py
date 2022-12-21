import cv2
import time
import numpy as np
import HandsTrackingModule as hand
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def main():
    cap = cv2.VideoCapture(1)
    detector = hand.HandDetector(min_detection_confidence=0.9, max_hands=1)
    pTime = 0

    # pyCaw
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Getting the current volume and the volume range of the system
    cVol = volume.GetMasterVolumeLevel()
    minVol, maxVol, temp = volume.GetVolumeRange()

    while True:
        success, img = cap.read()

        # Drawing landmarks and connections
        detector.DrawHands(img=img, draw=False)

        # Getting all the position of the landmarks
        points = detector.givePosition(img=img, draw=False)

        # check that the returned list is empty or not
        if len(points) != 0:

            # Getting the position of landmark 4,8
            x1, y1 = points[4][1], points[4][2]
            x2, y2 = points[8][1], points[8][2]
            cx, cy = (x1+x2)//2, (y1+y2)//2

            # Drawing circles on the particular positions
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 225), 3)

            # Getting the distance between the landmark 4 and 8
            length = math.hypot(x2-x1, y2-y1)

            # Converting the distance to the specified system volume range
            cVol = np.interp(length, [60, 280], [minVol, maxVol])

            # Setting the volume
            volume.SetMasterVolumeLevel(cVol, None)

            if length < 50:
                cv2.circle(img, (cx, cy), 10, (0, 225, 0), cv2.FILLED)

        # Converting the volume to the specified range for the bar and volume percentage
        volRange = int(np.interp(cVol, [minVol, maxVol], [400, 150]))
        volValue = int(np.interp(cVol, [minVol, maxVol], [0, 100]))

        # Drawing the volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 225, 0), 3)
        cv2.rectangle(img, (50, volRange), (85, 400), (0, 225, 0), cv2.FILLED)
        cv2.putText(img, f"{volValue}%", (45, 440), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 225, 0), 2)

        # Calculating FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS-{int(fps)}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)

        # Showing the Image
        cv2.imshow("Image", img)

        # On 'q' press the loop breaks
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
