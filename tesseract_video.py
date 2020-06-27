# uzycie
# python text_recognition.py --east frozen_east_text_detection.pb --image zdjecia/costam.jpg


# import paczek
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import pytesseract
import imutils
import time
import cv2

def decode_predictions(scores, geometry):
	# wczytaj parametr scores i policz ilosc wierszy oraz kolumn. Potem inicjalizuj ramki opisujace oraz wyniki prawdopodobienstwa
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loopuj po wierszach
	for y in range(0, numRows):
		# wyciągnij prawdopodobieństo, oraz ksztalt, aby go obrysować jeśli jest taka możliwość
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loopuj po kolumnach
		for x in range(0, numCols):
			# jeśli nasz wyniki nie ma wystarczającego prawdopodobieństwa to go zignoruj
			if scoresData[x] < args["min_confidence"]:
				continue

			# oblicz współczynnik przesunięcia, ponieważ nasze wynikowe mapy obiektów będą czterokrotnie mniejsze niż obraz wejściowy
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# wyciągnij kąty obiektów oraz policz ich sin/cos
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# użyj geometrii aby obliczyc jak duza musi byc ramka
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# obliczyć początkowe i końcowe współrzędne (x, y) dla ramki tekstu
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# dodaj współrzędne ramki granicznej i wynik prawdopodobieństwa do listy obiektów
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# zwróc tupla z danymi czyli ramkami i prawdopodobienstwem
	return (rects, confidences)



# parser argumentow
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

# ustaw nową szerokość i wysokość, a następnie określ zmienny stosunek zarówno dla szerokości, jak i wysokości
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# zdefiniuj dwie warswt modelu EAST - detekcji - Sigmoid i ciecia - concat
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# zaladuj silnik detekcji EAST
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])



# jeżeli ścieżka do wideo nie została podana, użyj kamery 
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args["video"])

fps = FPS().start()


while True:
	
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	if frame is None:
		break
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)
	frame = cv2.resize(frame, (newW, newH))



	# skonstruuj bloba i przekaz do dalej. Dla każdej komutacji wykonaj porównanie rgb
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# deserializuj znaliziska i zabezpiecz ramki w pozytywnych predykcjach
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# inicjalizuj liste
	results = []
	for (startX, startY, endX, endY) in boxes:
		# skaluj ramke aby opisywala element
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

		# w celu uzyskania lepszego OCR tekstu możemy potencjalnie zastosować nieco dopełnienia otaczającego ramkę - tutaj obliczamy delty w obu kierunkach xi y
		dX = int((endX - startX) * args["padding"])
		dY = int((endY - startY) * args["padding"])
		
		# zastosuje dopelnienie po kazdej ze stron ramki
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(W, endX + (dX * 2))
		endY = min(H, endY + (dY * 2))

		# wyciagnij aktualne dopelnienie ROI
		roi = orig[startY:endY, startX:endX]

		# w celu zastosowania Tesseract v4 do tekstu OCR musimy dostarczyć (1) język, (2) flagę OEM 4, wskazującą, że chcemy użyć modelu sieci neuronowej LSTM dla OCR, a na koniec (3) wartość OEM , w tym przypadku 7, co oznacza, że traktujemy ROI jako pojedynczą linię tekstu
		config = ("-l eng --oem 1 --psm 7")
		text = pytesseract.image_to_string(roi, config=config)

		# dodaj współrzędne ramki granicznej i tekst OCR do listy wyników
		results.append(((startX, startY, endX, endY), text))

		# posortuj wyniki współrzędnych ramki granicznej od góry do dołu
		results = sorted(results, key=lambda r:r[0][1])

		# loopuj po wynikach
	for ((startX, startY, endX, endY), text) in results:
		# wyświetl tekst OCR
			print("znaleziony tekst OCR")
			print("========")
			print("{}\n".format(text))

		# usuń tekst spoza ASCII, abyśmy mogli narysować tekst na obrazie za pomocą OpenCV, a następnie narysować tekst i ramkę otaczającą obszar tekstowy obrazu wejściowego
			text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
			output = orig.copy()
			cv2.rectangle(output, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(output, text, (startX, startY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

			# wyswietl sprasowany obraz
			cv2.imshow("Text Detection", output)
			cv2.waitKey(0)

	fps.update()

	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
	if key == ord("z"):
		exit()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
if not args.get("video", False):
	vs.stop()

else:
	vs.release()
# zamknij wszystkie okna
cv2.destroyAllWindows()