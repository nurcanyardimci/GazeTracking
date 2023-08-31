import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time, random

def eyeAspectRatio(points):  # 2 dizi arasındaki öklid mesafesi hesaplama

    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])

    return (A + B) / (2.0 * C)  # göz en boy oranını hesaplayan denklem


def getROI(frame, image, landmarks, eye):
    if eye == 0:
        points = [36, 37, 38, 39, 40, 41]  # sağ göz noktaları
    else:
        points = [42, 43, 44, 45, 46, 47]  # sol göz noktaları

    region = np.array([[landmarks.part(point).x, landmarks.part(point).y] for point in points])
    margin = 7  # değer arttıkça gözün dairesi sola kayar

    left = np.min(region[:, 0])  # dizideki min ve maks değerleri bulur
    top = np.min(region[:, 1])
    right = np.max(region[:, 0])
    bottom = np.max(region[:, 1])

    height = abs(top - bottom)  # yükseklik
    width = abs(left - right)  # genişlik
    grayEye = image[top:bottom, left + margin:right - margin]
    roi = frame[top:bottom, left + margin:right - margin]
    thresh = calibrate(grayEye)  # eşik değeri hesaplama
    _, threshEye = cv2.threshold(grayEye, thresh, 255, cv2.THRESH_BINARY)  # ilk girdi gri görüntü,2.girdi eşik değeri,3.girdi maks değer,4.girdi opencv eşikleme türü
    prepEye = preprocess(threshEye)
    x, y = getIris(prepEye, roi)
    # text = str((x*left)/(width*100.0))
    # cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # print(height)
    cv2.circle(frame, (x + left, y + top), 10, (0, 255, 0), 1)  # göze çizilen daire

    ear = eyeAspectRatio(region)  # göz en boy oranı

    return (x * left) / (width * 100.0), (y * top) / (height * 100.0), ear


def getSize(eye, t):
    height, width = eye.shape
    _, thresh = cv2.threshold(eye, t, 255, cv2.THRESH_BINARY)
    n_pixels = height * width
    # print(n_pixels)

    black_pixels = n_pixels - cv2.countNonZero(thresh)
    # print("->", black_pixels)
    try:
        ratio = black_pixels * 1.0 / n_pixels
        return ratio
    except ZeroDivisionError:  # sonuç sonsuz olursa,sıfıra bölme hatası verir bu satır çalışır
        return None


def calibrate(eye):
    iris_size = 0.48  # iris takip hassasiyeti
    trials = {}

    for t in range(5, 100, 5):  # değerlerle oynayınca iristeki daire konumu değişir
        trials[t] = getSize(eye, t)

    try:
        best_threshold, size = min(trials.items(), key=lambda x: abs(x[1] - iris_size))
        # print(best_threshold, size)
        return best_threshold
    except TypeError:
        return None


def preprocess(image):
    kernel = np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]], dtype=np.uint8)
    blur = cv2.bilateralFilter(image, 5, 10, 10)  # bir alandaki ortalama piksel değeri
    # leftEroded = cv2.erode(leftBlur, kernel, iterations = 1)
    dilated = cv2.dilate(blur, kernel)  # ön plandaki nesne boyutunu arttırır
    return cv2.bitwise_not(dilated)  # siyah ve beyazı yer değiştirir.Tek resim ile uygulanır.


def getIris(image, roi):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key=cv2.contourArea,
                          reverse=True)  # konturleri boyutlarına göre en büyükten en küçüğe sıralamayı sağlar
    margin = 6  # değer arttıkça gözün dairesi sağa kayar
    # return max_contour
    # for contour in contours:    #göz konturlerini gösteriyor
    # cv2.drawContours(roi, contour, -1, (255, 0, 0), 2)
    # cv2.drawContours(roi, max_contour, -1, (255, 0, 0), 2)
    try:
        max_contour = all_contours[0]  # değer arttırılırsa ya da azaltılırsa iris daire sola doğru kayar
        M = cv2.moments(max_contour)
        x = int(M['m10'] / M['m00']) + margin  # iristeki dairenin sağa sola göre ortalama konumu
        y = int(M['m01'] / M['m00'])  # iristeki dairenin aşağı yukarıya göre ortalama konumu
        roi = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.circle(roi, (x, y), 3, (0, 0, 255), -1)  # başka bir pencerede göz roisini alır
        cv2.circle(roi, (150, 63), 50, (0, 0, 255), -1)

        # cv2.imshow("ROI", roi)  #alınan roi leri başka pencerede gösterir,göz roi
        return x, y
    except (IndexError, ZeroDivisionError):
        return 0, 0


def printText(frame, text):
    width, height, _ = frame.shape
    cv2.putText(frame, text, (width // 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # yönlerin yazıldığı kısım


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # kamera
    detector = dlib.get_frontal_face_detector()  # yüz bulma
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    total = 0
    previousRatio = 1



    merkezebak = True  #ilk olarak merkeze bakıyo varsayalım
    control = False  # kırmızı daire çizme durumu önce false olur, çünkü merkez kontrolü yapmadan daire çizmesin
    i = 0  # frameler için i değişkeni tanımlama
    while True:

        retr, frame = cap.read()  # yenibir frame geçme
        frame = cv2.flip(frame, 1)  # kamerayı ters yapma,çevirme
        i = i + 1
        if i % 15 == 0:  # 15 frame bölme
            if control:

                control = False  # birden fazla kırmızı daire üretmesin yeşil olduktan sonra
                time.sleep(0.25)  # yeşil daireden sonra bekleme


            if merkezebak:
                time.sleep(0.50)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                faces = detector(gray)
                landmarks = predictor(gray, faces[0])
            # cv2.circle(frame, (landmarks.part(0).x, landmarks.part(1).y), 3, (255, 0, 0), -1)  #solda yüzün uç noktasına mavi nokta çizer
            except:
                continue

            margin = 7
            Lhori, Lverti, Lear = getROI(frame, gray, landmarks, 0)  # sol göz için
            # hori yatay, verti dikey, ear göz açıklığı(en-boy oranı),avg de ortalama değerlerini bulur
            Rhori, Rverti, Rear = getROI(frame, gray, landmarks, 1)  # sağ göz için


            avgEAR = (Lear + Rear) / 2.0
            avgHori = (Lhori + Rhori) / 2.0
            avgVerti = (Lverti + Rverti) / 2.0


            if avgHori < 0.92 :
                printText(frame, "LEFT")
                cv2.circle(frame, (30, 228), 28, (0, 255, 0), -1)  # sol yeşil daire çizme
                cv2.putText(frame, "ogrenci ayaga kalkti", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                merkezebak = True   #merkezebak true ya git

            elif avgHori > 1.780 :
                printText(frame, "RIGHT")
                cv2.circle(frame, (610, 228), 28, (0, 255, 0), -1)  # sağ yeşil daire çizme
                cv2.putText(frame, "ogrenci ayaga kalkti", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                merkezebak = True


            elif avgVerti < 0.515 :
                printText(frame, "UP")
                cv2.circle(frame, (320, 30), 28, (0, 255, 0), -1)  # yukarı yeşil daire çizme
                cv2.putText(frame, "ogrenci ayaga kalkti", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                merkezebak = True


            elif avgVerti > 0.800 :
                printText(frame, "DOWN")
                cv2.circle(frame, (320, 410), 28, (0, 255, 0), -1)  # aşağı yeşil daire çizme
                cv2.putText(frame, "ogrenci ayaga kalkti", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                merkezebak = True


            elif avgEAR >0.240 or avgEAR <0.350:
                printText(frame, "CENTER ")
                cv2.circle(frame, (320, 200), 140, (0, 0, 255), 2)
                cv2.putText(frame, "ogrenci oturuyor", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                if merkezebak:   #eğer merkeze bakılmışsa
                    control = True  #control true, merkezebak false olur ve 0.75 ms bekleyip yeni kırmızı daire çizer
                    merkezebak = False


            if (avgEAR < 0.20):
                if (previousRatio >= 0.20):
                    total += 1
            previousRatio = avgEAR

            # cv2.putText(frame, "Counter: " + str(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) #counter
            # time.sleep(0.4)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
