import cv2
import numpy as np
import utlis

CANDIDATES = [
    "Bùi Hoàng Sơn", "Phạm Thị Hương Nhài", "Hà Mạnh Dũng", "Nguyễn Thạc Anh",
    "Nguyễn Nhật Thành", "Nguyễn Thị Ngân", "Trịnh Lê Xuân Bách", "Nguyễn Trí Dũng",
    "Bùi Hồng Hà", "Đinh Việt Dũng", "Trần Thu Thiên", "Nguyễn Thị Tuyết Anh"
]

path = "image1.jpg"
widthImg = 700
heighthImg = 700

# PREPROCESSING
img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heighthImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# TIM TAT CA COUNTOURS
contours, heirarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

# TIM CAC HCN
recCont = utlis.rectContours(contours)
biggestCont = utlis.getCornerPoints(recCont[0])
gradePoint = utlis.getCornerPoints(recCont[1])

# Ktra xem chung co duoc nhan dang hay khong
if biggestCont.size != 0 and gradePoint.size != 0:
    cv2.drawContours(imgBiggestContours, biggestCont, -1, (0, 255, 0), 20)
    cv2.drawContours(imgBiggestContours, gradePoint, -1, (255, 0, 0), 20)

    biggestCont = utlis.reorder(biggestCont)
    gradePoint = utlis.reorder(gradePoint)

    pt1 = np.float32(biggestCont)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heighthImg], [widthImg, heighthImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heighthImg))

    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]


    boxes = utlis.splitBoxes(imgThresh)
    # cv2.imshow("Test", boxes[2])
    # boxes = utlis.splitBoxes(imgThresh, 12, 4)  # 12 hàng và 4 cột (Bí thư, Phó Bí thư, Ủy viên, Phiếu trắng)
     # 12 hàng và 4 cột (Bí thư, Phó Bí thư, Ủy viên, Phiếu trắng)

    myPixelValue = np.zeros((12, 4))
    for i, box in enumerate(boxes):
        totalPixels = cv2.countNonZero(box)
        row = i // 4
        col = i % 4
        myPixelValue[row][col] = totalPixels

    # Xác định các vị trí đã được tô
    myIndex = []
    for x in range(12):
        arr = myPixelValue[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])

    # Các danh sách để lưu Bí thư, Phó Bí thư, và Ủy viên
    bi_thu = []
    pho_bi_thu = []
    uy_vien = []

    # Gán các giá trị từ myIndex vào các danh sách tương ứng
    for idx, val in enumerate(myIndex):
        candidate = CANDIDATES[idx]
        if val == 0:
            bi_thu.append(candidate)
        elif val == 1:
            pho_bi_thu.append(candidate)
        elif val == 2:
            uy_vien.append(candidate)

    # Kiểm tra tính hợp lệ của phiếu bầu
    is_valid = len(bi_thu) == 1 and len(pho_bi_thu) == 2 and len(uy_vien) == 7

    print("Bí thư:", ", ".join(bi_thu))
    print("Phó Bí thư:", ", ".join(pho_bi_thu))
    print("Ủy viên:", ", ".join(uy_vien))
    print("Phiếu bầu hợp lệ:", is_valid)

    imgBlank = np.zeros_like(img)
    imgArray = ([img, imgGray, imgBlur, imgCanny],
                [imgContours, imgBiggestContours, imgWarpColored, imgThresh])
    imgStack = utlis.stackImages(imgArray, 0.5)

    cv2.imshow("Stack Image", imgStack)
    cv2.waitKey(0)
