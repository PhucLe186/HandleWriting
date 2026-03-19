from PIL import Image
import matplotlib.pyplot as plt

def resize_with_aspect_show(img, imgH=32, imgW=512):

    # kích thước ảnh gốc
    w, h = img.size

    # tính width mới giữ tỉ lệ
    new_w = int(imgH * w / h)

    if new_w > imgW:
        new_w = imgW

    # resize
    resized = img.resize((new_w, imgH), Image.BILINEAR)

    # tạo canvas trắng
    canvas = Image.new("L", (imgW, imgH), 255)

    # nếu ảnh RGB → chuyển grayscale
    resized = resized.convert("L")

    # dán vào canvas
    canvas.paste(resized, (0, 0))

    # # ===== SHOW IMAGE =====
    # plt.figure(figsize=(10,3))
    #
    # plt.subplot(1,2,1)
    # plt.imshow(img)
    # plt.title(f"Original: {img.size}")
    # plt.axis("off")
    #
    # plt.subplot(1,2,2)
    # plt.imshow(canvas, cmap="gray")
    # plt.title(f"After resize+pad: {canvas.size}")
    # plt.axis("off")
    #
    # plt.show()

    return canvas