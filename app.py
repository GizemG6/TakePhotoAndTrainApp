import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CameraApp:
    def __init__(self, root, camera):
        self.root = root
        self.root.title("Kamera Uygulaması")

        self.camera = camera
        self.video_source = 0  

        self.vid = cv2.VideoCapture(self.video_source)
        self.photo_count = 0  # Counter for the number of photos taken

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.entry_name = tk.Entry(root, width=20)
        self.entry_name.pack(pady=5)

        self.btn_capture = tk.Button(root, text="Fotoğraf Çek", command=self.capture)
        self.btn_capture.pack(pady=10)

        self.btn_train = tk.Button(root, text="Eğit", command=self.train)
        self.btn_train.pack(pady=5)
        self.btn_train["state"] = "disabled"

        self.roi = (180, 100, 260, 220)

        self.update()
        self.root.mainloop()

    def capture(self):
        ret, frame = self.vid.read()
        if ret and self.photo_count < 3:
            cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[0] + self.roi[2], self.roi[1] + self.roi[3]), 2)

            name = self.entry_name.get()
            path_directory = f".../camera/photos/{name}"

            if not os.path.exists(path_directory):
                os.makedirs(path_directory)
            save_path = f"{path_directory}/{name}{self.photo_count + 1}.jpg"

            cropped_frame = frame[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]

            cv2.imwrite(save_path, cropped_frame)
            print(f"Fotoğraf başarıyla kaydedildi: {save_path}")


            self.photo_count += 1

            if self.photo_count == 3:
                self.btn_capture["state"] = "disabled"
                self.btn_train["state"] = "normal"


    def update(self):
        ret, frame = self.vid.read()
        cv2.rectangle(frame, (200, 300), (400, 128), (0, 255, 0), 3)
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(10, self.update)

    def train(self):
        name = self.entry_name.get()
        train_dir = f".../camera/photos"
        datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

        batch_size = 96
        img_height = 224
        img_width = 224

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        epochs = 5
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator
        )

        save_path = f".../train_keras/{name}.h5"
        model.save(save_path)
        print(f"Eğitim tamamlandı, model başarıyla kaydedildi: {save_path}")

    def update(self):
        ret, frame = self.vid.read()
        cv2.rectangle(frame, (200, 300), (400, 128), (0, 255, 0), 3)
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(10, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, cv2.VideoCapture(0))
