import cv2 as cv
from ultralytics import YOLO
import sys
import os
import ffmpeg


def main():
    video_file = sys.argv[1]
    model = YOLO("yolov8n.pt")

    input_video = video_file
    temp_video = "temp.mp4"
    output_video = "output.mp4"


    cap = cv.VideoCapture(input_video)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(temp_video, fourcc, fps, (width, height))

    print("Starting to writes file....")

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print("Video processed. Merging audio with ffmpeg-python...")
    video_stream = ffmpeg.input(temp_video)
    audio_stream = ffmpeg.input(input_video)

    (
        ffmpeg
        .output(
            video_stream.video,
            audio_stream.audio,
            output_video,
            vcodec="copy",
            acodec="copy"
        )
        .overwrite_output()
        .run()
    )


    print("Done! Saved as:", output_video)
    


if __name__ == "__main__":
    main()
