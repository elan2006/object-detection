import cv2 as cv
from ultralytics import YOLO
import sys
import os
import ffmpeg
import argparse


def object_detection(input_video_filepath, output_video_filepath) -> str:
    model = YOLO("yolov8n.pt")
    input_video = input_video_filepath
    temp_video = output_video_filepath


    cap = cv.VideoCapture(input_video)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(temp_video, fourcc, fps, (width, height))

    print("Starting to write file....")

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    return temp_video


def copy_audio_to_output(input_video, video_with_audio, output_video):
    video_stream = ffmpeg.input(input_video)
    audio_stream = ffmpeg.input(video_with_audio)

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
        .run(quiet=True)
    )



def main():

    parser = argparse.ArgumentParser(
        description="Run object detection and store it in a video file."
    )
    parser.add_argument("--input-video",
                        "-i",
                        help="Path to the input video file")
    parser.add_argument("--output-video",
                        "-o",
                        default="output.mkv",
                        help="Path to save the processed output video")
    parser.add_argument("--temp-video",
                        "-t",
                        default="temp.mkv",
                        help="Store a temporary video for further processing")

    args = parser.parse_args()

    input_video = args.input_video
    output_video = args.output_video
    temp_video = object_detection(input_video, "temp.mkv")

    copy_audio_to_output(input_video=temp_video, video_with_audio=input_video, output_video=output_video)
    print("Done! Saved as:", output_video)


if __name__ == "__main__":
    main()
