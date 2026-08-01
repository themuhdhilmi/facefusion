import os
import subprocess
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Iterator, List

import cv2
from tqdm import tqdm

from facefusion import ffmpeg_builder, logger, state_manager, translator
from facefusion.audio import create_empty_audio_frame
from facefusion.content_analyser import analyse_stream
from facefusion.ffmpeg import open_ffmpeg
from facefusion.filesystem import is_directory
from facefusion.processors.core import get_processors_modules
from facefusion.types import Fps, StreamMode, VisionFrame
from facefusion.vision import extract_vision_mask, is_vision_frame, read_static_images


def multi_process_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps) -> Iterator[VisionFrame]:
	capture_deque : Deque[VisionFrame] = deque()
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	execution_thread_count = state_manager.get_item('execution_thread_count')

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		with ThreadPoolExecutor(max_workers = execution_thread_count) as executor:
			futures : Deque[Future[VisionFrame]] = deque()

			while camera_capture and camera_capture.isOpened():
				has_vision_frame, capture_vision_frame = camera_capture.read()

				if has_vision_frame and is_vision_frame(capture_vision_frame):
					if analyse_stream(capture_vision_frame, camera_fps):
						logger.warn(translator.get('stream_stopped_by_analyser'), __name__)
						camera_capture.release()
					else:
						futures.append(executor.submit(process_stream_frame, source_vision_frames, capture_vision_frame))

				while len(futures) > execution_thread_count:
					capture_deque.append(futures.popleft().result())

				while futures and futures[0].done():
					capture_deque.append(futures.popleft().result())

				while capture_deque:
					progress.update()
					yield capture_deque.popleft()


def process_stream_frame(source_vision_frames : List[VisionFrame], target_vision_frame : VisionFrame) -> VisionFrame:
	source_audio_frame = create_empty_audio_frame()
	source_voice_frame = create_empty_audio_frame()
	temp_vision_frame = target_vision_frame.copy()
	temp_vision_mask = extract_vision_mask(temp_vision_frame)

	for processor_module in get_processors_modules(state_manager.get_item('processors')):
		try:
			with logger.suppress():
				is_pre_processed = processor_module.pre_process('stream')

			if is_pre_processed:
				temp_vision_frame, temp_vision_mask = processor_module.process_frame(
				{
					'source_vision_frames': source_vision_frames,
					'source_audio_frame': source_audio_frame,
					'source_voice_frame': source_voice_frame,
					'target_vision_frames': [ target_vision_frame ],
					'temp_vision_frame': temp_vision_frame,
					'temp_vision_mask': temp_vision_mask
				})
		except Exception as exception:
			logger.warn(translator.get('stream_frame_not_processed').format(error = exception), __name__)

	return temp_vision_frame


def open_stream(stream_mode : StreamMode, stream_resolution : str, stream_fps : Fps) -> subprocess.Popen[bytes]:
	commands = ffmpeg_builder.chain(
		ffmpeg_builder.capture_video(),
		ffmpeg_builder.set_media_resolution(stream_resolution),
		ffmpeg_builder.set_input_fps(stream_fps)
	)

	if stream_mode == 'udp':
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('udp'))
		commands.extend(ffmpeg_builder.set_stream_quality(2000))
		commands.extend(ffmpeg_builder.set_output('udp://localhost:27000?pkt_size=1316'))

	if stream_mode == 'v4l2':
		device_directory_path = '/sys/devices/virtual/video4linux'
		commands.extend(ffmpeg_builder.set_input('-'))
		commands.extend(ffmpeg_builder.set_stream_mode('v4l2'))

		if is_directory(device_directory_path):
			device_names = os.listdir(device_directory_path)

			for device_name in device_names:
				device_path = '/dev/' + device_name
				commands.extend(ffmpeg_builder.set_output(device_path))

		else:
			logger.error(translator.get('stream_not_loaded').format(stream_mode = stream_mode), __name__)

	return open_ffmpeg(commands)
