import subprocess
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Deque, Iterator, List, Optional, Tuple

import cv2
import numpy
from tqdm import tqdm

from facefusion import ffmpeg_builder, logger, state_manager, translator
from facefusion.audio import create_empty_audio_frame
from facefusion.content_analyser import analyse_stream, get_inference_pool as get_content_analyser_pool
from facefusion.face_creator import get_static_faces
from facefusion.ffmpeg import open_ffmpeg
from facefusion.processors.core import get_processors_modules
from facefusion.types import Fps, StreamMode, VisionFrame
from facefusion.vision import extract_vision_mask, is_vision_frame, read_static_images


def multi_process_capture(camera_capture : cv2.VideoCapture, camera_fps : Fps, freeze_on_face_loss : bool = False, freeze_recovery_delay : float = 0.5) -> Iterator[VisionFrame]:
	capture_deque : Deque[Tuple[VisionFrame, bool]] = deque()
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	execution_thread_count = state_manager.get_item('execution_thread_count')
	recovery_frame_total = max(1, int(round(freeze_recovery_delay * camera_fps)))
	recovery_frame_count = 0
	capture_failure_total = max(1, int(round(camera_fps * 2)))
	capture_failure_count = 0
	freeze_vision_frame = None
	is_frozen = False

	get_content_analyser_pool()

	with tqdm(desc = translator.get('streaming'), unit = 'frame', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
		with ThreadPoolExecutor(max_workers = execution_thread_count) as executor:
			futures : Deque[Future[Tuple[VisionFrame, bool]]] = deque()

			while (camera_capture and camera_capture.isOpened()) or futures:
				is_capturing = bool(camera_capture and camera_capture.isOpened())

				if is_capturing:
					has_vision_frame, capture_vision_frame = read_camera_frame(camera_capture)

					if has_vision_frame and is_vision_frame(capture_vision_frame):
						capture_failure_count = 0

						if analyse_stream_frame(capture_vision_frame, camera_fps):
							logger.warn(translator.get('stream_stopped_by_analyser'), __name__)
							camera_capture.release()
						else:
							futures.append(executor.submit(process_stream_frame, source_vision_frames, capture_vision_frame, freeze_on_face_loss))

					else:
						capture_failure_count = capture_failure_count + 1

						if capture_failure_count >= capture_failure_total:
							logger.error(translator.get('stream_stopped_by_camera'), __name__)
							camera_capture.release()

				while futures and (not is_capturing or len(futures) > execution_thread_count or futures[0].done()):
					capture_deque.append(futures.popleft().result())

				while capture_deque:
					progress.update()
					capture_vision_frame, has_target_face = capture_deque.popleft()

					if not freeze_on_face_loss:
						yield capture_vision_frame
						continue

					if has_target_face:
						recovery_frame_count = recovery_frame_count + 1
					else:
						recovery_frame_count = 0
						is_frozen = True

					if is_frozen and recovery_frame_count < recovery_frame_total:
						if freeze_vision_frame is None:
							continue
						yield freeze_vision_frame
					else:
						is_frozen = False
						freeze_vision_frame = capture_vision_frame
						yield capture_vision_frame


def read_camera_frame(camera_capture : cv2.VideoCapture) -> Tuple[bool, Optional[VisionFrame]]:
	try:
		return camera_capture.read()
	except cv2.error as exception:
		logger.debug(translator.get('stream_frame_not_captured').format(error = exception), __name__)
		return False, None


def analyse_stream_frame(capture_vision_frame : VisionFrame, camera_fps : Fps) -> bool:
	try:
		return analyse_stream(capture_vision_frame, camera_fps)
	except Exception as exception:
		logger.warn(translator.get('stream_frame_not_analysed').format(error = exception), __name__)
		return False


def process_stream_frame(source_vision_frames : List[VisionFrame], target_vision_frame : VisionFrame, detect_target_face : bool = False) -> Tuple[VisionFrame, bool]:
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
			return target_vision_frame, False

	has_target_face = bool(get_static_faces([ target_vision_frame ])) if detect_target_face else True
	return temp_vision_frame, has_target_face


def open_stream(stream_mode : StreamMode, stream_resolution : str, stream_fps : Fps) -> Optional[subprocess.Popen[bytes]]:
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
		return open_ffmpeg(commands)

	return None


def write_stream_frame(stream : subprocess.Popen[bytes], stream_mode : StreamMode, vision_frame : VisionFrame) -> bool:
	if stream.poll() is not None:
		logger.error(translator.get('stream_closed').format(stream_mode = stream_mode, return_code = stream.returncode), __name__)
		return False

	try:
		stream.stdin.write(numpy.ascontiguousarray(vision_frame).data)
		return True
	except (BrokenPipeError, OSError, ValueError) as exception:
		logger.error(translator.get('stream_frame_not_written').format(error = exception), __name__)
		return False


def close_stream(stream : Optional[subprocess.Popen[bytes]]) -> None:
	if stream:
		try:
			if stream.stdin:
				stream.stdin.close()
			stream.terminate()
			stream.wait(timeout = 2)
		except Exception:
			stream.kill()
