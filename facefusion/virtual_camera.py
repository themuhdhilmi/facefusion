import atexit
import errno
import fcntl
import os
import struct
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy

from facefusion import logger, translator
from facefusion.filesystem import is_directory, is_file
from facefusion.types import Fps, VisionFrame
from facefusion.vision import unpack_resolution

VIDIOC_G_FMT = 0xC0D05604
VIDIOC_S_FMT = 0xC0D05605
VIDIOC_S_PARM = 0xC0CC5616
V4L2_BUF_TYPE_VIDEO_OUTPUT = 2
V4L2_FIELD_NONE = 1
V4L2_COLORSPACE_SRGB = 8
V4L2_FORMAT_SIZE = 208
V4L2_STREAMPARM_SIZE = 204
PIXEL_FORMAT_BYTES_PER_PIXEL =\
{
	'YU12': 1.5,
	'YV12': 1.5,
	'YUYV': 2,
	'UYVY': 2,
	'RGB3': 3,
	'BGR3': 3,
	'RGB4': 4,
	'BGR4': 4
}
VIRTUAL_CAMERA_PIXEL_FORMAT = 'YU12'
VIRTUAL_CAMERA_IDLE_FPS = 2
VIRTUAL_CAMERA_LOCK = threading.Lock()
VIRTUAL_CAMERA_FD : Optional[int] = None
VIRTUAL_CAMERA_DEVICE_PATH : Optional[str] = None
VIRTUAL_CAMERA_RESOLUTION : Optional[str] = None
VIRTUAL_CAMERA_FORMAT : Optional[Tuple[int, int, str]] = None
VIRTUAL_CAMERA_IDLE_EVENT = threading.Event()
VIRTUAL_CAMERA_IDLE_THREAD : Optional[threading.Thread] = None


def detect_virtual_camera_devices() -> List[Tuple[str, str]]:
	device_directory_path = '/sys/devices/virtual/video4linux'
	virtual_camera_devices = []

	if is_directory(device_directory_path):
		for device_name in sorted(os.listdir(device_directory_path)):
			device_path = '/dev/' + device_name
			device_label_path = os.path.join(device_directory_path, device_name, 'name')
			device_label = device_name

			if is_file(device_label_path):
				with open(device_label_path, 'r') as device_label_file:
					device_label = device_label_file.read().strip() or device_name

			if os.path.exists(device_path):
				virtual_camera_devices.append((device_path, device_label))

	return virtual_camera_devices


def resolve_virtual_camera_device() -> Optional[Tuple[str, str]]:
	for device_path, device_label in detect_virtual_camera_devices():
		if 'facefusion' in device_label.lower():
			return device_path, device_label
	return None


def pack_format(width : int, height : int, pixel_format : str) -> bytearray:
	bytes_per_pixel = PIXEL_FORMAT_BYTES_PER_PIXEL.get(pixel_format, 1.5)
	buffer = bytearray(V4L2_FORMAT_SIZE)
	struct.pack_into('I', buffer, 0, V4L2_BUF_TYPE_VIDEO_OUTPUT)
	struct.pack_into('IIIIIII', buffer, 8, width, height, int.from_bytes(pixel_format.encode(), 'little'), V4L2_FIELD_NONE, width, int(width * height * bytes_per_pixel), V4L2_COLORSPACE_SRGB)
	return buffer


def unpack_format(buffer : bytearray) -> Tuple[int, int, str]:
	width, height, pixel_format = struct.unpack_from('III', buffer, 8)
	return width, height, pixel_format.to_bytes(4, 'little').decode(errors = 'replace')


def read_device_format(fd : int) -> Optional[Tuple[int, int, str]]:
	buffer = bytearray(V4L2_FORMAT_SIZE)
	struct.pack_into('I', buffer, 0, V4L2_BUF_TYPE_VIDEO_OUTPUT)

	try:
		fcntl.ioctl(fd, VIDIOC_G_FMT, buffer)
		return unpack_format(buffer)
	except OSError:
		return None


def write_device_format(fd : int, width : int, height : int, pixel_format : str) -> Optional[Tuple[int, int, str]]:
	buffer = pack_format(width, height, pixel_format)

	try:
		fcntl.ioctl(fd, VIDIOC_S_FMT, buffer)
	except OSError:
		pass
	return read_device_format(fd)


def write_device_fps(fd : int, fps : Fps) -> None:
	buffer = bytearray(V4L2_STREAMPARM_SIZE)
	struct.pack_into('I', buffer, 0, V4L2_BUF_TYPE_VIDEO_OUTPUT)
	struct.pack_into('II', buffer, 12, 1000, int(round(fps * 1000)))

	try:
		fcntl.ioctl(fd, VIDIOC_S_PARM, buffer)
	except OSError:
		pass


def open_virtual_camera(resolution : str, fps : Fps) -> bool:
	global VIRTUAL_CAMERA_FD
	global VIRTUAL_CAMERA_DEVICE_PATH
	global VIRTUAL_CAMERA_RESOLUTION
	global VIRTUAL_CAMERA_FORMAT
	global VIRTUAL_CAMERA_IDLE_THREAD

	with VIRTUAL_CAMERA_LOCK:
		width, height = unpack_resolution(resolution)

		if is_virtual_camera_open() and VIRTUAL_CAMERA_FORMAT and (VIRTUAL_CAMERA_FORMAT[0], VIRTUAL_CAMERA_FORMAT[1]) == (width, height):
			VIRTUAL_CAMERA_IDLE_EVENT.clear()
			return True

		if is_virtual_camera_open() and VIRTUAL_CAMERA_RESOLUTION != resolution:
			logger.warn(translator.get('virtual_camera_format_changed').format(previous_resolution = VIRTUAL_CAMERA_RESOLUTION, resolution = resolution), __name__)
		terminate_virtual_camera()
		virtual_camera_device = resolve_virtual_camera_device()

		if not virtual_camera_device:
			logger.error(translator.get('virtual_camera_not_found'), __name__)
			return False

		device_path, device_label = virtual_camera_device

		try:
			fd = os.open(device_path, os.O_RDWR)
		except OSError as exception:
			logger.error(translator.get('virtual_camera_not_opened').format(device_path = device_path, error = exception), __name__)
			return False

		device_format = write_device_format(fd, width, height, VIRTUAL_CAMERA_PIXEL_FORMAT)

		if not device_format or device_format[2] not in PIXEL_FORMAT_BYTES_PER_PIXEL:
			logger.error(translator.get('virtual_camera_format_not_supported').format(device_path = device_path, device_format = device_format), __name__)
			os.close(fd)
			return False

		write_device_fps(fd, fps)
		VIRTUAL_CAMERA_FD = fd
		VIRTUAL_CAMERA_DEVICE_PATH = device_path
		VIRTUAL_CAMERA_RESOLUTION = resolution
		VIRTUAL_CAMERA_FORMAT = device_format
		VIRTUAL_CAMERA_IDLE_EVENT.clear()
		device_resolution = str(device_format[0]) + 'x' + str(device_format[1])
		logger.info(translator.get('virtual_camera_opened').format(device_path = device_path, device_label = device_label, resolution = device_resolution, pixel_format = device_format[2]), __name__)

		if (device_format[0], device_format[1]) != (width, height):
			logger.warn(translator.get('virtual_camera_format_locked').format(device_path = device_path, resolution = resolution, device_resolution = device_resolution), __name__)

		if not VIRTUAL_CAMERA_IDLE_THREAD or not VIRTUAL_CAMERA_IDLE_THREAD.is_alive():
			VIRTUAL_CAMERA_IDLE_THREAD = threading.Thread(target = feed_idle_virtual_camera, daemon = True)
			VIRTUAL_CAMERA_IDLE_THREAD.start()
		return True


def is_virtual_camera_open() -> bool:
	return VIRTUAL_CAMERA_FD is not None


def convert_frame(vision_frame : VisionFrame, device_format : Tuple[int, int, str]) -> bytes:
	width, height, pixel_format = device_format

	if vision_frame.shape[1] != width or vision_frame.shape[0] != height:
		vision_frame = cv2.resize(vision_frame, (width, height), interpolation = cv2.INTER_AREA)
	if pixel_format == 'YU12':
		return cv2.cvtColor(vision_frame, cv2.COLOR_RGB2YUV_I420).tobytes()
	if pixel_format == 'YV12':
		return cv2.cvtColor(vision_frame, cv2.COLOR_RGB2YUV_YV12).tobytes()
	if pixel_format == 'YUYV':
		return cv2.cvtColor(vision_frame, cv2.COLOR_RGB2YUV_YUYV).tobytes()
	if pixel_format == 'UYVY':
		return cv2.cvtColor(vision_frame, cv2.COLOR_RGB2YUV_UYVY).tobytes()
	if pixel_format == 'BGR3':
		return cv2.cvtColor(vision_frame, cv2.COLOR_RGB2BGR).tobytes()
	if pixel_format == 'BGR4':
		return cv2.cvtColor(vision_frame, cv2.COLOR_RGB2BGRA).tobytes()
	if pixel_format == 'RGB4':
		return cv2.cvtColor(vision_frame, cv2.COLOR_RGB2RGBA).tobytes()
	return numpy.ascontiguousarray(vision_frame).tobytes()


def write_virtual_camera_frame(vision_frame : VisionFrame) -> bool:
	with VIRTUAL_CAMERA_LOCK:
		return write_virtual_camera_frame_unlocked(vision_frame)


def write_virtual_camera_frame_unlocked(vision_frame : VisionFrame) -> bool:
	if VIRTUAL_CAMERA_FD is None or VIRTUAL_CAMERA_FORMAT is None:
		return False

	try:
		os.write(VIRTUAL_CAMERA_FD, convert_frame(vision_frame, VIRTUAL_CAMERA_FORMAT))
		return True
	except OSError as exception:
		if exception.errno == errno.EBUSY:
			logger.error(translator.get('virtual_camera_busy').format(device_path = VIRTUAL_CAMERA_DEVICE_PATH), __name__)
		else:
			logger.error(translator.get('virtual_camera_frame_not_written').format(error = exception), __name__)
		terminate_virtual_camera()
		return False


def idle_virtual_camera() -> None:
	with VIRTUAL_CAMERA_LOCK:
		if is_virtual_camera_open():
			write_virtual_camera_frame_unlocked(create_blank_frame())
		VIRTUAL_CAMERA_IDLE_EVENT.set()


def feed_idle_virtual_camera() -> None:
	while True:
		with VIRTUAL_CAMERA_LOCK:
			if not is_virtual_camera_open():
				break
			if VIRTUAL_CAMERA_IDLE_EVENT.is_set():
				write_virtual_camera_frame_unlocked(create_blank_frame())
		time.sleep(1 / VIRTUAL_CAMERA_IDLE_FPS)


def create_blank_frame() -> VisionFrame:
	width, height = unpack_resolution(VIRTUAL_CAMERA_RESOLUTION or '640x480')
	return numpy.zeros((height, width, 3), dtype = numpy.uint8)


def terminate_virtual_camera() -> None:
	global VIRTUAL_CAMERA_FD
	global VIRTUAL_CAMERA_FORMAT

	if VIRTUAL_CAMERA_FD is not None:
		try:
			os.close(VIRTUAL_CAMERA_FD)
		except OSError:
			pass
	VIRTUAL_CAMERA_FD = None
	VIRTUAL_CAMERA_FORMAT = None


def close_virtual_camera() -> None:
	with VIRTUAL_CAMERA_LOCK:
		VIRTUAL_CAMERA_IDLE_EVENT.clear()
		terminate_virtual_camera()


atexit.register(close_virtual_camera)
