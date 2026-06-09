"""
vulkan_probe.py - Vulkan GPU enumeration via ctypes.

No Python Vulkan package required — uses vulkan-1.dll directly.
On systems without Vulkan support, returns an empty list without raising.
"""

import ctypes
import ctypes.util
import logging
import sys

logger = logging.getLogger(__name__)

VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256
VK_UUID_SIZE = 16


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class _VkPhysicalDeviceProperties(ctypes.Structure):
    # VkPhysicalDeviceLimits is ~500 bytes; pad rather than enumerate all fields.
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * VK_MAX_PHYSICAL_DEVICE_NAME_SIZE),
        ("pipelineCacheUUID", ctypes.c_uint8 * VK_UUID_SIZE),
        ("_limits_padding", ctypes.c_uint8 * 504),
        ("_sparse_padding", ctypes.c_uint8 * 32),
    ]


_cached_gpus: list[dict] | None = None


def detect_vulkan_gpus() -> list[dict]:
    """
    Return a list of Vulkan-capable GPUs: [{"index": int, "name": str}, ...]
    Returns [] if Vulkan is unavailable or enumeration fails.
    Result is cached after first call.
    """
    global _cached_gpus
    if _cached_gpus is not None:
        return _cached_gpus
    _cached_gpus = _enumerate()
    return _cached_gpus


def _enumerate() -> list[dict]:
    try:
        if sys.platform == "win32":
            vk = ctypes.WinDLL("vulkan-1.dll")
        else:
            lib = ctypes.util.find_library("vulkan") or "libvulkan.so.1"
            vk = ctypes.CDLL(lib)
    except OSError:
        logger.debug("Vulkan library not found — no GPU enumeration")
        return []

    try:
        app_info = _VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=b"EasyScribe",
            applicationVersion=1,
            pEngineName=b"none",
            engineVersion=1,
            apiVersion=(1 << 22),  # Vulkan 1.0
        )
        create_info = _VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=ctypes.byref(app_info),
        )

        instance = ctypes.c_void_p()
        if vk.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance)) != VK_SUCCESS:
            logger.debug("vkCreateInstance failed")
            return []

        count = ctypes.c_uint32(0)
        vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)

        if count.value == 0:
            vk.vkDestroyInstance(instance, None)
            return []

        devices = (ctypes.c_void_p * count.value)()
        vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), devices)

        gpus: list[dict] = []
        for i in range(count.value):
            props = _VkPhysicalDeviceProperties()
            vk.vkGetPhysicalDeviceProperties(devices[i], ctypes.byref(props))
            name = props.deviceName.decode("utf-8", errors="replace").rstrip("\x00")
            gpus.append({"index": i, "name": name})
            logger.debug(f"Vulkan GPU {i}: {name}")

        vk.vkDestroyInstance(instance, None)
        logger.info(f"Vulkan: {len(gpus)} device(s) found")
        return gpus

    except Exception as exc:
        logger.debug(f"Vulkan enumeration error: {exc}")
        return []
