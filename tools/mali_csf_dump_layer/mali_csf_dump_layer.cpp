// ============================================================================
// mali_csf_dump_layer.cpp
// Vulkan Layer Entrypoints & Dispatch Table for VK_LAYER_MALI_csf_dump.
//
// Core subsystems are cleanly factored into separate translation units:
//   - csf_kbase_uapi.h      : Mali kbase CSF UAPI structs & ioctl constants
//   - csf_config_log.{h,cpp}: Android props/env config & structured diagnostics
//   - csf_va_tracker.{h,cpp}: SAME_VA / cookie / CUSTOM_VA / DMA-BUF VA tracker
//   - csf_interpreter.{h,cpp}: Per-queue CSF v10–v14 ISA interpreter
//   - csf_dumper.{h,cpp}    : Dynamic VA-bounded multi-draw & compute dumper
//   - csf_elf_hooks.{h,cpp} : ELF GOT/PLT patching + lazy dlopen interception
//   - csf_layer_state.{h,cpp}: Central LayerState & queue kick/present orchestration
// ============================================================================

#include "csf_config_log.h"
#include "csf_elf_hooks.h"
#include "csf_layer_state.h"

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

using namespace mali_csf_dump;

struct InstanceDispatch {
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkDestroyInstance destroy_instance = nullptr;
    PFN_vkCreateDevice create_device = nullptr;
};

struct DeviceDispatch {
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;
};

struct VkDispatchGlobals {
    std::mutex mu;
    std::unordered_map<void*, InstanceDispatch> inst_disp;
    std::unordered_map<void*, DeviceDispatch> dev_disp;
};

VkDispatchGlobals& vk_globals() {
    static VkDispatchGlobals g;
    return g;
}

void* dispatch_key(void* handle) {
    return *reinterpret_cast<void**>(handle);
}

__attribute__((constructor)) void on_layer_load() {
    install_mali_hooks();
}

VKAPI_ATTR VkResult VKAPI_CALL layer_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance) {
    install_mali_hooks();

    const auto* layer_info = reinterpret_cast<const VkLayerInstanceCreateInfo*>(pCreateInfo->pNext);
    while (layer_info && (layer_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
                          layer_info->function != VK_LAYER_LINK_INFO)) {
        layer_info = reinterpret_cast<const VkLayerInstanceCreateInfo*>(layer_info->pNext);
    }
    if (!layer_info || !layer_info->u.pLayerInfo) {
        csf_log(LogLevel::Error, "vk", "Missing VK_LAYER_LINK_INFO in vkCreateInstance");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr next_gipa = layer_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const_cast<VkLayerInstanceCreateInfo*>(layer_info)->u.pLayerInfo = layer_info->u.pLayerInfo->pNext;
    auto next_create = reinterpret_cast<PFN_vkCreateInstance>(next_gipa(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = next_create(pCreateInfo, pAllocator, pInstance);
    install_mali_hooks();
    if (res == VK_SUCCESS && *pInstance) {
        InstanceDispatch d{};
        d.gipa = next_gipa;
        d.destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(next_gipa(*pInstance, "vkDestroyInstance"));
        d.create_device = reinterpret_cast<PFN_vkCreateDevice>(next_gipa(*pInstance, "vkCreateDevice"));
        std::lock_guard<std::mutex> lk(vk_globals().mu);
        vk_globals().inst_disp[dispatch_key(*pInstance)] = d;
    }
    return res;
}

VKAPI_ATTR void VKAPI_CALL layer_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    InstanceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(vk_globals().mu);
        auto it = vk_globals().inst_disp.find(dispatch_key(instance));
        if (it != vk_globals().inst_disp.end()) {
            d = it->second;
            vk_globals().inst_disp.erase(it);
        }
    }
    if (d.destroy_instance) d.destroy_instance(instance, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL layer_CreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice* pDevice) {
    install_mali_hooks();

    const auto* layer_info = reinterpret_cast<const VkLayerDeviceCreateInfo*>(pCreateInfo->pNext);
    while (layer_info && (layer_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
                          layer_info->function != VK_LAYER_LINK_INFO)) {
        layer_info = reinterpret_cast<const VkLayerDeviceCreateInfo*>(layer_info->pNext);
    }
    if (!layer_info || !layer_info->u.pLayerInfo) {
        csf_log(LogLevel::Error, "vk", "Missing VK_LAYER_LINK_INFO in vkCreateDevice");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr next_gipa = layer_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_gdpa = layer_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    const_cast<VkLayerDeviceCreateInfo*>(layer_info)->u.pLayerInfo = layer_info->u.pLayerInfo->pNext;
    auto next_create = reinterpret_cast<PFN_vkCreateDevice>(next_gipa(VK_NULL_HANDLE, "vkCreateDevice"));
    if (!next_create) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = next_create(physicalDevice, pCreateInfo, pAllocator, pDevice);
    if (res == VK_SUCCESS && *pDevice) {
        DeviceDispatch d{};
        d.gdpa = next_gdpa;
        d.destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(next_gdpa(*pDevice, "vkDestroyDevice"));
        d.queue_present = reinterpret_cast<PFN_vkQueuePresentKHR>(next_gdpa(*pDevice, "vkQueuePresentKHR"));
        std::lock_guard<std::mutex> lk(vk_globals().mu);
        vk_globals().dev_disp[dispatch_key(*pDevice)] = d;
    }
    return res;
}

VKAPI_ATTR void VKAPI_CALL layer_DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
    DeviceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(vk_globals().mu);
        auto it = vk_globals().dev_disp.find(dispatch_key(device));
        if (it != vk_globals().dev_disp.end()) {
            d = it->second;
            vk_globals().dev_disp.erase(it);
        }
    }
    if (d.destroy_device) d.destroy_device(device, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL layer_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    DeviceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(vk_globals().mu);
        auto it = vk_globals().dev_disp.find(dispatch_key(queue));
        if (it != vk_globals().dev_disp.end()) d = it->second;
    }
    VkResult res = d.queue_present ? d.queue_present(queue, pPresentInfo) : VK_SUCCESS;

    {
        std::lock_guard<std::mutex> lk(layer_state().mu);
        process_frame_present_locked(layer_state());
    }
    return res;
}

} // namespace

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetDeviceProcAddr(VkDevice device, const char* pName) {
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_GetDeviceProcAddr);
    if (std::strcmp(pName, "vkDestroyDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_DestroyDevice);
    if (std::strcmp(pName, "vkQueuePresentKHR") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_QueuePresentKHR);
    DeviceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(vk_globals().mu);
        auto it = vk_globals().dev_disp.find(dispatch_key(device));
        if (it != vk_globals().dev_disp.end()) d = it->second;
    }
    return d.gdpa ? d.gdpa(device, pName) : nullptr;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL layer_GetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (std::strcmp(pName, "vkGetInstanceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_GetInstanceProcAddr);
    if (std::strcmp(pName, "vkCreateInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateInstance);
    if (std::strcmp(pName, "vkDestroyInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_DestroyInstance);
    if (std::strcmp(pName, "vkCreateDevice") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_CreateDevice);
    if (std::strcmp(pName, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&layer_GetDeviceProcAddr);
    InstanceDispatch d{};
    {
        std::lock_guard<std::mutex> lk(vk_globals().mu);
        auto it = vk_globals().inst_disp.find(dispatch_key(instance));
        if (it != vk_globals().inst_disp.end()) d = it->second;
    }
    return d.gipa ? d.gipa(instance, pName) : nullptr;
}

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    return layer_GetDeviceProcAddr(device, pName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    return layer_GetInstanceProcAddr(instance, pName);
}

VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    mali_csf_dump::install_mali_hooks();
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (pVersionStruct->loaderLayerInterfaceVersion >= 2) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetInstanceProcAddr = &layer_GetInstanceProcAddr;
        pVersionStruct->pfnGetDeviceProcAddr = &layer_GetDeviceProcAddr;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    return VK_SUCCESS;
}

// ============================================================================
// Self-Test Entrypoint (Backward-Compatible C ABI)
// ============================================================================
bool mali_csf_dump_layer_selftest(const char* out_dir, uint32_t target_frame) {
    setenv("MALI_CSF_DUMP_ENABLE", "1", 1);
    setenv("MALI_CSF_DUMP_DIR", out_dir, 1);
    char fbuf[16];
    std::snprintf(fbuf, sizeof(fbuf), "%u", target_frame);
    setenv("MALI_CSF_DUMP_FRAME", fbuf, 1);

    mali_csf_dump::reset_layer_state_for_test(target_frame);
    return mali_csf_dump::layer_state().hooks_installed;
}

} // extern "C"
