#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <cassert>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "=== [Test 7] VK_LAYER_MALI_csf_dump Load & Vulkan Loader Chain Verification ===\n";

    void* handle = dlopen(CSF_DUMP_LAYER_PATH, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "dlopen failed: " << dlerror() << "\n";
        return 1;
    }

    auto neg_fn = reinterpret_cast<PFN_vkNegotiateLoaderLayerInterfaceVersion>(
        dlsym(handle, "vkNegotiateLoaderLayerInterfaceVersion"));
    assert(neg_fn != nullptr);

    VkNegotiateLayerInterface neg{};
    neg.sType = LAYER_NEGOTIATE_INTERFACE_STRUCT;
    neg.loaderLayerInterfaceVersion = 2;
    VkResult nr = neg_fn(&neg);
    assert(nr == VK_SUCCESS);
    assert(neg.loaderLayerInterfaceVersion == 2);
    assert(neg.pfnGetInstanceProcAddr != nullptr);
    assert(neg.pfnGetDeviceProcAddr != nullptr);
    std::cout << "[PASS] vkNegotiateLoaderLayerInterfaceVersion negotiated v2\n";

    auto selftest_fn = reinterpret_cast<bool (*)(const char*, uint32_t)>(
        dlsym(handle, "mali_csf_dump_layer_selftest"));
    assert(selftest_fn != nullptr);
    bool hooks_ok = selftest_fn("/tmp/mali_csf_dump_test", 1u);
    assert(hooks_ok);
    std::cout << "[PASS] GOT/PLT hook installer initialized cleanly\n";

    // Verify Vulkan loader discovers VK_LAYER_MALI_csf_dump via VK_LAYER_PATH
    setenv("VK_LAYER_PATH", CSF_DUMP_LAYER_DIR, 1);
    const char* default_icd = (access("/usr/share/vulkan/icd.d/lvp_icd.json", R_OK) == 0)
        ? "/usr/share/vulkan/icd.d/lvp_icd.json"
        : "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json";
    setenv("VK_ICD_FILENAMES", default_icd, 0);
    setenv("MALI_CSF_DUMP_ENABLE", "1", 1);

    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
    bool found = false;
    for (const auto& lp : layers) {
        if (std::strcmp(lp.layerName, "VK_LAYER_MALI_csf_dump") == 0) {
            found = true;
            break;
        }
    }
    assert(found);
    std::cout << "[PASS] Vulkan loader enumerated VK_LAYER_MALI_csf_dump\n";

    const char* req_layers[] = { "VK_LAYER_MALI_csf_dump" };
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "test_csf_dump_layer";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = 1;
    ici.ppEnabledLayerNames = req_layers;

    VkInstance inst = VK_NULL_HANDLE;
    VkResult ir = vkCreateInstance(&ici, nullptr, &inst);
    assert(ir == VK_SUCCESS && inst != VK_NULL_HANDLE);
    vkDestroyInstance(inst, nullptr);
    std::cout << "[PASS] vkCreateInstance / vkDestroyInstance chained through VK_LAYER_MALI_csf_dump\n";

    dlclose(handle);
    return 0;
}
