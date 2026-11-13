// sr_probe: checks that the GPU/driver exposes what this renderer needs.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <vector>

static bool hasExt(const std::vector<VkExtensionProperties>& v, const char* n) {
  for (auto& e : v) if (!std::strcmp(e.extensionName, n)) return true;
  return false;
}

int main() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "sr_probe";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;

  VkInstance inst = VK_NULL_HANDLE;
  if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
    std::fprintf(stderr, "vkCreateInstance failed (is a Vulkan 1.3 driver installed?)\n");
    return 1;
  }

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(inst, &n, devs.data());

  // bufferDeviceAddress is core in Vulkan 1.2+, so it is not listed here.
  const char* required[] = {
      "VK_KHR_acceleration_structure",
      "VK_KHR_ray_tracing_pipeline",
      "VK_KHR_ray_query",
      "VK_KHR_deferred_host_operations",
  };

  int ok = 0;
  for (auto d : devs) {
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p.pNext = &rt;
    vkGetPhysicalDeviceProperties2(d, &p);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(d, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    vkEnumerateDeviceExtensionProperties(d, nullptr, &ne, exts.data());

    std::printf("GPU: %s (API %u.%u)\n", p.properties.deviceName,
                VK_VERSION_MAJOR(p.properties.apiVersion),
                VK_VERSION_MINOR(p.properties.apiVersion));
    bool all = true;
    for (auto r : required) {
      bool h = hasExt(exts, r);
      all = all && h;
      std::printf("  [%s] %s\n", h ? "x" : " ", r);
    }
    if (all) {
      ++ok;
      std::printf("  maxRayRecursionDepth=%u shaderGroupHandleSize=%u\n",
                  rt.maxRayRecursionDepth, rt.shaderGroupHandleSize);
    }
  }
  vkDestroyInstance(inst, nullptr);
  std::printf(ok ? "\nReady: %d RT-capable GPU(s).\n" : "\nNo RT-capable GPU found.\n", ok);
  return ok ? 0 : 2;
}
