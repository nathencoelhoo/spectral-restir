// sr_render: headless Vulkan ray-query renderer (milestone 1).
// Usage: sr_render <mode 0=furnace|1=prism+NEE|2=prism brute-force|3=prism+NEE with ReSTIR caustic channel|4=caustic channel only|
//                  5=prism+NEE with light-traced caustic channel|6=light-traced caustic channel only>
//                  <width> <height> <frames> <pathtrace.spv> <out.ppm> [raw.pfm|-] [frameOffset] [candidates] [reuse] [photons]
// reuse: bit 0 = temporal, bit 1 = spatial (modes 3/4)
// raw.pfm holds linear-sRGB means (no tone map) for error measurement; frameOffset shifts the
// RNG seeds so independent renders (e.g. reference vs test) never share samples.
// NOT YET COMPILED against a Vulkan SDK. Expect small fixes on first build.
// Resources are not freed: the process exits right after rendering.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
  std::fprintf(stderr, "%s failed (%d) at %s:%d\n", #x, (int)r_, __FILE__, __LINE__); std::exit(1); } } while (0)

// ---- tiny math + mesh -------------------------------------------------------
struct V3 { float x, y, z; };
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

enum Mat : uint32_t { MAT_DIFFUSE = 0, MAT_EMISSIVE = 1, MAT_GLASS = 2 };

struct Mesh {
  std::vector<float> verts;       // xyzw
  std::vector<uint32_t> idx, mat;
  uint32_t vert(V3 p) { verts.insert(verts.end(), {p.x, p.y, p.z, 1.f}); return uint32_t(verts.size() / 4 - 1); }
  V3 at(uint32_t i) const { return {verts[4 * i], verts[4 * i + 1], verts[4 * i + 2]}; }
  // Orient so the geometric normal (cross(b-a, c-a)) points toward `facing`.
  void tri(uint32_t a, uint32_t b, uint32_t c, V3 facing, Mat m) {
    if (dot(cross(at(b) - at(a), at(c) - at(a)), facing) < 0) std::swap(b, c);
    idx.insert(idx.end(), {a, b, c}); mat.push_back(m);
  }
  void quad(V3 p0, V3 p1, V3 p2, V3 p3, V3 facing, Mat m) {
    uint32_t a = vert(p0), b = vert(p1), c = vert(p2), d = vert(p3);
    tri(a, b, c, facing, m); tri(a, c, d, facing, m);
  }
};

static Mesh buildScene(int mode) {
  Mesh m;
  // Equilateral glass prism (side 2, length 2), outward-oriented via body centroid.
  const float s3 = std::sqrt(3.f);
  uint32_t v[6] = {m.vert({-1, 0, -1}), m.vert({1, 0, -1}), m.vert({0, s3, -1}),
                   m.vert({-1, 0, 1}),  m.vert({1, 0, 1}),  m.vert({0, s3, 1})};
  V3 c{0, s3 / 3.f, 0};
  auto body = [&](uint32_t a, uint32_t b, uint32_t d) {
    V3 cen = (m.at(a) + m.at(b) + m.at(d));
    cen = {cen.x / 3, cen.y / 3, cen.z / 3};
    m.tri(a, b, d, cen - c, MAT_GLASS);
  };
  body(v[0], v[1], v[2]); body(v[3], v[4], v[5]);                       // caps
  body(v[0], v[1], v[4]); body(v[0], v[4], v[3]);                       // bottom
  body(v[1], v[2], v[5]); body(v[1], v[5], v[4]);                       // right
  body(v[2], v[0], v[3]); body(v[2], v[3], v[5]);                       // left
  if (mode >= 1) {
    m.quad({-6, 0, -6}, {6, 0, -6}, {6, 0, 6}, {-6, 0, 6}, {0, 1, 0}, MAT_DIFFUSE);     // floor
    m.quad({-1.5f, 4, -1.5f}, {1.5f, 4, -1.5f}, {1.5f, 4, 1.5f}, {-1.5f, 4, 1.5f}, {0, -1, 0}, MAT_EMISSIVE); // light
  }
  return m;
}

// ---- Vulkan helpers ---------------------------------------------------------
static VkDevice gDev;
static VkPhysicalDeviceMemoryProperties gMem;
static PFN_vkCreateAccelerationStructureKHR pCreateAS;
static PFN_vkGetAccelerationStructureBuildSizesKHR pBuildSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR pCmdBuildAS;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pASAddr;

static uint32_t findMem(uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < gMem.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (gMem.memoryTypes[i].propertyFlags & want) == want) return i;
  std::fprintf(stderr, "no suitable memory type\n"); std::exit(1);
}

struct Buf { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; VkDeviceAddress addr = 0; void* map = nullptr; };

static Buf makeBuf(VkDeviceSize size, VkBufferUsageFlags usage) {
  Buf r;
  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ci.size = size; ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK(vkCreateBuffer(gDev, &ci, nullptr, &r.b));
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(gDev, r.b, &mr);
  VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &fi; ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK(vkAllocateMemory(gDev, &ai, nullptr, &r.m));
  VK(vkBindBufferMemory(gDev, r.b, r.m, 0));
  VK(vkMapMemory(gDev, r.m, 0, VK_WHOLE_SIZE, 0, &r.map));
  VkBufferDeviceAddressInfo bi{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; bi.buffer = r.b;
  r.addr = vkGetBufferDeviceAddress(gDev, &bi);
  return r;
}

struct AS { VkAccelerationStructureKHR h = VK_NULL_HANDLE; Buf buf; VkDeviceAddress addr = 0; };

static AS buildAS(VkCommandBuffer cmd, VkAccelerationStructureTypeKHR type,
                  const VkAccelerationStructureGeometryKHR& geom, uint32_t primCount, uint32_t scratchAlign) {
  VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bi.type = type; bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR; bi.geometryCount = 1; bi.pGeometries = &geom;
  VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  pBuildSizes(gDev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &primCount, &sz);

  AS as;
  as.buf = makeBuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
  VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  ci.buffer = as.buf.b; ci.size = sz.accelerationStructureSize; ci.type = type;
  VK(pCreateAS(gDev, &ci, nullptr, &as.h));
  VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  ai.accelerationStructure = as.h; as.addr = pASAddr(gDev, &ai);

  Buf scratch = makeBuf(sz.buildScratchSize + scratchAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  VkDeviceAddress a = scratchAlign ? (scratch.addr + scratchAlign - 1) & ~VkDeviceAddress(scratchAlign - 1) : scratch.addr;
  bi.dstAccelerationStructure = as.h; bi.scratchData.deviceAddress = a;
  VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount = primCount;
  const VkAccelerationStructureBuildRangeInfoKHR* pr = &range;
  pCmdBuildAS(cmd, 1, &bi, &pr);
  return as;
}

static double yBar(double l) {
  auto lobe = [](double x, double mu, double s1, double s2) { double t = (x - mu) / (x < mu ? s1 : s2); return std::exp(-0.5 * t * t); };
  return 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1);
}

struct PC { uint32_t width, height, frame, mode, passId, cands, reuse, photons; };   // 32 bytes, matches the shader

int main(int argc, char** argv) {
  int mode = argc > 1 ? std::atoi(argv[1]) : 0;
  uint32_t W = argc > 2 ? std::atoi(argv[2]) : 512, H = argc > 3 ? std::atoi(argv[3]) : 512;
  int frames = argc > 4 ? std::atoi(argv[4]) : 64;
  std::string spvPath = argc > 5 ? argv[5] : "pathtrace.spv", outPath = argc > 6 ? argv[6] : "out.ppm";
  std::string rawPath = argc > 7 ? argv[7] : "";
  uint32_t frameOffset = argc > 8 ? uint32_t(std::strtoul(argv[8], nullptr, 10)) : 0;
  uint32_t cands = argc > 9 ? uint32_t(std::strtoul(argv[9], nullptr, 10)) : 4;   // ReSTIR candidates per pixel per frame
  uint32_t reuse = argc > 10 ? uint32_t(std::strtoul(argv[10], nullptr, 10)) : 0;  // reuse bits
  uint32_t photons = argc > 11 ? uint32_t(std::strtoul(argv[11], nullptr, 10)) : 4; // photons per pixel per frame (modes 5, 6)

  // Instance + device
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
  VkInstance inst; VK(vkCreateInstance(&ici, nullptr, &inst));
  uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
  std::vector<VkPhysicalDevice> devs(n); vkEnumeratePhysicalDevices(inst, &n, devs.data());

  const char* exts[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  for (auto d : devs) {
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(d, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ex(ne); vkEnumerateDeviceExtensionProperties(d, nullptr, &ne, ex.data());
    int found = 0;
    for (auto r : exts) for (auto& e : ex) if (!std::strcmp(e.extensionName, r)) { ++found; break; }
    if (found == 3) { phys = d; break; }
  }
  if (!phys) { std::fprintf(stderr, "No GPU with ray query support found (run sr_probe).\n"); return 1; }
  VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &asp;
  vkGetPhysicalDeviceProperties2(phys, &p2);
  std::printf("GPU: %s\n", p2.properties.deviceName);
  vkGetPhysicalDeviceMemoryProperties(phys, &gMem);

  uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
  std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
  uint32_t qfi = 0; for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }

  float prio = 1.f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.bufferDeviceAddress = VK_TRUE;
  VkPhysicalDeviceAccelerationStructureFeaturesKHR fas{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  fas.accelerationStructure = VK_TRUE; fas.pNext = &f12;
  VkPhysicalDeviceRayQueryFeaturesKHR frq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  frq.rayQuery = VK_TRUE; frq.pNext = &fas;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &frq; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 3; dci.ppEnabledExtensionNames = exts;
  VK(vkCreateDevice(phys, &dci, nullptr, &gDev));
  VkQueue queue; vkGetDeviceQueue(gDev, qfi, 0, &queue);

  pCreateAS = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(gDev, "vkCreateAccelerationStructureKHR");
  pBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(gDev, "vkGetAccelerationStructureBuildSizesKHR");
  pCmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(gDev, "vkCmdBuildAccelerationStructuresKHR");
  pASAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(gDev, "vkGetAccelerationStructureDeviceAddressKHR");

  // Scene buffers
  Mesh mesh = buildScene(mode);
  const VkBufferUsageFlags asInput = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  Buf vbuf = makeBuf(mesh.verts.size() * 4, asInput);  std::memcpy(vbuf.map, mesh.verts.data(), mesh.verts.size() * 4);
  Buf ibuf = makeBuf(mesh.idx.size() * 4, asInput);    std::memcpy(ibuf.map, mesh.idx.data(), mesh.idx.size() * 4);
  Buf mbuf = makeBuf(mesh.mat.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); std::memcpy(mbuf.map, mesh.mat.data(), mesh.mat.size() * 4);
  VkDeviceSize accumBytes = VkDeviceSize(W) * H * 16;
  Buf abuf = makeBuf(accumBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); std::memset(abuf.map, 0, accumBytes);
  // ReSTIR caustic channel (mode 3): one 32-byte reservoir and one vec4 primary-hit record per pixel.
  // Two slots each (ping-pong by frame parity): current frame is written, previous frame is read.
  Buf rbuf = makeBuf(VkDeviceSize(W) * H * 32 * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); std::memset(rbuf.map, 0, size_t(W) * H * 32 * 2);
  Buf gbuf = makeBuf(VkDeviceSize(W) * H * 16 * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); std::memset(gbuf.map, 0, size_t(W) * H * 16 * 2);
  // Light-tracing splat accumulator: 3 fixed-point ints per pixel, cleared every frame by pass 3.
  Buf sbuf = makeBuf(VkDeviceSize(W) * H * 3 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); std::memset(sbuf.map, 0, size_t(W) * H * 3 * 4);

  // Command pool
  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = qfi;
  VkCommandPool pool; VK(vkCreateCommandPool(gDev, &cpi, nullptr, &pool));
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
  VkCommandBuffer cmd; VK(vkAllocateCommandBuffers(gDev, &cai, &cmd));
  VkCommandBufferBeginInfo bgi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bgi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  // BLAS + TLAS
  VK(vkBeginCommandBuffer(cmd, &bgi));
  VkAccelerationStructureGeometryKHR bg{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  bg.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR; bg.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  auto& tri = bg.geometry.triangles;
  tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; tri.vertexData.deviceAddress = vbuf.addr;
  tri.vertexStride = 16; tri.maxVertex = uint32_t(mesh.verts.size() / 4 - 1);
  tri.indexType = VK_INDEX_TYPE_UINT32; tri.indexData.deviceAddress = ibuf.addr;
  uint32_t align = std::max(1u, asp.minAccelerationStructureScratchOffsetAlignment);
  AS blas = buildAS(cmd, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, bg, uint32_t(mesh.mat.size()), align);

  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR; mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &mb, 0, nullptr, 0, nullptr);

  Buf instBuf = makeBuf(sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  VkAccelerationStructureInstanceKHR inst0{};
  inst0.transform.matrix[0][0] = inst0.transform.matrix[1][1] = inst0.transform.matrix[2][2] = 1.f;
  inst0.mask = 0xFF; inst0.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  inst0.accelerationStructureReference = blas.addr;
  std::memcpy(instBuf.map, &inst0, sizeof(inst0));
  VkAccelerationStructureGeometryKHR tg{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR; tg.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  tg.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  tg.geometry.instances.arrayOfPointers = VK_FALSE; tg.geometry.instances.data.deviceAddress = instBuf.addr;
  AS tlas = buildAS(cmd, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tg, 1, align);
  VK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
  VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK(vkQueueWaitIdle(queue));

  // Descriptors
  VkDescriptorSetLayoutBinding lb[8]{};
  lb[0].binding = 0; lb[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  for (int i = 1; i < 8; ++i) { lb[i].binding = i; lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; }
  for (auto& b : lb) { b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dli.bindingCount = 8; dli.pBindings = lb;
  VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(gDev, &dli, nullptr, &dsl));
  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7}};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 1; dpi.poolSizeCount = 2; dpi.pPoolSizes = ps;
  VkDescriptorPool dpool; VK(vkCreateDescriptorPool(gDev, &dpi, nullptr, &dpool));
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = dpool; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
  VkDescriptorSet set; VK(vkAllocateDescriptorSets(gDev, &dai, &set));

  VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  asw.accelerationStructureCount = 1; asw.pAccelerationStructures = &tlas.h;
  VkDescriptorBufferInfo bi[7] = {{abuf.b, 0, VK_WHOLE_SIZE}, {vbuf.b, 0, VK_WHOLE_SIZE}, {ibuf.b, 0, VK_WHOLE_SIZE}, {mbuf.b, 0, VK_WHOLE_SIZE},
                                  {rbuf.b, 0, VK_WHOLE_SIZE}, {gbuf.b, 0, VK_WHOLE_SIZE}, {sbuf.b, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet w[8]{};
  for (int i = 0; i < 8; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1; }
  w[0].pNext = &asw; w[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  for (int i = 1; i < 8; ++i) { w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i - 1]; }
  vkUpdateDescriptorSets(gDev, 8, w, 0, nullptr);

  // Pipeline
  std::ifstream f(spvPath, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", spvPath.c_str()); return 1; }
  std::vector<char> spv(size_t(f.tellg())); f.seekg(0); f.read(spv.data(), spv.size());
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spv.size(); smi.pCode = reinterpret_cast<const uint32_t*>(spv.data());
  VkShaderModule sm; VK(vkCreateShaderModule(gDev, &smi, nullptr, &sm));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
  VkPipelineLayout pl; VK(vkCreatePipelineLayout(gDev, &pli, nullptr, &pl));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main"; cpci.layout = pl;
  VkPipeline pipe; VK(vkCreateComputePipelines(gDev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

  // Render: one sample per pixel per dispatch
  for (int fr = 0; fr < frames; ++fr) {
    VK(vkResetCommandBuffer(cmd, 0));
    VK(vkBeginCommandBuffer(cmd, &bgi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
    PC pc{W, H, uint32_t(fr) + frameOffset, uint32_t(mode), 1, cands, reuse, photons};
    if (mode == 5) pc.cands = 0;           // mode 5 only needs the primary-hit flag from pass 0
    auto runPass = [&](uint32_t passId) {
      pc.passId = passId;
      vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(cmd, (W + 7) / 8, (H + 7) / 8, 1);
    };
    auto barrier = [&]() {
      VkMemoryBarrier cbar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      cbar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; cbar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &cbar, 0, nullptr, 0, nullptr);
    };
    if (mode == 3 || mode == 4) {          // pass 0: candidates, pass 2: reuse (optional), pass 1: shading
      runPass(0); barrier();
      if (reuse) { runPass(2); barrier(); }
    }
    if (mode == 5 || mode == 6) {          // pass 3: clear splats, pass 0 (mode 5): primary hit, pass 4: photons
      runPass(3);
      if (mode == 5) runPass(0);
      barrier();
      runPass(4); barrier();
    }
    runPass(1);
    VK(vkEndCommandBuffer(cmd));
    VK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE)); VK(vkQueueWaitIdle(queue));
  }

  // Readback, normalise by integral of ybar so a flat spectrum of 1 gives Y = 1
  double yInt = 0; for (double l = 360; l < 830; l += 0.01) yInt += yBar(l) * 0.01;
  const float* acc = static_cast<const float*>(abuf.map);
  double sumY = 0, sumY2 = 0; size_t np = size_t(W) * H;
  std::vector<unsigned char> rgb(np * 3);
  std::vector<float> linear(np * 3);
  for (size_t i = 0; i < np; ++i) {
    double cnt = std::max(1.0f, acc[4 * i + 3]);
    double X = acc[4 * i] / cnt / yInt, Y = acc[4 * i + 1] / cnt / yInt, Z = acc[4 * i + 2] / cnt / yInt;
    sumY += Y; sumY2 += Y * Y;
    double lin[3] = {3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z,
                    -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z,
                     0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z};
    for (int c = 0; c < 3; ++c) {
      linear[3 * i + c] = float(lin[c]);
      double v = lin[c];
      if (mode >= 1) {   // ACES filmic fit (Narkowicz) so the bright floor does not clip
        v = std::max(0.0, v);
        v = (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14);
      }
      v = std::min(1.0, std::max(0.0, v));
      v = v <= 0.0031308 ? 12.92 * v : 1.055 * std::pow(v, 1 / 2.4) - 0.055;
      rgb[3 * i + c] = (unsigned char)std::lround(v * 255.0);
    }
  }
  std::ofstream out(outPath, std::ios::binary);
  out << "P6\n" << W << " " << H << "\n255\n"; out.write(reinterpret_cast<char*>(rgb.data()), rgb.size());
  double meanY = sumY / np, sd = std::sqrt(std::max(0.0, sumY2 / np - meanY * meanY));
  std::printf("wrote %s | mean Y = %.4f, std over pixels = %.4f\n", outPath.c_str(), meanY, sd);

  if (!rawPath.empty() && rawPath != "-") {   // linear float image for tools/compare.cpp
    std::ofstream pf(rawPath, std::ios::binary);
    pf << "PF\n" << W << " " << H << "\n-1.0\n";
    for (int y = int(H) - 1; y >= 0; --y)
      pf.write(reinterpret_cast<const char*>(&linear[size_t(y) * W * 3]), std::streamsize(W) * 3 * sizeof(float));
  }
  if (mode == 0) {   // white-furnace: lossless glass in a flat env must give Y = 1
    bool pass = std::fabs(meanY - 1.0) < 0.02;
    std::printf("furnace test: %s (expect mean Y = 1.0 +/- 0.02)\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 3;
  }
  return 0;
}
