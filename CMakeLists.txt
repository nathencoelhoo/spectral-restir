cmake_minimum_required(VERSION 3.20)
project(spectral_restir LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CPU reference tests for the spectral math (no GPU needed) ---------------
enable_testing()
add_executable(spectral_test tests/spectral_test.cpp)
add_test(NAME spectral COMMAND spectral_test)
add_executable(transport_test tests/transport_test.cpp)
add_test(NAME transport COMMAND transport_test)
add_executable(nee_test tests/nee_test.cpp)
add_test(NAME nee COMMAND nee_test)

# --- Vulkan ray-tracing capability probe ------------------------------------
find_package(Vulkan)
if(Vulkan_FOUND)
  add_executable(sr_probe src/main.cpp)
  target_link_libraries(sr_probe PRIVATE Vulkan::Vulkan)
else()
  message(STATUS "Vulkan SDK not found: skipping sr_probe")
endif()

# --- Headless spectral path tracer (needs Vulkan SDK + glslc + RT-capable GPU) --
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin $ENV{VULKAN_SDK}/Bin)
if(Vulkan_FOUND AND GLSLC)
  add_executable(sr_render src/render.cpp)
  target_link_libraries(sr_render PRIVATE Vulkan::Vulkan)
  set(SPV ${CMAKE_BINARY_DIR}/pathtrace.spv)
  add_custom_command(OUTPUT ${SPV}
    COMMAND ${GLSLC} --target-env=vulkan1.3 -O -I ${CMAKE_SOURCE_DIR}/shaders
            ${CMAKE_SOURCE_DIR}/shaders/pathtrace.comp -o ${SPV}
    DEPENDS shaders/pathtrace.comp shaders/spectral.glsl shaders/reservoir.glsl)
  add_custom_target(shaders ALL DEPENDS ${SPV})
  add_dependencies(sr_render shaders)
  # GPU furnace test: glass in a flat-spectrum environment must render as Y = 1.
  add_test(NAME furnace_gpu COMMAND sr_render 0 256 256 64 ${SPV} furnace.ppm)
else()
  message(STATUS "sr_render skipped (need Vulkan SDK and glslc)")
endif()
