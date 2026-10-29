cmake_minimum_required(VERSION 3.21)

# OpenCV 来自 vcpkg / Homebrew / apt；Windows 下若未显式指定工具链则自动使用 E:/Lib/vcpkg
set(VCPKG_ROOT_DIR "E:/Lib/vcpkg" CACHE PATH "vcpkg root directory")
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${VCPKG_ROOT_DIR}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file")
endif()

project(UseAiForWinML VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(MSVC)
    add_compile_options(/utf-8)
endif()

# onnxruntime 解压目录，需包含 include/ 与 lib/
# Windows 保留原有默认值；Linux/macOS 请通过 -DONNXRUNTIME_ROOT=... 指定
if(WIN32)
    set(_ONNXRUNTIME_DEFAULT "E:/Lib/onnxruntime-win-x64-1.31.1")
else()
    set(_ONNXRUNTIME_DEFAULT "")
endif()
set(ONNXRUNTIME_ROOT "${_ONNXRUNTIME_DEFAULT}" CACHE PATH "ONNX Runtime root directory")

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "请用 -DONNXRUNTIME_ROOT=<onnxruntime 解压目录> 指定 onnxruntime 路径（需包含 include/ 与 lib/）")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include"
    REQUIRED)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED)
message(STATUS "onnxruntime 库: ${ONNXRUNTIME_LIBRARY}")

find_package(OpenCV CONFIG REQUIRED)

add_library(cnocr SHARED
    cnocr.h cnocr.cpp
    cnstd.h cnstd.cpp
    modle.h modle.cpp
    det_box.h
    db_postprocess.h db_postprocess.cpp
    image_io.h image_io.cpp
)
target_include_directories(cnocr PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(cnocr PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

add_executable(UseAiForWinML main.cpp)
target_link_libraries(UseAiForWinML PRIVATE cnocr ${OpenCV_LIBS})

# 运行时依赖：onnxruntime 库拷到输出目录（Windows 拷 DLL，其他平台拷 so/dylib 并设置 RPATH）；
# OpenCV 的 DLL 由 vcpkg applocal 自动部署，Linux/macOS 使用系统包即可
if(WIN32)
    set(ONNXRUNTIME_RUNTIME_FILES
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
else()
    file(GLOB ONNXRUNTIME_RUNTIME_FILES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.*")
endif()

foreach(target cnocr UseAiForWinML)
    if(ONNXRUNTIME_RUNTIME_FILES)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_RUNTIME_FILES}
                "$<TARGET_FILE_DIR:${target}>"
            VERBATIM
        )
    endif()
    if(NOT WIN32)
        if(APPLE)
            set_target_properties(${target} PROPERTIES
                BUILD_RPATH "@loader_path"
                INSTALL_RPATH "@loader_path")
        else()
            set_target_properties(${target} PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN")
        endif()
    endif()
endforeach()

# 程序按相对路径读取 modle/*.onnx、dict/*.txt 与示例图片，拷到可执行文件旁便于直接运行
file(GLOB CNOCR_MODEL_FILES "${PROJECT_SOURCE_DIR}/modle/*.onnx")
file(GLOB CNOCR_DICT_FILES "${PROJECT_SOURCE_DIR}/dict/*.txt")
file(GLOB CNOCR_SAMPLE_IMAGES "${PROJECT_SOURCE_DIR}/*.png")
add_custom_command(TARGET UseAiForWinML POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:UseAiForWinML>/modle"
        "$<TARGET_FILE_DIR:UseAiForWinML>/dict"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CNOCR_MODEL_FILES} "$<TARGET_FILE_DIR:UseAiForWinML>/modle"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CNOCR_DICT_FILES} "$<TARGET_FILE_DIR:UseAiForWinML>/dict"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CNOCR_SAMPLE_IMAGES} "$<TARGET_FILE_DIR:UseAiForWinML>"
    VERBATIM
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
