cmake_minimum_required(VERSION 3.21)

# OpenCV 来自 vcpkg，若未显式指定工具链则自动使用 E:/Lib/vcpkg
set(VCPKG_ROOT_DIR "E:/Lib/vcpkg" CACHE PATH "vcpkg root directory")
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${VCPKG_ROOT_DIR}/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file")
endif()

project(UseAiForWinML VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

# onnxruntime 解压目录，需包含 include/ 与 lib/
set(ONNXRUNTIME_ROOT "E:/Lib/onnxruntime-win-x64-1.31.1" CACHE PATH "ONNX Runtime root directory")
foreach(required_file
        "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "onnxruntime 文件缺失: ${required_file}\n请用 -DONNXRUNTIME_ROOT=<解压目录> 指定正确路径")
    endif()
endforeach()

find_package(OpenCV CONFIG REQUIRED)

add_library(cnocr SHARED
    cnocr.h cnocr.cpp
    cnstd.h cnstd.cpp
    modle.h modle.cpp
    det_box.h
    db_postprocess.h db_postprocess.cpp
    text_utils.h text_utils.cpp
)
target_include_directories(cnocr PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
)
target_link_libraries(cnocr PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
)

add_executable(UseAiForWinML main.cpp)
target_link_libraries(UseAiForWinML PRIVATE cnocr ${OpenCV_LIBS})

# 运行时依赖：onnxruntime 的 DLL 拷到输出目录；OpenCV 的 DLL 由 vcpkg applocal 自动部署
foreach(target cnocr UseAiForWinML)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:${target}>"
        VERBATIM
    )
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
