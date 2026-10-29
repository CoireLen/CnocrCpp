cmake_minimum_required(VERSION 3.21)

# ---------------------------------------------------------------------------
# vcpkg：优先使用环境变量 VCPKG_ROOT 或显式 CMAKE_TOOLCHAIN_FILE；
# Windows 下再尝试常见安装目录。Linux/macOS 默认使用系统包（apt / Homebrew）。
# 必须在 project() 之前设置工具链。
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(_vcpkg_candidates "")
    if(DEFINED ENV{VCPKG_ROOT})
        list(APPEND _vcpkg_candidates "$ENV{VCPKG_ROOT}")
    endif()
    if(WIN32)
        list(APPEND _vcpkg_candidates
            "E:/Lib/vcpkg"
            "C:/vcpkg"
            "${CMAKE_CURRENT_LIST_DIR}/vcpkg")
    endif()
    foreach(_candidate IN LISTS _vcpkg_candidates)
        if(EXISTS "${_candidate}/scripts/buildsystems/vcpkg.cmake")
            set(CMAKE_TOOLCHAIN_FILE "${_candidate}/scripts/buildsystems/vcpkg.cmake"
                CACHE FILEPATH "vcpkg toolchain file")
            message(STATUS "使用 vcpkg 工具链: ${_candidate}")
            break()
        endif()
    endforeach()
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

# ---------------------------------------------------------------------------
# onnxruntime：自动查找，找不到时按平台下载官方包（或使用本地压缩包）
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "" CACHE PATH "onnxruntime 解压目录；留空则自动查找或下载")
set(ONNXRUNTIME_VERSION "1.31.1" CACHE STRING "自动下载使用的 onnxruntime 版本")
option(ONNXRUNTIME_AUTO_DOWNLOAD "找不到 onnxruntime 时自动下载对应平台的官方包" ON)
set(ONNXRUNTIME_PACKAGE_URL "" CACHE STRING "onnxruntime 下载地址；留空按平台与版本生成")
set(ONNXRUNTIME_PACKAGE_ARCHIVE "" CACHE FILEPATH "本地 onnxruntime 压缩包（离线安装时使用）")

# 已解压的 onnxruntime 包目录（包含 include/ 与 lib/），使用系统安装时为空
set(ONNXRUNTIME_PACKAGE_DIR "")

set(_ort_candidates "")
if(ONNXRUNTIME_ROOT)
    list(APPEND _ort_candidates "${ONNXRUNTIME_ROOT}")
endif()
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    file(TO_CMAKE_PATH "$ENV{ONNXRUNTIME_ROOT}" _ort_env_root)
    list(APPEND _ort_candidates "${_ort_env_root}")
endif()
file(GLOB _ort_local_dirs
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime*"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime*")
list(APPEND _ort_candidates ${_ort_local_dirs})
file(GLOB _ort_common_dirs
    "E:/Lib/onnxruntime-*"
    "C:/Lib/onnxruntime-*"
    "$ENV{HOME}/onnxruntime-*"
    "/opt/onnxruntime-*"
    "/usr/local/onnxruntime-*")
list(APPEND _ort_candidates ${_ort_common_dirs})

foreach(_candidate IN LISTS _ort_candidates)
    if(EXISTS "${_candidate}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_PACKAGE_DIR "${_candidate}")
        break()
    endif()
endforeach()

# 显式指定了根目录但内容不对时直接报错，避免悄悄改用其他版本
if(ONNXRUNTIME_ROOT AND NOT ONNXRUNTIME_PACKAGE_DIR)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT 下未找到 include/onnxruntime_cxx_api.h: ${ONNXRUNTIME_ROOT}\n"
        "请检查路径，或清空 ONNXRUNTIME_ROOT 让 CMake 自动查找/下载。")
endif()

# 优先使用探测到的解压包，其次系统 / vcpkg 中已安装的版本
if(ONNXRUNTIME_PACKAGE_DIR)
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_PACKAGE_DIR}/include" NO_DEFAULT_PATH)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS "${ONNXRUNTIME_PACKAGE_DIR}/lib" NO_DEFAULT_PATH)
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)

# 仍未找到时下载对应平台的官方包
if((NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
   AND (ONNXRUNTIME_AUTO_DOWNLOAD OR ONNXRUNTIME_PACKAGE_ARCHIVE))
    if(WIN32)
        if(CMAKE_SIZEOF_VOID_P EQUAL 4)
            set(_ort_platform "win-x86")
        else()
            set(_ort_platform "win-x64")
        endif()
        set(_ort_ext "zip")
    elseif(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(_ort_platform "osx-arm64")
        else()
            message(FATAL_ERROR
                "onnxruntime ${ONNXRUNTIME_VERSION} 官方未提供 macOS x86_64 安装包。\n"
                "请通过 -DONNXRUNTIME_ROOT=<本地安装目录> 或 "
                "-DONNXRUNTIME_PACKAGE_ARCHIVE=<压缩包> 指定，"
                "或自行从源码构建。")
        endif()
        set(_ort_ext "tgz")
    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(_ort_platform "linux-aarch64")
        else()
            set(_ort_platform "linux-x64")
        endif()
        set(_ort_ext "tgz")
    endif()

    set(_ort_archive_name "onnxruntime-${_ort_platform}-${ONNXRUNTIME_VERSION}.${_ort_ext}")
    set(_ort_deps_dir "${CMAKE_BINARY_DIR}/_deps/onnxruntime")
    file(MAKE_DIRECTORY "${_ort_deps_dir}")

    if(ONNXRUNTIME_PACKAGE_ARCHIVE)
        if(NOT EXISTS "${ONNXRUNTIME_PACKAGE_ARCHIVE}")
            message(FATAL_ERROR "ONNXRUNTIME_PACKAGE_ARCHIVE 指向的文件不存在: ${ONNXRUNTIME_PACKAGE_ARCHIVE}")
        endif()
        set(_ort_archive "${ONNXRUNTIME_PACKAGE_ARCHIVE}")
    else()
        set(_ort_archive "${_ort_deps_dir}/${_ort_archive_name}")
        if(ONNXRUNTIME_PACKAGE_URL)
            set(_ort_url "${ONNXRUNTIME_PACKAGE_URL}")
        else()
            set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${_ort_archive_name}")
        endif()
        if(NOT EXISTS "${_ort_archive}")
            message(STATUS "未找到 onnxruntime，正在下载: ${_ort_url}")
            file(DOWNLOAD "${_ort_url}" "${_ort_archive}"
                STATUS _ort_download_status
                SHOW_PROGRESS
                TLS_VERIFY ON
                INACTIVITY_TIMEOUT 60
                TIMEOUT 1800)
            list(GET _ort_download_status 0 _ort_download_code)
            if(NOT _ort_download_code EQUAL 0)
                list(GET _ort_download_status 1 _ort_download_error)
                file(REMOVE "${_ort_archive}")
                message(FATAL_ERROR
                    "onnxruntime 下载失败: ${_ort_download_error}\n"
                    "可手动下载 ${_ort_archive_name} 后使用 "
                    "-DONNXRUNTIME_PACKAGE_ARCHIVE=<压缩包路径>，"
                    "或解压后使用 -DONNXRUNTIME_ROOT=<目录>。")
            endif()
        endif()
    endif()

    message(STATUS "解压 onnxruntime: ${_ort_archive}")
    file(ARCHIVE_EXTRACT INPUT "${_ort_archive}" DESTINATION "${_ort_deps_dir}")

    file(GLOB _ort_extracted_dirs "${_ort_deps_dir}/*")
    foreach(_dir IN LISTS _ort_extracted_dirs)
        if(EXISTS "${_dir}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_PACKAGE_DIR "${_dir}")
            break()
        endif()
    endforeach()
    if(NOT ONNXRUNTIME_PACKAGE_DIR)
        message(FATAL_ERROR "解压后未在 ${_ort_deps_dir} 中找到 onnxruntime 头文件")
    endif()

    # 强制使用刚解压的包，避免与之前找到的头文件混用
    set(ONNXRUNTIME_INCLUDE_DIR "ONNXRUNTIME_INCLUDE_DIR-NOTFOUND" CACHE FILEPATH "" FORCE)
    set(ONNXRUNTIME_LIBRARY "ONNXRUNTIME_LIBRARY-NOTFOUND" CACHE FILEPATH "" FORCE)
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_PACKAGE_DIR}/include" NO_DEFAULT_PATH)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS "${ONNXRUNTIME_PACKAGE_DIR}/lib" NO_DEFAULT_PATH)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "未找到 onnxruntime。请通过 -DONNXRUNTIME_ROOT=<解压目录> 指定，"
        "或允许自动下载（默认开启）。")
endif()
message(STATUS "onnxruntime: ${ONNXRUNTIME_LIBRARY}")

# 从缓存的头文件路径反推解压包目录（便于重新生成时继续拷贝运行库）
if(NOT ONNXRUNTIME_PACKAGE_DIR AND ONNXRUNTIME_INCLUDE_DIR)
    string(FIND "${ONNXRUNTIME_INCLUDE_DIR}" "onnxruntime" _ort_include_tag)
    if(_ort_include_tag GREATER -1)
        get_filename_component(_ort_pkg_dir "${ONNXRUNTIME_INCLUDE_DIR}" DIRECTORY)
        if(EXISTS "${_ort_pkg_dir}/lib" AND
           EXISTS "${_ort_pkg_dir}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_PACKAGE_DIR "${_ort_pkg_dir}")
        endif()
    endif()
endif()
if(ONNXRUNTIME_PACKAGE_DIR)
    message(STATUS "onnxruntime 目录: ${ONNXRUNTIME_PACKAGE_DIR}")
endif()

# 运行时文件：来自解压包时拷到输出目录；系统安装则交给系统 / vcpkg applocal
set(ONNXRUNTIME_RUNTIME_FILES "")
if(ONNXRUNTIME_PACKAGE_DIR)
    if(WIN32)
        file(GLOB ONNXRUNTIME_RUNTIME_FILES "${ONNXRUNTIME_PACKAGE_DIR}/lib/onnxruntime*.dll")
    else()
        file(GLOB ONNXRUNTIME_RUNTIME_FILES "${ONNXRUNTIME_PACKAGE_DIR}/lib/libonnxruntime.*")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenCV：vcpkg / 系统包 / Homebrew 自动查找
# ---------------------------------------------------------------------------
find_package(OpenCV CONFIG QUIET)
if(NOT OpenCV_FOUND)
    set(_opencv_hints "")
    if(DEFINED ENV{VCPKG_ROOT} AND DEFINED VCPKG_TARGET_TRIPLET)
        list(APPEND _opencv_hints "$ENV{VCPKG_ROOT}/installed/${VCPKG_TARGET_TRIPLET}")
    endif()
    if(APPLE)
        find_program(_brew_executable brew)
        if(_brew_executable)
            execute_process(COMMAND "${_brew_executable}" --prefix opencv
                OUTPUT_VARIABLE _brew_opencv_prefix
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(_brew_opencv_prefix)
                list(APPEND _opencv_hints "${_brew_opencv_prefix}")
            endif()
        endif()
        list(APPEND _opencv_hints "/opt/homebrew/opt/opencv" "/usr/local/opt/opencv")
    endif()
    find_package(OpenCV CONFIG QUIET HINTS ${_opencv_hints})
endif()
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "未找到 OpenCV。Windows 可用 vcpkg install opencv4；"
        "Ubuntu/Debian 可 apt install libopencv-dev；macOS 可 brew install opencv；"
        "也可通过 -DOpenCV_DIR=<OpenCVConfig.cmake 所在目录> 指定。")
endif()
message(STATUS "OpenCV: ${OpenCV_DIR}")

# ---------------------------------------------------------------------------
# zlib：解压自动下载的模型压缩包（OpenCV 已依赖，通常无需额外安装）
# ---------------------------------------------------------------------------
find_package(ZLIB REQUIRED)
message(STATUS "zlib: ${ZLIB_VERSION_STRING}")

# ---------------------------------------------------------------------------
# 目标
# ---------------------------------------------------------------------------
add_library(cnocr SHARED
    cnocr.h cnocr.cpp
    cnstd.h cnstd.cpp
    modle.h modle.cpp
    det_box.h
    db_postprocess.h db_postprocess.cpp
    image_io.h image_io.cpp
    model_files.h model_files.cpp
    model_registry.h model_registry.cpp
)
target_include_directories(cnocr PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(cnocr PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    ZLIB::ZLIB
)

add_executable(UseAiForWinML main.cpp)
target_link_libraries(UseAiForWinML PRIVATE cnocr ${OpenCV_LIBS})

# 运行时依赖：onnxruntime 库拷到输出目录并设置 RPATH；
# OpenCV 的 DLL 由 vcpkg applocal 自动部署，Linux/macOS 使用系统包即可
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
