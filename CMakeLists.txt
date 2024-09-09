cmake_minimum_required(VERSION 3.14.0)
project(UseAiForWinML VERSION 0.1.0)
#仅支持release 模式
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV CONFIG REQUIRED)
include_directories(D:/Lib/vcpkg/installed/x64-windows/includ D:/Lib/onnxruntime-win-x64-1.19.0/include)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(cnocr SHARED  cnocr.h cnocr.cpp modle.h modle.cpp cnstd.h cnstd.cpp)

add_executable(UseAiForWinML main.cpp)
target_include_directories(UseAiForWinML PUBLIC D:/Lib/vcpkg/installed/x64-windows/includ D:/Lib/onnxruntime-win-x64-1.19.0/include D:/Lib/vcpkg/installed/x64-windows/include)
target_link_directories(cnocr PUBLIC D:/Lib/onnxruntime-win-x64-1.19.0/lib)

target_link_libraries(cnocr PRIVATE ${OpenCV_LIBS} onnxruntime )
target_include_directories(cnocr
    PUBLIC 
        ${PROJECT_SOURCE_DIR}
)
target_link_libraries(UseAiForWinML PUBLIC cnocr ${OpenCV_LIBS} onnxruntime)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
