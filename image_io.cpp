#include "image_io.h"

#include <fstream>
#include <iterator>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace imageio {

cv::Mat imread_utf8(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return cv::Mat();
    }
    const std::vector<uchar> bytes((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        return cv::Mat();
    }
    return cv::imdecode(bytes, cv::IMREAD_COLOR);
}

}  // namespace imageio
