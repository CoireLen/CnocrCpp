#ifndef CNOCR_TEXT_UTILS_H
#define CNOCR_TEXT_UTILS_H

#include <string>

namespace textutils {

std::wstring utf8_to_wstring(const std::string& text);
std::string wstring_to_utf8(const std::wstring& text);

}  // namespace textutils

#endif  // CNOCR_TEXT_UTILS_H
