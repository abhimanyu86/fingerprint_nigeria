// fp_test.cpp — Standalone CLI test for the fingerprint C++ core
//
// Usage (human-readable):
//   fp_test <finger.jpg>
//   fp_test <finger1.jpg> <finger2.jpg>
//
// Usage (JSON — for the web UI):
//   fp_test --json <finger.jpg>
//   fp_test --json <finger1.jpg> <finger2.jpg>
//
// Build:  cmake -B build && cmake --build build --config Release
// Run:    ./build/Release/fp_test.exe finger.jpg

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include "fingerprint_core.h"

// ---------------------------------------------------------------------------
// Helper: read a file into bytes
// ---------------------------------------------------------------------------
static std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[ERROR] Cannot open file: " << path << "\n";
        return {};
    }
    return {std::istreambuf_iterator<char>(f), {}};
}

static std::string escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

static std::string decisionStr(fingerprint::QualityDecision d) {
    switch (d) {
        case fingerprint::QualityDecision::ACCEPT: return "ACCEPT";
        case fingerprint::QualityDecision::RETRY:  return "RETRY";
        case fingerprint::QualityDecision::REJECT: return "REJECT";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// JSON output — one CaptureResult node
// ---------------------------------------------------------------------------
static void writeResultJson(std::ostream& o, const fingerprint::CaptureResult& r) {
    auto b = [](bool v) -> const char* { return v ? "true" : "false"; };
    o << std::fixed << std::setprecision(4);

    o << "{\n";
    o << "  \"errorMessage\": \""  << escapeJson(r.errorMessage) << "\",\n";

    o << "  \"quality\": {\n";
    o << "    \"compositeScore\":    " << r.quality.compositeScore    << ",\n";
    o << "    \"blurScore\":         " << r.quality.blurScore         << ",\n";
    o << "    \"contrastScore\":     " << r.quality.contrastScore     << ",\n";
    o << "    \"ridgeClarityScore\": " << r.quality.ridgeClarityScore << ",\n";
    o << "    \"coverageScore\":     " << r.quality.coverageScore     << ",\n";
    o << "    \"orientationScore\":  " << r.quality.orientationScore  << ",\n";
    o << "    \"decision\":          \"" << decisionStr(r.quality.decision) << "\",\n";
    o << "    \"guidance\":          \"" << escapeJson(r.quality.guidance) << "\"\n";
    o << "  },\n";

    o << "  \"liveness\": {\n";
    o << "    \"confidence\":    " << r.liveness.confidence              << ",\n";
    o << "    \"isLive\":        " << b(r.liveness.isLive)               << ",\n";
    o << "    \"glareDetected\": " << b(r.liveness.glareDetected)        << ",\n";
    o << "    \"textureScore\":  " << r.liveness.textureScore            << ",\n";
    o << "    \"skinScore\":     " << r.liveness.skinScore               << "\n";
    o << "  },\n";

    o << "  \"template\": {\n";
    o << "    \"success\":       " << b(r.templ.success)                 << ",\n";
    o << "    \"minutiaeCount\": " << r.templ.minutiae.size()            << "\n";
    o << "  },\n";

    o << "  \"accepted\": " << b(r.accepted) << "\n";
    o << "}";
}

// ---------------------------------------------------------------------------
// Human-readable output
// ---------------------------------------------------------------------------
static void printResult(const fingerprint::CaptureResult& r, const std::string& label) {
    auto sep = []{ std::cout << std::string(55, '-') << "\n"; };

    sep();
    std::cout << "  RESULT: " << label << "\n";
    sep();

    if (!r.errorMessage.empty()) {
        std::cout << "  [PIPELINE ERROR] " << r.errorMessage << "\n";
        sep();
        return;
    }

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "  QUALITY\n";
    std::cout << "    compositeScore    : " << r.quality.compositeScore    << " / 100\n";
    std::cout << "    blurScore         : " << r.quality.blurScore         << "\n";
    std::cout << "    contrastScore     : " << r.quality.contrastScore     << "\n";
    std::cout << "    ridgeClarityScore : " << r.quality.ridgeClarityScore << "\n";
    std::cout << "    coverageScore     : " << r.quality.coverageScore     << "\n";
    std::cout << "    orientationScore  : " << r.quality.orientationScore  << "\n";
    std::cout << "    decision          : " << decisionStr(r.quality.decision) << "\n";
    std::cout << "    guidance          : \"" << r.quality.guidance << "\"\n";

    std::cout << "\n  LIVENESS\n";
    std::cout << "    confidence        : " << r.liveness.confidence   << " (threshold 0.50)\n";
    std::cout << "    isLive            : " << (r.liveness.isLive        ? "YES" : "NO") << "\n";
    std::cout << "    glareDetected     : " << (r.liveness.glareDetected ? "YES" : "NO") << "\n";
    std::cout << "    textureScore      : " << r.liveness.textureScore << "\n";
    std::cout << "    skinScore         : " << r.liveness.skinScore    << "\n";

    std::cout << "\n  TEMPLATE\n";
    std::cout << "    templateSuccess   : " << (r.templ.success ? "YES" : "NO") << "\n";
    if (r.templ.success) {
        std::cout << "    minutiaeCount     : " << r.templ.minutiae.size() << "\n";
        const std::string& b64 = r.templ.base64Template;
        std::cout << "    base64 (first 64) : "
                  << b64.substr(0, std::min<size_t>(64, b64.size())) << "...\n";
    }

    std::cout << "\n  OVERALL\n";
    std::cout << "    accepted          : " << (r.accepted ? "YES" : "NO") << "\n";
    sep();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Parse --json flag
    bool jsonMode = false;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--json") jsonMode = true;
        else                 files.push_back(arg);
    }

    if (files.empty()) {
        std::cerr << "Usage:\n"
                  << "  fp_test [--json] <finger.jpg>\n"
                  << "  fp_test [--json] <finger1.jpg> <finger2.jpg>  (also runs match)\n";
        return 1;
    }

    // --- Process image 1 ---
    auto bytes1 = readFile(files[0]);
    if (bytes1.empty()) {
        if (jsonMode) std::cout << "{\"errorMessage\":\"Cannot open file: " << files[0] << "\"}\n";
        return 1;
    }
    auto result1 = fingerprint::processFingerImage(bytes1);

    // --- Single image mode ---
    if (files.size() == 1) {
        if (jsonMode) {
            writeResultJson(std::cout, result1);
            std::cout << "\n";
        } else {
            printResult(result1, files[0]);
        }
        return 0;
    }

    // --- Two-image mode: process + match ---
    auto bytes2 = readFile(files[1]);
    if (bytes2.empty()) {
        if (jsonMode) std::cout << "{\"errorMessage\":\"Cannot open file: " << files[1] << "\"}\n";
        return 1;
    }
    auto result2 = fingerprint::processFingerImage(bytes2);

    if (jsonMode) {
        bool canMatch = result1.templ.success && result2.templ.success;
        float matchScore = 0.0f;
        if (canMatch) {
            fingerprint::MatchResult m = fingerprint::Matcher::match(
                result1.templ.base64Template, result2.templ.base64Template);
            matchScore = m.score;
        }

        std::cout << "{\n";
        std::cout << "  \"image1\": ";
        writeResultJson(std::cout, result1);
        std::cout << ",\n";
        std::cout << "  \"image2\": ";
        writeResultJson(std::cout, result2);
        std::cout << ",\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  \"match\": {\n";
        std::cout << "    \"possible\": " << (canMatch ? "true" : "false") << ",\n";
        std::cout << "    \"score\":    " << matchScore << ",\n";
        std::cout << "    \"isMatch\": " << (matchScore >= 0.35f ? "true" : "false") << "\n";
        std::cout << "  }\n";
        std::cout << "}\n";
    } else {
        printResult(result1, files[0]);
        printResult(result2, files[1]);

        std::cout << "\n  MATCH RESULT\n";
        std::cout << std::string(55, '-') << "\n";
        if (result1.templ.success && result2.templ.success) {
            fingerprint::MatchResult m = fingerprint::Matcher::match(
                result1.templ.base64Template, result2.templ.base64Template);
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "    score   : " << m.score << "  (threshold 0.35)\n";
            std::cout << "    matched : " << (m.score >= 0.35f ? "YES" : "NO") << "\n";
        } else {
            std::cout << "    [SKIP] One or both templates failed — cannot match.\n";
        }
        std::cout << std::string(55, '-') << "\n";
    }

    return 0;
}
