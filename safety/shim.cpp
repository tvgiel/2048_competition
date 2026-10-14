// shim.cpp - trusted harness code, compiled TOGETHER with a participant's agent_<name>.cpp.
// It owns main() and speaks the JSON-lines protocol; the agent only provides:
//
//     void train_and_instantiate();
//     int  act(const int state[4][4], const std::vector<int>& valid_actions);
//
// Keep your trained model in global/static variables (there is no agent object).
// stdout is re-pointed to stderr and stdin to /dev/null, so printing from the
// agent cannot corrupt the protocol.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

void train_and_instantiate();
int act(const int (*state)[4], const std::vector<int>& valid_actions);

static FILE* OUT = nullptr;

static void send_line(const std::string& s) {
    fputs(s.c_str(), OUT);
    fputc('\n', OUT);
    fflush(OUT);
}

static void send_error(const std::string& what) {
    std::string esc = "{\"error\":\"";
    size_t start = what.size() > 800 ? what.size() - 800 : 0;
    for (size_t i = start; i < what.size(); ++i) {
        char c = what[i];
        if (c == '"' || c == '\\') { esc += '\\'; esc += c; }
        else if (c == '\n') esc += "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) esc += ' ';
        else esc += c;
    }
    esc += "\"}";
    send_line(esc);
}

// Extract all (non-negative) integers from line[from, to).
static std::vector<int> ints_in(const std::string& line, size_t from, size_t to) {
    std::vector<int> v;
    size_t i = from;
    while (i < to) {
        if (line[i] >= '0' && line[i] <= '9') {
            long x = 0;
            while (i < to && line[i] >= '0' && line[i] <= '9') x = x * 10 + (line[i++] - '0');
            v.push_back(static_cast<int>(x));
        } else {
            ++i;
        }
    }
    return v;
}

int main() {
    OUT = fdopen(dup(1), "w");
    FILE* in = fdopen(dup(0), "r");
    dup2(2, 1);
    int devnull = open("/dev/null", O_RDWR);
    dup2(devnull, 0);

    try {
        train_and_instantiate();
    } catch (const std::exception& e) {
        send_error(std::string("train_and_instantiate threw: ") + e.what());
        return 3;
    } catch (...) {
        send_error("train_and_instantiate threw an unknown exception");
        return 3;
    }
    send_line("{\"ready\":true}");

    char* buf = nullptr;
    size_t cap = 0;
    while (getline(&buf, &cap, in) != -1) {
        std::string line(buf);
        // Harness format: {"state": [[..],[..],[..],[..]], "valid": [..]}
        size_t p = line.find("\"valid\"");
        if (p == std::string::npos) { send_error("malformed request"); continue; }
        std::vector<int> flat = ints_in(line, 0, p);
        std::vector<int> valid = ints_in(line, p + 7, line.size());
        if (flat.size() != 16) { send_error("malformed state"); continue; }
        int state[4][4];
        for (int i = 0; i < 16; ++i) state[i / 4][i % 4] = flat[i];
        try {
            int a = act(state, valid);
            send_line("{\"action\":" + std::to_string(a) + "}");
        } catch (const std::exception& e) {
            send_error(std::string("act threw: ") + e.what());
        } catch (...) {
            send_error("act threw an unknown exception");
        }
    }
    return 0;
}
